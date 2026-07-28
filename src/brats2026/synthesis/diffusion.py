"""Synthesis Stage 2 — a mask-conditioned latent DDPM that inpaints real tumours (from scratch).

Stage 2 of the DiffTumor track is a **denoising diffusion** model that operates inside the frozen
Stage-1 AutoencoderKL latent (:mod:`brats2026.synthesis.autoencoder`). It is trained as a
**conditional inpainter on REAL-tumour patches**: encode a genuine tumour-bearing patch to its
latent, **hole** the tumour region of that latent (keep the surrounding healthy context and the
Stage-3 mask as conditioning), and learn to denoise the held-out real tumour latent back into the
hole. At generation time the same model paints a *synthetic* tumour into a *healthy* latent, guided
by a :mod:`brats2026.synthesis.masks` concentric mask — appearance is learned only from real GoAT
tumours, geometry comes from the sampled mask. The patch is aligned to the AE / nnU-Net plan patch.

Rule-compliant by construction: trained **from scratch on GoAT data only** — no external/prior-BraTS
weights (see :mod:`brats2026.compliance`).

Like :mod:`brats2026.synthesis.autoencoder` this module invents **no** numbers: every tunable is a
``# SPECIALIST:`` hook on :class:`LatentDiffusionConfig` defaulting to ``None`` (reference ranges in
comments only). The pure schedule / noising helpers (β schedules, ᾱ, q-sample coefficients, latent
holing) are numpy/stdlib only — imported *inside* the functions — so the module and its tests load
and run without torch / MONAI. The torch :class:`GoATLatentDiffusion` wrapper is defined only when
torch is importable, mirroring :class:`brats2026.synthesis.autoencoder.GoATAutoencoderKL`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# The knobs the specialist owns. Names mirror LatentDiffusionConfig fields below.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "num_train_timesteps",
    "beta_schedule",
    "beta_start",
    "beta_end",
    "cosine_s",
    "unet_channels",
    "unet_attention_levels",
    "num_res_blocks",
    "patch_size",
    "initial_lr",
    "weight_decay",
    "warmup_epochs",
    "num_epochs",
    "batch_size",
    "seed",
)

# Supported β schedules (the *mechanism*; the choice is the ``beta_schedule`` hook).
BETA_SCHEDULES: tuple[str, ...] = ("linear", "cosine")


@dataclass
class LatentDiffusionConfig:
    """Stage-2 latent-DDPM hyperparameters — all ``# SPECIALIST:`` hooks.

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied. ``patch_size`` should match
    the Stage-1 AE / nnU-Net plan patch so the latent geometry lines up. ``beta_schedule`` selects
    one of :data:`BETA_SCHEDULES`; ``beta_start`` / ``beta_end`` govern the linear schedule and
    ``cosine_s`` the cosine schedule.
    """

    num_train_timesteps: Optional[int] = None            # SPECIALIST: diffusion steps T, ~1000
    beta_schedule: Optional[str] = None                  # SPECIALIST: linear|cosine (see BETA_SCHEDULES)
    beta_start: Optional[float] = None                   # SPECIALIST: linear-schedule β_0, ~1e-4
    beta_end: Optional[float] = None                     # SPECIALIST: linear-schedule β_T, ~2e-2
    cosine_s: Optional[float] = None                     # SPECIALIST: cosine offset s (Nichol & Dhariwal), ~8e-3
    unet_channels: Optional[tuple[int, ...]] = None      # SPECIALIST: DiffusionModelUNet widths per level, e.g. (128, 256, 256)
    unet_attention_levels: Optional[tuple[bool, ...]] = None  # SPECIALIST: self-attention per level (parallel to unet_channels)
    num_res_blocks: Optional[int] = None                 # SPECIALIST: residual blocks per level, ~1–3
    patch_size: Optional[tuple[int, int, int]] = None    # SPECIALIST: 3D patch, aligned to the AE / plan patch (e.g. 128³)
    initial_lr: Optional[float] = None                   # SPECIALIST: diffusion LR (offline), ~1e-4
    weight_decay: Optional[float] = None                 # SPECIALIST: diffusion weight decay
    warmup_epochs: Optional[int] = None                  # SPECIALIST: LR warm-up epochs
    num_epochs: Optional[int] = None                     # SPECIALIST: diffusion training epochs (offline budget)
    batch_size: Optional[int] = None                     # SPECIALIST: watch training-GPU memory (latent patches)
    seed: Optional[int] = None                           # SPECIALIST: RNG seed (noise + provenance)


def unset_hooks(config: LatentDiffusionConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided)."""
    return [f.name for f in fields(config) if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None]


def assert_configured(config: LatentDiffusionConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before launching a Stage-2 diffusion run."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to launch Stage-2 latent-diffusion training: the AI specialist must set "
            f"these # SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- noise schedule (pure numpy; imported inside so the module loads in a bare env) --------

def linear_beta_schedule(num_timesteps, beta_start, beta_end):
    """Linear β schedule: ``num_timesteps`` values evenly spaced from ``beta_start`` to ``beta_end``.

    The classic DDPM (Ho et al.) variance schedule. Returns a 1-D array of length ``num_timesteps``
    with ``betas[0] == beta_start`` and ``betas[-1] == beta_end``.
    """
    import numpy as np

    return np.linspace(beta_start, beta_end, num_timesteps, dtype=float)


def cosine_beta_schedule(num_timesteps, s):
    """Cosine β schedule (Nichol & Dhariwal, *Improved DDPM*), clipped to ``[0, 0.999]``.

    Derives ᾱ from a shifted cosine, then ``β_t = 1 - ᾱ_t / ᾱ_{t-1}``. Returns a 1-D array of
    length ``num_timesteps`` with values in ``(0, 0.999]``. ``s`` is the small offset that prevents
    β from being too small near ``t = 0``.
    """
    import numpy as np

    steps = num_timesteps + 1
    t = np.linspace(0, num_timesteps, steps, dtype=float) / num_timesteps
    acp = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
    acp = acp / acp[0]
    betas = 1.0 - acp[1:] / acp[:-1]
    return np.clip(betas, 0.0, 0.999)


def alphas_cumprod(betas):
    """Cumulative product ᾱ_t = ∏(1 - β) up to each step.

    With every ``β ∈ (0, 1)`` the factors are in ``(0, 1)`` so ᾱ is strictly decreasing and lies in
    ``(0, 1]``. This is the signal-retention curve the forward (q-sample) process follows.
    """
    import numpy as np

    betas = np.asarray(betas, dtype=float)
    return np.cumprod(1.0 - betas)


def q_sample_coefficients(acp, t):
    """Forward-process coefficients at step ``t``: ``(sqrt(ᾱ_t), sqrt(1 - ᾱ_t))``.

    These scale the clean latent and the added Gaussian noise in ``x_t = sqrt(ᾱ_t) x_0 +
    sqrt(1 - ᾱ_t) ε``. At ``t = 0`` with ``ᾱ_0 ≈ 1`` this is ``≈ (1, 0)`` — almost no noise yet.
    """
    import numpy as np

    acp = np.asarray(acp, dtype=float)
    acp_t = float(acp[t])
    return float(np.sqrt(acp_t)), float(np.sqrt(1.0 - acp_t))


def hole_latent(latent, mask):
    """Zero the masked (tumour) region of a latent, leaving the healthy context intact.

    ``latent`` is a ``(C, *spatial)`` array; ``mask`` is a boolean spatial array (broadcast across
    channels) marking the region to hole. Returns a **copy** with the masked locations set to 0 on
    every channel; the input is never mutated. This is the conditional-inpainting hole the diffusion
    model learns to fill from the surrounding context + the conditioning mask.
    """
    import numpy as np

    out = np.array(latent, dtype=float, copy=True)
    m = np.broadcast_to(np.asarray(mask, dtype=bool), out.shape)
    out[m] = 0.0
    return out


def load_diffusion_config(path: str | Path) -> LatentDiffusionConfig:
    """Build a :class:`LatentDiffusionConfig` from ``configs/synthesis/diffusion.yaml``.

    Reads the SPECIALIST hook values from the YAML. The sequence hooks ``unet_channels``,
    ``unet_attention_levels`` and ``patch_size`` are coerced from YAML lists to tuples.
    :func:`assert_configured` still governs whether the result may launch a run.
    """
    from ..config import load_config

    cfg = load_config(Path(path))
    kwargs: dict = {}
    tuple_hooks = {"unet_channels", "unet_attention_levels", "patch_size"}
    for hook in SPECIALIST_HOOKS:
        if hook in cfg and cfg[hook] is not None:
            value = cfg[hook]
            if hook in tuple_hooks and isinstance(value, list):
                value = tuple(value)
            kwargs[hook] = value
    return LatentDiffusionConfig(**kwargs)


# --- torch latent-diffusion module (only when torch is available) --------------------------
try:  # pragma: no cover - exercised only in a full training environment
    import torch
    from torch import nn

    class GoATLatentDiffusion(nn.Module):
        """Stage-2 mask-conditioned latent DDPM wrapper (scaffold).

        Composition — numbers from a configured :class:`LatentDiffusionConfig`, architecture from
        the injected MONAI ``unet`` (``DiffusionModelUNet``), ``scheduler`` (``DDPMScheduler`` built
        from the β-schedule hooks) and the **frozen** Stage-1 ``autoencoder``::

            real patch -> frozen AE.encode -> latent -> hole_latent(tumour) + mask/context
                       -> DDPM(unet) denoise the held-out real tumour latent in the hole

        This scaffold fixes the structure and the config contract; the tensor-level forward and
        train step are wired in the training env (mirroring
        :class:`brats2026.synthesis.autoencoder.GoATAutoencoderKL`).
        """

        def __init__(
            self,
            config: LatentDiffusionConfig,
            unet: "nn.Module",
            scheduler: "object",
            autoencoder: "nn.Module",
        ):
            super().__init__()
            assert_configured(config)
            self.config = config
            self.unet = unet                # SPECIALIST: MONAI DiffusionModelUNet (unet_channels / attention_levels / num_res_blocks)
            self.scheduler = scheduler      # SPECIALIST: DDPMScheduler (num_train_timesteps / beta_schedule / beta_start|end / cosine_s)
            self.autoencoder = autoencoder  # FROZEN Stage-1 AE (encode/decode; no grad)

        def forward(self, x):  # noqa: D401
            # SPECIALIST wiring (training env): encode the REAL tumour patch via the frozen AE ->
            # latent; hole_latent the tumour region (keep healthy context + mask conditioning);
            # q_sample noise at t using q_sample_coefficients(alphas_cumprod(betas), t); unet
            # predicts the noise; MSE on the held-out tumour latent. The pure helpers above define
            # the schedule / holing mechanism.
            raise NotImplementedError(
                "GoATLatentDiffusion.forward is a scaffold: nnunet-trainer wires the MONAI "
                "DiffusionModelUNet + DDPMScheduler onto the frozen Stage-1 AE and the pure "
                "schedule/holing helpers in this module, and ai-specialist supplies "
                "configs/synthesis/diffusion.yaml. Do not invent the tensor ops here."
            )

        def training_step(self, batch):  # noqa: D401
            raise NotImplementedError(
                "GoATLatentDiffusion.training_step is a scaffold: the encode-via-frozen-AE -> hole "
                "-> condition on mask+context -> DDPM denoise loop is wired by nnunet-trainer/Opus "
                "in the training env. Do not invent the tensor ops here."
            )

except ImportError:  # torch not installed (e.g. CI / planning env)
    GoATLatentDiffusion = None  # type: ignore[assignment]
