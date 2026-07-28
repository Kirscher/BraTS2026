"""Synthesis Stage 1 — a 3D AutoencoderKL that compresses → reconstructs a GoAT volume.

The DiffTumor synthesis track is a **three-stage** pipeline that grows extra training data
by *painting* synthetic tumours into GoAT volumes. This module is **Stage 1 only**: a
convolutional variational autoencoder (MONAI ``AutoencoderKL``, patch-GAN + LPIPS refinement)
that learns a compact latent for a whole volume and reconstructs it. It is **frozen** after
training — Stage 2 (mask sampling) and Stage 3 (a latent diffusion model conditioned on those
masks) run *inside* this frozen latent space, and a downstream nnU-Net scores the augmented set.

Stage 1 by itself does **not** synthesise tumours: it neither invents lesions nor needs labels.
Reconstruction is a self-supervised objective, so the autoencoder trains on **all** GoAT patches
— healthy *and* tumour-bearing, every cohort — which is exactly why it can be label-free. Its
``patch_size`` is aligned to the nnU-Net plan patch so the latent geometry matches the segmenter
the whole track ultimately feeds.

Rule-compliant by construction: the autoencoder trains **from scratch on GoAT data only** — no
pretrained/external weights, no prior-BraTS checkpoints (see :mod:`brats2026.compliance`).

Like :mod:`brats2026.ssl.mae`, this module invents **no** numbers: every tunable is a
``# SPECIALIST:`` hook on :class:`AutoencoderKLConfig` defaulting to ``None`` (reference ranges
live in comments only). The two fixed fields (``spatial_dims``, ``in_channels``) are data facts,
not tunables. The pure helpers (downsample factor, latent geometry, L1 / KL losses) are
numpy/stdlib only — imported *inside* the functions — so the module and its tests load and run
without torch / MONAI. The torch :class:`GoATAutoencoderKL` module is defined only when torch is
importable, mirroring :class:`brats2026.ssl.mae.MAE3D`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# The knobs the specialist owns. Names mirror AutoencoderKLConfig fields below.
# ``spatial_dims`` / ``in_channels`` are deliberately absent: they are data facts, not tunables.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "patch_size",
    "num_downsamplings",
    "latent_channels",
    "channels",
    "num_res_blocks",
    "attention_levels",
    "norm_num_groups",
    "kl_weight",
    "perceptual_weight",
    "adversarial_weight",
    "discriminator_start_epoch",
    "initial_lr",
    "weight_decay",
    "warmup_epochs",
    "num_epochs",
    "batch_size",
    "seed",
)


@dataclass
class AutoencoderKLConfig:
    """Stage-1 AutoencoderKL hyperparameters — all ``# SPECIALIST:`` hooks (plus 2 data facts).

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied. ``patch_size`` should be
    aligned with the nnU-Net plan patch so the latent geometry matches the segmenter this track
    ultimately augments, and every axis must be divisible by the latent downsample factor
    (``2 ** num_downsamplings``). ``spatial_dims`` and ``in_channels`` are fixed GoAT data facts,
    not tunables, so they carry real defaults and are excluded from :data:`SPECIALIST_HOOKS`.
    """

    patch_size: Optional[tuple[int, int, int]] = None      # SPECIALIST: 3D patch, plan-aligned & divisible by 2**num_downsamplings (e.g. 128³)
    num_downsamplings: Optional[int] = None                # SPECIALIST: encoder down-levels; latent factor = 2**n, ~2–4
    latent_channels: Optional[int] = None                  # SPECIALIST: latent channel count, ~3–8
    channels: Optional[tuple[int, ...]] = None             # SPECIALIST: encoder widths per level, e.g. (64, 128, 256)
    num_res_blocks: Optional[int] = None                   # SPECIALIST: residual blocks per level, ~1–3
    attention_levels: Optional[tuple[bool, ...]] = None    # SPECIALIST: self-attention per level (parallel to channels), usually only deepest
    norm_num_groups: Optional[int] = None                  # SPECIALIST: GroupNorm groups; each channels[i] must be divisible by it (e.g. 32)
    kl_weight: Optional[float] = None                      # SPECIALIST: KL regulariser weight, ~1e-6 (keep latent near N(0,1) without over-smoothing)
    perceptual_weight: Optional[float] = None              # SPECIALIST: LPIPS perceptual loss weight, ~0.5–1.0
    adversarial_weight: Optional[float] = None             # SPECIALIST: patch-GAN loss weight, ~0.1–0.5
    discriminator_start_epoch: Optional[int] = None        # SPECIALIST: warm up the AE before the GAN kicks in, ~20–50
    initial_lr: Optional[float] = None                     # SPECIALIST: AE learning rate (offline), ~1e-4
    weight_decay: Optional[float] = None                   # SPECIALIST: AE weight decay, ~0
    warmup_epochs: Optional[int] = None                    # SPECIALIST: LR warm-up epochs
    num_epochs: Optional[int] = None                       # SPECIALIST: AE training epochs (offline budget)
    batch_size: Optional[int] = None                       # SPECIALIST: watch training-GPU memory (3D volumes are heavy)
    seed: Optional[int] = None                             # SPECIALIST: RNG seed (init + provenance)
    spatial_dims: int = 3                                  # data fact: volumetric MRI (NOT a hook)
    in_channels: int = 4                                   # data fact: t1n/t1c/t2f/t2w (NOT a hook)


def unset_hooks(config: AutoencoderKLConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided).

    ``spatial_dims`` / ``in_channels`` are data facts, not hooks, so they are never reported here
    even though they are dataclass fields.
    """
    return [f.name for f in fields(config) if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None]


def assert_configured(config: AutoencoderKLConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before launching a Stage-1 AE run."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to launch Stage-1 autoencoder training: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- latent geometry (pure numpy/stdlib; imported inside so the module loads in a bare env) ---

def downsample_factor(num_downsamplings: int) -> int:
    """Spatial downsample factor of the encoder: ``2 ** num_downsamplings``.

    Each encoder down-level halves every spatial axis, so ``n`` levels shrink each axis by
    ``2 ** n``. Raises if ``num_downsamplings`` is negative (a configuration error).
    """
    if num_downsamplings < 0:
        raise ValueError(f"num_downsamplings must be >= 0, got {num_downsamplings}")
    return 2 ** num_downsamplings


def latent_spatial_shape(
    volume_spatial: tuple[int, int, int], factor: int
) -> tuple[int, int, int]:
    """Spatial shape of the latent: each axis ``// factor``.

    ``volume_spatial`` is the spatial ``(D, H, W)`` (channels excluded). Raises if any axis is
    not an exact multiple of ``factor`` — the convolutional encoder assumes a clean division, so
    a non-divisible shape is a configuration error to surface, not to pad silently.
    """
    if len(volume_spatial) != 3:
        raise ValueError("volume_spatial must be a 3-tuple (D, H, W)")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    latent = []
    for axis, dim in enumerate(volume_spatial):
        if dim % factor != 0:
            raise ValueError(
                f"volume axis {axis} of size {dim} is not divisible by the latent factor {factor}; "
                "pad/crop the volume to a factor-multiple before encoding"
            )
        latent.append(dim // factor)
    return (latent[0], latent[1], latent[2])


def assert_patch_divisible(patch_size: tuple[int, int, int], factor: int) -> None:
    """Raise if any ``patch_size`` axis is not divisible by the latent downsample ``factor``.

    The plan-aligned patch must tile cleanly into the latent grid; a stray axis would leave the
    autoencoder unable to reconstruct at the patch edge. A configuration guard, not a silent crop.
    """
    if len(patch_size) != 3:
        raise ValueError("patch_size must be a 3-tuple (D, H, W)")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    for axis, dim in enumerate(patch_size):
        if dim % factor != 0:
            raise ValueError(
                f"patch_size axis {axis} of size {dim} is not divisible by the latent factor "
                f"{factor}; choose a plan-aligned patch that is a multiple of 2**num_downsamplings"
            )


def num_latent_elements(latent_channels: int, latent_spatial: tuple[int, int, int]) -> int:
    """Total elements in the latent tensor: ``latent_channels * prod(latent_spatial)``."""
    import numpy as np

    return int(latent_channels * np.prod(latent_spatial))


def latent_shape_from_config(
    config: AutoencoderKLConfig, volume_spatial: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    """Full latent shape ``(latent_channels, *latent_spatial)`` for ``volume_spatial``.

    Combines :func:`downsample_factor` (from ``num_downsamplings``) and :func:`latent_spatial_shape`
    with the configured ``latent_channels``. Raises a clear error if either hook is still ``None``,
    so callers never derive a latent shape from an undecided config.
    """
    if config.num_downsamplings is None:
        raise ValueError("latent_shape_from_config needs num_downsamplings set (still None)")
    if config.latent_channels is None:
        raise ValueError("latent_shape_from_config needs latent_channels set (still None)")
    factor = downsample_factor(config.num_downsamplings)
    spatial = latent_spatial_shape(volume_spatial, factor)
    return (config.latent_channels, spatial[0], spatial[1], spatial[2])


# --- reconstruction / KL objectives (pure numpy; CPU-testable end-to-end) ------------------

def reconstruction_l1(pred, target) -> float:
    """Mean absolute error between predicted and target voxels (the AE reconstruction term).

    ``pred`` / ``target`` are any broadcastable arrays (e.g. ``(C, D, H, W)``). L1 is the standard
    pixel reconstruction loss for AutoencoderKL — less blur-prone than L2 — before the perceptual
    (LPIPS) and adversarial (patch-GAN) terms are added. Returns a plain float.
    """
    import numpy as np

    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.abs(pred - target).mean())


def kl_divergence_standard_normal(mu, logvar) -> float:
    """KL divergence of the encoder posterior ``N(mu, exp(logvar))`` from ``N(0, 1)``.

    Computes ``0.5 * mean(mu**2 + exp(logvar) - 1 - logvar)`` — the closed-form KL between two
    diagonal Gaussians when the prior is the unit normal. This is the variational regulariser that
    keeps the latent close to the standard normal Stage-3 diffusion samples from; its weight is the
    ``kl_weight`` SPECIALIST hook. Returns ``0.0`` at ``mu=0, logvar=0`` (the posterior IS the prior).
    """
    import numpy as np

    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    return float(0.5 * np.mean(mu ** 2 + np.exp(logvar) - 1.0 - logvar))


def load_ae_config(path: str | Path) -> AutoencoderKLConfig:
    """Build an :class:`AutoencoderKLConfig` from a ``configs/synthesis/autoencoder.yaml`` file.

    Reads the SPECIALIST hook values from the YAML and returns a config object. The sequence hooks
    ``patch_size``, ``channels`` and ``attention_levels`` are coerced from YAML lists to tuples.
    :func:`assert_configured` still governs whether the result may launch a run.
    """
    from ..config import load_config

    cfg = load_config(Path(path))
    kwargs: dict = {}
    tuple_hooks = {"patch_size", "channels", "attention_levels"}
    for hook in SPECIALIST_HOOKS:
        if hook in cfg and cfg[hook] is not None:
            value = cfg[hook]
            if hook in tuple_hooks and isinstance(value, list):
                value = tuple(value)
            kwargs[hook] = value
    return AutoencoderKLConfig(**kwargs)


# --- torch AutoencoderKL module (only when torch is available) -----------------------------
try:  # pragma: no cover - exercised only in a full training environment
    import torch
    from torch import nn

    class GoATAutoencoderKL(nn.Module):
        """Stage-1 AutoencoderKL wrapper (scaffold).

        Composition — numbers from a configured :class:`AutoencoderKLConfig`, architecture from the
        injected MONAI ``autoencoder`` (the ``AutoencoderKL``) and ``discriminator`` (a patch-GAN),
        both built by ``nnunet-trainer`` + ``ai-specialist``::

            volume -> autoencoder.encode -> latent (KL to N(0,1)) -> decode -> reconstruction
            loss = L1 + kl_weight*KL + perceptual_weight*LPIPS + adversarial_weight*patchGAN

        This scaffold fixes the structure and the config contract; the tensor-level forward and
        train step are wired in the training env (mirroring :class:`brats2026.ssl.mae.MAE3D`).
        """

        def __init__(
            self,
            config: AutoencoderKLConfig,
            autoencoder: "nn.Module",
            discriminator: "nn.Module",
        ):
            super().__init__()
            assert_configured(config)
            self.config = config
            self.autoencoder = autoencoder      # SPECIALIST: MONAI AutoencoderKL (channels / latent_channels / attention_levels)
            self.discriminator = discriminator  # SPECIALIST: patch-GAN (adversarial_weight, discriminator_start_epoch)

        def forward(self, x):  # noqa: D401
            # SPECIALIST wiring (training env): autoencoder.encode x -> (mu, logvar); sample latent;
            # decode -> reconstruction; combine reconstruction_l1 + kl_divergence_standard_normal +
            # LPIPS + patch-GAN per the loss-weight hooks. The pure helpers above define the terms.
            raise NotImplementedError(
                "GoATAutoencoderKL.forward is a scaffold: nnunet-trainer wires the MONAI "
                "AutoencoderKL + patch-GAN discriminator onto the pure latent/loss helpers in this "
                "module, and ai-specialist supplies configs/synthesis/autoencoder.yaml. Do not "
                "invent the tensor ops here."
            )

        def training_step(self, batch):  # noqa: D401
            raise NotImplementedError(
                "GoATAutoencoderKL.training_step is a scaffold: the generator/discriminator "
                "alternation (discriminator_start_epoch) is wired by nnunet-trainer/Opus in the "
                "training env. Do not invent the tensor ops here."
            )

except ImportError:  # torch not installed (e.g. CI / planning env)
    GoATAutoencoderKL = None  # type: ignore[assignment]
