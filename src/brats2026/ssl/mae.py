"""Phase 4b — MAE self-supervised pre-training on GoAT's own volumes (label-free).

A **Masked Autoencoder** pre-trains the *same* ResEnc encoder that
:class:`brats2026.nnunet.trainer.nnUNetTrainerGoAT` uses: a case is patchified, a large
fraction of 3D patches is masked, the encoder sees only the *visible* patches, a light
throwaway decoder reconstructs the hidden voxels, and the loss is MSE on the **masked** patches
only. The learned encoder weights then **warm-start** the GoAT segmentation trainer instead of a
random init — cohort-agnostic anatomy / contrast priors that lift the LODO *worst-cohort* score.

Rule-compliant by construction: pre-training uses ONLY GoAT-provided volumes (labelled +
unlabelled, all 5 cohorts) and trains **from scratch** — importing any external / prior-BraTS
encoder weights would disqualify (see :mod:`brats2026.compliance`).

Like :mod:`brats2026.ssl` (self-training) and :mod:`brats2026.nnunet.trainer`, this module
invents **no** numbers: every tunable is a ``# SPECIALIST:`` hook on :class:`MAE3DConfig`
defaulting to ``None``. The pure functions (patch grid, patchify, random masking, reconstruction
loss) are numpy/stdlib only — imported *inside* the functions — so the module and its tests load
and run without torch / nnU-Net. The torch :class:`MAE3D` module is defined only when torch is
importable, mirroring ``nnUNetTrainerGoAT``'s guarded ``nnunetv2`` import.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

# The knobs the specialist owns. Names mirror MAE3DConfig fields below.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "mask_ratio",
    "patch_size",
    "decoder_depth",
    "decoder_embed_dim",
    "norm_pix_loss",
    "initial_lr",
    "weight_decay",
    "warmup_epochs",
    "num_epochs",
    "batch_size",
    "seed",
)


@dataclass
class MAE3DConfig:
    """MAE pre-training hyperparameters — all ``# SPECIALIST:`` hooks.

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied. ``patch_size`` should be
    aligned with the nnU-Net plan patch/stride so the pre-trained encoder transfers cleanly.
    """

    mask_ratio: Optional[float] = None                 # SPECIALIST: fraction of patches hidden, ~0.6–0.75
    patch_size: Optional[tuple[int, int, int]] = None  # SPECIALIST: 3D token size, plan-aligned (e.g. 16³)
    decoder_depth: Optional[int] = None                # SPECIALIST: light decoder blocks, ~2–4
    decoder_embed_dim: Optional[int] = None            # SPECIALIST: decoder width (< encoder width)
    norm_pix_loss: Optional[bool] = None               # SPECIALIST: per-patch target normalisation (He et al.)
    initial_lr: Optional[float] = None                 # SPECIALIST: pre-train LR (cosine, offline)
    weight_decay: Optional[float] = None               # SPECIALIST: pre-train weight decay
    warmup_epochs: Optional[int] = None                # SPECIALIST: LR warm-up epochs
    num_epochs: Optional[int] = None                   # SPECIALIST: pre-train epochs (offline budget)
    batch_size: Optional[int] = None                   # SPECIALIST: watch training-GPU memory (offline)
    seed: Optional[int] = None                         # SPECIALIST: RNG seed (masking + provenance)


def unset_hooks(config: MAE3DConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided)."""
    return [f.name for f in fields(config) if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None]


def assert_configured(config: MAE3DConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before launching an MAE pre-train run."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to launch MAE pre-training: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- patch geometry (pure numpy; imported inside so the module loads in a bare env) -------

def patch_grid_shape(
    volume_shape: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Number of non-overlapping patches per spatial axis.

    ``volume_shape`` is the spatial ``(D, H, W)`` (channels excluded). Raises if any axis is not
    an exact multiple of the corresponding ``patch_size`` — MAE tokenisation assumes a clean
    grid, so a non-divisible shape is a configuration error to surface, not to pad silently.
    """
    if len(volume_shape) != 3 or len(patch_size) != 3:
        raise ValueError("volume_shape and patch_size must both be 3-tuples (D, H, W)")
    grid = []
    for axis, (dim, patch) in enumerate(zip(volume_shape, patch_size)):
        if patch <= 0:
            raise ValueError(f"patch_size axis {axis} must be positive, got {patch}")
        if dim % patch != 0:
            raise ValueError(
                f"volume axis {axis} of size {dim} is not divisible by patch_size {patch}; "
                "pad/crop the volume to a patch-multiple before MAE tokenisation"
            )
        grid.append(dim // patch)
    return (grid[0], grid[1], grid[2])


def num_patches(volume_shape: tuple[int, int, int], patch_size: tuple[int, int, int]) -> int:
    """Total number of patches (product of the per-axis grid)."""
    gd, gh, gw = patch_grid_shape(volume_shape, patch_size)
    return gd * gh * gw


def num_masked(n_patches: int, mask_ratio: float) -> int:
    """How many patches to hide, MAE-style (``L - int(L*(1-ratio))``).

    Deriving the masked count from the *kept* count (``int(L*(1-ratio))``) matches the reference
    MAE and guarantees ``0 <= n_masked <= n_patches``. ``mask_ratio`` is a SPECIALIST hook; this
    only defines the count rule.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    n_visible = int(n_patches * (1.0 - mask_ratio))
    return n_patches - n_visible


def random_masking(n_patches: int, mask_ratio: float, seed: int):
    """Deterministic random patch mask given a seed.

    Returns ``(mask, visible_idx, masked_idx)`` where ``mask`` is a boolean array of length
    ``n_patches`` (``True`` = *hidden*), and the two index arrays are ascending. Determinism is
    keyed on ``seed`` so the same masking is reproducible from provenance. Both ``mask_ratio`` and
    ``seed`` are SPECIALIST hooks — this only defines the sampling.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_patches)
    n_mask = num_masked(n_patches, mask_ratio)
    masked_idx = np.sort(perm[:n_mask])
    visible_idx = np.sort(perm[n_mask:])
    mask = np.zeros(n_patches, dtype=bool)
    mask[masked_idx] = True
    return mask, visible_idx, masked_idx


def patchify(volume, patch_size: tuple[int, int, int]):
    """Split a ``(C, D, H, W)`` volume into ``(n_patches, C*pd*ph*pw)`` flat patch vectors.

    Patch order is row-major over the ``(gd, gh, gw)`` grid. Round-trips exactly with
    :func:`unpatchify`. Pure numpy so masking + reconstruction loss are CPU-testable end-to-end.
    """
    import numpy as np

    volume = np.asarray(volume)
    if volume.ndim != 4:
        raise ValueError(f"volume must be 4D (C, D, H, W), got shape {volume.shape}")
    c, d, h, w = volume.shape
    gd, gh, gw = patch_grid_shape((d, h, w), patch_size)
    pd, ph, pw = patch_size
    grid = volume.reshape(c, gd, pd, gh, ph, gw, pw)
    grid = grid.transpose(1, 3, 5, 0, 2, 4, 6)  # (gd, gh, gw, C, pd, ph, pw)
    return grid.reshape(gd * gh * gw, c * pd * ph * pw)


def unpatchify(patches, volume_shape: tuple[int, int, int, int], patch_size: tuple[int, int, int]):
    """Inverse of :func:`patchify`: reassemble ``(C, D, H, W)`` from flat patch vectors."""
    import numpy as np

    patches = np.asarray(patches)
    c, d, h, w = volume_shape
    gd, gh, gw = patch_grid_shape((d, h, w), patch_size)
    pd, ph, pw = patch_size
    grid = patches.reshape(gd, gh, gw, c, pd, ph, pw)
    grid = grid.transpose(3, 0, 4, 1, 5, 2, 6)  # (C, gd, pd, gh, ph, gw, pw)
    return grid.reshape(c, d, h, w)


def reconstruction_loss(pred, target, mask, norm_pix_loss: bool = False) -> float:
    """MSE between predicted and target patch vectors, over the **masked** patches only.

    ``pred`` / ``target`` are ``(n_patches, patch_dim)``; ``mask`` is the boolean array from
    :func:`random_masking` (``True`` = masked). When ``norm_pix_loss`` is True each *target* patch
    is standardised (per-patch mean/var) before the MSE — the MAE trick that makes the objective
    reconstruct local *contrast* rather than absolute intensity (more robust across scanners /
    cohorts). Returns ``0.0`` if nothing is masked. ``norm_pix_loss`` is a SPECIALIST hook.
    """
    import numpy as np

    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    tgt = target[mask]
    prd = pred[mask]
    if norm_pix_loss:
        mean = tgt.mean(axis=1, keepdims=True)
        var = tgt.var(axis=1, keepdims=True)
        tgt = (tgt - mean) / np.sqrt(var + 1e-6)
    return float(((prd - tgt) ** 2).mean())


# --- encoder warm-start (pure dict; the bridge into segmentation) -------------------------

def select_encoder_weights(state_dict: dict, encoder_prefix: str = "encoder.") -> dict:
    """Extract the encoder sub-state-dict from a trained MAE checkpoint, stripping the prefix.

    The light decoder is discarded; only the encoder tensors (keys under ``encoder_prefix``) carry
    into ``nnUNetTrainerGoAT.initialize()`` as a warm start. Pure dict op — no torch — so the
    transfer contract is unit-testable. Returns an empty dict if no key matches (surfaced by the
    caller as "nothing to warm-start", never a silent partial load).
    """
    n = len(encoder_prefix)
    return {k[n:]: v for k, v in state_dict.items() if k.startswith(encoder_prefix)}


# --- torch MAE module (only when torch is available) --------------------------------------
try:  # pragma: no cover - exercised only in a full training environment
    import torch
    from torch import nn

    class MAE3D(nn.Module):
        """Masked-autoencoder wrapper around the shared ResEnc encoder (scaffold).

        Composition — numbers from a configured :class:`MAE3DConfig`, architecture from the
        injected ``encoder`` / ``decoder`` built by ``nnunet-trainer`` + ``ai-specialist``::

            patchify → random-mask(mask_ratio) → encoder(visible) → light decoder → MSE(masked)

        The ``encoder`` MUST be the same ResEnc-L backbone class ``nnUNetTrainerGoAT`` uses, so its
        trained weights transfer via :func:`select_encoder_weights`. This scaffold fixes the
        structure and the config contract; the tensor-level forward is wired in the training env
        (mirroring how ``nnUNetTrainerGoAT.initialize`` defers to ``# SPECIALIST:`` hooks).
        """

        def __init__(self, config: MAE3DConfig, encoder: "nn.Module", decoder: "nn.Module"):
            super().__init__()
            assert_configured(config)
            self.config = config
            self.encoder = encoder  # nnunet-trainer: the SHARED ResEnc-L backbone (weights transfer out)
            self.decoder = decoder  # SPECIALIST: light throwaway decoder (decoder_depth / decoder_embed_dim)

        def forward(self, x):  # noqa: D401
            # SPECIALIST wiring (training env): patch_embed x -> tokens; random_masking(mask_ratio,
            # seed); encode VISIBLE tokens only; decode to voxels; return reconstruction_loss over
            # MASKED patches (norm_pix_loss). The pure helpers above define this mechanism.
            raise NotImplementedError(
                "MAE3D.forward is a scaffold: nnunet-trainer wires the ResEnc encoder + light "
                "decoder onto the pure masking/loss helpers in this module, and ai-specialist "
                "supplies configs/mae.yaml. Do not invent the tensor ops here."
            )

except ImportError:  # torch not installed (e.g. CI / planning env)
    MAE3D = None  # type: ignore[assignment]
