"""Synthesis Stage 2/3 core — sample a synthetic tumour mask and place it on a healthy canvas.

The DiffTumor track paints synthetic tumours into GoAT volumes. This module is the **mask**
half: it draws a plausible **concentric NCR/ED/ET** lesion and decides *where* on a healthy
volume it may sit, so Stage-3 (:mod:`brats2026.synthesis.diffusion` +
:mod:`brats2026.synthesis.generate`) can inpaint appearance inside the frozen Stage-1 latent and
paint the *label map* returned here. Unlike the diffusion / autoencoder scaffolds this is **real,
complete logic** — pure numpy, fully unit-tested — because the geometry has no learned weights.

The lesion is a set of concentric spheres: an enhancing core (ET=3), a necrotic shell (NCR=1) and
an edema rim (ED=2), which makes the region model **nested by construction** —
``ET({3}) ⊂ TC({1,3}) ⊂ WT({1,2,3})`` — matching how the segmenter is scored (see
:mod:`brats2026.nnunet.convert`). Placement respects the brain mask, a keep-out margin around the
*real* tumour, and (optionally) the contralateral hemisphere, so a synthetic lesion never lands on
or beside genuine pathology.

Like the rest of the package this module invents **no** numbers: every tunable is a
``# SPECIALIST:`` hook on :class:`SyntheticMaskConfig` defaulting to ``None`` (reference ranges in
comments only). ``placement_mode`` is a *choice* over :data:`PLACEMENT_MODES`, not a tunable, so it
carries a safe default and is excluded from :data:`SPECIALIST_HOOKS` (mirroring how
:mod:`brats2026.ssl` treats ``case_confidence_metric``'s allowed-value set). All helpers import
numpy *inside* the function so the module and its tests load without torch / any imaging stack.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# The knobs the specialist owns. Names mirror SyntheticMaskConfig fields below.
# ``placement_mode`` is deliberately absent: it is a choice over PLACEMENT_MODES, not a tunable.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "radius_range_mm",
    "et_core_fraction",
    "ncr_fraction",
    "ed_rim_fraction",
    "irregularity",
    "n_blobs",
    "margin_mm",
    "seed",
)

# Where a synthetic lesion may be placed (the *mechanism*; the choice is a config field).
PLACEMENT_MODES: tuple[str, ...] = ("healthy_anywhere", "contralateral")

# GoAT left-right axis of a (D, H, W) volume — the default midline for contralateral placement.
DEFAULT_MIDLINE_AXIS: int = 2


@dataclass
class SyntheticMaskConfig:
    """Synthetic-tumour geometry hyperparameters — all ``# SPECIALIST:`` hooks (plus one choice).

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied. ``placement_mode`` is a
    choice over :data:`PLACEMENT_MODES` (a mechanism switch, not a tuned number), so it carries a
    safe default and is excluded from :data:`SPECIALIST_HOOKS`.
    """

    radius_range_mm: Optional[tuple[float, float]] = None  # SPECIALIST: (min, max) overall tumour radius in mm, sampled per case
    et_core_fraction: Optional[float] = None               # SPECIALIST: enhancing-core radius as a fraction of R, ~0.3–0.5
    ncr_fraction: Optional[float] = None                   # SPECIALIST: necrotic-shell thickness as a fraction of R (et+ncr <= 1)
    ed_rim_fraction: Optional[float] = None                # SPECIALIST: edema-rim thickness as a fraction of R (et+ncr+ed ~= 1)
    irregularity: Optional[float] = None                   # SPECIALIST: boundary irregularity, 0 = clean ellipsoid; ~0–1
    n_blobs: Optional[int] = None                          # SPECIALIST: number of merged sub-blobs for a multi-focal look, ~1–3
    margin_mm: Optional[float] = None                      # SPECIALIST: min distance to the real tumour / brain edge, mm
    seed: Optional[int] = None                             # SPECIALIST: RNG seed (geometry sampling + provenance)
    placement_mode: str = "healthy_anywhere"               # CHOICE over PLACEMENT_MODES (NOT a hook)


def unset_hooks(config: SyntheticMaskConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided).

    ``placement_mode`` is a choice, not a hook, so it is never reported here even though it is a
    dataclass field.
    """
    return [f.name for f in fields(config) if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None]


def assert_configured(config: SyntheticMaskConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before sampling a synthetic mask."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to sample a synthetic tumour mask: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- geometry primitives (pure numpy; imported inside so the module loads in a bare env) ---

def ellipsoid_mask(shape, center, radii):
    """Boolean ellipsoid: voxels with ``sum(((x - c) / r) ** 2) <= 1``.

    ``shape`` / ``center`` / ``radii`` are 3-tuples over ``(D, H, W)``. Anisotropic ``radii`` give
    an axis-aligned ellipsoid (per-axis half-extent), so it composes with :func:`mm_to_voxels`
    under anisotropic spacing. Raises if any radius is non-positive.
    """
    import numpy as np

    if len(shape) != 3 or len(center) != 3 or len(radii) != 3:
        raise ValueError("shape, center and radii must all be 3-tuples (D, H, W)")
    if any(r <= 0 for r in radii):
        raise ValueError(f"radii must be positive, got {radii}")
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dz = (zz - center[0]) / radii[0]
    dy = (yy - center[1]) / radii[1]
    dx = (xx - center[2]) / radii[2]
    return (dz * dz + dy * dy + dx * dx) <= 1.0


def concentric_labels(shape, center, total_radius, et_core_fraction, ncr_fraction):
    """Concentric NCR/ED/ET label map, nested by construction.

    By Euclidean voxel distance ``d`` from ``center`` and overall radius ``R = total_radius``::

        d <= et_core_fraction * R                       -> 3 (ET, enhancing core)
        d <= (et_core_fraction + ncr_fraction) * R      -> 1 (NCR, necrotic shell)
        d <= R                                          -> 2 (ED, edema rim)
        else                                            -> 0 (background)

    Painting outward-in (ED, then NCR, then ET overwrite) makes the region model nested:
    ``ET({3}) ⊂ TC({1,3}) ⊂ WT({1,2,3})`` — the invariant the whole track relies on.
    """
    import numpy as np

    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    seg = np.zeros(shape, dtype=int)
    r = float(total_radius)
    seg[d <= r] = 2                                          # ED (whole tumour)
    seg[d <= (et_core_fraction + ncr_fraction) * r] = 1      # NCR (tumour core minus ET)
    seg[d <= et_core_fraction * r] = 3                       # ET (enhancing core)
    return seg


def mm_to_voxels(radius_mm, spacing):
    """Convert a physical radius (mm) to per-axis voxel radii via ``spacing``.

    ``spacing`` is the ``(D, H, W)`` voxel size in mm. ``radius_mm`` may be a scalar (same physical
    radius on every axis) or a per-axis 3-tuple. Returns a 3-tuple of voxel radii ``r / spacing``.
    """
    import numpy as np

    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3:
        raise ValueError("spacing must be a 3-tuple (D, H, W)")
    if np.isscalar(radius_mm):
        radii = (float(radius_mm),) * 3
    else:
        radii = tuple(float(r) for r in radius_mm)
        if len(radii) != 3:
            raise ValueError("radius_mm must be a scalar or a 3-tuple")
    return (radii[0] / spacing[0], radii[1] / spacing[1], radii[2] / spacing[2])


def brain_mask_from_volume(volume):
    """Foreground (brain) mask = any nonzero voxel — headers / intensity only, never labels.

    Accepts a single-channel ``(D, H, W)`` volume or a multi-channel ``(C, D, H, W)`` volume
    (foreground = nonzero in *any* channel). Returns a boolean ``(D, H, W)`` array.
    """
    import numpy as np

    volume = np.asarray(volume)
    if volume.ndim == 4:
        return np.any(volume != 0, axis=0)
    if volume.ndim == 3:
        return volume != 0
    raise ValueError(f"volume must be 3D (D, H, W) or 4D (C, D, H, W), got shape {volume.shape}")


def contralateral_region(brain_mask, real_tumor_mask, midline_axis=DEFAULT_MIDLINE_AXIS):
    """The brain hemisphere *opposite* the real tumour's centroid along ``midline_axis``.

    The midline is the brain's own centre of mass along ``midline_axis`` (robust to an off-centre
    or cropped volume). Voxels on the far side of that midline from the tumour centroid, and inside
    the brain, form the returned mask. With no real tumour the low side is returned by convention.
    """
    import numpy as np

    brain = np.asarray(brain_mask, dtype=bool)
    tumor = np.asarray(real_tumor_mask, dtype=bool)
    coords = np.indices(brain.shape)[midline_axis]
    if brain.any():
        midline = float(coords[brain].mean())
    else:
        midline = brain.shape[midline_axis] / 2.0
    tumor_centroid = float(coords[tumor].mean()) if tumor.any() else midline
    if tumor_centroid >= midline:
        opposite = coords < midline
    else:
        opposite = coords >= midline
    return brain & opposite


def _dilate_once(mask):
    """One step of 6-connected binary dilation (no wrap-around at the volume edge)."""
    import numpy as np

    out = mask.copy()
    for axis in range(mask.ndim):
        dst = [slice(None)] * mask.ndim
        src = [slice(None)] * mask.ndim
        dst[axis] = slice(1, None)
        src[axis] = slice(0, -1)
        out[tuple(dst)] |= mask[tuple(src)]
        dst[axis] = slice(0, -1)
        src[axis] = slice(1, None)
        out[tuple(dst)] |= mask[tuple(src)]
    return out


def _dilate(mask, margin_voxels):
    """Binary dilation by ``margin_voxels`` steps (rounded up); a keep-out ball around ``mask``."""
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    steps = int(np.ceil(margin_voxels))
    out = mask.copy()
    for _ in range(max(steps, 0)):
        out = _dilate_once(out)
    return out


def healthy_candidate_mask(
    brain_mask, real_tumor_mask, margin_voxels, placement_mode, midline_axis=DEFAULT_MIDLINE_AXIS
):
    """Voxels a synthetic lesion centre may occupy: healthy tissue, clear of the real tumour.

    ``brain ∧ ¬dilate(real_tumor, margin_voxels)`` — and, when ``placement_mode`` is
    ``"contralateral"``, further intersected with :func:`contralateral_region`. ``margin_voxels``
    is the keep-out radius (in voxels) around the real tumour. Raises on an unknown
    ``placement_mode``.
    """
    import numpy as np

    if placement_mode not in PLACEMENT_MODES:
        raise ValueError(
            f"unknown placement_mode {placement_mode!r}; expected one of {PLACEMENT_MODES}"
        )
    brain = np.asarray(brain_mask, dtype=bool)
    forbidden = _dilate(real_tumor_mask, margin_voxels)
    candidate = brain & ~forbidden
    if placement_mode == "contralateral":
        candidate = candidate & contralateral_region(brain, real_tumor_mask, midline_axis)
    return candidate


def assert_region_nesting(seg) -> None:
    """Raise unless ``seg`` is a valid, region-nested GoAT label map.

    Verifies every voxel is a valid label (``{0, 1, 2, 3}``) and that the region model is nested:
    ``ET({3}) ⊂ TC({1,3}) ⊂ WT({1,2,3})``. A stray label breaks the region decomposition the
    scorer relies on, so it is surfaced here rather than silently shipped.
    """
    import numpy as np

    seg = np.asarray(seg)
    valid = np.isin(seg, (0, 1, 2, 3))
    if not valid.all():
        bad = sorted(set(np.unique(seg)) - {0, 1, 2, 3})
        raise ValueError(f"seg contains non-GoAT labels {bad}; expected only {{0, 1, 2, 3}}")
    et = seg == 3
    tc = np.isin(seg, (1, 3))
    wt = np.isin(seg, (1, 2, 3))
    if not np.all(tc[et]):
        raise ValueError("region nesting violated: ET ⊄ TC")
    if not np.all(wt[tc]):
        raise ValueError("region nesting violated: TC ⊄ WT")


def place_synthetic_tumor(brain_mask, real_tumor_mask, config, spacing, seed,
                          midline_axis=DEFAULT_MIDLINE_AXIS):
    """Deterministically draw and place a concentric synthetic tumour on a healthy canvas.

    Keyed on ``seed``: samples an overall radius from ``config.radius_range_mm`` (converted to
    voxels via ``spacing``), builds the healthy candidate region (brain, minus a ``margin_mm``
    keep-out ball around the real tumour, restricted to the contralateral hemisphere when
    ``config.placement_mode == "contralateral"``), then picks a centre whose whole lesion fits
    inside that candidate region and paints :func:`concentric_labels`. ``config.et_core_fraction``
    and ``config.ncr_fraction`` set the inner radii; ``irregularity`` / ``n_blobs`` are reserved
    boundary-shaping hooks the specialist tunes (the tested core is the nested concentric draw).

    Returns ``(seg, center)`` where ``seg`` is an ``int`` ``(D, H, W)`` label map (0 elsewhere) and
    ``center`` is the placed centre. Raises ``ValueError`` if no centre admits a fitting lesion.
    """
    import numpy as np

    assert_configured(config)
    brain = np.asarray(brain_mask, dtype=bool)
    rng = np.random.default_rng(seed)

    lo, hi = config.radius_range_mm
    radius_mm = float(rng.uniform(lo, hi))
    voxel_radii = mm_to_voxels(radius_mm, spacing)
    radius_voxels = float(min(voxel_radii))  # conservative sphere radius (fits every axis)
    if radius_voxels <= 0:
        raise ValueError(f"sampled radius collapsed to {radius_voxels} voxels; check spacing / radius_range_mm")

    margin_voxels = config.margin_mm / float(min(spacing))
    candidate = healthy_candidate_mask(
        brain, real_tumor_mask, margin_voxels, config.placement_mode, midline_axis
    )
    coords = np.argwhere(candidate)
    if coords.shape[0] == 0:
        raise ValueError(
            "no valid placement: the healthy candidate region is empty "
            "(brain too small, or margin / contralateral gate excluded everything)"
        )

    radii = (radius_voxels, radius_voxels, radius_voxels)
    order = rng.permutation(coords.shape[0])
    for idx in order:
        center = tuple(int(c) for c in coords[idx])
        lesion = ellipsoid_mask(brain.shape, center, radii)
        if not (lesion & ~candidate).any():  # the whole lesion sits on healthy tissue
            seg = concentric_labels(
                brain.shape, center, radius_voxels,
                config.et_core_fraction, config.ncr_fraction,
            )
            return seg, center

    raise ValueError(
        "no valid placement: a lesion of the sampled radius does not fit inside the healthy "
        "candidate region at any centre (reduce radius_range_mm or relax margin_mm)"
    )


def load_mask_config(path: str | Path) -> SyntheticMaskConfig:
    """Build a :class:`SyntheticMaskConfig` from the ``mask_*`` keys of a synthesis config file.

    Reads ``configs/synthesis/synthesis.yaml`` keys of the form ``mask_<hook>`` (the mask config
    shares the file with the Stage-3 pool config, which owns the un-prefixed pool keys).
    ``mask_radius_range_mm`` is coerced from a YAML list to a tuple; ``mask_placement_mode`` fills
    the choice field. :func:`assert_configured` still governs whether the result may sample a mask.
    """
    from ..config import load_config

    cfg = load_config(Path(path))
    kwargs: dict = {}
    for hook in SPECIALIST_HOOKS:
        key = f"mask_{hook}"
        if key in cfg and cfg[key] is not None:
            value = cfg[key]
            if hook == "radius_range_mm" and isinstance(value, list):
                value = tuple(value)
            kwargs[hook] = value
    if cfg.get("mask_placement_mode") is not None:
        kwargs["placement_mode"] = cfg["mask_placement_mode"]
    return SyntheticMaskConfig(**kwargs)
