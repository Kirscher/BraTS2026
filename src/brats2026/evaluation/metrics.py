"""Phase 3 — region segmentation metrics for BraTS-GoAT.

Scored regions are derived from the harmonised integer labels (NCR=1, ED=2, ET=3):

- ``ET = {3}``   (enhancing tumor)
- ``TC = {1,3}`` (tumor core = NCR + ET)
- ``WT = {1,2,3}`` (whole tumor)

We provide **Dice**, **HD95** and **NSD** (Normalized Surface Dice — the BraTS leaderboard's
surface metric) in two flavours, because the choice changes post-processing (Phase 5):
*legacy-overlap* (one global score per region) and *lesion-wise* (per connected component,
false-positive lesions penalised). numpy only — no scipy/torch dependency.
"""
from __future__ import annotations

from collections import deque

import numpy as np

REGIONS: dict[str, tuple[int, ...]] = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
ALLOWED_LABELS: tuple[int, ...] = (0, 1, 2, 3)

# BraTS lesion-wise scoring pairs Dice with the Normalized Surface Dice (NSD): the fraction of
# each region's surface lying within a tolerance τ (mm) of the other surface. τ is a fixed,
# challenge-PUBLISHED per-region distance (NOT a per-model tunable) reflecting the acceptable
# boundary / inter-annotator uncertainty for that structure — identical for every submission so
# scores are comparable. The values below are a REFERENCE default (1 mm isotropic); confirm the
# authoritative per-region τ against the GoAT 2026 spec and override via config (ai-specialist /
# compliance-guard) rather than tuning them to flatter a model.
DEFAULT_NSD_TOLERANCES_MM: dict[str, float] = {"ET": 1.0, "TC": 1.0, "WT": 1.0}


def validate_labels(arr, allowed: tuple[int, ...] = ALLOWED_LABELS) -> bool:
    """True iff every value in ``arr`` is an allowed label (sanity gate before scoring)."""
    return set(int(v) for v in np.unique(arr)).issubset(set(allowed))


def region_mask(arr, region: tuple[int, ...]):
    arr = np.asarray(arr)
    mask = np.zeros(arr.shape, dtype=bool)
    for value in region:
        mask |= arr == value
    return mask


def dice_coefficient(pred_mask, gt_mask) -> float:
    """Dice on two boolean masks. Both-empty → 1.0; exactly-one-empty → 0.0 (BraTS rule)."""
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    p, g = int(pred_mask.sum()), int(gt_mask.sum())
    if p == 0 and g == 0:
        return 1.0
    if p == 0 or g == 0:
        return 0.0
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return 2.0 * inter / (p + g)


def _surface_points(mask, spacing):
    """Coordinates (scaled by spacing) of foreground voxels touching background."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.empty((0, mask.ndim))
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = np.ones_like(mask)
    for axis in range(mask.ndim):
        for shift in (-1, 1):
            sl = [slice(1, -1)] * mask.ndim
            sl[axis] = slice(1 + shift, (-1 + shift) or None)
            interior &= padded[tuple(sl)]
    surface = mask & ~interior
    coords = np.argwhere(surface).astype(float)
    return coords * np.asarray(spacing, dtype=float)


def _directed_percentile(a, b, q):
    # min distance from every point in a to set b, then the q-th percentile
    diff = a[:, None, :] - b[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    nearest = dists.min(axis=1)
    return float(np.percentile(nearest, q))


def hausdorff95(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)) -> float:
    """Symmetric 95th-percentile Hausdorff distance.

    Both-empty → 0.0; exactly-one-empty → ``inf`` (a true miss, penalised in aggregation).
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return float("inf")
    sp = _surface_points(pred_mask, spacing)
    sg = _surface_points(gt_mask, spacing)
    return max(_directed_percentile(sp, sg, 95), _directed_percentile(sg, sp, 95))


def _directed_within_tolerance(a, b, tolerance_mm: float) -> tuple[int, int]:
    """``(#points of a whose nearest point in b is <= tolerance_mm, len(a))``.

    Guards the empty degenerate cases (a fully-solid mask has no surface): an empty ``a``
    contributes nothing; against an empty ``b`` no point can be within tolerance.
    """
    import numpy as np

    if a.shape[0] == 0:
        return 0, 0
    if b.shape[0] == 0:
        return 0, int(a.shape[0])
    diff = a[:, None, :] - b[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    nearest = dists.min(axis=1)
    return int((nearest <= tolerance_mm).sum()), int(a.shape[0])


def normalized_surface_dice(pred_mask, gt_mask, tolerance_mm: float, spacing=(1.0, 1.0, 1.0)) -> float:
    """Normalized Surface Dice at tolerance ``tolerance_mm`` (mm).

    Fraction of the two region surfaces lying within ``tolerance_mm`` of each other::

        NSD = (|S_pred within τ of S_gt| + |S_gt within τ of S_pred|) / (|S_pred| + |S_gt|)

    Bounded to ``[0, 1]`` (1 = surfaces agree within tolerance everywhere) — this is what the
    BraTS leaderboard rank-aggregates alongside Dice, and unlike :func:`hausdorff95` it is
    unaffected by a single distant outlier voxel. Both-empty → 1.0 (nothing to segment, correctly
    predicted); exactly-one-empty → 0.0 (a miss). A voxel-surface-count approximation of surface
    *area* (numpy only, no scipy), consistent with :func:`hausdorff95`; distances honour ``spacing``
    so anisotropic voxels are handled in real mm. ``tolerance_mm`` is a challenge-set distance, not
    a per-model tunable — see :data:`DEFAULT_NSD_TOLERANCES_MM`.
    """
    import numpy as np

    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    if not pred_mask.any() or not gt_mask.any():
        return 0.0
    sp = _surface_points(pred_mask, spacing)
    sg = _surface_points(gt_mask, spacing)
    within_p, n_p = _directed_within_tolerance(sp, sg, tolerance_mm)
    within_g, n_g = _directed_within_tolerance(sg, sp, tolerance_mm)
    total = n_p + n_g
    if total == 0:  # both masks fully solid (no surface) — treat as agreeing
        return 1.0
    return (within_p + within_g) / total


def region_scores(
    pred_arr,
    gt_arr,
    spacing=(1.0, 1.0, 1.0),
    nsd_tolerances_mm: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Legacy-overlap Dice + HD95 + NSD for every region.

    ``nsd_tolerances_mm`` maps region name → tolerance (mm); defaults to
    :data:`DEFAULT_NSD_TOLERANCES_MM` (a reference value — confirm against the GoAT 2026 spec).
    """
    tolerances = nsd_tolerances_mm or DEFAULT_NSD_TOLERANCES_MM
    out: dict[str, dict[str, float]] = {}
    for name, region in REGIONS.items():
        pm = region_mask(pred_arr, region)
        gm = region_mask(gt_arr, region)
        out[name] = {
            "dice": dice_coefficient(pm, gm),
            "hd95": hausdorff95(pm, gm, spacing),
            "nsd": normalized_surface_dice(pm, gm, tolerances[name], spacing),
        }
    return out


def connected_components(mask):
    """Label 6/8/26-connected (full neighbourhood) components; returns int array, n_components."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    current = 0
    offsets = _neighbour_offsets(mask.ndim)
    it = np.argwhere(mask)
    for start in map(tuple, it):
        if labels[start]:
            continue
        current += 1
        queue = deque([start])
        labels[start] = current
        while queue:
            vox = queue.popleft()
            for off in offsets:
                nb = tuple(v + o for v, o in zip(vox, off))
                if all(0 <= nb[d] < mask.shape[d] for d in range(mask.ndim)):
                    if mask[nb] and not labels[nb]:
                        labels[nb] = current
                        queue.append(nb)
    return labels, current


def _neighbour_offsets(ndim):
    import itertools

    offsets = [o for o in itertools.product((-1, 0, 1), repeat=ndim) if any(o)]
    return offsets


def lesion_wise_dice(pred_arr, gt_arr, region: tuple[int, ...]) -> float:
    """Mean lesion-wise Dice for one region.

    Each GT connected component is matched to overlapping predicted voxels (Dice per lesion);
    predicted components with no GT overlap count as false-positive lesions scoring 0. With no
    GT and no prediction the score is 1.0; with no GT but some prediction it is 0.0.
    """
    pm = region_mask(pred_arr, region)
    gm = region_mask(gt_arr, region)
    gt_labels, n_gt = connected_components(gm)
    pred_labels, n_pred = connected_components(pm)

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0

    scores: list[float] = []
    matched_pred: set[int] = set()
    for gi in range(1, n_gt + 1):
        gmask = gt_labels == gi
        overlap = pred_labels[gmask]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            scores.append(0.0)
            continue
        hit_ids = set(int(x) for x in np.unique(overlap))
        matched_pred |= hit_ids
        pmask = np.isin(pred_labels, list(hit_ids))
        scores.append(dice_coefficient(pmask, gmask))

    false_positives = [p for p in range(1, n_pred + 1) if p not in matched_pred]
    scores.extend(0.0 for _ in false_positives)
    return float(np.mean(scores))
