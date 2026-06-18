"""Phase 5 — inference post-processing: mirroring-TTA merge, small-component filtering
and ET-suppression.

Pure numpy/stdlib so ``pytest`` runs without torch/nnU-Net. The only volume threshold
(``min_voxels`` for ET-suppression) is a ``# SPECIALIST:`` hook with NO baked-in default —
the value is supplied by the ai-specialist via ``configs/infer.yaml``. We REUSE the
6/8/26-connected component labeller from :mod:`brats2026.evaluation.metrics`.
"""
from __future__ import annotations

import numpy as np

from brats2026.evaluation.metrics import connected_components

# Harmonised integer labels (NCR=1, ED=2, ET=3). ET-suppression only ever touches ET.
ET_LABEL = 3
NCR_LABEL = 1


def merge_tta(prob_maps) -> np.ndarray:
    """Average a list/stack of probability arrays (mirroring-TTA merge).

    ``prob_maps`` is a sequence (or stack) of equally-shaped probability arrays — one per
    test-time augmentation (e.g. axis mirrors). Returns their element-wise mean. Raises on an
    empty input or on a shape mismatch between members.
    """
    maps = [np.asarray(p, dtype=float) for p in prob_maps]
    if not maps:
        raise ValueError("merge_tta() requires at least one probability map")
    ref_shape = maps[0].shape
    for i, m in enumerate(maps):
        if m.shape != ref_shape:
            raise ValueError(
                f"merge_tta() shape mismatch: member {i} has shape {m.shape}, expected {ref_shape}"
            )
    return np.mean(np.stack(maps, axis=0), axis=0)


def connected_component_filter(mask, min_voxels: int) -> np.ndarray:
    """Remove connected components smaller than ``min_voxels`` from a boolean mask.

    Returns a cleaned boolean mask; components whose size is ``>= min_voxels`` are kept
    (the boundary size ``== min_voxels`` is kept).
    """
    mask = np.asarray(mask, dtype=bool)
    if min_voxels <= 1:
        return mask.copy()
    labels, n = connected_components(mask)
    if n == 0:
        return mask.copy()
    counts = np.bincount(labels.ravel())
    keep = np.zeros(mask.shape, dtype=bool)
    for comp in range(1, n + 1):
        if counts[comp] >= min_voxels:
            keep |= labels == comp
    return keep


def et_suppression(label_array, min_voxels: int, replace_with: int = NCR_LABEL) -> np.ndarray:
    """Relabel tiny ET (value 3) connected components to ``replace_with`` (default 1 = NCR).

    Small enhancing-tumor blobs are a common cross-cohort false positive; relabelling them to
    NCR keeps the tumor-core mask intact while removing them from the ET region. Components of
    size ``>= min_voxels`` are preserved (boundary kept). Labels 1 (NCR) and 2 (ED) are never
    touched.

    ``min_voxels`` is required and has no default — it is the
    # SPECIALIST: ET volume threshold, supplied via configs/infer.yaml.
    """
    out = np.asarray(label_array).copy()
    et_mask = out == ET_LABEL
    if not et_mask.any() or min_voxels <= 1:
        return out
    labels, n = connected_components(et_mask)
    counts = np.bincount(labels.ravel())
    for comp in range(1, n + 1):
        if counts[comp] < min_voxels:
            out[labels == comp] = replace_with
    return out
