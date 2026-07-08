import math

import numpy as np
import pytest

from brats2026.evaluation.metrics import (
    DEFAULT_NSD_TOLERANCES_MM,
    REGIONS,
    connected_components,
    dice_coefficient,
    hausdorff95,
    lesion_wise_dice,
    normalized_surface_dice,
    region_mask,
    region_scores,
    validate_labels,
)


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #
def test_dice_both_empty_is_one():
    z = np.zeros((4, 4), dtype=bool)
    assert dice_coefficient(z, z) == 1.0


def test_dice_exactly_one_empty_is_zero():
    full = np.ones((4, 4), dtype=bool)
    empty = np.zeros((4, 4), dtype=bool)
    assert dice_coefficient(full, empty) == 0.0
    assert dice_coefficient(empty, full) == 0.0


def test_dice_perfect_overlap_is_one():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    assert dice_coefficient(m, m) == 1.0


def test_dice_known_partial_overlap():
    # pred: 4 voxels, gt: 4 voxels, intersection 2 -> 2*2/(4+4) = 0.5
    pred = np.zeros((1, 8), dtype=bool)
    gt = np.zeros((1, 8), dtype=bool)
    pred[0, 0:4] = True
    gt[0, 2:6] = True
    assert dice_coefficient(pred, gt) == 0.5


# --------------------------------------------------------------------------- #
# Region membership
# --------------------------------------------------------------------------- #
def test_region_masks_from_labelled_array():
    # one voxel of each label 0..3
    arr = np.array([0, 1, 2, 3])
    et = region_mask(arr, REGIONS["ET"])
    tc = region_mask(arr, REGIONS["TC"])
    wt = region_mask(arr, REGIONS["WT"])
    assert et.tolist() == [False, False, False, True]  # {3}
    assert tc.tolist() == [False, True, False, True]  # {1,3}
    assert wt.tolist() == [False, True, True, True]  # {1,2,3}


def test_region_constants():
    assert REGIONS["ET"] == (3,)
    assert REGIONS["TC"] == (1, 3)
    assert REGIONS["WT"] == (1, 2, 3)


# --------------------------------------------------------------------------- #
# HD95
# --------------------------------------------------------------------------- #
def test_hd95_both_empty_is_zero():
    z = np.zeros((4, 4, 4), dtype=bool)
    assert hausdorff95(z, z) == 0.0


def test_hd95_one_empty_is_inf():
    full = np.zeros((4, 4, 4), dtype=bool)
    full[1:3, 1:3, 1:3] = True
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert math.isinf(hausdorff95(full, empty))
    assert math.isinf(hausdorff95(empty, full))


def test_hd95_identical_masks_is_zero():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[1:4, 1:4, 1:4] = True
    assert hausdorff95(m, m) == 0.0


def test_hd95_anisotropic_spacing_scales_distance():
    # two single-voxel masks separated by 1 step along the last axis.
    pred = np.zeros((1, 1, 3), dtype=bool)
    gt = np.zeros((1, 1, 3), dtype=bool)
    pred[0, 0, 0] = True
    gt[0, 0, 1] = True
    iso = hausdorff95(pred, gt, spacing=(1.0, 1.0, 1.0))
    aniso = hausdorff95(pred, gt, spacing=(1.0, 1.0, 3.0))
    assert iso == 1.0
    # the separating axis is now 3x as long
    assert aniso == 3.0


# --------------------------------------------------------------------------- #
# NSD (Normalized Surface Dice)
# --------------------------------------------------------------------------- #
def test_nsd_both_empty_is_one():
    z = np.zeros((4, 4, 4), dtype=bool)
    assert normalized_surface_dice(z, z, tolerance_mm=1.0) == 1.0


def test_nsd_one_empty_is_zero():
    full = np.zeros((4, 4, 4), dtype=bool)
    full[1:3, 1:3, 1:3] = True
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert normalized_surface_dice(full, empty, tolerance_mm=1.0) == 0.0
    assert normalized_surface_dice(empty, full, tolerance_mm=1.0) == 0.0


def test_nsd_identical_masks_is_one():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[1:4, 1:4, 1:4] = True
    assert normalized_surface_dice(m, m, tolerance_mm=0.0) == 1.0


def test_nsd_tolerance_gate():
    # two single-voxel masks one step apart: inside τ=1 they agree, at τ=0.5 they do not.
    pred = np.zeros((1, 1, 3), dtype=bool)
    gt = np.zeros((1, 1, 3), dtype=bool)
    pred[0, 0, 0] = True
    gt[0, 0, 1] = True
    assert normalized_surface_dice(pred, gt, tolerance_mm=1.0) == 1.0
    assert normalized_surface_dice(pred, gt, tolerance_mm=0.5) == 0.0


def test_nsd_uses_spacing_in_mm():
    # the separating axis is 3x as long, so τ=1 mm no longer bridges the gap; τ=3 mm does.
    pred = np.zeros((1, 1, 3), dtype=bool)
    gt = np.zeros((1, 1, 3), dtype=bool)
    pred[0, 0, 0] = True
    gt[0, 0, 1] = True
    assert normalized_surface_dice(pred, gt, tolerance_mm=1.0, spacing=(1.0, 1.0, 3.0)) == 0.0
    assert normalized_surface_dice(pred, gt, tolerance_mm=3.0, spacing=(1.0, 1.0, 3.0)) == 1.0


def test_nsd_partial_surface_agreement():
    # pred has two surface points (one matching gt, one 10 mm away); gt has one.
    # pred→gt within τ: 1 of 2 ; gt→pred within τ: 1 of 1  →  (1+1)/(2+1) = 2/3
    pred = np.zeros((1, 1, 11), dtype=bool)
    gt = np.zeros((1, 1, 11), dtype=bool)
    pred[0, 0, 0] = True
    pred[0, 0, 10] = True
    gt[0, 0, 0] = True
    assert normalized_surface_dice(pred, gt, tolerance_mm=1.0) == pytest.approx(2 / 3)


def test_nsd_default_tolerances_cover_all_regions():
    assert set(DEFAULT_NSD_TOLERANCES_MM) == set(REGIONS)


# --------------------------------------------------------------------------- #
# Connected components
# --------------------------------------------------------------------------- #
def test_connected_components_2d_two_blobs():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True  # blob 1
    mask[4, 4] = True  # blob 2 (not touching)
    _, n = connected_components(mask)
    assert n == 2


def test_connected_components_2d_single_blob():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    _, n = connected_components(mask)
    assert n == 1


def test_connected_components_3d_two_blobs():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0:2, 0:2, 0:2] = True  # blob 1
    mask[3:5, 3:5, 3:5] = True  # blob 2
    labels, n = connected_components(mask)
    assert n == 2
    # the two blobs carry different labels
    assert labels[0, 0, 0] != labels[4, 4, 4]


# --------------------------------------------------------------------------- #
# Lesion-wise Dice
# --------------------------------------------------------------------------- #
def test_lesion_wise_false_positive_drags_mean_down():
    # GT has one lesion (label 3); pred matches it perfectly AND adds a separate FP lesion.
    gt = np.zeros((1, 9), dtype=int)
    gt[0, 0:3] = 3
    pred = np.zeros((1, 9), dtype=int)
    pred[0, 0:3] = 3  # perfect match -> 1.0
    pred[0, 6:9] = 3  # false-positive lesion -> 0.0
    # mean of [1.0, 0.0] = 0.5
    assert lesion_wise_dice(pred, gt, REGIONS["ET"]) == 0.5


def test_lesion_wise_missed_gt_scores_zero():
    gt = np.zeros((1, 9), dtype=int)
    gt[0, 0:3] = 3
    pred = np.zeros((1, 9), dtype=int)  # nothing predicted
    assert lesion_wise_dice(pred, gt, REGIONS["ET"]) == 0.0


def test_lesion_wise_no_gt_no_pred_is_one():
    gt = np.zeros((1, 9), dtype=int)
    pred = np.zeros((1, 9), dtype=int)
    assert lesion_wise_dice(pred, gt, REGIONS["ET"]) == 1.0


def test_lesion_wise_no_gt_but_pred_is_zero():
    gt = np.zeros((1, 9), dtype=int)
    pred = np.zeros((1, 9), dtype=int)
    pred[0, 0:3] = 3
    assert lesion_wise_dice(pred, gt, REGIONS["ET"]) == 0.0


# --------------------------------------------------------------------------- #
# Label validation
# --------------------------------------------------------------------------- #
def test_validate_labels_accepts_harmonized():
    arr = np.array([0, 1, 2, 3, 0, 1])
    assert validate_labels(arr) is True


def test_validate_labels_rejects_stray_label():
    arr = np.array([0, 1, 2, 3, 4])
    assert validate_labels(arr) is False


# --------------------------------------------------------------------------- #
# region_scores wiring
# --------------------------------------------------------------------------- #
def test_region_scores_returns_all_regions():
    # 3D to match the default (1,1,1) spacing used in scoring.
    arr = np.array([[[0, 1, 2, 3]]])
    out = region_scores(arr, arr)
    assert set(out) == {"ET", "TC", "WT"}
    for region in out.values():
        assert region["dice"] == 1.0
        assert region["hd95"] == 0.0
        assert region["nsd"] == 1.0
