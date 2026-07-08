import numpy as np

from brats2026.evaluation.score import (
    build_report,
    score_case_arrays,
)


def _cube(shape, region_slices, label):
    arr = np.zeros(shape, dtype=int)
    arr[region_slices] = label
    return arr


def test_score_case_arrays_perfect_match_is_dice_one():
    gt = np.zeros((10, 10, 10), dtype=int)
    gt[3:7, 3:7, 3:7] = 3  # ET (also part of TC and WT)
    score = score_case_arrays("BraTS-GLI-0001-000", gt.copy(), gt)
    for family in ("legacy", "lesion"):
        fam = score.family(family)
        for region in ("ET", "TC", "WT"):
            assert fam[region]["dice"] == 1.0
            assert fam[region]["nsd"] == 1.0
            assert fam[region]["hd95"] == 0.0


def test_score_case_arrays_cohort_decoded_from_id():
    z = np.zeros((4, 4, 4), dtype=int)
    assert score_case_arrays("BraTS-PED-0009-000", z, z).cohort == "PED"


def test_score_case_arrays_legacy_and_lesion_dice_differ_on_false_positive():
    # GT: one ET blob. Pred: same blob PLUS a spurious far-away blob (false-positive lesion).
    gt = _cube((20, 20, 20), np.s_[2:6, 2:6, 2:6], 3)
    pred = gt.copy()
    pred[14:18, 14:18, 14:18] = 3  # extra predicted lesion with no GT overlap
    score = score_case_arrays("BraTS-MEN-0001-000", pred, gt)
    # Lesion-wise penalises the false-positive lesion harder than global overlap does.
    assert score.family("lesion")["ET"]["dice"] < score.family("legacy")["ET"]["dice"]


def test_build_report_shapes_and_worst_cohort():
    gt = _cube((10, 10, 10), np.s_[3:7, 3:7, 3:7], 3)
    good = score_case_arrays("BraTS-GLI-0001-000", gt.copy(), gt)          # perfect
    miss = score_case_arrays("BraTS-SSA-0002-000", np.zeros_like(gt), gt)  # total miss
    report = build_report([good, miss])
    assert report["n_cases"] == 2
    assert set(report["per_cohort"]) == {"legacy", "lesion"}
    # The missed SSA case must be the worst cohort on WT.
    assert report["worst_cohort"]["WT"]["cohort"] == "SSA"
    assert report["worst_cohort"]["WT"]["dice"] == 0.0
