import numpy as np
import pytest

from brats2026.evaluation.threshold_sweep import (
    LesionCounts,
    apply_operating_point,
    best_operating_point,
    lesion_confusion,
    summarise,
    sweep_operating_points,
)


def _blob(shape, slices, value=1.0):
    a = np.zeros(shape, dtype=float)
    a[slices] = value
    return a


# --- lesion confusion -----------------------------------------------------------------------

def test_one_matched_lesion_is_a_true_positive():
    gt = _blob((10, 10, 10), np.s_[2:5, 2:5, 2:5]).astype(bool)
    assert lesion_confusion(gt, gt) == LesionCounts(tp=1, fp=0, fn=0)


def test_a_missed_lesion_is_a_false_negative():
    gt = _blob((10, 10, 10), np.s_[2:5, 2:5, 2:5]).astype(bool)
    pred = np.zeros_like(gt)
    assert lesion_confusion(pred, gt) == LesionCounts(tp=0, fp=0, fn=1)


def test_a_spurious_component_is_a_false_positive():
    gt = np.zeros((10, 10, 10), dtype=bool)
    pred = _blob((10, 10, 10), np.s_[7:9, 7:9, 7:9]).astype(bool)
    assert lesion_confusion(pred, gt) == LesionCounts(tp=0, fp=1, fn=0)


def test_fragmented_prediction_on_one_lesion_is_one_tp_not_extra_fps():
    """Fragmentation is already penalised by Dice/NSD; charging it here too would bias the
    search toward over-suppression, which is the failure we are undoing."""
    gt = _blob((10, 10, 10), np.s_[2:8, 2:4, 2:4]).astype(bool)
    pred = np.zeros_like(gt)
    pred[2:4, 2:4, 2:4] = True      # two disconnected fragments,
    pred[6:8, 2:4, 2:4] = True      # both inside the single GT lesion
    assert lesion_confusion(pred, gt) == LesionCounts(tp=1, fp=0, fn=0)


# --- the metric asymmetry that motivates the whole module -----------------------------------

def test_f2_prefers_recall_where_f1_is_indifferent():
    """Our data has FN:FP ~ 2.6:1, so the objective must be able to weight recall higher."""
    recall_heavy = LesionCounts(tp=80, fp=40, fn=20)   # recall .80  precision .67
    precision_heavy = LesionCounts(tp=67, fp=13, fn=33)  # recall .67  precision .84
    assert recall_heavy.fbeta(2.0) > precision_heavy.fbeta(2.0)


def test_recall_and_precision_are_one_when_nothing_exists():
    empty = LesionCounts(0, 0, 0)
    assert empty.recall == 1.0 and empty.precision == 1.0


# --- operating point ------------------------------------------------------------------------

def test_lowering_the_threshold_recovers_a_faint_lesion():
    """The core claim: a faint lesion missed at a high threshold is found at a lower one."""
    prob = _blob((10, 10, 10), np.s_[2:5, 2:5, 2:5], value=0.35)
    gt = prob > 0
    assert not apply_operating_point(prob, threshold=0.5, min_voxels=1).any()
    assert apply_operating_point(prob, threshold=0.3, min_voxels=1).any()


def test_min_voxels_filter_deletes_small_true_lesions():
    """This is very likely where the missing ET recall went — the filter is not free."""
    prob = _blob((10, 10, 10), np.s_[2:4, 2:4, 2:4], value=0.9)  # an 8-voxel lesion
    assert apply_operating_point(prob, 0.5, min_voxels=1).sum() == 8
    assert apply_operating_point(prob, 0.5, min_voxels=27).sum() == 0


def test_min_voxels_of_one_disables_the_filter():
    prob = _blob((6, 6, 6), np.s_[1:3, 1:3, 1:3], value=0.9)
    assert apply_operating_point(prob, 0.5, 1).sum() == apply_operating_point(prob, 0.5, 0).sum()


# --- sweep ----------------------------------------------------------------------------------

def _two_cases():
    # The blobs must not touch even diagonally: connected_components uses the full 26-neighbour
    # stencil, so [1:4] and [4:7] would merge into a single component through their corner.
    strong = _blob((9, 9, 9), np.s_[1:4, 1:4, 1:4], value=0.9)
    faint = _blob((9, 9, 9), np.s_[6:9, 6:9, 6:9], value=0.4)
    prob = strong + faint
    gt = prob > 0
    return [(prob, gt), (prob, gt)]


def test_sweep_covers_the_whole_grid():
    results = sweep_operating_points(_two_cases(), [0.3, 0.5], [1, 5])
    assert len(results) == 4
    assert {(r.threshold, r.min_voxels) for r in results} == {
        (0.3, 1), (0.3, 5), (0.5, 1), (0.5, 5)
    }


def test_sweep_finds_the_threshold_that_recovers_the_faint_lesion():
    results = sweep_operating_points(_two_cases(), [0.3, 0.5], [1])
    best = best_operating_point(results, "recall")
    assert best.threshold == 0.3
    assert best.recall == 1.0                     # both lesions in both cases
    low = next(r for r in results if r.threshold == 0.5)
    assert low.recall == 0.5                      # faint lesion lost at the higher threshold


def test_counts_are_summed_across_cases_not_averaged():
    """A 1-lesion case must not outweigh a 20-lesion one when counting total detections."""
    results = sweep_operating_points(_two_cases(), [0.3], [1])
    assert results[0].counts.tp == 4              # 2 lesions x 2 cases


def test_ties_break_toward_the_conservative_rule():
    results = sweep_operating_points(_two_cases(), [0.2, 0.3], [1])
    assert best_operating_point(results, "recall").threshold == 0.3


def test_unknown_objective_is_rejected():
    results = sweep_operating_points(_two_cases(), [0.5], [1])
    with pytest.raises(ValueError):
        best_operating_point(results, "dice")


def test_empty_results_are_rejected():
    with pytest.raises(ValueError):
        best_operating_point([], "f1")


# --- report ---------------------------------------------------------------------------------

def test_summary_reports_a_delta_against_the_shipped_operating_point():
    results = sweep_operating_points(_two_cases(), [0.3, 0.5], [1])
    baseline = next(r for r in results if r.threshold == 0.5)
    report = summarise(results, baseline=baseline)
    assert report["baseline"]["threshold"] == 0.5
    assert report["best"]["recall"]["delta_vs_baseline"]["recall"] == pytest.approx(0.5)
    assert len(report["grid"]) == 2


def test_summary_without_a_baseline_omits_deltas():
    report = summarise(sweep_operating_points(_two_cases(), [0.5], [1]))
    assert "baseline" not in report
    assert "delta_vs_baseline" not in report["best"]["f1"]
