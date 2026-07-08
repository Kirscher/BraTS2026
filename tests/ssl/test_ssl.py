from pathlib import Path

import numpy as np
import pytest

from brats2026.ssl import (
    SPECIALIST_HOOKS,
    CaseStats,
    GoATSelfTrainingConfig,
    accept_case,
    assert_configured,
    assert_goat_pool,
    case_confidence_score,
    labeled_unlabeled_sampling_plan,
    load_ssl_config,
    pseudo_label_predict_commands,
    round_dataset_id,
    select_from_stats_records,
    select_pseudo_cases,
    self_training_plan,
    self_training_round_commands,
    unset_hooks,
    voxel_confidence_mask,
)


def _full_config(**overrides):
    """A fully-resolved config (every SPECIALIST hook set) for happy-path tests."""
    base = dict(
        n_rounds=2,
        confidence_threshold_voxel=0.8,
        confidence_threshold_case=0.85,
        case_confidence_metric="mean_fg_softmax",
        min_pseudo_foreground_voxels=100,
        max_pseudo_foreground_fraction=0.5,
        labeled_unlabeled_ratio=2.0,
        per_cohort_pseudo_quota=True,
        pseudo_label_softmax_temperature=1.0,
        ignore_label_index=4,
        pseudo_use_tta=True,
        pseudo_use_ensemble=True,
        teacher_folds=(0, 1, 2, 3, 4),
        recompute_pseudo_each_round=True,
        seed=1234,
    )
    base.update(overrides)
    return GoATSelfTrainingConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = GoATSelfTrainingConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_allow_validation_pool_is_not_a_specialist_hook():
    # the conformance gate is not a tunable: excluded from hooks and from the unset count
    assert "allow_validation_pool" not in SPECIALIST_HOOKS
    cfg = GoATSelfTrainingConfig()
    assert "allow_validation_pool" not in unset_hooks(cfg)
    assert cfg.allow_validation_pool is False


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(GoATSelfTrainingConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)  # no raise


def test_assert_configured_ignores_allow_validation_pool_default():
    # a config with everything set but allow_validation_pool left at its default still passes
    cfg = _full_config()
    assert cfg.allow_validation_pool is False
    assert_configured(cfg)


# --- voxel_confidence_mask ---------------------------------------------------------------

def test_voxel_confidence_mask_thresholds():
    softmax = np.array([0.6, 0.8, 0.95, 0.79])
    mask = voxel_confidence_mask(softmax, threshold=0.8)
    assert mask.tolist() == [False, True, True, False]


def test_voxel_confidence_mask_returns_bool_array():
    mask = voxel_confidence_mask(np.array([0.9, 0.1]), threshold=0.5)
    assert mask.dtype == bool


def test_voxel_confidence_mask_temperature_sharpens_and_softens():
    softmax = np.array([0.9])
    # higher temperature softens the (max) probability -> fewer voxels pass a fixed threshold
    hot = voxel_confidence_mask(softmax, threshold=0.85, temperature=4.0)
    cold = voxel_confidence_mask(softmax, threshold=0.85, temperature=1.0)
    assert cold.tolist() == [True]
    assert hot.tolist() == [False]


# --- case_confidence_score / accept_case -------------------------------------------------

def _stats(**overrides):
    base = dict(
        case_id="BraTS-MEN-0001-000",
        mean_fg_softmax=0.9,
        n_fg_voxels=500,
        fg_fraction=0.2,
        frac_confident_voxels=0.7,
    )
    base.update(overrides)
    return CaseStats(**base)


def test_case_confidence_score_mean_fg_softmax():
    assert case_confidence_score(_stats(mean_fg_softmax=0.9), "mean_fg_softmax") == 0.9


def test_case_confidence_score_frac_confident_voxels():
    assert case_confidence_score(_stats(frac_confident_voxels=0.7), "frac_confident_voxels") == 0.7


def test_case_confidence_score_unknown_metric_raises():
    with pytest.raises(ValueError):
        case_confidence_score(_stats(), "no_such_metric")


def test_accept_case_passes_when_all_criteria_met():
    cfg = _full_config()
    assert accept_case(_stats(mean_fg_softmax=0.9, n_fg_voxels=500, fg_fraction=0.2), cfg)


def test_accept_case_rejects_low_confidence():
    cfg = _full_config(confidence_threshold_case=0.95)
    assert not accept_case(_stats(mean_fg_softmax=0.9), cfg)


def test_accept_case_rejects_too_few_foreground_voxels():
    cfg = _full_config(min_pseudo_foreground_voxels=1000)
    assert not accept_case(_stats(n_fg_voxels=500), cfg)


def test_accept_case_rejects_hallucinated_oversized_foreground():
    cfg = _full_config(max_pseudo_foreground_fraction=0.1)
    assert not accept_case(_stats(fg_fraction=0.6), cfg)


# --- select_pseudo_cases -----------------------------------------------------------------

def test_select_pseudo_cases_excludes_rejected():
    cfg = _full_config(per_cohort_pseudo_quota=False)
    stats = [
        _stats(case_id="BraTS-MEN-0001-000", mean_fg_softmax=0.9),
        _stats(case_id="BraTS-MEN-0002-000", mean_fg_softmax=0.1),  # rejected: low conf
    ]
    cohort_of = {s.case_id: "MEN" for s in stats}
    selected = select_pseudo_cases(stats, cohort_of, cfg)
    assert selected["MEN"] == ["BraTS-MEN-0001-000"]


def test_select_pseudo_cases_quota_off_keeps_all_accepted():
    cfg = _full_config(per_cohort_pseudo_quota=False)
    stats = [_stats(case_id=f"BraTS-MET-{i:04d}-000") for i in range(5)]
    cohort_of = {s.case_id: "MET" for s in stats}
    selected = select_pseudo_cases(stats, cohort_of, cfg)
    assert len(selected["MET"]) == 5


def test_select_pseudo_cases_quota_on_caps_per_cohort():
    # quota caps each cohort at the size of the smallest accepted cohort (balanced)
    cfg = _full_config(per_cohort_pseudo_quota=True)
    stats = (
        [_stats(case_id=f"BraTS-MET-{i:04d}-000") for i in range(5)]
        + [_stats(case_id="BraTS-MEN-0001-000")]
    )
    cohort_of = {s.case_id: ("MEN" if "MEN" in s.case_id else "MET") for s in stats}
    selected = select_pseudo_cases(stats, cohort_of, cfg)
    assert len(selected["MEN"]) == 1
    assert len(selected["MET"]) == 1  # capped down to the smallest cohort


# --- labeled_unlabeled_sampling_plan -----------------------------------------------------

def test_sampling_plan_normalises_to_one():
    plan = labeled_unlabeled_sampling_plan(n_labeled=100, n_pseudo=100, ratio=1.0)
    assert plan["labeled"] + plan["pseudo"] == pytest.approx(1.0)
    assert plan["labeled"] == pytest.approx(0.5)


def test_sampling_plan_ratio_favours_labeled():
    plan = labeled_unlabeled_sampling_plan(n_labeled=100, n_pseudo=100, ratio=3.0)
    assert plan["labeled"] == pytest.approx(0.75)
    assert plan["pseudo"] == pytest.approx(0.25)


def test_sampling_plan_no_pseudo_is_all_labeled():
    plan = labeled_unlabeled_sampling_plan(n_labeled=50, n_pseudo=0, ratio=2.0)
    assert plan["labeled"] == pytest.approx(1.0)
    assert plan["pseudo"] == pytest.approx(0.0)


# --- pseudo_label_predict_commands -------------------------------------------------------

def test_pseudo_label_commands_single_when_ensemble_off():
    cfg = _full_config(pseudo_use_ensemble=False, pseudo_use_tta=True, teacher_folds=(0, 1, 2))
    cmds = pseudo_label_predict_commands("teacher", "/unlab", "/pseudo", cfg)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd[0] == "nnUNetv2_predict"
    assert cmd[cmd.index("-i") + 1] == "/unlab"
    assert cmd[cmd.index("-o") + 1] == "/pseudo"
    assert cmd[cmd.index("-d") + 1] == "teacher"


def test_pseudo_label_commands_one_per_member_when_ensemble_on():
    cfg = _full_config(pseudo_use_ensemble=True, teacher_folds=(0, 1, 2, 3, 4))
    cmds = pseudo_label_predict_commands(["t0", "t1"], "/unlab", "/pseudo", cfg)
    assert len(cmds) == 2
    for member, cmd in zip(["t0", "t1"], cmds):
        assert cmd[cmd.index("-d") + 1] == member


def test_pseudo_label_commands_tta_toggle():
    on = pseudo_label_predict_commands("t", "/u", "/p", _full_config(pseudo_use_tta=True, pseudo_use_ensemble=False))
    off = pseudo_label_predict_commands("t", "/u", "/p", _full_config(pseudo_use_tta=False, pseudo_use_ensemble=False))
    assert "--disable_tta" not in on[0]
    assert "--disable_tta" in off[0]


def test_pseudo_label_commands_use_teacher_folds():
    cfg = _full_config(pseudo_use_ensemble=False, teacher_folds=(1, 3))
    cmd = pseudo_label_predict_commands("t", "/u", "/p", cfg)[0]
    fi = cmd.index("-f")
    assert cmd[fi + 1 : fi + 3] == ["1", "3"]


# --- self_training_round_commands / self_training_plan -----------------------------------

def test_round_commands_emit_predict_filter_then_train():
    cfg = _full_config(pseudo_use_ensemble=False)
    cmds = self_training_round_commands(0, "teacher", "/unlab", "/work", cfg)
    predicts = [c for c in cmds if c[0] == "nnUNetv2_predict"]
    selects = [c for c in cmds if c[:2] == ["brats2026", "ssl-select"]]
    trains = [c for c in cmds if c[0] == "nnUNetv2_train"]
    assert predicts
    assert selects  # the filter step is now IN the plan (was missing)
    assert trains
    # the train command trains this round's own dataset and pins the ResEnc-L plans with -p
    assert trains[0][1] == str(round_dataset_id(501, 0))
    assert "-p" in trains[0]


def test_round_commands_order_is_predict_then_filter_then_train():
    cfg = _full_config(pseudo_use_ensemble=False)
    cmds = self_training_round_commands(0, "teacher", "/unlab", "/work", cfg)
    kinds = [c[0] if c[0] != "brats2026" else "ssl-select" for c in cmds]
    assert kinds.index("nnUNetv2_predict") < kinds.index("ssl-select") < kinds.index("nnUNetv2_train")


def test_self_training_plan_single_round():
    cfg = _full_config(n_rounds=1, pseudo_use_ensemble=False)
    plan = self_training_plan(cfg, seed_teacher="teacher0",
                              unlabeled_input_dir="/unlab", work_root="/work")
    assert len(plan) == 1


def test_self_training_plan_two_rounds_chains_teacher_to_prev_dataset():
    cfg = _full_config(n_rounds=2, pseudo_use_ensemble=False)
    plan = self_training_plan(cfg, seed_teacher="teacher0",
                              unlabeled_input_dir="/unlab", work_root="/work")
    assert len(plan) == 2
    # round 0 predicts with the seed teacher
    round0_predict = [c for c in plan[0] if c[0] == "nnUNetv2_predict"][0]
    assert round0_predict[round0_predict.index("-d") + 1] == "teacher0"
    # round 1 teacher = the student trained in round 0, i.e. round_dataset_id(501, 0)
    round1_predict = [c for c in plan[1] if c[0] == "nnUNetv2_predict"][0]
    assert round1_predict[round1_predict.index("-d") + 1] == str(round_dataset_id(501, 0))


def test_round_dataset_id_is_distinct_per_round():
    assert round_dataset_id(501, 0) != round_dataset_id(501, 1)


def test_select_from_stats_records_applies_filters_and_cohort():
    cfg = _full_config(per_cohort_pseudo_quota=False)
    records = [
        {"case_id": "BraTS-MEN-0001-000", "mean_fg_softmax": 0.95, "n_fg_voxels": 500,
         "fg_fraction": 0.2, "frac_confident_voxels": 0.9},
        {"case_id": "BraTS-MEN-0002-000", "mean_fg_softmax": 0.1, "n_fg_voxels": 500,
         "fg_fraction": 0.2, "frac_confident_voxels": 0.1},  # rejected: low confidence
    ]
    selected = select_from_stats_records(records, cfg)
    assert selected["MEN"] == ["BraTS-MEN-0001-000"]


def test_load_ssl_config_reads_repo_yaml():
    pytest.importorskip("yaml")
    ssl_yaml = Path(__file__).resolve().parents[2] / "configs" / "ssl.yaml"
    cfg = load_ssl_config(ssl_yaml)
    # the repo ssl.yaml is fully resolved -> no unset hooks, folds coerced to a tuple
    assert unset_hooks(cfg) == []
    assert isinstance(cfg.teacher_folds, tuple)


# --- assert_goat_pool (fatal guards) -----------------------------------------------------

def test_assert_goat_pool_passes_for_known_cohorts():
    assert_goat_pool(["BraTS-MEN-0001-000", "BraTS-MET-0002-000"], allow_validation_pool=False)


def test_assert_goat_pool_rejects_unknown_cohort():
    with pytest.raises(ValueError) as exc:
        assert_goat_pool(["BraTS-XXX-0001-000"], allow_validation_pool=False)
    assert "UNK" in str(exc.value) or "cohort" in str(exc.value).lower()


def test_assert_goat_pool_rejects_validation_collision_when_not_allowed():
    with pytest.raises(ValueError):
        assert_goat_pool(
            ["BraTS-MEN-0001-000"],
            allow_validation_pool=False,
            validation_ids=["BraTS-MEN-0001-000"],
        )


def test_assert_goat_pool_allows_validation_collision_when_allowed():
    # no raise when the rule gate is explicitly opened
    assert_goat_pool(
        ["BraTS-MEN-0001-000"],
        allow_validation_pool=True,
        validation_ids=["BraTS-MEN-0001-000"],
    )


def test_assert_goat_pool_rejects_outer_lodo_collision():
    with pytest.raises(ValueError):
        assert_goat_pool(
            ["BraTS-MEN-0001-000"],
            allow_validation_pool=True,
            outer_lodo_ids=["BraTS-MEN-0001-000"],
        )
