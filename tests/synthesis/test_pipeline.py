import pytest

from brats2026.nnunet.convert import DATASET_ID
from brats2026.synthesis.pipeline import (
    SPECIALIST_HOOKS,
    SynthesisPoolConfig,
    allocate_per_cohort,
    assert_configured,
    load_pool_config,
    n_synthetic_from_ratio,
    synth_dataset_id,
    synthesis_plan,
    unset_hooks,
)


def _full_config(**overrides):
    base = dict(
        synthetic_ratio=0.5,
        per_cohort_quota=True,
        et_emphasis=1.0,
        seed=20260725,
    )
    base.update(overrides)
    return SynthesisPoolConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = SynthesisPoolConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(SynthesisPoolConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)


# --- n_synthetic_from_ratio --------------------------------------------------------------

def test_n_synthetic_from_ratio_rounds():
    assert n_synthetic_from_ratio(1000, 0.5) == 500
    assert n_synthetic_from_ratio(1000, 1.0) == 1000
    assert n_synthetic_from_ratio(3, 0.5) == 2  # round(1.5) -> 2


# --- allocate_per_cohort -----------------------------------------------------------------

def test_allocate_per_cohort_sums_to_total():
    counts = {"GLI": 100, "SSA": 10, "MEN": 40, "MET": 60, "PED": 20}
    alloc = allocate_per_cohort(counts, 300)
    assert sum(alloc.values()) == 300
    assert set(alloc) == set(counts)


def test_allocate_per_cohort_rare_cohort_gets_more_than_majority():
    counts = {"GLI": 100, "SSA": 10}
    alloc = allocate_per_cohort(counts, 90)
    assert alloc["SSA"] > alloc["GLI"]  # rare cohort favoured (deficit vs largest)


def test_allocate_per_cohort_is_deterministic():
    counts = {"GLI": 100, "SSA": 10, "MEN": 40, "MET": 60, "PED": 20}
    a = allocate_per_cohort(counts, 137, et_emphasis=1.5)
    b = allocate_per_cohort(counts, 137, et_emphasis=1.5)
    assert a == b
    assert sum(a.values()) == 137


def test_allocate_per_cohort_zero_budget():
    counts = {"GLI": 100, "SSA": 10}
    assert allocate_per_cohort(counts, 0) == {"GLI": 0, "SSA": 0}


# --- synth_dataset_id --------------------------------------------------------------------

def test_synth_dataset_id_is_base_plus_200():
    assert synth_dataset_id() == DATASET_ID + 200
    assert synth_dataset_id() == 701
    assert synth_dataset_id(600) == 800


# --- synthesis_plan ----------------------------------------------------------------------

def test_synthesis_plan_argv_shape(tmp_path):
    cfg = _full_config()
    counts = {"GLI": 100, "SSA": 10, "MEN": 40, "MET": 60, "PED": 20}
    plan = synthesis_plan(cfg, counts, work_root=tmp_path, folds=(0, 1))
    assert isinstance(plan, list)
    assert all(isinstance(cmd, list) for cmd in plan)

    flat = [" ".join(cmd) for cmd in plan]
    # at least one generate step
    assert any(cmd[:2] == ["brats2026", "synthesize"] for cmd in plan)
    # a build-Dataset701 step
    assert any("701" in " ".join(cmd) and cmd[0] == "brats2026" and "build" in cmd[1] for cmd in plan)
    # plan_and_preprocess on 701
    assert any(cmd[0] == "nnUNetv2_plan_and_preprocess" and "701" in cmd for cmd in plan)
    # one train command per fold, on dataset 701 with the GoAT trainer
    train_cmds = [cmd for cmd in plan if cmd[0] == "nnUNetv2_train"]
    assert len(train_cmds) == 2
    for cmd in train_cmds:
        assert "701" in cmd
        assert "nnUNetTrainerGoAT" in cmd


def test_synthesis_plan_requires_configured_pool():
    with pytest.raises(ValueError):
        synthesis_plan(SynthesisPoolConfig(), {"GLI": 10}, work_root="work")


# --- load_pool_config --------------------------------------------------------------------

def test_load_pool_config_reads_pool_keys(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    cfg_dict = {
        "kind": "synthesis",
        "synthetic_ratio": 0.5,
        "per_cohort_quota": True,
        "et_emphasis": 1.2,
        "seed": 20260725,
    }
    path = tmp_path / "synthesis.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_pool_config(path)
    assert isinstance(cfg, SynthesisPoolConfig)
    assert cfg.synthetic_ratio == 0.5
    assert cfg.et_emphasis == 1.2
    assert unset_hooks(cfg) == []


def test_load_pool_config_leaves_absent_hooks_none(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "synthesis.yaml"
    path.write_text(yaml.safe_dump({"kind": "synthesis"}), encoding="utf-8")
    cfg = load_pool_config(path)
    for hook in SPECIALIST_HOOKS:
        assert getattr(cfg, hook) is None
