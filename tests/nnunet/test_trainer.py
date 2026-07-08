import pytest

from brats2026.config import ConfigError, dump_config, make_template
from brats2026.nnunet.trainer import (
    SPECIALIST_HOOKS,
    GoATTrainerConfig,
    apply_goat_config,
    assert_configured,
    config_from_mapping,
    domain_sampling_weights,
    load_goat_config,
    unset_hooks,
)


def _full_mapping():
    """A fully-filled train-config mapping (provenance header + every hook)."""
    return {
        "kind": "train",
        "provenance": {
            "git_sha": "abc123",
            "dataset_fingerprint": "fp-deadbeef",
            "seed": 20260708,
            "parent_run_id": "root",
            "date": "2026-07-08",
            "config_schema_version": "brats2026-config/1",
        },
        "initial_lr": 1e-2,
        "weight_decay": 3e-5,
        "num_epochs": 1000,
        "batch_size": 2,
        "patch_size": [128, 128, 128],           # YAML list -> coerced to tuple
        "oversample_foreground_percent": 0.33,
        "loss_dice_weight": 1.0,
        "loss_ce_weight": 1.0,
        "domain_sampling_alpha": 0.5,
        "domain_rand_intensity_sigma": 0.1,
        "domain_rand_contrast_range": [0.7, 1.3],
        "domain_rand_noise_sigma": 0.05,
    }


class _DummyTrainer:
    """Duck-typed stand-in for an nnUNetTrainer (the attributes apply_goat_config sets)."""

    initial_lr = None
    weight_decay = None
    num_epochs = None
    oversample_foreground_percent = None


def test_fresh_config_has_all_hooks_unset():
    cfg = GoATTrainerConfig()
    missing = unset_hooks(cfg)
    # every declared SPECIALIST hook is among the unset fields
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(GoATTrainerConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = GoATTrainerConfig(
        initial_lr=1e-2,
        weight_decay=3e-5,
        num_epochs=1000,
        batch_size=2,
        patch_size=(128, 128, 128),
        oversample_foreground_percent=0.33,
        loss_dice_weight=1.0,
        loss_ce_weight=1.0,
        domain_sampling_alpha=0.5,
        domain_rand_intensity_sigma=0.1,
        domain_rand_contrast_range=(0.7, 1.3),
        domain_rand_noise_sigma=0.05,
    )
    assert unset_hooks(cfg) == []
    assert_configured(cfg)  # no raise


def test_domain_sampling_alpha0_is_natural_alpha1_is_uniform():
    counts = {"GLI": 80, "SSA": 10, "MEN": 5, "MET": 3, "PED": 2}
    natural = domain_sampling_weights(counts, alpha=0.0)
    assert natural["GLI"] == pytest.approx(0.8)
    uniform = domain_sampling_weights(counts, alpha=1.0)
    for w in uniform.values():
        assert w == pytest.approx(0.2)
    assert sum(uniform.values()) == pytest.approx(1.0)


def test_domain_sampling_ignores_absent_cohorts():
    counts = {"GLI": 10, "SSA": 0, "MEN": 0, "MET": 0, "PED": 0}
    weights = domain_sampling_weights(counts, alpha=1.0)
    assert set(weights) == {"GLI"}
    assert weights["GLI"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# config.yaml -> trainer wiring
# --------------------------------------------------------------------------- #
def test_config_from_mapping_maps_and_coerces_tuples():
    cfg = config_from_mapping(_full_mapping())
    assert unset_hooks(cfg) == []
    assert cfg.patch_size == (128, 128, 128)           # list -> tuple
    assert cfg.domain_rand_contrast_range == (0.7, 1.3)
    assert cfg.initial_lr == 1e-2


def test_config_from_mapping_leaves_missing_hooks_none():
    cfg = config_from_mapping({"initial_lr": 0.01})
    assert cfg.initial_lr == 0.01
    assert cfg.num_epochs is None
    assert "num_epochs" in unset_hooks(cfg)


def test_apply_goat_config_sets_nnunet_attributes():
    trainer = _DummyTrainer()
    apply_goat_config(trainer, config_from_mapping(_full_mapping()))
    assert trainer.initial_lr == 1e-2
    assert trainer.weight_decay == 3e-5
    assert trainer.num_epochs == 1000
    assert trainer.oversample_foreground_percent == 0.33


def test_apply_goat_config_refuses_unset_config():
    with pytest.raises(ValueError):
        apply_goat_config(_DummyTrainer(), GoATTrainerConfig())


def test_load_goat_config_reads_and_validates(tmp_path):
    pytest.importorskip("yaml")
    path = dump_config(_full_mapping(), tmp_path / "train.yaml")
    cfg = load_goat_config(path)
    assert unset_hooks(cfg) == []
    assert cfg.patch_size == (128, 128, 128)


def test_load_goat_config_refuses_stub(tmp_path):
    pytest.importorskip("yaml")
    path = dump_config(make_template("train"), tmp_path / "stub.yaml")
    with pytest.raises(ConfigError):
        load_goat_config(path)
