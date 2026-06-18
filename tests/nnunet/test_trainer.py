import pytest

from brats2026.nnunet.trainer import (
    SPECIALIST_HOOKS,
    GoATTrainerConfig,
    assert_configured,
    domain_sampling_weights,
    unset_hooks,
)


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
