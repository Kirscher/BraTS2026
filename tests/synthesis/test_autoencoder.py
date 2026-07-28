import importlib.util

import numpy as np
import pytest

from brats2026.synthesis.autoencoder import (
    SPECIALIST_HOOKS,
    AutoencoderKLConfig,
    GoATAutoencoderKL,
    assert_configured,
    assert_patch_divisible,
    downsample_factor,
    kl_divergence_standard_normal,
    latent_shape_from_config,
    latent_spatial_shape,
    load_ae_config,
    num_latent_elements,
    reconstruction_l1,
    unset_hooks,
)


def _full_config(**overrides):
    """A fully-resolved config (every SPECIALIST hook set) for happy-path tests."""
    base = dict(
        patch_size=(128, 128, 128),
        num_downsamplings=3,
        latent_channels=4,
        channels=(64, 128, 256),
        num_res_blocks=2,
        attention_levels=(False, False, True),
        norm_num_groups=32,
        kl_weight=1e-6,
        perceptual_weight=1.0,
        adversarial_weight=0.5,
        discriminator_start_epoch=50,
        initial_lr=1e-4,
        weight_decay=0.0,
        warmup_epochs=10,
        num_epochs=200,
        batch_size=2,
        seed=20260725,
    )
    base.update(overrides)
    return AutoencoderKLConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = AutoencoderKLConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_fixed_data_fields_are_not_hooks():
    cfg = AutoencoderKLConfig()
    # data facts, not tunables — real defaults, excluded from SPECIALIST_HOOKS
    assert cfg.spatial_dims == 3
    assert cfg.in_channels == 4
    assert "spatial_dims" not in SPECIALIST_HOOKS
    assert "in_channels" not in SPECIALIST_HOOKS


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(AutoencoderKLConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)  # no raise


def test_assert_configured_names_the_missing_hook():
    cfg = _full_config(kl_weight=None)
    with pytest.raises(ValueError) as exc:
        assert_configured(cfg)
    assert "kl_weight" in str(exc.value)


# --- downsample_factor -------------------------------------------------------------------

def test_downsample_factor_is_two_to_the_n():
    assert downsample_factor(0) == 1
    assert downsample_factor(3) == 8


def test_downsample_factor_rejects_negative():
    with pytest.raises(ValueError):
        downsample_factor(-1)


# --- latent_spatial_shape ----------------------------------------------------------------

def test_latent_spatial_shape_divides_each_axis():
    assert latent_spatial_shape((128, 128, 96), 8) == (16, 16, 12)


def test_latent_spatial_shape_rejects_non_divisible_axis():
    with pytest.raises(ValueError) as exc:
        latent_spatial_shape((130, 128, 128), 8)
    assert "divisible" in str(exc.value)


# --- assert_patch_divisible --------------------------------------------------------------

def test_assert_patch_divisible_passes_on_multiple():
    assert_patch_divisible((128, 128, 128), 8)  # no raise


def test_assert_patch_divisible_rejects_non_multiple():
    with pytest.raises(ValueError) as exc:
        assert_patch_divisible((128, 100, 128), 8)
    assert "divisible" in str(exc.value)


# --- num_latent_elements -----------------------------------------------------------------

def test_num_latent_elements_is_product():
    assert num_latent_elements(4, (16, 16, 12)) == 4 * 16 * 16 * 12


# --- latent_shape_from_config ------------------------------------------------------------

def test_latent_shape_from_config_prepends_channels():
    cfg = _full_config(num_downsamplings=3, latent_channels=4)
    assert latent_shape_from_config(cfg, (128, 128, 128)) == (4, 16, 16, 16)


def test_latent_shape_from_config_raises_when_downsamplings_none():
    cfg = _full_config(num_downsamplings=None)
    with pytest.raises(ValueError):
        latent_shape_from_config(cfg, (128, 128, 128))


def test_latent_shape_from_config_raises_when_latent_channels_none():
    cfg = _full_config(latent_channels=None)
    with pytest.raises(ValueError):
        latent_shape_from_config(cfg, (128, 128, 128))


# --- reconstruction_l1 -------------------------------------------------------------------

def test_reconstruction_l1_zero_on_equal_arrays():
    x = np.random.default_rng(0).standard_normal((2, 4, 4, 4))
    assert reconstruction_l1(x, x) == pytest.approx(0.0)


def test_reconstruction_l1_known_value():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 0.0, 0.0])
    # mean(|1|, |2|, |3|) = 2.0
    assert reconstruction_l1(pred, target) == pytest.approx(2.0)


# --- kl_divergence_standard_normal -------------------------------------------------------

def test_kl_divergence_zero_at_standard_normal():
    mu = np.zeros((4, 8))
    logvar = np.zeros((4, 8))
    assert kl_divergence_standard_normal(mu, logvar) == pytest.approx(0.0)


def test_kl_divergence_known_nonzero_value():
    # single element mu=1, logvar=0: 0.5*(1 + 1 - 1 - 0) = 0.5
    mu = np.array([1.0])
    logvar = np.array([0.0])
    assert kl_divergence_standard_normal(mu, logvar) == pytest.approx(0.5)


# --- load_ae_config ----------------------------------------------------------------------

def test_load_ae_config_reads_hooks_and_coerces_tuples(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    cfg_dict = {
        "kind": "synthesis-ae",
        "provenance": {"seed": 20260725},
        "patch_size": [128, 128, 128],
        "num_downsamplings": 3,
        "latent_channels": 4,
        "channels": [64, 128, 256],
        "num_res_blocks": 2,
        "attention_levels": [False, False, True],
        "norm_num_groups": 32,
        "kl_weight": 1e-6,
        "perceptual_weight": 1.0,
        "adversarial_weight": 0.5,
        "discriminator_start_epoch": 50,
        "initial_lr": 1e-4,
        "weight_decay": 0.0,
        "warmup_epochs": 10,
        "num_epochs": 200,
        "batch_size": 2,
        "seed": 20260725,
    }
    path = tmp_path / "autoencoder.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_ae_config(path)
    assert isinstance(cfg, AutoencoderKLConfig)
    # list -> tuple coercion
    assert cfg.patch_size == (128, 128, 128)
    assert cfg.channels == (64, 128, 256)
    assert cfg.attention_levels == (False, False, True)
    # scalars pass through
    assert cfg.num_downsamplings == 3
    assert cfg.latent_channels == 4
    assert unset_hooks(cfg) == []


def test_load_ae_config_leaves_absent_hooks_none(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "autoencoder.yaml"
    path.write_text(yaml.safe_dump({"kind": "synthesis-ae"}), encoding="utf-8")
    cfg = load_ae_config(path)
    for hook in SPECIALIST_HOOKS:
        assert getattr(cfg, hook) is None


# --- torch scaffold ----------------------------------------------------------------------

def test_torch_class_presence_tracks_torch_availability():
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch:
        pytest.skip("torch not installed")
    assert GoATAutoencoderKL is not None
