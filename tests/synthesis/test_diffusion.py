import importlib.util

import numpy as np
import pytest

from brats2026.synthesis.diffusion import (
    BETA_SCHEDULES,
    SPECIALIST_HOOKS,
    GoATLatentDiffusion,
    LatentDiffusionConfig,
    alphas_cumprod,
    assert_configured,
    cosine_beta_schedule,
    hole_latent,
    linear_beta_schedule,
    load_diffusion_config,
    q_sample_coefficients,
    unset_hooks,
)


def _full_config(**overrides):
    base = dict(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        beta_start=1e-4,
        beta_end=2e-2,
        cosine_s=8e-3,
        unet_channels=(128, 256, 256),
        unet_attention_levels=(False, True, True),
        num_res_blocks=2,
        patch_size=(128, 128, 128),
        initial_lr=1e-4,
        weight_decay=0.0,
        warmup_epochs=10,
        num_epochs=300,
        batch_size=2,
        seed=20260725,
    )
    base.update(overrides)
    return LatentDiffusionConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = LatentDiffusionConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(LatentDiffusionConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)


def test_beta_schedules_constant():
    assert BETA_SCHEDULES == ("linear", "cosine")


# --- linear_beta_schedule ----------------------------------------------------------------

def test_linear_beta_schedule_endpoints_and_length():
    betas = linear_beta_schedule(100, 1e-4, 2e-2)
    assert betas.shape == (100,)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(2e-2)
    # monotonically increasing
    assert np.all(np.diff(betas) > 0)


# --- cosine_beta_schedule ----------------------------------------------------------------

def test_cosine_beta_schedule_length_and_bounds():
    betas = cosine_beta_schedule(1000, 8e-3)
    assert betas.shape == (1000,)
    assert np.all(betas > 0.0)
    assert np.all(betas < 1.0)
    # clipped below the 0.999 ceiling
    assert np.all(betas <= 0.999)


# --- alphas_cumprod ----------------------------------------------------------------------

def test_alphas_cumprod_strictly_decreasing_in_unit_interval():
    betas = linear_beta_schedule(50, 1e-4, 2e-2)
    acp = alphas_cumprod(betas)
    assert acp.shape == (50,)
    assert np.all(acp > 0.0) and np.all(acp <= 1.0)
    assert np.all(np.diff(acp) < 0)


# --- q_sample_coefficients ---------------------------------------------------------------

def test_q_sample_coefficients_at_t_zero():
    acp = np.array([1.0, 0.5, 0.25])
    sqrt_acp, sqrt_om = q_sample_coefficients(acp, 0)
    assert sqrt_acp == pytest.approx(1.0)
    assert sqrt_om == pytest.approx(0.0)


def test_q_sample_coefficients_general_t():
    acp = np.array([1.0, 0.5, 0.25])
    sqrt_acp, sqrt_om = q_sample_coefficients(acp, 1)
    assert sqrt_acp == pytest.approx(np.sqrt(0.5))
    assert sqrt_om == pytest.approx(np.sqrt(0.5))


# --- hole_latent -------------------------------------------------------------------------

def test_hole_latent_zeros_masked_region_leaves_context():
    latent = np.arange(2 * 3 * 3 * 3, dtype=float).reshape(2, 3, 3, 3) + 1.0
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    holed = hole_latent(latent, mask)
    # masked spatial location zeroed across every channel
    assert np.all(holed[:, 1, 1, 1] == 0)
    # everything else untouched
    keep = np.ones((3, 3, 3), dtype=bool)
    keep[1, 1, 1] = False
    assert np.array_equal(holed[:, keep], latent[:, keep])
    # original not mutated
    assert latent[0, 1, 1, 1] != 0


# --- load_diffusion_config ---------------------------------------------------------------

def test_load_diffusion_config_reads_hooks_and_coerces_tuples(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    cfg_dict = {
        "kind": "synthesis-diffusion",
        "num_train_timesteps": 1000,
        "beta_schedule": "cosine",
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "cosine_s": 8e-3,
        "unet_channels": [128, 256, 256],
        "unet_attention_levels": [False, True, True],
        "num_res_blocks": 2,
        "patch_size": [128, 128, 128],
        "initial_lr": 1e-4,
        "weight_decay": 0.0,
        "warmup_epochs": 10,
        "num_epochs": 300,
        "batch_size": 2,
        "seed": 20260725,
    }
    path = tmp_path / "diffusion.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_diffusion_config(path)
    assert isinstance(cfg, LatentDiffusionConfig)
    assert cfg.unet_channels == (128, 256, 256)
    assert cfg.unet_attention_levels == (False, True, True)
    assert cfg.patch_size == (128, 128, 128)
    assert unset_hooks(cfg) == []


def test_load_diffusion_config_leaves_absent_hooks_none(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "diffusion.yaml"
    path.write_text(yaml.safe_dump({"kind": "synthesis-diffusion"}), encoding="utf-8")
    cfg = load_diffusion_config(path)
    for hook in SPECIALIST_HOOKS:
        assert getattr(cfg, hook) is None


# --- torch scaffold ----------------------------------------------------------------------

def test_torch_class_presence_tracks_torch_availability():
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch:
        pytest.skip("torch not installed")
    assert GoATLatentDiffusion is not None
