import numpy as np
import pytest

from brats2026.ssl.mae import (
    SPECIALIST_HOOKS,
    MAE3DConfig,
    assert_configured,
    num_masked,
    num_patches,
    patch_grid_shape,
    patchify,
    random_masking,
    reconstruction_loss,
    select_encoder_weights,
    unpatchify,
    unset_hooks,
)


def _full_config(**overrides):
    """A fully-resolved config (every SPECIALIST hook set) for happy-path tests."""
    base = dict(
        mask_ratio=0.75,
        patch_size=(16, 16, 16),
        decoder_depth=2,
        decoder_embed_dim=256,
        norm_pix_loss=True,
        initial_lr=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=40,
        num_epochs=800,
        batch_size=2,
        seed=1234,
    )
    base.update(overrides)
    return MAE3DConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = MAE3DConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(MAE3DConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)  # no raise


def test_assert_configured_names_the_missing_hook():
    cfg = _full_config(mask_ratio=None)
    with pytest.raises(ValueError) as exc:
        assert_configured(cfg)
    assert "mask_ratio" in str(exc.value)


# --- patch geometry ----------------------------------------------------------------------

def test_patch_grid_shape_divides_each_axis():
    assert patch_grid_shape((64, 32, 16), (16, 16, 16)) == (4, 2, 1)


def test_num_patches_is_grid_product():
    assert num_patches((64, 32, 16), (16, 16, 16)) == 8


def test_patch_grid_shape_rejects_non_divisible_axis():
    with pytest.raises(ValueError) as exc:
        patch_grid_shape((30, 32, 16), (16, 16, 16))
    assert "divisible" in str(exc.value)


def test_patch_grid_shape_rejects_non_3d():
    with pytest.raises(ValueError):
        patch_grid_shape((64, 32), (16, 16, 16))


# --- num_masked / random_masking ---------------------------------------------------------

def test_num_masked_matches_mae_keep_rule():
    # keep = int(100 * 0.25) = 25 -> masked = 75
    assert num_masked(100, 0.75) == 75


def test_num_masked_extremes():
    assert num_masked(10, 0.0) == 0
    assert num_masked(10, 1.0) == 10


def test_num_masked_rejects_out_of_range_ratio():
    with pytest.raises(ValueError):
        num_masked(10, 1.5)


def test_random_masking_hides_expected_count():
    mask, visible_idx, masked_idx = random_masking(64, mask_ratio=0.75, seed=0)
    assert mask.dtype == bool
    assert mask.sum() == 48                 # 64 - int(64*0.25)
    assert len(masked_idx) == 48
    assert len(visible_idx) == 16


def test_random_masking_is_deterministic_by_seed():
    a = random_masking(50, 0.6, seed=7)[0]
    b = random_masking(50, 0.6, seed=7)[0]
    c = random_masking(50, 0.6, seed=8)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_masking_visible_and_masked_are_disjoint_partition():
    mask, visible_idx, masked_idx = random_masking(40, 0.7, seed=3)
    assert set(visible_idx).isdisjoint(set(masked_idx))
    assert sorted([*visible_idx, *masked_idx]) == list(range(40))


# --- patchify / unpatchify ---------------------------------------------------------------

def test_patchify_shape():
    vol = np.arange(2 * 4 * 4 * 4, dtype=float).reshape(2, 4, 4, 4)
    patches = patchify(vol, (2, 2, 2))
    # grid 2x2x2 = 8 patches; patch_dim = C*pd*ph*pw = 2*8 = 16
    assert patches.shape == (8, 16)


def test_patchify_unpatchify_roundtrip():
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((3, 8, 4, 4))
    patches = patchify(vol, (2, 2, 2))
    back = unpatchify(patches, vol.shape, (2, 2, 2))
    assert np.array_equal(back, vol)


def test_patchify_rejects_non_4d():
    with pytest.raises(ValueError):
        patchify(np.zeros((4, 4, 4)), (2, 2, 2))


# --- reconstruction_loss -----------------------------------------------------------------

def test_reconstruction_loss_zero_on_perfect_prediction():
    target = np.random.default_rng(0).standard_normal((8, 16))
    mask = np.array([True, False] * 4)
    assert reconstruction_loss(target, target, mask) == pytest.approx(0.0)


def test_reconstruction_loss_counts_masked_patches_only():
    target = np.zeros((4, 3))
    pred = np.zeros((4, 3))
    pred[0] = 2.0            # masked patch: contributes
    pred[1] = 5.0            # visible patch: must be ignored
    mask = np.array([True, False, False, False])
    # only patch 0 counts: mean of (2^2) over its 3 elements = 4.0
    assert reconstruction_loss(pred, target, mask) == pytest.approx(4.0)


def test_reconstruction_loss_empty_mask_is_zero():
    target = np.ones((4, 3))
    pred = np.zeros((4, 3))
    mask = np.zeros(4, dtype=bool)
    assert reconstruction_loss(pred, target, mask) == 0.0


def test_reconstruction_loss_norm_pix_standardises_target():
    # a constant-offset target has zero per-patch variance in its *pattern*; norm_pix makes the
    # objective about pattern, not absolute level. Here pred already matches the standardised
    # target pattern, so the normalised loss is lower than the raw loss.
    target = np.array([[10.0, 12.0, 14.0, 16.0]])   # mean 13, has spread
    pred = np.array([[-1.0, -0.4, 0.4, 1.0]])       # roughly the standardised shape
    mask = np.array([True])
    raw = reconstruction_loss(pred, target, mask, norm_pix_loss=False)
    normed = reconstruction_loss(pred, target, mask, norm_pix_loss=True)
    assert normed < raw


# --- select_encoder_weights (warm-start bridge) ------------------------------------------

def test_select_encoder_weights_strips_prefix_and_drops_decoder():
    state = {
        "encoder.stem.weight": 1,
        "encoder.block0.bias": 2,
        "decoder.up0.weight": 3,        # discarded
        "mask_token": 4,                # discarded
    }
    enc = select_encoder_weights(state)
    assert enc == {"stem.weight": 1, "block0.bias": 2}


def test_select_encoder_weights_empty_when_no_match():
    assert select_encoder_weights({"decoder.x": 1}, encoder_prefix="encoder.") == {}
