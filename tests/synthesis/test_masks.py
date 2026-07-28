import numpy as np
import pytest

from brats2026.synthesis.masks import (
    PLACEMENT_MODES,
    SPECIALIST_HOOKS,
    SyntheticMaskConfig,
    assert_configured,
    assert_region_nesting,
    brain_mask_from_volume,
    concentric_labels,
    contralateral_region,
    ellipsoid_mask,
    healthy_candidate_mask,
    load_mask_config,
    mm_to_voxels,
    place_synthetic_tumor,
    unset_hooks,
)


def _full_config(**overrides):
    """A fully-resolved SyntheticMaskConfig (every SPECIALIST hook set)."""
    base = dict(
        radius_range_mm=(3.0, 3.0),
        et_core_fraction=0.4,
        ncr_fraction=0.3,
        ed_rim_fraction=0.3,
        irregularity=0.0,
        n_blobs=1,
        margin_mm=2.0,
        seed=20260725,
    )
    base.update(overrides)
    return SyntheticMaskConfig(**base)


# --- config hooks ------------------------------------------------------------------------

def test_fresh_config_has_all_specialist_hooks_unset():
    cfg = SyntheticMaskConfig()
    missing = unset_hooks(cfg)
    for hook in SPECIALIST_HOOKS:
        assert hook in missing


def test_placement_mode_is_not_a_hook():
    cfg = SyntheticMaskConfig()
    assert cfg.placement_mode == "healthy_anywhere"
    assert cfg.placement_mode in PLACEMENT_MODES
    assert "placement_mode" not in SPECIALIST_HOOKS


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(SyntheticMaskConfig())
    assert "SPECIALIST" in str(exc.value)


def test_assert_configured_passes_when_all_set():
    cfg = _full_config()
    assert unset_hooks(cfg) == []
    assert_configured(cfg)  # no raise


# --- ellipsoid_mask ----------------------------------------------------------------------

def test_ellipsoid_mask_center_is_inside():
    m = ellipsoid_mask((21, 21, 21), (10, 10, 10), (5, 5, 5))
    assert m.dtype == bool
    assert m[10, 10, 10]
    assert not m[0, 0, 0]


def test_ellipsoid_mask_respects_anisotropic_radii():
    m = ellipsoid_mask((21, 21, 21), (10, 10, 10), (2, 8, 8))
    # tight along axis 0, wide along axes 1/2
    assert m[10, 10, 17]      # within 8 of centre on axis 2
    assert not m[13, 10, 10]  # 3 away on axis 0, radius only 2


# --- concentric_labels + nesting ---------------------------------------------------------

def test_concentric_labels_are_region_nested():
    seg = concentric_labels((41, 41, 41), (20, 20, 20), 18.0, 0.4, 0.3)
    et = seg == 3
    tc = np.isin(seg, (1, 3))
    wt = np.isin(seg, (1, 2, 3))
    assert et.any() and tc.any() and wt.any()
    # ET subset TC subset WT
    assert np.all(tc[et])
    assert np.all(wt[tc])
    # strict growth of the regions
    assert et.sum() < tc.sum() < wt.sum()


def test_concentric_labels_only_valid_labels():
    seg = concentric_labels((31, 31, 31), (15, 15, 15), 12.0, 0.4, 0.3)
    assert set(np.unique(seg)).issubset({0, 1, 2, 3})


# --- assert_region_nesting ---------------------------------------------------------------

def test_assert_region_nesting_passes_on_concentric():
    seg = concentric_labels((31, 31, 31), (15, 15, 15), 12.0, 0.4, 0.3)
    assert_region_nesting(seg)  # no raise


def test_assert_region_nesting_rejects_invalid_label():
    seg = np.zeros((5, 5, 5), dtype=int)
    seg[2, 2, 2] = 5  # not a GoAT label
    with pytest.raises(ValueError):
        assert_region_nesting(seg)


# --- mm_to_voxels ------------------------------------------------------------------------

def test_mm_to_voxels_scalar_radius_per_axis_spacing():
    assert mm_to_voxels(10.0, (1.0, 1.0, 2.0)) == pytest.approx((10.0, 10.0, 5.0))


def test_mm_to_voxels_tuple_radius():
    assert mm_to_voxels((10.0, 20.0, 10.0), (1.0, 2.0, 2.0)) == pytest.approx((10.0, 10.0, 5.0))


# --- brain_mask_from_volume --------------------------------------------------------------

def test_brain_mask_from_volume_multichannel():
    vol = np.zeros((4, 6, 6, 6))
    vol[0, 2:4, 2:4, 2:4] = 1.0  # only one channel nonzero -> still foreground
    bm = brain_mask_from_volume(vol)
    assert bm.shape == (6, 6, 6)
    assert bm.dtype == bool
    assert bm[2, 2, 2]
    assert not bm[0, 0, 0]


# --- contralateral_region ----------------------------------------------------------------

def test_contralateral_region_picks_opposite_hemisphere():
    brain = np.zeros((20, 20, 20), dtype=bool)
    brain[5:15, 5:15, 2:18] = True
    tumor = np.zeros((20, 20, 20), dtype=bool)
    tumor[8:12, 8:12, 14:17] = True  # high side of axis 2
    region = contralateral_region(brain, tumor, midline_axis=2)
    # opposite (low side of axis 2) is populated, tumour side is not
    assert region[8, 8, 4]
    assert not region[8, 8, 16]
    assert np.all(region <= brain)  # never leaves the brain


# --- healthy_candidate_mask --------------------------------------------------------------

def test_healthy_candidate_excludes_tumour_and_margin():
    brain = np.zeros((30, 30, 30), dtype=bool)
    brain[3:27, 3:27, 3:27] = True
    tumor = np.zeros((30, 30, 30), dtype=bool)
    tumor[10:14, 10:14, 10:14] = True
    cand = healthy_candidate_mask(brain, tumor, margin_voxels=2,
                                  placement_mode="healthy_anywhere", midline_axis=2)
    assert np.all(cand <= brain)
    # tumour voxels and a 2-voxel margin around them are excluded
    assert not cand[12, 12, 12]
    assert not cand[9, 12, 12]   # 1 voxel outside tumour, within margin
    assert cand[20, 20, 20]      # far healthy tissue survives


def test_healthy_candidate_contralateral_is_subset_of_that_hemisphere():
    brain = np.zeros((30, 30, 30), dtype=bool)
    brain[3:27, 3:27, 3:27] = True
    tumor = np.zeros((30, 30, 30), dtype=bool)
    tumor[10:14, 10:14, 20:24] = True  # high side of axis 2
    contra = contralateral_region(brain, tumor, midline_axis=2)
    cand = healthy_candidate_mask(brain, tumor, margin_voxels=2,
                                  placement_mode="contralateral", midline_axis=2)
    assert np.all(cand <= contra)
    assert cand.any()


# --- place_synthetic_tumor ---------------------------------------------------------------

def _brain_and_tumor():
    brain = np.zeros((60, 60, 60), dtype=bool)
    brain[5:55, 5:55, 5:55] = True
    tumor = np.zeros((60, 60, 60), dtype=bool)
    tumor[10:15, 10:15, 10:15] = True
    return brain, tumor


def test_place_synthetic_tumor_inside_brain_off_real_tumour():
    brain, tumor = _brain_and_tumor()
    cfg = _full_config()
    seg, center = place_synthetic_tumor(brain, tumor, cfg, spacing=(1.0, 1.0, 1.0), seed=0)
    placed = seg > 0
    assert placed.any()
    assert np.all(placed <= brain)          # strictly inside brain
    assert not (placed & tumor).any()       # never overlaps the real tumour
    assert_region_nesting(seg)              # the drawn seg is region-nested
    assert brain[center]                    # centre lands in the brain


def test_place_synthetic_tumor_is_seed_deterministic():
    brain, tumor = _brain_and_tumor()
    cfg = _full_config()
    seg_a, ca = place_synthetic_tumor(brain, tumor, cfg, spacing=(1.0, 1.0, 1.0), seed=7)
    seg_b, cb = place_synthetic_tumor(brain, tumor, cfg, spacing=(1.0, 1.0, 1.0), seed=7)
    assert np.array_equal(seg_a, seg_b)
    assert ca == cb


def test_place_synthetic_tumor_raises_when_no_room():
    brain = np.zeros((30, 30, 30), dtype=bool)
    brain[14:16, 14:16, 14:16] = True  # a tiny brain
    tumor = np.zeros((30, 30, 30), dtype=bool)
    cfg = _full_config(radius_range_mm=(12.0, 12.0))  # far too large to fit
    with pytest.raises(ValueError):
        place_synthetic_tumor(brain, tumor, cfg, spacing=(1.0, 1.0, 1.0), seed=0)


# --- load_mask_config --------------------------------------------------------------------

def test_load_mask_config_reads_prefixed_keys_and_coerces_tuple(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    cfg_dict = {
        "kind": "synthesis",
        "mask_radius_range_mm": [5.0, 20.0],
        "mask_et_core_fraction": 0.4,
        "mask_ncr_fraction": 0.3,
        "mask_ed_rim_fraction": 0.3,
        "mask_irregularity": 0.2,
        "mask_n_blobs": 2,
        "mask_margin_mm": 3.0,
        "mask_seed": 20260725,
        "mask_placement_mode": "contralateral",
    }
    path = tmp_path / "synthesis.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_mask_config(path)
    assert isinstance(cfg, SyntheticMaskConfig)
    assert cfg.radius_range_mm == (5.0, 20.0)   # list -> tuple
    assert cfg.et_core_fraction == 0.4
    assert cfg.placement_mode == "contralateral"
    assert unset_hooks(cfg) == []


def test_load_mask_config_leaves_absent_hooks_none(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "synthesis.yaml"
    path.write_text(yaml.safe_dump({"kind": "synthesis"}), encoding="utf-8")
    cfg = load_mask_config(path)
    for hook in SPECIALIST_HOOKS:
        assert getattr(cfg, hook) is None
