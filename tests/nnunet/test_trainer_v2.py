import random
from dataclasses import fields

import numpy as np
import pytest

from brats2026.compliance import Status, check_no_prior_weights
from brats2026.config import TRAIN_HOOK_KEYS, TRAIN_V2_HOOK_KEYS, make_template
from brats2026.nnunet.plan import GOAT_TRAINER, GOAT_TRAINER_V2, train_command, train_v2_command
from brats2026.nnunet.trainer import GOAT_TRAIN_CONFIG_ENV
from brats2026.nnunet.trainer import SPECIALIST_HOOKS as V1_HOOKS
from brats2026.nnunet.trainer_v2 import (
    _CHOICE_KEYS,
    GOAT_V2_SHIM_BODY,
    GOAT_V2_SHIM_NAME,
    BIAS_FIELD_EVAL_GRID,
    GOAT_V2_TRAIN_CONFIG_ENV,
    SPECIALIST_HOOKS_V2,
    GoATTrainerConfigV2,
    apply_goat_v2_config,
    assert_configured,
    assert_transferable,
    bias_field_multiplier,
    bias_field_term_count,
    config_from_mapping,
    plan_weight_transfer,
    resolve_v2_config,
    retune_augmentation,
    sample_bias_field_coefficients,
    unset_hooks,
)

_CKPT = "/work/nnUNet_results/Dataset501_BraTSGoAT/nnUNetTrainerGoAT__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth"


def _full_v2_mapping():
    """A fully-filled train_v2 mapping (provenance header + every hook + the warm-start choice)."""
    return {
        "kind": "train_v2",
        "provenance": {
            "git_sha": "abc123",
            "dataset_fingerprint": "fp-deadbeef",
            "seed": 42,
            "parent_run_id": "goat-v1-fold0",
            "date": "2026-07-27",
            "config_schema_version": "brats2026-config/1",
        },
        "initial_lr": 1e-3,
        "weight_decay": 3e-5,
        "num_epochs": 100,
        "batch_size": 2,
        "patch_size": [128, 160, 112],          # YAML list -> coerced to tuple
        "oversample_foreground_percent": 0.33,
        "loss_dice_weight": 1.0,
        "loss_ce_weight": 1.0,
        "da_scale_range": [0.6, 1.6],           # YAML list -> coerced to tuple
        "da_noise_p": 0.2,
        "da_blur_p": 0.3,
        "da_brightness_p": 0.3,
        "da_contrast_p": 0.3,
        "da_low_res_p": 0.35,
        "da_gamma_p": 0.3,
        "da_bias_field_p": 0.3,
        "da_bias_field_order": 3,
        "da_bias_field_coeff_range": [0.0, 0.1],  # YAML list -> coerced to tuple
        "init_weights_from": _CKPT,
    }


class _DummyTrainer:
    """Duck-typed stand-in for an nnUNetTrainer (the attributes apply_goat_v2_config sets)."""

    initial_lr = None
    weight_decay = None
    num_epochs = None
    oversample_foreground_percent = None

# The v1 hook tuple, frozen here verbatim. configs/train.yaml (the IN-FLIGHT run) is validated
# against it via config.TRAIN_HOOK_KEYS, so any drift would invalidate a running config.
_V1_HOOKS_FROZEN = (
    "initial_lr",
    "weight_decay",
    "num_epochs",
    "batch_size",
    "patch_size",
    "oversample_foreground_percent",
    "loss_dice_weight",
    "loss_ce_weight",
    "domain_sampling_alpha",
    "domain_rand_intensity_sigma",
    "domain_rand_contrast_range",
    "domain_rand_noise_sigma",
)


def test_importing_v2_does_not_mutate_v1_hooks():
    """In-flight guard: importing v2 must not shift what configs/train.yaml is validated against."""
    assert V1_HOOKS == _V1_HOOKS_FROZEN


def test_fresh_v2_config_has_every_hook_unset():
    missing = unset_hooks(GoATTrainerConfigV2())
    for hook in SPECIALIST_HOOKS_V2:
        assert hook in missing


def test_assert_configured_refuses_unset_config():
    with pytest.raises(ValueError) as exc:
        assert_configured(GoATTrainerConfigV2())
    assert "SPECIALIST" in str(exc.value)


def test_init_weights_from_is_a_choice_not_a_required_hook():
    """``None`` must stay legal (= train from scratch), which assert_configured would forbid."""
    field_names = {f.name for f in fields(GoATTrainerConfigV2())}
    assert "init_weights_from" in field_names
    assert "init_weights_strict" in field_names
    assert "init_weights_from" not in SPECIALIST_HOOKS_V2
    assert "init_weights_strict" not in SPECIALIST_HOOKS_V2


def test_v2_drops_the_unusable_cohort_sampling_hook():
    """On-disk ids are BraTS-GoAT-NNNNN -> cohort UNK for all 1351 cases, so it cannot work."""
    assert "domain_sampling_alpha" not in SPECIALIST_HOOKS_V2


def test_train_v2_kind_is_registered_with_every_v2_hook():
    tpl = make_template("train_v2")
    assert tpl["kind"] == "train_v2"
    for hook in SPECIALIST_HOOKS_V2:
        assert hook in tpl


def test_train_v2_hook_keys_in_lockstep_with_v2_specialist_hooks():
    # Regression guard against code/config drift, mirroring the v1 guard in tests/test_config.py.
    assert TRAIN_V2_HOOK_KEYS == SPECIALIST_HOOKS_V2


def test_registering_train_v2_leaves_the_train_kind_untouched():
    """In-flight guard: the running configs/train.yaml is validated against kind 'train'."""
    assert TRAIN_HOOK_KEYS == _V1_HOOKS_FROZEN
    tpl = make_template("train")
    assert set(tpl) - {"kind", "provenance"} == set(_V1_HOOKS_FROZEN)


def test_v2_declares_the_augmentation_hooks_it_wires():
    for hook in (
        "da_scale_range",
        "da_noise_p",
        "da_blur_p",
        "da_brightness_p",
        "da_contrast_p",
        "da_low_res_p",
        "da_gamma_p",
        "da_bias_field_p",
        "da_bias_field_order",
        "da_bias_field_coeff_range",
    ):
        assert hook in SPECIALIST_HOOKS_V2


def test_config_from_mapping_coerces_yaml_lists_to_tuples():
    cfg = config_from_mapping(_full_v2_mapping())
    assert cfg.patch_size == (128, 160, 112)
    assert cfg.da_scale_range == (0.6, 1.6)
    assert cfg.da_bias_field_coeff_range == (0.0, 0.1)


def test_config_from_mapping_reads_the_warm_start_choice():
    """init_weights_* are not hooks, so they need explicit handling or they are silently dropped."""
    cfg = config_from_mapping(_full_v2_mapping())
    assert cfg.init_weights_from == _CKPT


def test_config_from_mapping_leaves_warm_start_none_when_absent():
    mapping = _full_v2_mapping()
    del mapping["init_weights_from"]
    assert config_from_mapping(mapping).init_weights_from is None


def test_a_fully_filled_mapping_is_configured():
    assert unset_hooks(config_from_mapping(_full_v2_mapping())) == []


def test_apply_v2_config_sets_the_optimiser_attributes():
    trainer = _DummyTrainer()
    apply_goat_v2_config(trainer, config_from_mapping(_full_v2_mapping()))
    assert trainer.initial_lr == 1e-3
    assert trainer.weight_decay == 3e-5
    assert trainer.num_epochs == 100
    assert trainer.oversample_foreground_percent == 0.33


def test_apply_v2_config_refuses_an_unconfigured_config():
    with pytest.raises(ValueError):
        apply_goat_v2_config(_DummyTrainer(), GoATTrainerConfigV2())


# --- warm start ---------------------------------------------------------------------------

_MODEL = {"enc.0.weight": (32, 4, 3, 3, 3), "enc.0.bias": (32,), "seg.weight": (3, 32, 1, 1, 1)}


def test_identical_shapes_transfer_completely():
    plan = plan_weight_transfer(dict(_MODEL), dict(_MODEL))
    assert set(plan.transferable) == set(_MODEL)
    assert plan.shape_mismatched == []
    assert plan.missing_in_checkpoint == []


def test_a_shape_mismatch_is_reported_not_transferred():
    ckpt = dict(_MODEL) | {"seg.weight": (4, 32, 1, 1, 1)}  # different number of output classes
    plan = plan_weight_transfer(ckpt, dict(_MODEL))
    assert plan.shape_mismatched == ["seg.weight"]
    assert "seg.weight" not in plan.transferable


def test_keys_absent_from_the_checkpoint_are_reported():
    ckpt = {"enc.0.weight": (32, 4, 3, 3, 3)}
    plan = plan_weight_transfer(ckpt, dict(_MODEL))
    assert set(plan.missing_in_checkpoint) == {"enc.0.bias", "seg.weight"}


def test_assert_transferable_refuses_a_checkpoint_that_shares_nothing():
    """The failure mode that must never be silent: a wrong checkpoint -> random init -> wasted GPU days."""
    plan = plan_weight_transfer({"totally.different": (1,)}, dict(_MODEL))
    with pytest.raises(ValueError) as exc:
        assert_transferable(plan, strict=False)
    assert "no tensor" in str(exc.value).lower()


def test_assert_transferable_tolerates_a_partial_match_when_not_strict():
    plan = plan_weight_transfer({"enc.0.weight": (32, 4, 3, 3, 3)}, dict(_MODEL))
    assert_transferable(plan, strict=False)  # must not raise


def test_assert_transferable_refuses_a_partial_match_when_strict():
    plan = plan_weight_transfer({"enc.0.weight": (32, 4, 3, 3, 3)}, dict(_MODEL))
    with pytest.raises(ValueError):
        assert_transferable(plan, strict=True)


def test_assert_transferable_refuses_a_shape_mismatch_even_when_not_strict():
    """A mismatch means the architectures differ — silently skipping it would corrupt the warm start."""
    ckpt = dict(_MODEL) | {"seg.weight": (4, 32, 1, 1, 1)}
    with pytest.raises(ValueError):
        assert_transferable(plan_weight_transfer(ckpt, dict(_MODEL)), strict=False)


def test_v2_hook_names_pass_the_prior_weights_compliance_gate():
    # Guard: naming a hook `pretrained_*` would trip our own GoAT gate (compliance.py:18-22).
    result = check_no_prior_weights(list(SPECIALIST_HOOKS_V2) + list(_CHOICE_KEYS))
    assert result.status is Status.PASS


# --- discovery shim -----------------------------------------------------------------------

def test_v2_shim_filename_differs_from_v1():
    """In-flight guard: reusing brats2026_goat.py would clobber the running trainer's shim."""
    assert GOAT_V2_SHIM_NAME != "brats2026_goat.py"


def test_v2_shim_body_imports_the_v2_trainer():
    assert "trainer_v2" in GOAT_V2_SHIM_BODY
    assert "nnUNetTrainerGoATv2" in GOAT_V2_SHIM_BODY


def test_v2_config_env_var_differs_from_v1():
    """Both must be exportable in one shell without the v2 config hijacking the v1 run."""
    assert GOAT_V2_TRAIN_CONFIG_ENV != GOAT_TRAIN_CONFIG_ENV


# --- bias field (the one corruption nnU-Net's default DA does not model) --------------------

def test_bias_field_has_the_requested_shape():
    field = bias_field_multiplier((4, 5, 6), order=3, coefficients=[0.05] * 20)
    assert field.shape == (4, 5, 6)


def test_bias_field_is_strictly_positive():
    """It multiplies the image, so a non-positive value would flip or annihilate intensities."""
    field = bias_field_multiplier((6, 6, 6), order=3, coefficients=[0.3] * 20)
    assert (field > 0).all()


def test_zero_coefficients_give_an_identity_field():
    field = bias_field_multiplier((4, 4, 4), order=3, coefficients=[0.0] * 20)
    assert np.allclose(field, 1.0)


def test_order_zero_gives_a_spatially_constant_field():
    field = bias_field_multiplier((3, 4, 5), order=0, coefficients=[0.2])
    assert np.allclose(field, field.flat[0])


def test_bias_field_is_smooth_not_noise():
    """A bias field models slow scanner inhomogeneity; neighbouring voxels must stay close."""
    field = bias_field_multiplier((16, 16, 16), order=3, coefficients=[0.1] * 20)
    assert np.abs(np.diff(field, axis=0)).max() < 0.1 * field.mean()


def test_bias_field_rejects_too_few_coefficients():
    with pytest.raises(ValueError):
        bias_field_multiplier((4, 4, 4), order=3, coefficients=[0.1])


def test_bias_field_term_count_matches_the_cubic_20():
    """order=3 in 3D is 20 terms — the number the config's coeff vector must supply."""
    assert bias_field_term_count(3) == 20
    assert bias_field_term_count(0) == 1


def test_sampled_coefficients_fill_exactly_one_polynomial():
    rng = random.Random(0)
    coeffs = sample_bias_field_coefficients(3, (0.0, 0.1), rng)
    assert len(coeffs) == bias_field_term_count(3)
    bias_field_multiplier((4, 4, 4), order=3, coefficients=coeffs)  # must not raise


def test_sampled_coefficients_take_both_signs():
    """A one-sided field is a systematic brightening, not scanner inhomogeneity."""
    coeffs = sample_bias_field_coefficients(3, (0.05, 0.1), random.Random(0))
    assert min(coeffs) < 0 < max(coeffs)


# --- augmentation retuning ------------------------------------------------------------------

class _Wrapped:
    """Stand-in for a batchgeneratorsv2 transform (identified by class name)."""


class GaussianNoiseTransform(_Wrapped):
    pass


class SimulateLowResolutionTransform(_Wrapped):
    pass


class SpatialTransform(_Wrapped):
    def __init__(self):
        self.scaling = (0.7, 1.4)


class _Random:
    """Stand-in for RandomTransform: wraps a transform and carries apply_probability."""

    def __init__(self, transform, apply_probability):
        self.transform = transform
        self.apply_probability = apply_probability


class _Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)


def _pipeline():
    return _Compose([
        SpatialTransform(),
        _Random(GaussianNoiseTransform(), apply_probability=0.1),
        _Random(SimulateLowResolutionTransform(), apply_probability=0.25),
    ])


def test_retune_overrides_nnunet_default_probabilities():
    """da_*_p REPLACE nnU-Net's defaults (0.1 noise / 0.25 low-res), they don't add to them."""
    pipeline = _pipeline()
    retune_augmentation(pipeline, config_from_mapping(_full_v2_mapping()))
    assert pipeline.transforms[1].apply_probability == 0.2
    assert pipeline.transforms[2].apply_probability == 0.35


def test_retune_replaces_the_spatial_scaling_range():
    pipeline = _pipeline()
    retune_augmentation(pipeline, config_from_mapping(_full_v2_mapping()))
    assert pipeline.transforms[0].scaling == (0.6, 1.6)


def test_retune_reports_hooks_that_matched_no_transform():
    """A renamed upstream transform must not silently disable an override."""
    report = retune_augmentation(_pipeline(), config_from_mapping(_full_v2_mapping()))
    assert "da_gamma_p" in report["missing"]
    assert "da_noise_p" in report["retuned"]


class GammaTransform(_Wrapped):
    """nnU-Net ships two of these; only the non-inverted one is `da_gamma_p`."""

    def __init__(self, p_invert_image):
        self.p_invert_image = p_invert_image


def test_retune_leaves_the_inverted_gamma_pass_alone():
    """Matching on class name alone would raise nnU-Net's inverted gamma 0.10 -> 0.45 unasked."""
    inverted = _Random(GammaTransform(p_invert_image=1), apply_probability=0.10)
    plain = _Random(GammaTransform(p_invert_image=0), apply_probability=0.30)
    retune_augmentation(_Compose([inverted, plain]), config_from_mapping(_full_v2_mapping()))
    assert inverted.apply_probability == 0.10   # untouched — the config does not expose it
    assert plain.apply_probability == 0.3       # da_gamma_p from the mapping


def test_coarse_bias_field_grid_is_smaller_than_a_real_patch():
    """The field is evaluated coarsely and upsampled; at patch resolution it dominates the loader."""
    assert BIAS_FIELD_EVAL_GRID < min(_full_v2_mapping()["patch_size"])


def test_retune_leaves_unset_hooks_alone():
    """An unset hook keeps nnU-Net's default rather than being coerced to 0."""
    pipeline = _pipeline()
    mapping = _full_v2_mapping()
    del mapping["da_noise_p"]
    retune_augmentation(pipeline, config_from_mapping(mapping))
    assert pipeline.transforms[1].apply_probability == 0.1


# --- config resolution ----------------------------------------------------------------------

def test_resolve_config_prefers_an_already_filled_config(monkeypatch):
    monkeypatch.setenv(GOAT_V2_TRAIN_CONFIG_ENV, "/nonexistent/train_v2.yaml")
    filled = config_from_mapping(_full_v2_mapping())
    assert resolve_v2_config(filled) is filled  # env var must not be read when already configured


def test_resolve_config_refuses_when_unset_and_no_env(monkeypatch):
    monkeypatch.delenv(GOAT_V2_TRAIN_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError):
        resolve_v2_config(GoATTrainerConfigV2())


# --- launch wiring --------------------------------------------------------------------------

def test_train_v2_command_uses_the_v2_trainer():
    argv = train_v2_command(fold=0)
    assert argv[argv.index("-tr") + 1] == GOAT_TRAINER_V2 == "nnUNetTrainerGoATv2"


def test_train_v2_keeps_plans_and_configuration_of_v1():
    """The warm start is shape-compatible only if plans + configuration match the parent run."""
    v1, v2 = train_command(fold=0), train_v2_command(fold=0)
    assert v1[v1.index("-p") + 1] == v2[v2.index("-p") + 1]
    assert v1[1:3] == v2[1:3]  # dataset id + configuration


def test_v1_and_v2_write_to_different_results_directories():
    """nnU-Net derives its output dir from <trainer>__<plans>__<config>."""
    assert GOAT_TRAINER != GOAT_TRAINER_V2
