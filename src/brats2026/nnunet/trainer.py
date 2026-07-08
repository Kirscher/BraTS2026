"""Phase 2 — ``nnUNetTrainerGoAT`` scaffold.

This module defines the *shape* of the GoAT trainer:

- **domain-balanced sampling** across the 5 cohorts (counter the GLI majority),
- **domain-randomization** augmentation (intensity / contrast / noise) for cross-scanner
  generalisation,
- **region-based deep supervision** for ET/TC/WT.

It invents **no** numbers. Every tunable is a ``# SPECIALIST:`` hook held in
:class:`GoATTrainerConfig` with the field defaulting to ``None`` (meaning "specialist must
set"). :data:`SPECIALIST_HOOKS` lists exactly what the human AI specialist owns, and
:func:`unset_hooks` reports which are still missing so a run can refuse to start unconfigured.

The concrete ``nnUNetTrainerGoAT`` class is only defined when ``nnunetv2`` is importable, so
this module (and its tests) load fine in a bare environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# nnU-Net locates a trainer by NAME, scanning only its own package tree — so the config a run
# should use can't be passed as an object. Instead the trainer reads this env var (a path to a
# filled configs/train.yaml) at initialize() time. Set it before `nnUNetv2_train`.
GOAT_TRAIN_CONFIG_ENV = "BRATS_GOAT_TRAIN_CONFIG"

# SPECIALIST tuple-valued hooks: YAML carries them as lists; coerce to tuples when building.
_TUPLE_HOOKS: tuple[str, ...] = ("patch_size", "domain_rand_contrast_range")

# The knobs the specialist owns. Names mirror GoATTrainerConfig fields below.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "initial_lr",
    "weight_decay",
    "num_epochs",
    "batch_size",
    "patch_size",
    "oversample_foreground_percent",
    "loss_dice_weight",
    "loss_ce_weight",
    "domain_sampling_alpha",      # 0 = natural freq, 1 = uniform across cohorts
    "domain_rand_intensity_sigma",
    "domain_rand_contrast_range",
    "domain_rand_noise_sigma",
)


@dataclass
class GoATTrainerConfig:
    """Hyperparameters for ``nnUNetTrainerGoAT`` — all ``# SPECIALIST:`` hooks.

    Defaults are ``None`` on purpose: a ``None`` means "not yet decided by the specialist".
    The nnU-Net stock default is noted in the comment only, never silently applied.
    """

    initial_lr: Optional[float] = None           # SPECIALIST: nnU-Net default 1e-2
    weight_decay: Optional[float] = None         # SPECIALIST: nnU-Net default 3e-5
    num_epochs: Optional[int] = None             # SPECIALIST: nnU-Net default 1000
    batch_size: Optional[int] = None             # SPECIALIST: planner-derived; watch A10G 24GB
    patch_size: Optional[tuple[int, int, int]] = None  # SPECIALIST: planner-derived
    oversample_foreground_percent: Optional[float] = None  # SPECIALIST: nnU-Net default 0.33
    loss_dice_weight: Optional[float] = None     # SPECIALIST: region Dice weight
    loss_ce_weight: Optional[float] = None       # SPECIALIST: region BCE/CE weight
    domain_sampling_alpha: Optional[float] = None  # SPECIALIST: cohort re-balancing strength
    domain_rand_intensity_sigma: Optional[float] = None  # SPECIALIST: intensity jitter
    domain_rand_contrast_range: Optional[tuple[float, float]] = None  # SPECIALIST: gamma range
    domain_rand_noise_sigma: Optional[float] = None  # SPECIALIST: additive noise


def unset_hooks(config: GoATTrainerConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided)."""
    return [f.name for f in fields(config) if getattr(config, f.name) is None]


def assert_configured(config: GoATTrainerConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before launching a real run."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to launch nnUNetTrainerGoAT: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


def config_from_mapping(cfg: dict) -> GoATTrainerConfig:
    """Build a :class:`GoATTrainerConfig` from a train-config mapping (a loaded ``train.yaml``).

    Only :data:`SPECIALIST_HOOKS` keys are read; list-valued tuple hooks (``patch_size``,
    ``domain_rand_contrast_range``) are coerced from YAML lists to tuples. Missing / ``None`` keys
    stay ``None`` so :func:`unset_hooks` / :func:`assert_configured` still gate an unfilled config.
    """
    kwargs: dict = {}
    for hook in SPECIALIST_HOOKS:
        value = cfg.get(hook)
        if value is None:
            continue
        if hook in _TUPLE_HOOKS and isinstance(value, list):
            value = tuple(value)
        kwargs[hook] = value
    return GoATTrainerConfig(**kwargs)


def load_goat_config(path: str | Path) -> GoATTrainerConfig:
    """Load, VALIDATE and build a :class:`GoATTrainerConfig` from a ``configs/train.yaml`` file.

    Provenance-header + all-hooks validation is delegated to
    :func:`brats2026.config.assert_valid_config` (kind ``"train"``), so a stub train.yaml (empty
    header / ``None`` hooks) is refused BEFORE any training starts — no run launches unconfigured.
    """
    from ..config import assert_valid_config, load_config

    cfg = load_config(Path(path))
    assert_valid_config(cfg, "train")
    return config_from_mapping(cfg)


def apply_goat_config(trainer, config: GoATTrainerConfig) -> None:
    """Copy the directly-mappable SPECIALIST numbers onto a live ``nnUNetTrainer``.

    Sets ``initial_lr``, ``weight_decay``, ``num_epochs`` and ``oversample_foreground_percent`` —
    the standard nnU-Net attributes — and MUST run before ``super().initialize()`` so the optimizer
    and dataloaders pick them up. ``batch_size`` / ``patch_size`` come from the plans (planner),
    not here; loss weights + domain sampling/augmentation are the remaining TODO in
    :meth:`nnUNetTrainerGoAT.initialize`. Duck-typed on ``trainer`` so it is unit-testable without
    nnU-Net installed.
    """
    assert_configured(config)
    trainer.initial_lr = config.initial_lr
    trainer.weight_decay = config.weight_decay
    trainer.num_epochs = config.num_epochs
    trainer.oversample_foreground_percent = config.oversample_foreground_percent


def install_goat_trainer() -> str:
    """Make nnU-Net discover ``nnUNetTrainerGoAT`` by dropping an import shim into its trainer tree.

    nnU-Net v2 finds trainers by scanning ``nnunetv2/training/nnUNetTrainer`` only, so a trainer in
    this external package is not found by default. This writes a tiny module there that imports our
    class, so ``nnUNetv2_train … -tr nnUNetTrainerGoAT`` resolves. Returns the shim path; idempotent.
    Requires nnU-Net installed (import is local so the rest of the module stays import-light).
    """
    import nnunetv2

    variants = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    shim = variants / "brats2026_goat.py"
    shim.write_text(
        "# Auto-generated by `brats2026 install-trainer` so nnU-Net can discover the GoAT trainer.\n"
        "from brats2026.nnunet.trainer import nnUNetTrainerGoAT  # noqa: F401\n",
        encoding="utf-8",
    )
    return str(shim)


def domain_sampling_weights(cohort_counts: dict[str, int], alpha: float) -> dict[str, float]:
    """Per-cohort sampling weights interpolated between natural frequency and uniform.

    ``alpha=0`` keeps the natural distribution; ``alpha=1`` samples cohorts uniformly. The
    *value* of ``alpha`` is a SPECIALIST hook — this function only defines the interpolation.
    Weights are normalised to sum to 1 over cohorts with at least one case.
    """
    present = {c: n for c, n in cohort_counts.items() if n > 0}
    if not present:
        return {}
    total = sum(present.values())
    k = len(present)
    weights = {}
    for cohort, n in present.items():
        natural = n / total
        uniform = 1.0 / k
        weights[cohort] = (1 - alpha) * natural + alpha * uniform
    norm = sum(weights.values())
    return {c: w / norm for c, w in weights.items()}


# --- concrete trainer (only when nnU-Net is available) -----------------------------------
try:  # pragma: no cover - exercised only in a full training environment
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class nnUNetTrainerGoAT(nnUNetTrainer):
        """Region-based GoAT trainer with domain-balanced sampling + domain randomization.

        Hyperparameters must be injected from a configured :class:`GoATTrainerConfig`; this
        scaffold wires the structure, the specialist supplies the numbers.
        """

        goat_config: GoATTrainerConfig = GoATTrainerConfig()

        def initialize(self):  # noqa: D401
            # Resolve the config: if the class default is still unconfigured, load the filled
            # configs/train.yaml pointed to by $BRATS_GOAT_TRAIN_CONFIG (see GOAT_TRAIN_CONFIG_ENV).
            cfg = self.goat_config
            if unset_hooks(cfg):
                env_path = os.environ.get(GOAT_TRAIN_CONFIG_ENV)
                if env_path:
                    cfg = load_goat_config(env_path)
                    self.goat_config = cfg
            assert_configured(cfg)                 # refuses to launch if still unset
            apply_goat_config(self, cfg)           # lr / wd / epochs / oversample BEFORE super()
            super().initialize()
            # TODO(nnunet-trainer): install the domain-balanced sampler (domain_sampling_weights)
            # and the domain-randomization transforms (intensity/contrast/noise) + region loss
            # weights. Until then this is a CONFIGURED region-based nnU-Net (ET/TC/WT declared in
            # dataset.json), not yet the full GoAT recipe — see docs/RUN_HPC.md.

except ImportError:  # nnU-Net not installed (e.g. CI / planning env)
    nnUNetTrainerGoAT = None  # type: ignore[assignment]
