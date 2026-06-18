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

from dataclasses import dataclass, fields
from typing import Optional

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
            assert_configured(self.goat_config)
            # SPECIALIST: apply self.goat_config to lr/epochs/batch/patch/loss before super().
            super().initialize()
            # SPECIALIST: install domain-balanced sampler using domain_sampling_weights(...).
            # SPECIALIST: extend train transforms with domain-randomization (intensity/
            #             contrast/noise) at the configured magnitudes.

except ImportError:  # nnU-Net not installed (e.g. CI / planning env)
    nnUNetTrainerGoAT = None  # type: ignore[assignment]
