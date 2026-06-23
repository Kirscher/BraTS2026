"""Phase 4 — SSL / self-training (pseudo-labelling) on GoAT's own unlabeled cases.

A *teacher* (an ``nnUNetTrainerGoAT`` trained from scratch on the labelled inner-dev cases)
predicts on the unlabeled GoAT pool; predictions are filtered per-voxel (low-confidence
voxels become the nnU-Net ``ignore-label``) and per-case (near-empty / hallucinated /
low-confidence cases are rejected); a *student* is then re-trained on labelled + accepted
pseudo cases. Bounded to ``n_rounds=2`` and composed *on top of* ``nnUNetTrainerGoAT`` — no
new training loop.

Like :mod:`brats2026.nnunet.trainer` this module invents **no** numbers: every tunable is a
``# SPECIALIST:`` hook on :class:`GoATSelfTrainingConfig` defaulting to ``None``. The single
exception is :attr:`GoATSelfTrainingConfig.allow_validation_pool`, a *compliance* gate (not a
hyperparameter) with a safe default and excluded from :data:`SPECIALIST_HOOKS`.

The argv builders reuse :mod:`brats2026.inference.predict`; the cohort / compliance guards
reuse :mod:`brats2026.domains` and :mod:`brats2026.compliance`. Pure functions stay
numpy/stdlib only so this module (and its tests) load and run without torch / nnU-Net.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

from ..domains import UNKNOWN_COHORT, cohort_from_case_id
from ..inference.predict import predict_command, predict_ensemble_commands

# The knobs the specialist owns. Names mirror GoATSelfTrainingConfig fields below.
# ``allow_validation_pool`` is deliberately absent: it is a conformance gate, not a tunable.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "n_rounds",
    "confidence_threshold_voxel",
    "confidence_threshold_case",
    "case_confidence_metric",
    "min_pseudo_foreground_voxels",
    "max_pseudo_foreground_fraction",
    "labeled_unlabeled_ratio",
    "per_cohort_pseudo_quota",
    "pseudo_label_softmax_temperature",
    "ignore_label_index",
    "pseudo_use_tta",
    "pseudo_use_ensemble",
    "teacher_folds",
    "recompute_pseudo_each_round",
    "seed",
)

# Accepted ``case_confidence_metric`` values (the metric *name* is a SPECIALIST hook; the set
# of supported metrics is the mechanism this module provides).
CASE_CONFIDENCE_METRICS: tuple[str, ...] = (
    "mean_fg_softmax",
    "frac_confident_voxels",
)


@dataclass
class GoATSelfTrainingConfig:
    """Self-training hyperparameters — all ``# SPECIALIST:`` hooks except the rule gate.

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied.
    """

    n_rounds: Optional[int] = None                       # SPECIALIST: 1–2
    confidence_threshold_voxel: Optional[float] = None   # SPECIALIST: softmax max/voxel, ~0.75–0.95
    confidence_threshold_case: Optional[float] = None    # SPECIALIST: aggregated score/case, ~0.85
    case_confidence_metric: Optional[str] = None         # SPECIALIST: mean_fg_softmax|frac_confident_voxels
    min_pseudo_foreground_voxels: Optional[int] = None   # SPECIALIST: reject near-empty cases, ~100–500
    max_pseudo_foreground_fraction: Optional[float] = None  # SPECIALIST: anti-hallucination, ~0.5
    labeled_unlabeled_ratio: Optional[float] = None      # SPECIALIST: 1.0–3.0
    per_cohort_pseudo_quota: Optional[bool] = None       # SPECIALIST: cap #pseudo/cohort
    pseudo_label_softmax_temperature: Optional[float] = None  # SPECIALIST: default 1.0
    ignore_label_index: Optional[int] = None             # SPECIALIST: nnU-Net ignore-label index
    pseudo_use_tta: Optional[bool] = None                # SPECIALIST: mirroring at generation
    pseudo_use_ensemble: Optional[bool] = None           # SPECIALIST: ensemble inner-dev folds = teacher
    teacher_folds: Optional[tuple[int, ...]] = None      # SPECIALIST: teacher folds (inner-dev only)
    recompute_pseudo_each_round: Optional[bool] = None   # SPECIALIST: core of iterative self-training
    allow_validation_pool: bool = False                  # NON-SPECIALIST: rule gate; safe default
    seed: Optional[int] = None                           # SPECIALIST: RNG seed (provenance)


def unset_hooks(config: GoATSelfTrainingConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided).

    ``allow_validation_pool`` is a conformance gate, not a hook, so it is never reported here
    even though it is a dataclass field.
    """
    return [
        f.name
        for f in fields(config)
        if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None
    ]


def assert_configured(config: GoATSelfTrainingConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before launching a real self-training run."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to launch GoAT self-training: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- per-case aggregates (no raw pixels leave the predictor) ------------------------------

@dataclass
class CaseStats:
    """Per-case confidence aggregates derived from a teacher prediction.

    Holds only summary statistics — never raw pixels — so the selection logic stays pure and
    testable. The producer (data-pipeline / inference) computes these from the softmax maps.
    """

    case_id: str
    mean_fg_softmax: float        # mean over foreground voxels of the max softmax prob
    n_fg_voxels: int              # number of predicted-foreground voxels
    fg_fraction: float            # foreground voxels / total voxels in the volume
    frac_confident_voxels: float  # fraction of voxels with max softmax >= voxel threshold


def voxel_confidence_mask(max_softmax, threshold: float, temperature: float = 1.0):
    """Boolean mask of voxels whose (temperature-adjusted) max softmax meets ``threshold``.

    ``max_softmax`` is the per-voxel maximum class probability. ``temperature`` > 1 *softens*
    the (dominant-class) probability (less confident → fewer voxels pass), < 1 *sharpens* it;
    the adjusted probability is ``p ** temperature`` (the temperature-scaling effect on the
    dominant class). Both ``threshold`` and ``temperature`` are SPECIALIST hooks — this only
    defines the rule.
    """
    import numpy as np

    probs = np.asarray(max_softmax, dtype=float)
    if temperature != 1.0:
        probs = probs ** temperature
    return probs >= threshold


def case_confidence_score(stats: CaseStats, metric: str) -> float:
    """Aggregate a per-case confidence score according to ``metric``.

    Supported metrics are :data:`CASE_CONFIDENCE_METRICS`; the *choice* is a SPECIALIST hook.
    """
    if metric == "mean_fg_softmax":
        return stats.mean_fg_softmax
    if metric == "frac_confident_voxels":
        return stats.frac_confident_voxels
    raise ValueError(
        f"unknown case_confidence_metric {metric!r}; expected one of {CASE_CONFIDENCE_METRICS}"
    )


def accept_case(stats: CaseStats, config: GoATSelfTrainingConfig) -> bool:
    """Decide whether a pseudo-labelled case is good enough to train on.

    A case is accepted iff its aggregated confidence meets ``confidence_threshold_case``, it
    has at least ``min_pseudo_foreground_voxels`` predicted foreground, and its foreground
    fraction does not exceed ``max_pseudo_foreground_fraction`` (anti-hallucination). All three
    bounds are SPECIALIST hooks.
    """
    score = case_confidence_score(stats, config.case_confidence_metric)
    if score < config.confidence_threshold_case:
        return False
    if stats.n_fg_voxels < config.min_pseudo_foreground_voxels:
        return False
    if stats.fg_fraction > config.max_pseudo_foreground_fraction:
        return False
    return True


def select_pseudo_cases(
    case_stats: Sequence[CaseStats],
    cohort_of: dict[str, str],
    config: GoATSelfTrainingConfig,
) -> dict[str, list[str]]:
    """Group accepted pseudo cases by cohort, optionally enforcing a per-cohort quota.

    Rejected cases (see :func:`accept_case`) are dropped. When
    ``per_cohort_pseudo_quota`` is True, every cohort is capped at the size of the smallest
    non-empty accepted cohort so no cohort dominates the pseudo set (balanced augmentation).
    Order within each cohort follows the input order (deterministic).
    """
    accepted: dict[str, list[str]] = {}
    for stats in case_stats:
        if not accept_case(stats, config):
            continue
        cohort = cohort_of.get(stats.case_id, UNKNOWN_COHORT)
        accepted.setdefault(cohort, []).append(stats.case_id)

    if not config.per_cohort_pseudo_quota or not accepted:
        return accepted

    quota = min(len(ids) for ids in accepted.values())
    return {cohort: ids[:quota] for cohort, ids in accepted.items()}


def labeled_unlabeled_sampling_plan(
    n_labeled: int, n_pseudo: int, ratio: float
) -> dict[str, float]:
    """Normalised sampling probabilities for the labelled vs. pseudo pools.

    ``ratio`` is labelled:pseudo (e.g. ``2.0`` = draw labelled twice as often as pseudo). With
    no pseudo cases the plan is all-labelled. The two probabilities always sum to 1. ``ratio``
    is a SPECIALIST hook; this only defines the normalisation.
    """
    if n_pseudo <= 0 or n_labeled <= 0:
        return {"labeled": 1.0, "pseudo": 0.0}
    labeled_weight = ratio
    pseudo_weight = 1.0
    total = labeled_weight + pseudo_weight
    return {"labeled": labeled_weight / total, "pseudo": pseudo_weight / total}


# --- argv builders (reuse inference.predict; never run nnU-Net) ---------------------------

def pseudo_label_predict_commands(
    teacher_model_dir,
    unlabeled_input_dir: str | Path,
    pseudo_output_dir: str | Path,
    config: GoATSelfTrainingConfig,
) -> list[list[str]]:
    """Build the ``nnUNetv2_predict`` argv(s) that generate pseudo-labels with the teacher.

    When ``pseudo_use_ensemble`` is True, ``teacher_model_dir`` is treated as a sequence of
    ensemble members (one command each, mirroring :func:`predict_ensemble_commands`); otherwise
    a single command over ``teacher_folds`` is emitted. TTA mirroring is toggled by
    ``pseudo_use_tta``. Which models / folds form the teacher is config-driven — nothing here.
    """
    disable_tta = not bool(config.pseudo_use_tta)
    folds = tuple(config.teacher_folds)
    if config.pseudo_use_ensemble:
        return predict_ensemble_commands(
            teacher_model_dir,
            unlabeled_input_dir,
            pseudo_output_dir,
            folds=folds,
            disable_tta=disable_tta,
        )
    return [
        predict_command(
            model_dir=teacher_model_dir,
            input_dir=unlabeled_input_dir,
            output_dir=pseudo_output_dir,
            folds=folds,
            disable_tta=disable_tta,
        )
    ]


def self_training_round_commands(
    round_index: int,
    teacher_model_dir,
    train_data_dir: str | Path,
    unlabeled_input_dir: str | Path,
    work_root: str | Path,
    config: GoATSelfTrainingConfig,
) -> list[list[str]]:
    """Argv list for one self-training round: pseudo-label generation then a student train.

    The pseudo-labels for this round land under ``<work_root>/round<r>/pseudo`` and the student
    trains the GoAT dataset/folds via ``nnUNetv2_train``. Dataset id, plans and trainer are left
    as placeholders the data-pipeline / nnunet-trainer fill — this only sequences the argv.
    """
    work_root = Path(work_root)
    pseudo_output_dir = work_root / f"round{round_index}" / "pseudo"

    cmds: list[list[str]] = list(
        pseudo_label_predict_commands(
            teacher_model_dir, unlabeled_input_dir, pseudo_output_dir, config
        )
    )

    # SPECIALIST: dataset id / plans / configuration are owned by data-pipeline + ai-specialist.
    folds = tuple(config.teacher_folds)
    for fold in folds:
        cmds.append(
            [
                "nnUNetv2_train",
                "GOAT_DATASET_ID",   # SPECIALIST: nnU-Net dataset id of the labelled+pseudo set
                "3d_fullres",
                str(fold),
                "-tr",
                "nnUNetTrainerGoAT",
            ]
        )
    return cmds


def student_model_dir(work_root: str | Path, round_index: int) -> str:
    """Conventional output directory for the student trained in ``round_index``.

    The next round uses this as its teacher (teacher[r] = student[r-1]).
    """
    return str(Path(work_root) / f"round{round_index}" / "student")


def self_training_plan(
    config: GoATSelfTrainingConfig,
    teacher_model_dir,
    train_data_dir: str | Path,
    unlabeled_input_dir: str | Path,
    work_root: str | Path,
) -> list[list[list[str]]]:
    """Full bounded self-training plan: one argv-list per round, teacher[r] = student[r-1].

    Round 0 predicts with the seed ``teacher_model_dir``; each subsequent round predicts with
    the student trained in the previous round. ``n_rounds`` (1 or 2) is a SPECIALIST hook. No
    command is executed — the human-gated runner consumes this plan.
    """
    rounds: list[list[list[str]]] = []
    current_teacher = teacher_model_dir
    for r in range(int(config.n_rounds)):
        rounds.append(
            self_training_round_commands(
                r, current_teacher, train_data_dir, unlabeled_input_dir, work_root, config
            )
        )
        current_teacher = student_model_dir(work_root, r)
    return rounds


# --- compliance guards (fatal assertions) ------------------------------------------------

def assert_goat_pool(
    case_ids: Sequence[str],
    allow_validation_pool: bool,
    validation_ids: Optional[Sequence[str]] = None,
    outer_lodo_ids: Optional[Sequence[str]] = None,
) -> None:
    """Fatal anti-leakage guard for the unlabeled self-training pool.

    Raises if:
    - any case ID does not resolve to a known GoAT cohort (``UNK`` → fatal, foreign data);
    - ``allow_validation_pool`` is False and the pool intersects ``validation_ids`` (fatal);
    - the pool intersects ``outer_lodo_ids`` (always fatal — the outer-LODO fold is sacred).

    Provenance / from-scratch checks live in :mod:`brats2026.compliance` and must be run on the
    pool and teacher before the first pseudo-label; this guard covers cohort + split disjunction.
    """
    pool = list(case_ids)

    unknown = [cid for cid in pool if cohort_from_case_id(cid) == UNKNOWN_COHORT]
    if unknown:
        sample = ", ".join(unknown[:3])
        raise ValueError(
            f"self-training pool contains {len(unknown)} case(s) with an unknown cohort "
            f"({UNKNOWN_COHORT}) — only GoAT cohorts are allowed (e.g. {sample})"
        )

    pool_set = set(pool)

    if not allow_validation_pool and validation_ids:
        collision = pool_set & set(validation_ids)
        if collision:
            sample = ", ".join(sorted(collision)[:3])
            raise ValueError(
                f"self-training pool collides with {len(collision)} validation case(s) while "
                f"allow_validation_pool=False (e.g. {sample}) — leakage refused"
            )

    if outer_lodo_ids:
        collision = pool_set & set(outer_lodo_ids)
        if collision:
            sample = ", ".join(sorted(collision)[:3])
            raise ValueError(
                f"self-training pool collides with {len(collision)} outer-LODO case(s) "
                f"(e.g. {sample}) — the held-out fold must never enter training"
            )
