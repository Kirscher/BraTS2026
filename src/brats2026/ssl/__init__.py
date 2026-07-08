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
from ..nnunet.convert import DATASET_ID
from ..nnunet.plan import train_command

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


def select_from_stats_records(
    records: Sequence[dict], config: GoATSelfTrainingConfig
) -> dict[str, list[str]]:
    """Apply the pseudo-label filters to a list of per-case stats **records** (dicts).

    Each record must carry the :class:`CaseStats` fields (``case_id``, ``mean_fg_softmax``,
    ``n_fg_voxels``, ``fg_fraction``, ``frac_confident_voxels``). The cohort of each case is
    decoded from its ID via :func:`brats2026.domains.cohort_from_case_id`. Returns the accepted
    cohort→case-id mapping from :func:`select_pseudo_cases`. This is the pure core the
    ``brats2026 ssl-select`` CLI wraps: the producer (inference / data-pipeline) computes the
    stats from the teacher's softmax maps, this decides which cases survive the filters.
    """
    stats = [
        CaseStats(
            case_id=r["case_id"],
            mean_fg_softmax=float(r["mean_fg_softmax"]),
            n_fg_voxels=int(r["n_fg_voxels"]),
            fg_fraction=float(r["fg_fraction"]),
            frac_confident_voxels=float(r["frac_confident_voxels"]),
        )
        for r in records
    ]
    cohort_of = {s.case_id: cohort_from_case_id(s.case_id) for s in stats}
    return select_pseudo_cases(stats, cohort_of, config)


def load_ssl_config(path: str | Path) -> GoATSelfTrainingConfig:
    """Build a :class:`GoATSelfTrainingConfig` from a ``configs/ssl.yaml`` file.

    Reads the SPECIALIST hook values (and the ``allow_validation_pool`` rule gate) from the YAML
    and returns a config object. ``teacher_folds`` is coerced from a YAML list to a tuple.
    :func:`assert_configured` still governs whether the result may launch a run.
    """
    from ..config import load_config

    cfg = load_config(Path(path))
    kwargs: dict = {}
    for hook in SPECIALIST_HOOKS:
        if hook in cfg and cfg[hook] is not None:
            value = cfg[hook]
            if hook == "teacher_folds" and isinstance(value, list):
                value = tuple(value)
            kwargs[hook] = value
    if cfg.get("allow_validation_pool") is not None:
        kwargs["allow_validation_pool"] = bool(cfg["allow_validation_pool"])
    return GoATSelfTrainingConfig(**kwargs)


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


def round_dataset_id(base_dataset_id: int, round_index: int) -> int:
    """nnU-Net dataset id of the augmented (labelled + round-r accepted pseudo) training set.

    Each self-training round trains on its OWN dataset so students never clobber one another and
    the next round can use round r's student as its teacher (teacher[r+1] = the student trained on
    ``round_dataset_id(base, r)``). data-pipeline builds these augmented datasets under the ids
    this returns (``base + 100 + r``, e.g. 501 → 601, 602).
    """
    return base_dataset_id + 100 + round_index


def pseudo_select_command(
    stats_json: str | Path, config_path: str | Path, accepted_out: str | Path
) -> list[str]:
    """The explicit filter step: ``brats2026 ssl-select`` over the teacher's per-case stats.

    Sits BETWEEN pseudo-label generation and the student train in every round, so the plan
    *contains* the acceptance / rejection + cohort-quota decision instead of silently training on
    every teacher prediction (the previous gap). See :func:`select_from_stats_records`.
    """
    return [
        "brats2026", "ssl-select",
        "--stats-json", str(stats_json),
        "--config", str(config_path),
        "--out", str(accepted_out),
    ]


def self_training_round_commands(
    round_index: int,
    teacher: str,
    unlabeled_input_dir: str | Path,
    work_root: str | Path,
    config: GoATSelfTrainingConfig,
    base_dataset_id: int = DATASET_ID,
    config_path: str | Path = "configs/ssl.yaml",
) -> list[list[str]]:
    """Argv list for one self-training round: pseudo-gen → **filter** → student train.

    ``teacher`` is the nnU-Net dataset id / results identifier the teacher was trained on (passed
    to ``nnUNetv2_predict -d``): the seed teacher for round 0, then ``round_dataset_id(base, r-1)``
    for later rounds (see :func:`self_training_plan`). Pseudo-labels land under
    ``<work_root>/round<r>/pseudo``; the filter step reads ``round<r>/stats.json`` and writes
    ``round<r>/accepted.json``; the student then trains on this round's own
    ``round_dataset_id(base, round_index)`` via :func:`brats2026.nnunet.plan.train_command` (so the
    ResEnc-L plans are pinned with ``-p``), one command per fold in ``teacher_folds``.
    """
    work_root = Path(work_root)
    round_dir = work_root / f"round{round_index}"
    pseudo_output_dir = round_dir / "pseudo"

    cmds: list[list[str]] = list(
        pseudo_label_predict_commands(teacher, unlabeled_input_dir, pseudo_output_dir, config)
    )
    cmds.append(
        pseudo_select_command(round_dir / "stats.json", config_path, round_dir / "accepted.json")
    )

    dataset_id = round_dataset_id(base_dataset_id, round_index)
    for fold in tuple(config.teacher_folds):
        cmds.append(train_command(fold=fold, dataset_id=dataset_id))
    return cmds


def self_training_plan(
    config: GoATSelfTrainingConfig,
    seed_teacher: str,
    unlabeled_input_dir: str | Path,
    work_root: str | Path,
    base_dataset_id: int = DATASET_ID,
    config_path: str | Path = "configs/ssl.yaml",
) -> list[list[list[str]]]:
    """Full bounded self-training plan: one argv-list per round, teacher[r] = student[r-1].

    Round 0 predicts with ``seed_teacher`` (the from-scratch inner-dev model); round r>0 predicts
    with the student trained in round r-1, identified by ``round_dataset_id(base, r-1)`` — the
    dataset that student's model actually lives under in ``nnUNet_results`` (fixing the previous
    dangling ``work/round<r>/student`` path that nnU-Net never wrote to). ``n_rounds`` (1 or 2) is
    a SPECIALIST hook. No command is executed — the human-gated runner consumes this plan.
    """
    rounds: list[list[list[str]]] = []
    for r in range(int(config.n_rounds)):
        teacher = seed_teacher if r == 0 else str(round_dataset_id(base_dataset_id, r - 1))
        rounds.append(
            self_training_round_commands(
                r, teacher, unlabeled_input_dir, work_root, config,
                base_dataset_id=base_dataset_id, config_path=config_path,
            )
        )
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
