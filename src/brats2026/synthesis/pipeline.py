"""Synthesis Stage 3 — dataset sizing / allocation + the (unexecuted) generation→train argv plan.

This module decides **how much** synthetic data to make and **how to spread it across cohorts**,
then assembles the offline command plan that generates it, builds an nnU-Net dataset from it, and
trains on the augmented set. Like :func:`brats2026.ssl.self_training_plan` it only *assembles argv*
— nothing is executed and no hyperparameter is invented here.

Pool **sizing is a ratio** (``synthetic_ratio``), not a fixed count: how many synthetic cases to
add relative to the labelled set, found by sweep (the ai-specialist owns the value). Allocation is
**by cohort deficit** — rarer cohorts (further below the largest cohort) get a bigger share, so the
augmentation serves the worst-cohort objective rather than mean-Dice — optionally tilted toward
under-represented cohorts by ``et_emphasis``. All counts come in as **inputs** (from data-pipeline),
so this module needs no real numbers to be written or tested. The augmented dataset lives at a
distinct nnU-Net id (:func:`synth_dataset_id` → 701, ``base + 200``, clear of ssl's ``base + 100``).

Every tunable is a ``# SPECIALIST:`` hook on :class:`SynthesisPoolConfig` defaulting to ``None``.
Pure stdlib so the module and its tests load without torch / nnU-Net.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Sequence

from ..nnunet.convert import DATASET_ID
from ..nnunet.plan import plan_and_preprocess_command, train_command

# The knobs the specialist owns. Names mirror SynthesisPoolConfig fields below.
SPECIALIST_HOOKS: tuple[str, ...] = (
    "synthetic_ratio",
    "per_cohort_quota",
    "et_emphasis",
    "seed",
)

# Offset of the synthetic augmented dataset id from the base labelled dataset (distinct from
# ssl's +100 round datasets so nnU-Net results never collide).
SYNTH_DATASET_OFFSET: int = 200


@dataclass
class SynthesisPoolConfig:
    """Stage-3 synthetic-pool hyperparameters — all ``# SPECIALIST:`` hooks.

    Defaults are ``None`` on purpose: ``None`` means "not yet decided by the specialist".
    Reference values live in the comments only, never silently applied.
    """

    synthetic_ratio: Optional[float] = None   # SPECIALIST: #synthetic / #labelled, found by sweep, ~0.5–1.0
    per_cohort_quota: Optional[bool] = None    # SPECIALIST: allocate the budget by cohort deficit vs one lump
    et_emphasis: Optional[float] = None        # SPECIALIST: >1 tilts the budget toward under-represented cohorts, ~1.0–2.0
    seed: Optional[int] = None                 # SPECIALIST: RNG seed (generation + provenance)


def unset_hooks(config: SynthesisPoolConfig) -> list[str]:
    """Return the names of SPECIALIST hooks still ``None`` (i.e. not yet decided)."""
    return [f.name for f in fields(config) if f.name in SPECIALIST_HOOKS and getattr(config, f.name) is None]


def assert_configured(config: SynthesisPoolConfig) -> None:
    """Raise if any SPECIALIST hook is unset — a guard before building a synthesis plan."""
    missing = unset_hooks(config)
    if missing:
        raise ValueError(
            "Refusing to build a synthesis pool plan: the AI specialist must set these "
            f"# SPECIALIST hooks first: {', '.join(missing)}"
        )


# --- pool sizing / allocation (pure stdlib) -----------------------------------------------

def n_synthetic_from_ratio(n_labeled: int, synthetic_ratio: float) -> int:
    """Number of synthetic cases to generate: ``round(synthetic_ratio * n_labeled)``.

    Sizing is a *ratio* of the labelled set (found by sweep), never a hardcoded count. ``round``
    uses banker's rounding (Python default); the result is a non-negative int.
    """
    return int(round(synthetic_ratio * n_labeled))


def allocate_per_cohort(
    cohort_counts: dict, n_synthetic: int, et_emphasis: float = 1.0,
    et_relevant_cohorts: Optional[Sequence[str]] = None,
) -> dict:
    """Split ``n_synthetic`` across cohorts ∝ their **deficit vs the largest cohort**.

    Weight of a cohort is ``max_count - count`` — so the majority cohort (deficit 0) gets the least
    and the rarest gets the most, directly serving the worst-cohort objective. ``et_emphasis`` (>1)
    scales the weight of the under-represented cohorts further: by default "under-represented" means
    below the mean count (a data-driven proxy that needs no hardcoded cohort table), or an explicit
    ``et_relevant_cohorts`` set if the caller supplies a per-cohort ET-prevalence list. Integer
    seats are apportioned by the largest-remainder method with a deterministic cohort-name tie-break,
    so the result sums **exactly** to ``n_synthetic`` and is reproducible. Counts come in as input —
    no real numbers are needed here.
    """
    import math

    if not cohort_counts:
        return {}
    if n_synthetic <= 0:
        return {c: 0 for c in cohort_counts}

    max_count = max(cohort_counts.values())
    mean_count = sum(cohort_counts.values()) / len(cohort_counts)
    if et_relevant_cohorts is None:
        emphasised = {c for c, v in cohort_counts.items() if v < mean_count}
    else:
        emphasised = set(et_relevant_cohorts)

    weights: dict = {}
    for cohort, count in cohort_counts.items():
        w = float(max_count - count)  # deficit vs the largest cohort
        if cohort in emphasised:
            w *= et_emphasis
        weights[cohort] = w

    total_w = sum(weights.values())
    if total_w <= 0:  # all cohorts equal-sized -> spread uniformly
        weights = {c: 1.0 for c in cohort_counts}
        total_w = float(len(cohort_counts))

    raw = {c: n_synthetic * weights[c] / total_w for c in cohort_counts}
    alloc = {c: int(math.floor(v)) for c, v in raw.items()}
    remainder = n_synthetic - sum(alloc.values())
    # hand out the leftover seats to the largest fractional parts (ties broken by cohort name)
    order = sorted(cohort_counts, key=lambda c: (-(raw[c] - alloc[c]), c))
    for c in order[:remainder]:
        alloc[c] += 1
    return alloc


def synth_dataset_id(base_dataset_id: int = DATASET_ID) -> int:
    """nnU-Net dataset id of the labelled+synthetic augmented set: ``base + 200`` (→ 701).

    Distinct from ssl's ``base + 100`` round datasets so the two augmentation tracks never clobber
    one another's ``nnUNet_results``.
    """
    return base_dataset_id + SYNTH_DATASET_OFFSET


# --- argv builders (reuse nnunet.plan; never run anything) ---------------------------------

def _generate_command(
    cohort: str, n: int, config_path: str | Path, out_dir: str | Path, seed: int
) -> list[str]:
    """One ``brats2026 synthesize`` argv: make ``n`` synthetic cases for ``cohort``."""
    return [
        "brats2026", "synthesize",
        "--config", str(config_path),
        "--cohort", cohort,
        "--n", str(n),
        "--out", str(out_dir),
        "--seed", str(seed),
    ]


def _build_dataset_command(
    synthetic_root: str | Path,
    dataset_id: int,
    raw_root: str | Path,
    manifest: str | Path,
) -> list[str]:
    """The ``brats2026 build-synth-dataset`` argv that merges real + generated cases.

    ``-d`` is the id of the augmented dataset being built (701), matching the ``-d`` of the
    plan/preprocess and train steps that follow.
    """
    return [
        "brats2026", "build-synth-dataset",
        "--synthetic-root", str(synthetic_root),
        "--raw-root", str(raw_root),
        "--manifest", str(manifest),
        "-d", str(dataset_id),
    ]


def synthesis_plan(
    pool_config: SynthesisPoolConfig,
    cohort_counts: dict,
    work_root: str | Path,
    base_dataset_id: int = DATASET_ID,
    config_path: str | Path = "configs/synthesis/synthesis.yaml",
    folds: Sequence[int] = (0, 1, 2, 3, 4),
    manifest: str | Path = "work/manifests/task3_train.jsonl",
) -> list[list[str]]:
    """Full offline synthesis plan as an argv list: generate → build 701 → plan/preprocess → train.

    Mirrors :func:`brats2026.ssl.self_training_plan` (argv only, nothing executed): the sizing
    comes from ``pool_config.synthetic_ratio`` over the labelled ``cohort_counts``; when
    ``per_cohort_quota`` is True the budget is split by :func:`allocate_per_cohort` into one
    ``brats2026 synthesize`` step per cohort, otherwise a single lump step. Then a
    ``brats2026 build-synth-dataset`` step merges the real cases listed in ``manifest`` with the
    generated ones into :func:`synth_dataset_id` (701),
    ``nnUNetv2_plan_and_preprocess -d 701`` runs the ResEnc-L planner, and one
    :func:`brats2026.nnunet.plan.train_command` per fold trains the GoAT trainer on 701. The
    human-gated runner consumes this plan; no command runs here.
    """
    assert_configured(pool_config)
    work_root = Path(work_root)
    synthetic_root = work_root / "synthetic"
    dataset_id = synth_dataset_id(base_dataset_id)

    n_labeled = sum(cohort_counts.values())
    n_synthetic = n_synthetic_from_ratio(n_labeled, pool_config.synthetic_ratio)

    cmds: list[list[str]] = []
    if pool_config.per_cohort_quota:
        alloc = allocate_per_cohort(cohort_counts, n_synthetic, pool_config.et_emphasis)
        for cohort in cohort_counts:
            n = alloc.get(cohort, 0)
            if n > 0:
                cmds.append(
                    _generate_command(
                        cohort, n, config_path, synthetic_root / cohort, pool_config.seed
                    )
                )
    else:
        cmds.append(
            _generate_command(
                "all", n_synthetic, config_path, synthetic_root, pool_config.seed
            )
        )

    cmds.append(
        _build_dataset_command(
            synthetic_root,
            dataset_id,
            raw_root=work_root / "nnUNet_raw",
            manifest=manifest,
        )
    )
    cmds.append(plan_and_preprocess_command(dataset_id=dataset_id))
    for fold in folds:
        cmds.append(train_command(fold=fold, dataset_id=dataset_id))
    return cmds


def load_pool_config(path: str | Path) -> SynthesisPoolConfig:
    """Build a :class:`SynthesisPoolConfig` from the pool keys of ``configs/synthesis/synthesis.yaml``.

    Reads the un-prefixed pool keys (``synthetic_ratio``, ``per_cohort_quota``, ``et_emphasis``,
    ``seed``) — the mask config owns the ``mask_*`` keys in the same file. :func:`assert_configured`
    still governs whether the result may build a plan.
    """
    from ..config import load_config

    cfg = load_config(Path(path))
    kwargs: dict = {}
    for hook in SPECIALIST_HOOKS:
        if hook in cfg and cfg[hook] is not None:
            kwargs[hook] = cfg[hook]
    return SynthesisPoolConfig(**kwargs)
