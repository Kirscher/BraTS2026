"""Synthesis Stage 3 — assemble the augmented nnU-Net dataset (real + synthetic) safely.

:mod:`brats2026.synthesis.pipeline` decides *how many* synthetic cases to make and emits the
``brats2026 build-synth-dataset`` step; this module is what that step runs. It takes the generated
cases on disk, merges them with the real labelled cases into a **separate** nnU-Net dataset
(:func:`synth_dataset_name` → ``Dataset701_BraTSGoATSynth``, never clobbering the real 501), and
records exactly which cases are synthetic.

**The leak this module exists to prevent.** :mod:`brats2026.nnunet.splits` keys folds off the case
id alone, so a synthetic case dropped into the pool would be eligible for a *validation* fold — and
scoring a segmenter on tumours it was taught to paint is measuring the generator, not the model.
:func:`train_only_splits` therefore pins every synthetic case to the ``train`` side of every fold
and asserts none reached ``val``. Synthetic data augments training only; every reported number
stays computed on real GoAT cases.

Generated cases are written in the **native BraTS layout** —
``<synthetic_root>/<cohort>/<case_id>/<case_id>-<t1n|t1c|t2f|t2w|seg>.nii.gz`` — so they are
drop-in compatible with :mod:`brats2026.discover` and reuse the tested
:func:`brats2026.nnunet.convert.plan_conversion` planner rather than a parallel one.

Pure stdlib planning + a thin IO wrapper (mirroring :mod:`brats2026.nnunet.convert`), so this module
and its tests load without torch, numpy or nnU-Net.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from ..discover import find_case_file
from ..domains import cohort_from_case_id
from ..nnunet.convert import CHANNEL_ORDER, ConversionPlan, plan_conversion
from ..nnunet.splits import Fold, domain_balanced_kfold
from .generate import SYNTHETIC_MARKER, assert_synthetic_compliant
from .pipeline import synth_dataset_id

# Sidecar written next to dataset.json listing the synthetic case ids. This is the *authoritative*
# provenance record (the id convention below is only a secondary guard) and is what compliance-guard
# and the split builder read to keep synthetic cases out of validation.
SYNTH_MANIFEST_NAME: str = "synthetic_cases.json"

# Length of the numeric block minted by :func:`brats2026.synthesis.generate.synthetic_case_id`
# (marker digit + 5 zero-padded index digits).
_SYNTHETIC_BLOCK_LEN: int = 6

# The modality + label suffixes a generated case must carry to be complete.
_REQUIRED_SUFFIXES: tuple[str, ...] = CHANNEL_ORDER + ("seg",)


def synth_dataset_name(dataset_id: Optional[int] = None) -> str:
    """Directory name of the augmented dataset, e.g. ``Dataset701_BraTSGoATSynth``.

    ``dataset_id`` is the id of the augmented set itself and defaults to
    :func:`brats2026.synthesis.pipeline.synth_dataset_id` (``501 + 200`` → 701), so the augmented
    set never shares a directory with the real labelled 501 or ssl's ``base + 100`` rounds.
    """
    return f"Dataset{synth_dataset_id() if dataset_id is None else dataset_id}_BraTSGoATSynth"


def is_synthetic_case_id(case_id: str) -> bool:
    """True when ``case_id`` carries the generated-case marker minted by ``synthetic_case_id``.

    The convention is ``BraTS-<COHORT>-9NNNNN``: a final all-digit block of exactly
    :data:`_SYNTHETIC_BLOCK_LEN` characters opening with
    :data:`brats2026.synthesis.generate.SYNTHETIC_MARKER`. Real GoAT ids do not match this shape.
    This is a convenience/secondary guard — the sidecar manifest written by
    :func:`apply_synth_dataset` is the authoritative record of what is synthetic.
    """
    if not case_id:
        return False
    block = case_id.rsplit("-", 1)[-1]
    return (
        len(block) == _SYNTHETIC_BLOCK_LEN
        and block.isdigit()
        and block.startswith(SYNTHETIC_MARKER)
    )


def discover_synthetic_cases(synthetic_root: str | Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Scan generated cases under ``synthetic_root`` into discover-shaped records.

    Walks ``<synthetic_root>/<cohort>/<case_id>/`` and resolves each case's four modalities plus
    its ``seg`` via :func:`brats2026.discover.find_case_file`, so generated cases are read with the
    exact same file-matching rules as real BraTS cases. Cohort subdirectories are optional: a flat
    ``<synthetic_root>/<case_id>/`` layout is scanned too (the cohort is decoded from the id
    anyway).

    Returns ``(records, incomplete)`` — records carry ``case_id`` / ``inputs`` / ``target`` ready
    for :func:`brats2026.nnunet.convert.plan_conversion`, and ``incomplete`` maps a case id to the
    suffixes it is missing so a partial generation run is reported rather than silently dropped.
    """
    root = Path(synthetic_root)
    records: list[dict] = []
    incomplete: dict[str, list[str]] = {}
    if not root.is_dir():
        return records, incomplete

    # A directory is a case dir when it holds NIfTIs; otherwise treat it as a cohort level.
    case_dirs: list[Path] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(child.glob("*.nii.gz")):
            case_dirs.append(child)
        else:
            case_dirs.extend(sorted(p for p in child.iterdir() if p.is_dir()))

    for case_dir in case_dirs:
        case_id = case_dir.name
        inputs: dict[str, str] = {}
        missing: list[str] = []
        target: Optional[str] = None
        for suffix in _REQUIRED_SUFFIXES:
            path = find_case_file(case_dir, case_id, suffix)
            if path is None:
                missing.append(suffix)
            elif suffix != "seg":
                inputs[suffix] = str(path)
            else:
                target = str(path)
        if missing:
            incomplete[case_id] = missing
            continue
        records.append({"case_id": case_id, "inputs": inputs, "target": target})

    return records, incomplete


def plan_synth_dataset(
    real_records: Sequence[dict],
    synthetic_root: str | Path,
    raw_root: str | Path,
    dataset_id: Optional[int] = None,
) -> tuple[ConversionPlan, list[str]]:
    """Plan the augmented ``Dataset701`` from real manifest records + generated cases.

    Both sources are converted through the same tested
    :func:`brats2026.nnunet.convert.plan_conversion` (labels required on both sides — a synthetic
    case without its seg is worthless), targeting :func:`synth_dataset_name`.

    Enforces the GoAT no-external-data rule up front: every discovered synthetic id must decode to
    a real cohort via :func:`brats2026.synthesis.generate.assert_synthetic_compliant`, and a
    synthetic id colliding with a real one is fatal rather than silently overwriting it.

    Returns ``(plan, synthetic_ids)``; ``synthetic_ids`` is what gets written to the sidecar and
    handed to :func:`train_only_splits`.
    """
    synth_records, _incomplete = discover_synthetic_cases(synthetic_root)
    synthetic_ids = [r["case_id"] for r in synth_records]
    assert_synthetic_compliant(synthetic_ids)

    real_ids = {r["case_id"] for r in real_records}
    collisions = sorted(real_ids.intersection(synthetic_ids))
    if collisions:
        raise ValueError(
            f"{len(collisions)} synthetic case id(s) collide with real GoAT cases "
            f"(e.g. {', '.join(collisions[:3])}) — refusing to overwrite real data"
        )

    plan = plan_conversion(
        list(real_records) + synth_records,
        Path(raw_root),
        require_label=True,
        dataset_name=synth_dataset_name(dataset_id),
    )
    # Only the ids that actually made it through conversion are really in the dataset.
    converted = set(plan.converted)
    return plan, [cid for cid in synthetic_ids if cid in converted]


def apply_synth_dataset(
    plan: ConversionPlan,
    synthetic_ids: Sequence[str],
    link: bool = True,
    overwrite: bool = False,
) -> Path:
    """Materialise the augmented dataset and write the :data:`SYNTH_MANIFEST_NAME` sidecar.

    Delegates the link/copy + ``dataset.json`` to
    :func:`brats2026.nnunet.convert.apply_conversion`, then records the synthetic ids (with their
    per-cohort counts) so downstream steps can tell generated cases from real ones without relying
    on the id convention. Returns the sidecar path.
    """
    from ..nnunet.convert import apply_conversion

    apply_conversion(plan, link=link, overwrite=overwrite)

    per_cohort: dict[str, int] = {}
    for cid in synthetic_ids:
        cohort = cohort_from_case_id(cid)
        per_cohort[cohort] = per_cohort.get(cohort, 0) + 1

    sidecar = plan.dataset_dir / SYNTH_MANIFEST_NAME
    sidecar.write_text(
        json.dumps(
            {
                "synthetic_case_ids": sorted(synthetic_ids),
                "n_synthetic": len(synthetic_ids),
                "n_total": plan.num_training,
                "n_real": plan.num_training - len(synthetic_ids),
                "per_cohort": per_cohort,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def read_synthetic_ids(dataset_dir: str | Path) -> list[str]:
    """Read the synthetic ids recorded by :func:`apply_synth_dataset`.

    Returns an empty list when the sidecar is absent — a dataset with no synthetic provenance
    record is treated as fully real, which is the safe default for the split builder.
    """
    sidecar = Path(dataset_dir) / SYNTH_MANIFEST_NAME
    if not sidecar.is_file():
        return []
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    return list(body.get("synthetic_case_ids", []))


def train_only_splits(
    case_ids: Sequence[str],
    synthetic_ids: Optional[Sequence[str]] = None,
    k: int = 5,
    seed: int = 42,
) -> list[Fold]:
    """Domain-balanced K-fold in which synthetic cases only ever appear in ``train``.

    The real cases are split by the usual
    :func:`brats2026.nnunet.splits.domain_balanced_kfold` (so cohort proportions and the seeded
    ordering are unchanged, and folds stay comparable to a non-augmented run); the synthetic cases
    are then appended to *every* fold's train side. Validation therefore measures the model on real
    GoAT tumours only — scoring on painted tumours would grade the generator, not the segmenter.

    ``synthetic_ids`` defaults to whatever :func:`is_synthetic_case_id` flags, but callers should
    pass the sidecar list from :func:`read_synthetic_ids`, which is authoritative.
    """
    ids = list(case_ids)
    synth = set(synthetic_ids) if synthetic_ids is not None else {c for c in ids if is_synthetic_case_id(c)}
    real = [c for c in ids if c not in synth]
    extra = sorted(synth.intersection(ids))

    folds = domain_balanced_kfold(real, k=k, seed=seed)
    for fold in folds:
        fold["train"] = sorted(set(fold["train"]).union(extra))
        leaked = sorted(set(fold["val"]).intersection(synth))
        if leaked:  # unreachable by construction; a loud guard against future edits
            raise AssertionError(
                f"synthetic case(s) leaked into a validation fold: {', '.join(leaked[:3])}"
            )
    return folds
