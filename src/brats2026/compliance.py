"""Phase 0 — GoAT compliance gate.

The BraTS-GoAT rule is non-negotiable: **only** data provided through the GoAT
sub-challenge may be used, and **no** weights derived from a prior BraTS may seed training
(train from scratch). A single violation means disqualification, so these checks are meant
to run before any training run or submission and to fail *loud*.

Pure stdlib so the gate can run anywhere, including inside CI before heavy deps exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Tokens that, if they appear in a weights/config reference, suggest weights NOT trained
# from scratch on GoAT data. Deliberately broad — better a false UNSURE than a missed FAIL.
PRIOR_BRATS_TOKENS: tuple[str, ...] = (
    "brats2018", "brats2019", "brats2020", "brats2021", "brats2022", "brats2023",
    "brats2024", "brats2025", "pretrained", "pretrain", "finetune", "fine-tune",
    "imagenet", "totalsegmentator", "checkpoint_prior",
)


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNSURE = "UNSURE"


@dataclass(frozen=True)
class CheckResult:
    item: str
    status: Status
    detail: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.status.value}] {self.item}: {self.detail}"


def check_data_provenance(case_dirs: list[str], allowed_roots: list[str]) -> CheckResult:
    """All training case directories must live under an approved GoAT data root."""
    if not allowed_roots:
        return CheckResult("data_provenance", Status.UNSURE, "no allowed roots supplied")
    allowed = [Path(root).resolve() for root in allowed_roots]
    offenders: list[str] = []
    for raw in case_dirs:
        path = Path(raw).resolve()
        if not any(_is_relative_to(path, root) for root in allowed):
            offenders.append(raw)
    if offenders:
        sample = ", ".join(offenders[:3])
        return CheckResult(
            "data_provenance", Status.FAIL,
            f"{len(offenders)} case(s) outside allowed GoAT roots (e.g. {sample})",
        )
    return CheckResult(
        "data_provenance", Status.PASS,
        f"all {len(case_dirs)} cases under {len(allowed)} approved root(s)",
    )


def check_no_prior_weights(references: list[str]) -> CheckResult:
    """Flag any weight/config reference hinting at non-from-scratch initialisation."""
    hits: list[str] = []
    for ref in references:
        low = ref.lower()
        for token in PRIOR_BRATS_TOKENS:
            if token in low:
                hits.append(f"{ref} (~{token})")
                break
    if hits:
        return CheckResult(
            "no_prior_weights", Status.FAIL,
            f"{len(hits)} suspicious reference(s): " + "; ".join(hits[:3]),
        )
    return CheckResult(
        "no_prior_weights", Status.PASS,
        f"no prior-BraTS / pretrained tokens in {len(references)} reference(s)",
    )


def check_nas_not_committed(tracked_paths: list[str], nas_prefixes: list[str]) -> CheckResult:
    """No NAS path (controlled data) may be committed to the repo."""
    leaked = [p for p in tracked_paths if any(p.startswith(pre) for pre in nas_prefixes)]
    if leaked:
        return CheckResult(
            "nas_not_committed", Status.FAIL,
            f"{len(leaked)} NAS path(s) tracked: " + ", ".join(leaked[:3]),
        )
    return CheckResult("nas_not_committed", Status.PASS, "no NAS paths tracked")


def gate_status(results: list[CheckResult]) -> Status:
    """Aggregate: any FAIL → FAIL; else any UNSURE → UNSURE; else PASS."""
    statuses = {r.status for r in results}
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.UNSURE in statuses:
        return Status.UNSURE
    return Status.PASS


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
