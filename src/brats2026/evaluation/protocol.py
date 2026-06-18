"""Phase 3 — the nested inner/outer evaluation protocol, sealed in code.

The GoAT generalisation estimate is only honest if model selection / tuning never sees it.
We enforce that *mechanically* rather than by convention:

- **Inner dev** (:func:`score_inner_dev`): the 5-fold,
  :func:`splits.domain_balanced_kfold` scoring surface. Freely and repeatedly callable by the
  ai-specialist — this is where tuning happens. It can never read the outer estimate.
- **Outer LODO** (:func:`reveal_outer_lodo`): the sealed leave-one-domain-out estimate,
  revealed **exactly once** and only against a **pre-registered config hash**. Two guards:

  1. **Write-once flag.** ``<sealed_dir>/.outer_sealed`` is created on the first successful
     reveal. If it already exists, the reveal **refuses** — you cannot peek twice and then pick
     the better-looking number.
  2. **Frozen config hash.** ``reveal_outer_lodo`` requires the ``config_hash`` that was
     pre-registered via :func:`seal_config`. A mismatch (i.e. the config changed after the
     outer fold was sealed) **refuses** — you cannot retune and re-reveal under the same seal.

- :func:`seal_config` writes the pre-registered hash. It **refuses to overwrite** an existing
  sealed config (re-sealing would defeat guard 2), so the registration is itself write-once.

Pure stdlib + the in-repo ``report``/``splits`` modules (no torch/nnU-Net). Inner-dev scoring
takes already-computed :class:`CaseScore` objects — it never requires real predictions, so it
is unit-testable with hand-built scores.
"""
from __future__ import annotations

from pathlib import Path

from ..nnunet.splits import domain_balanced_kfold
from .report import CaseScore, leave_one_domain_out_table, per_cohort_table, worst_cohort

# Sealed-state filenames inside ``sealed_dir``.
CONFIG_HASH_FILE = "config_hash.txt"
OUTER_SEAL_FLAG = ".outer_sealed"


class ProtocolError(RuntimeError):
    """Raised when a protocol guard refuses an operation (seal/reveal violation)."""


# --------------------------------------------------------------------------------------- #
# Inner dev — freely callable tuning surface
# --------------------------------------------------------------------------------------- #
def score_inner_dev(
    scores: list[CaseScore],
    k: int = 5,
    seed: int = 42,
    family: str = "legacy",
) -> dict:
    """Inner-dev report over a domain-balanced K-fold split — callable as often as you like.

    Builds the K-fold split from the case IDs present in ``scores`` (so the split matches the
    data being scored), then reports, per fold, the per-cohort table and worst-cohort signal
    restricted to that fold's validation cases. This is the surface the ai-specialist tunes
    against; it carries **no** outer-LODO information.
    """
    by_id = {s.case_id: s for s in scores}
    folds = domain_balanced_kfold(sorted(by_id), k=k, seed=seed)
    fold_reports = []
    for i, fold in enumerate(folds):
        val_scores = [by_id[c] for c in fold["val"] if c in by_id]
        fold_reports.append(
            {
                "fold": i,
                "n_val": len(val_scores),
                "per_cohort": per_cohort_table(val_scores, family),
                "worst_cohort": worst_cohort(val_scores, family),
            }
        )
    pooled = per_cohort_table(list(by_id.values()), family)
    return {
        "stage": "inner_dev",
        "k": k,
        "seed": seed,
        "family": family,
        "folds": fold_reports,
        "pooled_per_cohort": pooled,
        "pooled_worst_cohort": worst_cohort(list(by_id.values()), family),
    }


# --------------------------------------------------------------------------------------- #
# Seal — pre-register the frozen config hash (write-once)
# --------------------------------------------------------------------------------------- #
def seal_config(config_hash: str, sealed_dir: Path) -> Path:
    """Pre-register the frozen ``config_hash`` that :func:`reveal_outer_lodo` checks against.

    **Refuses to overwrite** an existing sealed config: re-sealing under a new hash would let
    you retune after seeing (or before re-revealing) the outer estimate, defeating the point.
    To re-seal you must deliberately remove the file out-of-band. Returns the written path.
    """
    if not config_hash:
        raise ProtocolError("config_hash must be a non-empty pre-registered hash")
    sealed_dir = Path(sealed_dir)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    hash_path = sealed_dir / CONFIG_HASH_FILE
    if hash_path.exists():
        existing = hash_path.read_text(encoding="utf-8").strip()
        raise ProtocolError(
            f"config already sealed with hash {existing!r} at {hash_path}; "
            "refusing to overwrite (re-sealing would break the pre-registration guard)"
        )
    hash_path.write_text(config_hash.strip() + "\n", encoding="utf-8")
    return hash_path


def read_sealed_config(sealed_dir: Path) -> str | None:
    """Return the pre-registered config hash, or ``None`` if nothing is sealed yet."""
    hash_path = Path(sealed_dir) / CONFIG_HASH_FILE
    if not hash_path.exists():
        return None
    return hash_path.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------------------- #
# Reveal — the outer LODO estimate, once, against the frozen hash
# --------------------------------------------------------------------------------------- #
def reveal_outer_lodo(
    config_hash: str,
    sealed_dir: Path,
    scores_by_held_out: dict[str, list[CaseScore]],
    family: str = "legacy",
) -> dict:
    """Reveal the sealed outer leave-one-domain-out estimate — **exactly once**.

    Guards, both must pass:

    1. a sealed config must exist and ``config_hash`` must match it (pre-registration);
    2. the write-once flag ``<sealed_dir>/.outer_sealed`` must not yet exist.

    On success the flag is created (so a second call refuses) and the outer-LODO report is
    returned. Any guard failure raises :class:`ProtocolError` and leaves the flag untouched.
    """
    sealed_dir = Path(sealed_dir)

    # Guard 1: pre-registered config hash must exist and match.
    sealed_hash = read_sealed_config(sealed_dir)
    if sealed_hash is None:
        raise ProtocolError(
            f"no sealed config at {sealed_dir / CONFIG_HASH_FILE}; "
            "call seal_config(...) before revealing the outer estimate"
        )
    if config_hash.strip() != sealed_hash:
        raise ProtocolError(
            f"config_hash mismatch: supplied {config_hash.strip()!r} != sealed {sealed_hash!r}; "
            "the outer estimate is only valid for the pre-registered config"
        )

    # Guard 2: write-once. Refuse if already revealed.
    flag_path = sealed_dir / OUTER_SEAL_FLAG
    if flag_path.exists():
        raise ProtocolError(
            f"outer LODO already revealed (flag {flag_path} exists); refusing a second reveal "
            "(write-once: the generalisation estimate may be read only once)"
        )

    report = {
        "stage": "outer_lodo",
        "config_hash": sealed_hash,
        "family": family,
        "lodo": leave_one_domain_out_table(scores_by_held_out, family),
    }

    # Seal only after the report is built, so a failure above never burns the one-shot reveal.
    flag_path.write_text(f"revealed for config_hash {sealed_hash}\n", encoding="utf-8")
    return report
