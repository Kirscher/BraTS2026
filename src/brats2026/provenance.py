"""Phase 0/1 — provenance ledger + dataset fingerprint (root of trust).

This module proves training is **GoAT-only and reproducible-from-scratch**. It records, for
every training case, a *pixel-free* metadata digest and folds those into a single
**dataset fingerprint** that:

- depends ONLY on portable, recorded fields (relative paths, file sizes, NIfTI header
  geometry) plus an explicit ``extra`` block (label map, RNG seed, fold definitions);
- is **order-independent** (cases are sorted before hashing);
- **excludes wall-clock timestamps and absolute/machine-specific paths**, so the same hash
  is re-derivable inside the offline submission container from the ledger alone.

Downstream: compliance-guard audits the container against the ledger, the ai-specialist's
config header references the fingerprint, and paper-writer cites it.

Pure stdlib core (``hashlib``/``json``/``dataclasses``/``pathlib``/``datetime``) so the
hashing and fingerprint logic is unit-testable with plain dicts and tiny temp files — no
torch / nnU-Net / nibabel. The only optional dependency (nibabel, to read a NIfTI *header*)
sits behind a guarded import and is never required by the fingerprint path.

**Controlled-data rule:** there is NO code path here that reads image pixel payloads
(``get_fdata`` and friends). Digests use file metadata and NIfTI header geometry only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Version of the fingerprinting scheme. Bumping it intentionally changes every fingerprint,
# which is what we want if the digest definition ever changes.
FINGERPRINT_SCHEME = "brats2026-provenance/1"


# --------------------------------------------------------------------------------------- #
# Pure hashing core (no IO, fully unit-testable)
# --------------------------------------------------------------------------------------- #
def canonical_json(fields: dict) -> str:
    """Serialise ``fields`` to a canonical (sorted-key, compact) JSON string.

    Sorting keys and using a fixed separator/encoding makes the output independent of dict
    insertion order, so two dicts with the same content always serialise identically.
    """
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(fields: dict) -> str:
    """sha256 hex digest of the canonical JSON of ``fields``.

    Stable regardless of key order; changes whenever any value changes.
    """
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()


def case_fingerprint(relative_path: str, size: int, header: dict | None) -> str:
    """Per-case **metadata-only** digest.

    Computed from the case's *relative* source path (relative to the data root), its file
    size in bytes, and — when available — NIfTI header geometry (shape/dtype/affine). No
    pixel data is read. ``header`` may be ``None`` (e.g. size-only digest), which is folded
    explicitly so "no header" differs deterministically from "empty header".
    """
    return canonical_digest(
        {
            "scheme": FINGERPRINT_SCHEME,
            "relative_path": _as_posix(relative_path),
            "size": int(size),
            "header": header,
        }
    )


def dataset_fingerprint(case_fingerprints: list[str], extra: dict) -> str:
    """Order-independent digest over all per-case fingerprints plus an ``extra`` block.

    Cases are **sorted** before hashing, so reordering the input does not change the result.
    ``extra`` folds in portable, decision-defining fields (e.g. label map ``{NCR:1,ED:2,
    ET:3}``, RNG seed, fold definitions); changing any of them changes the fingerprint.
    """
    return canonical_digest(
        {
            "scheme": FINGERPRINT_SCHEME,
            "cases": sorted(case_fingerprints),
            "extra": extra,
        }
    )


# --------------------------------------------------------------------------------------- #
# Ledger entry + append-only IO
# --------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LedgerEntry:
    """One line of the append-only provenance ledger.

    ``recorded_at`` is informational only (an ISO-8601 UTC timestamp) and is **never** part
    of any fingerprint — it documents *when* the line was written, not *what* was recorded.
    """

    case_id: str
    relative_path: str
    size: int
    fingerprint: str
    label_harmonization: dict[str, int]
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "LedgerEntry":
        return cls(**json.loads(line))


def make_entry(
    case_id: str,
    relative_path: str,
    size: int,
    label_harmonization: dict[str, int],
    header: dict | None = None,
) -> LedgerEntry:
    """Build a :class:`LedgerEntry`, computing its per-case fingerprint from metadata only."""
    rel = _as_posix(relative_path)
    return LedgerEntry(
        case_id=case_id,
        relative_path=rel,
        size=int(size),
        fingerprint=case_fingerprint(rel, size, header),
        label_harmonization=label_harmonization,
    )


def append_entry(entry: LedgerEntry, ledger_path: Path) -> None:
    """Append one entry as a JSON line. Existing lines are never rewritten or truncated."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(entry.to_json() + "\n")


def read_ledger(ledger_path: Path) -> list[LedgerEntry]:
    """Read all entries back from an append-only ledger (round-trips :func:`append_entry`)."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(LedgerEntry.from_json(line))
    return entries


def fingerprint_from_ledger(entries: list[LedgerEntry], extra: dict) -> str:
    """Recompute the dataset fingerprint from ledger entries alone (container re-derivation).

    Uses only the recorded per-case ``fingerprint`` values and the supplied ``extra`` block —
    no filesystem access — so the offline container reproduces the training-time hash exactly.
    """
    return dataset_fingerprint([e.fingerprint for e in entries], extra)


# --------------------------------------------------------------------------------------- #
# Human-readable manifest
# --------------------------------------------------------------------------------------- #
def write_manifest(
    dataset_fp: str,
    entries_summary: dict,
    out_path: Path,
    git_sha: str | None = None,
) -> Path:
    """Write a small human-readable provenance manifest (JSON).

    ``git_sha`` is parameterised and optional — never auto-invented. ``generated_at`` is
    recorded for humans but is explicitly *not* part of any fingerprint.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scheme": FINGERPRINT_SCHEME,
        "dataset_fingerprint": dataset_fp,
        "git_sha": git_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": entries_summary,
    }
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------------------- #
# Optional NIfTI header probe (guarded; header geometry only, never pixels)
# --------------------------------------------------------------------------------------- #
def nifti_header_fields(path: Path) -> dict:
    """Return shape/dtype/affine from a NIfTI **header** — never the pixel payload.

    Behind a guarded ``nibabel`` import so the rest of the module loads in a bare env. We
    read only ``.shape``/``.affine`` and the header dtype; ``get_fdata`` is never called.
    """
    try:
        import nibabel as nib  # noqa: PLC0415 - optional, guarded
    except ImportError as exc:  # pragma: no cover - exercised only without nibabel
        raise RuntimeError("nibabel is required for nifti_header_fields()") from exc

    img = nib.load(str(path))  # lazy: does not load pixel data
    return {
        "shape": [int(d) for d in img.shape],
        "dtype": str(img.get_data_dtype()),
        "affine": [[round(float(v), 6) for v in row] for row in img.affine],
    }


# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #
def _as_posix(rel: str) -> str:
    """Normalise a relative path to POSIX form so fingerprints are OS-independent."""
    return Path(rel).as_posix()
