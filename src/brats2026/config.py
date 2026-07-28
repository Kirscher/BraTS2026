"""Phase 2 — provenance-bound config schema / validator (the config gate).

This module is the **file-level analogue** of :func:`brats2026.nnunet.trainer.assert_configured`.
The ai-specialist fills hyperparameters, but ONLY through a config that

1. carries a complete **provenance header** (git SHA, dataset fingerprint, seed, parent run-id,
   date, schema version), and
2. fills **every** ``# SPECIALIST:`` hook — train hooks come verbatim from
   :data:`brats2026.nnunet.trainer.SPECIALIST_HOOKS` (no hand re-listing, so code and config
   can never silently drift apart), and inference hooks from :data:`INFER_HOOK_KEYS`.

A fresh template (:func:`make_template`) is **born unusable**: every header field is empty and
every hook is ``None``, so :func:`assert_valid_config` refuses it on purpose. It only becomes
valid after the specialist fills it — *after* the dataset fingerprint exists.

:func:`check_fingerprint_binding` is the guard that a config was not tuned against different
data than what ships: the config's recorded ``dataset_fingerprint`` (see
:mod:`brats2026.provenance`) must equal the fingerprint of the data being trained/packaged.

Pure stdlib core so the whole validation surface is unit-testable on plain dicts — no torch /
nnU-Net / PyYAML. YAML I/O sits behind a guarded ``import yaml`` and is the only path that
needs PyYAML installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Stay in lockstep with the trainer: the train hook keys ARE the trainer's SPECIALIST_HOOKS.
# Reused verbatim (never re-listed by hand) so config drift against the code is impossible.
# Both trainer modules are import-light (stdlib only at module scope; torch / nnU-Net sit behind
# guarded imports), so importing them here keeps config.py loadable in a bare env.
from brats2026.nnunet.trainer import SPECIALIST_HOOKS
from brats2026.nnunet.trainer_v2 import SPECIALIST_HOOKS_V2

# Bump intentionally if the config schema changes (mirrors provenance.FINGERPRINT_SCHEME).
CONFIG_SCHEMA_VERSION = "brats2026-config/1"

# The provenance header every config must carry (mirrors the ai-specialist's contract and the
# fields provenance.py records). A config is only usable once all of these are filled.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "git_sha",              # SPECIALIST: commit the config was produced against
    "dataset_fingerprint",  # SPECIALIST: provenance.dataset_fingerprint of the training data
    "seed",                 # SPECIALIST: RNG seed (reproducible-from-scratch)
    "parent_run_id",        # SPECIALIST: run this config was derived from ("" for a fresh root)
    "date",                 # SPECIALIST: ISO date the config was written
    "config_schema_version",  # SPECIALIST: pin to CONFIG_SCHEMA_VERSION
)

# Training-time hooks: reuse the trainer's tuple verbatim. Kept as a named alias for symmetry
# with INFER_HOOK_KEYS; the test suite asserts TRAIN_HOOK_KEYS == SPECIALIST_HOOKS as a guard.
TRAIN_HOOK_KEYS: tuple[str, ...] = SPECIALIST_HOOKS

# Same contract for the v2 trainer (brats2026.nnunet.trainer_v2). Kept SEPARATE from
# TRAIN_HOOK_KEYS on purpose: configs/train.yaml is validated against kind "train" while a run is
# in flight, so the v1 tuple must never gain or lose a key.
TRAIN_V2_HOOK_KEYS: tuple[str, ...] = SPECIALIST_HOOKS_V2

# Inference-time hooks: the knobs the ai-specialist owns in configs/infer.yaml. These mirror
# brats2026.inference.{postprocess,predict}; the code keeps the mechanism, config the values.
INFER_HOOK_KEYS: tuple[str, ...] = (
    "tta_axes",                  # SPECIALIST: mirroring-TTA axes, e.g. [0, 1, 2] (or [] = off)
    "ensemble_members",          # SPECIALIST: list of model/fold dirs to ensemble
    "ensemble_weights",          # SPECIALIST: per-member weights (parallel to ensemble_members)
    "et_suppression_min_voxels",  # SPECIALIST: postprocess.et_suppression min component size
    "cc_min_voxels",             # SPECIALIST: connected_component_filter min component size
)

_HOOK_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "train": TRAIN_HOOK_KEYS,
    "train_v2": TRAIN_V2_HOOK_KEYS,
    "infer": INFER_HOOK_KEYS,
}


class ConfigError(ValueError):
    """Raised by :func:`assert_valid_config` when a config is incomplete or unbound."""


def _hook_keys(kind: str) -> tuple[str, ...]:
    try:
        return _HOOK_KEYS_BY_KIND[kind]
    except KeyError:
        raise ValueError(f"unknown config kind {kind!r}; expected one of {sorted(_HOOK_KEYS_BY_KIND)}")


def make_template(kind: str) -> dict:
    """Return a fresh, **intentionally invalid** config stub for ``kind`` in {"train","infer"}.

    The result has a ``provenance`` sub-dict with every :data:`PROVENANCE_FIELDS` present but
    empty, and every hook key for ``kind`` present with value ``None``. ``None`` means "an
    UNFILLABLE stub": :func:`validate_config` reports it as unset, so a fresh config is born
    unusable until the specialist fills it (post-fingerprint).
    """
    hooks = _hook_keys(kind)
    provenance = {f: None for f in PROVENANCE_FIELDS}
    cfg: dict = {"kind": kind, "provenance": provenance}
    for key in hooks:
        cfg[key] = None
    return cfg


def validate_provenance_header(cfg: dict) -> list[str]:
    """Return the names of missing or empty :data:`PROVENANCE_FIELDS` in ``cfg``.

    A field counts as empty when absent, ``None``, or an empty/whitespace-only string. The
    ``parent_run_id`` is allowed to be the literal string ``"root"`` for a fresh root run but
    must still be non-empty — there is no silent default.
    """
    header = cfg.get("provenance") or {}
    missing: list[str] = []
    for fieldname in PROVENANCE_FIELDS:
        value = header.get(fieldname)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(fieldname)
    return missing


def _unset_hooks(cfg: dict, kind: str) -> list[str]:
    """Hook keys that are absent or ``None`` (i.e. not yet decided by the specialist)."""
    return [key for key in _hook_keys(kind) if cfg.get(key) is None]


@dataclass(frozen=True)
class ConfigValidation:
    """Result of :func:`validate_config`.

    ``ok`` is True only when the provenance header is complete AND no hooks are unset.
    """

    ok: bool
    missing_header: list[str]
    unset_hooks: list[str]


def validate_config(cfg: dict, kind: str) -> ConfigValidation:
    """Validate a config dict for ``kind`` — header completeness + all hooks set."""
    missing_header = validate_provenance_header(cfg)
    unset = _unset_hooks(cfg, kind)
    return ConfigValidation(
        ok=not missing_header and not unset,
        missing_header=missing_header,
        unset_hooks=unset,
    )


def assert_valid_config(cfg: dict, kind: str) -> None:
    """Raise :class:`ConfigError` with a precise message if ``cfg`` is invalid for ``kind``.

    The file-level analogue of :func:`brats2026.nnunet.trainer.assert_configured`: it names the
    missing header fields and the unset hooks so the specialist knows exactly what to fill.
    """
    result = validate_config(cfg, kind)
    if result.ok:
        return
    parts: list[str] = []
    if result.missing_header:
        parts.append(
            "incomplete provenance header (missing/empty: "
            f"{', '.join(result.missing_header)})"
        )
    if result.unset_hooks:
        parts.append(
            "unset # SPECIALIST hooks (still None/absent: "
            f"{', '.join(result.unset_hooks)})"
        )
    raise ConfigError(
        f"Refusing {kind!r} config: " + "; ".join(parts) + ". "
        "The ai-specialist must fill these (after the dataset fingerprint exists)."
    )


def check_fingerprint_binding(cfg: dict, expected_fingerprint: str) -> bool:
    """True iff the config's recorded ``dataset_fingerprint`` equals ``expected_fingerprint``.

    Guards that a config was not tuned against different data than what ships. An empty/absent
    recorded fingerprint never matches.
    """
    recorded = (cfg.get("provenance") or {}).get("dataset_fingerprint")
    if not recorded:
        return False
    return recorded == expected_fingerprint


# --------------------------------------------------------------------------------------- #
# YAML I/O (guarded — the only path needing PyYAML; validation works on plain dicts)
# --------------------------------------------------------------------------------------- #
def _require_yaml():
    try:
        import yaml  # noqa: PLC0415 - optional, guarded
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ImportError(
            "PyYAML is required for config load/dump. Install it with: pip install pyyaml"
        ) from exc
    return yaml


def load_config(path: Path) -> dict:
    """Load a config dict from a YAML file (round-trips :func:`dump_config`)."""
    yaml = _require_yaml()
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"config at {path} did not parse to a mapping (got {type(data).__name__})")
    return data


def _yaml_scalar(value) -> str:
    """Render a Python scalar as a YAML scalar for in-place line editing (quotes strings)."""
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    return str(value)


def stamp_provenance_in_text(text: str, updates: dict) -> str:
    """Set ``provenance.<key>`` values in a config's YAML *text*, preserving comments/layout.

    Only lines inside the top-level ``provenance:`` mapping are touched; each ``key`` in
    ``updates`` has its value (and only its value) rewritten, keeping any trailing ``#`` comment
    on that line. Unlike a YAML load→dump round-trip this does NOT strip the file's comments — it
    is a targeted line edit, used by :func:`stamp_config_file`. Raises :class:`ConfigError` if a
    requested key is not found in the provenance block, so a silent no-op stamp can't happen.
    """
    lines = text.splitlines(keepends=True)
    in_block = False
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_block:
            if stripped.split("#", 1)[0].rstrip() == "provenance:":
                in_block = True
            continue
        # End of the provenance block: a non-indented, non-blank, non-comment line.
        if line[:1] not in (" ", "\t") and stripped and not stripped.startswith("#"):
            break
        indent = line[: len(line) - len(line.lstrip())]
        body = stripped.split("#", 1)[0]
        if ":" not in body:
            continue
        key = body.split(":", 1)[0].strip()
        if key in remaining:
            newline = line[-1] if line[-1:] in "\r\n" else ""
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):].rstrip("\r\n")
            lines[i] = f"{indent}{key}: {_yaml_scalar(remaining.pop(key))}{comment}{newline}"
    if remaining:
        raise ConfigError(
            "stamp targets not found in provenance block: " + ", ".join(sorted(remaining))
        )
    return "".join(lines)


def stamp_config_file(path: Path, updates: dict) -> Path:
    """Write ``updates`` into the ``provenance:`` block of the config file at ``path`` in place.

    Comment/layout-preserving (see :func:`stamp_provenance_in_text`). This is the mechanism
    behind ``brats2026 stamp-config``: it binds a shipped train config to the concrete data +
    commit on the HPC (git_sha / date / dataset_fingerprint) without disturbing the hand-written
    hyperparameter section.
    """
    path = Path(path)
    stamped = stamp_provenance_in_text(path.read_text(encoding="utf-8"), updates)
    path.write_text(stamped, encoding="utf-8")
    return path


def dump_config(cfg: dict, path: Path, header_comment: str | None = None) -> Path:
    """Write ``cfg`` to ``path`` as YAML, optionally prefixed with a ``#`` comment block."""
    yaml = _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    if header_comment:
        comment = "\n".join(f"# {line}" if line else "#" for line in header_comment.splitlines())
        body = comment + "\n" + body
    path.write_text(body, encoding="utf-8")
    return path
