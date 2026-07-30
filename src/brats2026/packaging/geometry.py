"""Exact NIfTI geometry preservation guard.

A GoAT submission MUST write predictions with the *exact* affine and shape of the input
NIfTI (per the challenge container rules). This module is the non-negotiable check:
shape must match exactly; the affine must match exactly (``atol=0.0`` by default), with a
tiny floating-point tolerance available only if a caller opts in. Pure numpy; nibabel is
behind a guarded import and only reads header geometry (never ``get_fdata``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class GeometryError(ValueError):
    """Raised when an output NIfTI's geometry does not match the reference input."""


def geometry_matches(ref_affine, ref_shape, out_affine, out_shape, atol: float = 0.0) -> bool:
    """True iff shape is exactly equal AND affine matches within ``atol``.

    Shape comparison is always exact. With the default ``atol=0.0`` the affine must be
    bit-exact (via ``np.array_equal``); a small positive ``atol`` allows a floating-point
    tolerance (``np.allclose`` with ``rtol=0``).
    """
    if tuple(int(d) for d in ref_shape) != tuple(int(d) for d in out_shape):
        return False
    ref_affine = np.asarray(ref_affine, dtype=float)
    out_affine = np.asarray(out_affine, dtype=float)
    if ref_affine.shape != out_affine.shape:
        return False
    if atol == 0.0:
        return bool(np.array_equal(ref_affine, out_affine))
    return bool(np.allclose(ref_affine, out_affine, rtol=0.0, atol=atol))


def assert_same_geometry(ref_affine, ref_shape, out_affine, out_shape, atol: float = 0.0) -> None:
    """Raise :class:`GeometryError` naming the mismatch (shape vs affine); else return None."""
    ref_shape_t = tuple(int(d) for d in ref_shape)
    out_shape_t = tuple(int(d) for d in out_shape)
    if ref_shape_t != out_shape_t:
        raise GeometryError(
            f"shape mismatch: output {out_shape_t} != reference {ref_shape_t}"
        )
    if not geometry_matches(ref_affine, ref_shape, out_affine, out_shape, atol=atol):
        raise GeometryError(
            f"affine mismatch (atol={atol}):\n"
            f"reference=\n{np.asarray(ref_affine, dtype=float)}\n"
            f"output=\n{np.asarray(out_affine, dtype=float)}"
        )


def read_geometry(path: str | Path):
    """Return ``(affine, shape)`` for a NIfTI, behind a guarded nibabel import.

    Reads header/affine/shape only — never ``get_fdata`` (controlled-data rule).
    """
    try:
        import nibabel as nib  # noqa: PLC0415 - optional, guarded
    except ImportError as exc:  # pragma: no cover - exercised only without nibabel
        raise RuntimeError("nibabel is required for read_geometry()") from exc

    img = nib.load(str(path))  # lazy: does not load pixel data
    return np.asarray(img.affine, dtype=float), tuple(int(d) for d in img.shape)
