"""Synthesis Stage 3 — assemble synthetic labelled cases from the frozen AE + latent diffusion.

This is the *generation* entrypoint of the DiffTumor track: given a healthy GoAT volume and a
sampled :mod:`brats2026.synthesis.masks` concentric mask, it encodes the volume with the frozen
Stage-1 autoencoder, replaces the masked latent region via Stage-2 diffusion sampling, decodes back
to a 4-channel volume, and returns that volume paired with the NCR/ED/ET label map — a new,
label-complete training case that feeds the nnU-Net.

**Decoupling:** this module must import cleanly without torch, so the torch-dependent pieces
(the AE / diffusion classes) are imported *inside* the scaffold body — which raises
``NotImplementedError`` anyway — never at module top level. The pure functions here are the case-id
naming + the GoAT-only compliance guard, both stdlib and fully tested.

Synthetic case IDs are minted in GoAT's ``BraTS-<COHORT>-...`` form with a synthetic marker in the
numeric block, so they still decode back to a known cohort via
:func:`brats2026.domains.cohort_from_case_id` (balanced sampling / LODO keep working) while staying
distinguishable from real cases. :func:`assert_synthetic_compliant` re-uses :mod:`brats2026.domains`
as a foreign-data guard — every synthetic id must resolve to a real GoAT cohort.
"""
from __future__ import annotations

from typing import Sequence

from ..domains import UNKNOWN_COHORT, cohort_from_case_id

# Marker digit that opens the numeric block of a synthetic case id (real GoAT numeric blocks are
# assigned by the challenge; a leading 9 flags "generated" while staying cohort-decodable).
SYNTHETIC_MARKER: str = "9"


def synthetic_case_id(cohort: str, index: int) -> str:
    """Mint a GoAT-style synthetic case id, e.g. ``BraTS-GLI-900042``.

    The cohort sits in its usual prefix slot so :func:`brats2026.domains.cohort_from_case_id`
    decodes it unchanged; the numeric block opens with :data:`SYNTHETIC_MARKER` (``9``) to flag the
    case as generated without colliding with real ids. ``index`` is zero-padded to keep ids sortable.
    """
    return f"BraTS-{cohort}-{SYNTHETIC_MARKER}{index:05d}"


def assert_synthetic_compliant(case_ids: Sequence[str]) -> None:
    """Fatal foreign-data guard: every id must resolve to a known GoAT cohort.

    Re-uses :func:`brats2026.domains.cohort_from_case_id`; any id that decodes to
    :data:`brats2026.domains.UNKNOWN_COHORT` means a non-GoAT / malformed case slipped into the
    synthetic pool, which would violate the GoAT no-external-data rule — so it is refused here.
    """
    foreign = [cid for cid in case_ids if cohort_from_case_id(cid) == UNKNOWN_COHORT]
    if foreign:
        sample = ", ".join(foreign[:3])
        raise ValueError(
            f"{len(foreign)} synthetic case id(s) do not resolve to a GoAT cohort "
            f"({UNKNOWN_COHORT}) — only GoAT cohorts are allowed (e.g. {sample})"
        )


def generate_synthetic_case(autoencoder, diffusion, healthy_volume, synthetic_seg, config):
    """Assemble one synthetic labelled case (scaffold).

    Wiring (training/generation env): encode ``healthy_volume`` with the frozen Stage-1
    ``autoencoder`` → latent; hole the ``synthetic_seg`` region and replace it via Stage-2
    ``diffusion`` sampling conditioned on the mask + surrounding context; decode → a 4-channel
    ``(t1n, t1c, t2f, t2w)`` volume; return ``(volume, synthetic_seg)`` with the NIfTI geometry of
    the input preserved. The torch classes are imported *inside* this body so the module stays
    importable without torch.
    """
    from .autoencoder import GoATAutoencoderKL  # noqa: F401 - imported inside to keep module torch-free
    from .diffusion import GoATLatentDiffusion   # noqa: F401

    raise NotImplementedError(
        "generate_synthetic_case is a scaffold: nnunet-trainer/Opus wires encode-healthy (frozen "
        "AE) -> replace the masked latent via diffusion sampling -> decode -> assemble the "
        "4-channel volume, returning it with the NCR/ED/ET seg and the input geometry preserved. "
        "Do not invent the tensor ops here."
    )
