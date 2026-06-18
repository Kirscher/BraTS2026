"""Phase 5 — nnU-Net v2 inference command builders (predict + ensemble).

Like :mod:`brats2026.nnunet.plan`, these only *assemble argv*; they never run nnU-Net, so
they stay unit-testable without torch/nnU-Net installed. Ensemble members (the model dirs)
and TTA toggling are parameters — nothing about which models to ensemble is invented here.
"""
from __future__ import annotations

from pathlib import Path

from ..nnunet.plan import DEFAULT_CONFIG, GOAT_TRAINER, RESENC_L_PLANS

DEFAULT_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def predict_command(
    model_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    folds: tuple[int, ...] = DEFAULT_FOLDS,
    trainer: str = GOAT_TRAINER,
    plans: str = RESENC_L_PLANS,
    configuration: str = DEFAULT_CONFIG,
    disable_tta: bool = False,
) -> list[str]:
    """Build a single ``nnUNetv2_predict`` argv.

    ``model_dir`` is passed via ``-d`` (an nnU-Net results directory / dataset). ``folds``
    expand after ``-f``. ``--disable_tta`` is appended only when ``disable_tta`` is True
    (mirroring TTA is the nnU-Net default; the ai-specialist may turn it off for the 8h budget).
    """
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(model_dir),
        "-f", *[str(f) for f in folds],
        "-tr", trainer,
        "-p", plans,
        "-c", configuration,
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    return cmd


def predict_ensemble_commands(
    model_dirs,
    input_dir: str | Path,
    output_root: str | Path,
    **kwargs,
) -> list[list[str]]:
    """One ``nnUNetv2_predict`` argv per ensemble member.

    Each member writes to its own subdirectory ``<output_root>/<model_dir name>`` so the
    per-member probability maps can later be merged (see :func:`postprocess.merge_tta`).
    ``model_dirs`` (the ensemble members) is a parameter — nothing is invented.
    """
    output_root = Path(output_root)
    cmds: list[list[str]] = []
    for model_dir in model_dirs:
        member_out = output_root / Path(str(model_dir)).name
        cmds.append(
            predict_command(
                model_dir=model_dir,
                input_dir=input_dir,
                output_dir=member_out,
                **kwargs,
            )
        )
    return cmds
