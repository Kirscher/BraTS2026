"""Phase 2 — nnU-Net v2 command builders (plan/preprocess + train).

These functions only *assemble argv and the environment*; they never run nnU-Net and never
pick hyperparameters. That keeps them unit-testable without torch/nnU-Net installed and keeps
all tunables behind the ``# SPECIALIST:`` hooks in :mod:`brats2026.nnunet.trainer`.
"""
from __future__ import annotations

from pathlib import Path

from .convert import DATASET_ID

# nnU-Net v2 ResEnc-L preset (Phase 2 architecture choice). The planner emits the
# "nnUNetResEncUNetLPlans" plans identifier consumed at train time.
RESENC_L_PLANNER = "nnUNetPlannerResEncL"
RESENC_L_PLANS = "nnUNetResEncUNetLPlans"
DEFAULT_CONFIG = "3d_fullres"

# Our custom trainer (scaffold lives in trainer.py).
GOAT_TRAINER = "nnUNetTrainerGoAT"


def nnunet_env(work_dir: Path) -> dict[str, str]:
    """The three env vars nnU-Net v2 needs, all rooted under ``work/`` (NAS is read-only)."""
    work_dir = Path(work_dir)
    return {
        "nnUNet_raw": str(work_dir / "nnUNet_raw"),
        "nnUNet_preprocessed": str(work_dir / "nnUNet_preprocessed"),
        "nnUNet_results": str(work_dir / "nnUNet_results"),
    }


def plan_and_preprocess_command(
    dataset_id: int = DATASET_ID,
    planner: str = RESENC_L_PLANNER,
    configurations: tuple[str, ...] = (DEFAULT_CONFIG,),
    verify: bool = True,
) -> list[str]:
    """Build the ``nnUNetv2_plan_and_preprocess`` argv for the ResEnc-L preset."""
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-pl", planner]
    if configurations:
        cmd += ["-c", *configurations]
    if verify:
        cmd.append("--verify_dataset_integrity")
    return cmd


def train_command(
    fold: int | str,
    dataset_id: int = DATASET_ID,
    configuration: str = DEFAULT_CONFIG,
    trainer: str = GOAT_TRAINER,
    plans: str = RESENC_L_PLANS,
) -> list[str]:
    """Build a single ``nnUNetv2_train`` argv."""
    return [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "-p", plans,
    ]


def kfold_train_commands(folds: int = 5, **kwargs) -> list[list[str]]:
    """Launch commands for the standard K-fold cross-validation (Phase 3)."""
    return [train_command(fold=i, **kwargs) for i in range(folds)]
