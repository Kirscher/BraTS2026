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

# Our custom trainers. v1 lives in trainer.py, v2 in trainer_v2.py. Kept as separate constants
# (not a default + override) because nnU-Net derives its results directory from
# "<trainer>__<plans>__<configuration>": the distinct name is what stops a v2 run from writing
# into the fold directories of a v1 run that is still training.
GOAT_TRAINER = "nnUNetTrainerGoAT"
GOAT_TRAINER_V2 = "nnUNetTrainerGoATv2"


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


def train_v2_command(fold: int | str, **kwargs) -> list[str]:
    """Build a single ``nnUNetv2_train`` argv for the **v2** trainer.

    A thin wrapper over :func:`train_command` that swaps in :data:`GOAT_TRAINER_V2`, so the v2
    trainer name is never hand-typed on the HPC (a typo there resolves to a missing trainer, or
    worse to v1, after the queue wait). ``plans`` and ``configuration`` must stay identical to the
    v1 run being warm-started from, or the checkpoint's shapes will not match.
    """
    kwargs.setdefault("trainer", GOAT_TRAINER_V2)
    return train_command(fold=fold, **kwargs)
