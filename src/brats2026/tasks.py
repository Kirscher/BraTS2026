from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/mnt/CPS-RADT/datasets/MICCAI/2026/BraTS2026")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    kind: str
    input_suffixes: tuple[str, ...]
    target_suffix: str | None = None
    train_roots: tuple[str, ...] = ()
    validation_roots: tuple[str, ...] = ()
    optional_suffixes: tuple[str, ...] = ()
    label_map: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @property
    def task_dir(self) -> str:
        return f"Task{self.task_id}"


TASKS: dict[str, TaskSpec] = {
    "task1": TaskSpec(
        name="Brain metastases segmentation",
        task_id=1,
        kind="mri_segmentation",
        input_suffixes=("t1n", "t1c", "t2f"),
        optional_suffixes=("t2w",),
        target_suffix="seg",
        train_roots=("MICCAI-LH-BraTS2025-MET-Challenge-Training",),
        validation_roots=("Validation",),
        label_map={"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
        notes="T2W is not mandatory for all MET cases; keep spatial metadata unchanged for submission.",
    ),
    "task2": TaskSpec(
        name="Pediatric brain tumor segmentation",
        task_id=2,
        kind="mri_segmentation",
        input_suffixes=("t1n", "t1c", "t2f", "t2w"),
        target_suffix="seg",
        train_roots=("BraTS26_PED_training", "BraTS-PEDs_Batch2_Release"),
        validation_roots=("BraTS26_PED_validation",),
        label_map={"ET": 1, "NET": 2, "CC": 3, "ED": 4},
        notes="Data are defaced but not skull-stripped; validation/test are also not skull-stripped.",
    ),
    "task3": TaskSpec(
        name="BraTS-GoAT generalizable tumor segmentation",
        task_id=3,
        kind="mri_segmentation",
        input_suffixes=("t1n", "t1c", "t2f", "t2w"),
        target_suffix="seg",
        train_roots=(
            "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth",
            "MICCAI2024-BraTS-GoAT-TrainingData-WithOut-GroundTruth",
        ),
        validation_roots=("MICCAI2024-BraTS-GoAT-ValidationData",),
        label_map={"NCR": 1, "ED": 2, "ET": 3},
        notes="Official rules forbid any data beyond the BraTS-GoAT sub-challenge data.",
    ),
    "task4": TaskSpec(
        name="Local MRI inpainting",
        task_id=4,
        kind="mri_inpainting",
        input_suffixes=("t1n-voided", "mask"),
        target_suffix="t1n",
        train_roots=("ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training",),
        validation_roots=("ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation",),
        optional_suffixes=("mask-healthy", "mask-unhealthy"),
        notes="Submission output is one NIfTI per case ending in -t1n-inference.nii.gz.",
    ),
    "task5": TaskSpec(
        name="BraTS-Path histology classification",
        task_id=5,
        kind="pathology_classification",
        input_suffixes=("jpg",),
        target_suffix="cls",
        train_roots=(".",),
        validation_roots=(),
        label_map={
            "CT": 0,
            "DM": 1,
            "IC": 2,
            "LI": 3,
            "MP": 4,
            "NC": 5,
            "PL": 6,
            "PN": 7,
            "WM": 8,
            "NOTA": 9,
        },
        notes="Training/validation are WebDataset tar shards plus class_map.json.",
    ),
}


def get_task(task: str) -> TaskSpec:
    key = task.lower()
    if key not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task {task!r}. Expected one of: {known}")
    return TASKS[key]


def task_base(data_root: Path, spec: TaskSpec) -> Path:
    return data_root / spec.task_dir
