"""Phase 1 — convert a discovery manifest into an nnU-Net v2 raw dataset.

Target layout (``work/nnUNet_raw/Dataset501_BraTSGoAT/``)::

    imagesTr/<case_id>_0000.nii.gz   # t1n
    imagesTr/<case_id>_0001.nii.gz   # t1c
    imagesTr/<case_id>_0002.nii.gz   # t2f
    imagesTr/<case_id>_0003.nii.gz   # t2w
    labelsTr/<case_id>.nii.gz        # seg (NCR=1, ED=2, ET=3)
    dataset.json

Region-based targets (ET/TC/WT) are declared in ``dataset.json``; the harmonised integer
labels NCR=1/ED=2/ET=3 are reconstructed via ``regions_class_order``.

Conversion *planning* is pure (testable without disk); the IO step (symlink/copy) is a thin
wrapper. We **symlink by default** so the read-only NAS NIfTI geometry is preserved bit-for-
bit and no pixels are copied.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# nnU-Net channel order is fixed by index; GoAT requires all four modalities per case.
CHANNEL_ORDER: tuple[str, ...] = ("t1n", "t1c", "t2f", "t2w")

# Region-based labels. Regions overlap; regions_class_order paints them in order to rebuild
# the integer labels: WT->2 (edema), then TC->1 (necrotic), then ET->3 (enhancing), so the
# later paints carve out NCR=1 (TC\ET) and ED=2 (WT\TC) and ET=3.
REGION_LABELS: dict[str, object] = {
    "background": 0,
    "whole_tumor": [1, 2, 3],
    "tumor_core": [1, 3],
    "enhancing_tumor": [3],
}
REGIONS_CLASS_ORDER: tuple[int, ...] = (2, 1, 3)

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_BraTSGoAT"


def build_dataset_json(num_training: int, file_ending: str = ".nii.gz") -> dict:
    """Return an nnU-Net v2 region-based ``dataset.json`` as a dict."""
    return {
        "channel_names": {str(i): name for i, name in enumerate(CHANNEL_ORDER)},
        "labels": REGION_LABELS,
        "regions_class_order": list(REGIONS_CLASS_ORDER),
        "numTraining": num_training,
        "file_ending": file_ending,
    }


@dataclass(frozen=True)
class LinkOp:
    """One planned filesystem link: copy/symlink ``src`` to ``dst``."""

    src: str
    dst: str


@dataclass(frozen=True)
class ConversionPlan:
    dataset_dir: Path
    ops: list[LinkOp]
    converted: list[str]
    skipped: dict[str, list[str]]  # case_id -> missing channels (incl. "seg")

    @property
    def num_training(self) -> int:
        return len(self.converted)


def plan_conversion(records: list[dict], raw_root: Path, require_label: bool = True) -> ConversionPlan:
    """Plan the link operations for converting manifest ``records`` to nnU-Net raw.

    A case is converted only when all four modalities (and, if ``require_label``, the seg)
    are present; otherwise it is recorded under ``skipped`` with the missing channel names.
    """
    dataset_dir = raw_root / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    ops: list[LinkOp] = []
    converted: list[str] = []
    skipped: dict[str, list[str]] = {}

    for record in records:
        case_id = record["case_id"]
        inputs = record.get("inputs", {})
        missing = [m for m in CHANNEL_ORDER if m not in inputs]
        target = record.get("target")
        if require_label and not target:
            missing = missing + ["seg"]
        if missing:
            skipped[case_id] = missing
            continue

        case_ops = [
            LinkOp(src=inputs[modality], dst=str(images_tr / f"{case_id}_{i:04d}.nii.gz"))
            for i, modality in enumerate(CHANNEL_ORDER)
        ]
        if target:
            case_ops.append(LinkOp(src=target, dst=str(labels_tr / f"{case_id}.nii.gz")))
        ops.extend(case_ops)
        converted.append(case_id)

    return ConversionPlan(dataset_dir=dataset_dir, ops=ops, converted=converted, skipped=skipped)


def apply_conversion(plan: ConversionPlan, link: bool = True, overwrite: bool = False) -> None:
    """Materialise a plan: symlink (default) or hard-copy each op, then write dataset.json."""
    (plan.dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (plan.dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    for op in plan.ops:
        dst = Path(op.dst)
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                continue
            dst.unlink()
        if link:
            os.symlink(os.path.realpath(op.src), dst)
        else:
            import shutil

            shutil.copy2(op.src, dst)

    dataset_json = plan.dataset_dir / "dataset.json"
    dataset_json.write_text(json.dumps(build_dataset_json(plan.num_training), indent=2) + "\n")
