import json
from pathlib import Path

from brats2026.nnunet.convert import (
    CHANNEL_ORDER,
    DATASET_NAME,
    REGIONS_CLASS_ORDER,
    apply_conversion,
    build_dataset_json,
    plan_conversion,
)


def _record(case_id, root: Path, with_label=True, drop=()):
    inputs = {
        m: str(root / f"{case_id}-{m}.nii.gz")
        for m in CHANNEL_ORDER
        if m not in drop
    }
    target = str(root / f"{case_id}-seg.nii.gz") if with_label and "seg" not in drop else None
    return {"case_id": case_id, "inputs": inputs, "target": target}


def test_dataset_json_is_region_based():
    dj = build_dataset_json(num_training=42)
    assert dj["channel_names"] == {"0": "t1n", "1": "t1c", "2": "t2f", "3": "t2w"}
    assert dj["labels"]["whole_tumor"] == [1, 2, 3]
    assert dj["labels"]["tumor_core"] == [1, 3]
    assert dj["labels"]["enhancing_tumor"] == [3]
    assert dj["regions_class_order"] == list(REGIONS_CLASS_ORDER)
    assert dj["numTraining"] == 42
    assert dj["file_ending"] == ".nii.gz"


def test_plan_conversion_emits_four_channels_plus_label(tmp_path):
    rec = _record("BraTS-GLI-00001-000", tmp_path)
    plan = plan_conversion([rec], tmp_path / "raw")
    assert plan.num_training == 1
    assert plan.dataset_dir.name == DATASET_NAME
    dsts = [Path(op.dst).name for op in plan.ops]
    assert "BraTS-GLI-00001-000_0000.nii.gz" in dsts  # t1n
    assert "BraTS-GLI-00001-000_0003.nii.gz" in dsts  # t2w
    assert "BraTS-GLI-00001-000.nii.gz" in dsts        # label
    assert len(plan.ops) == 5


def test_plan_conversion_skips_incomplete_cases(tmp_path):
    good = _record("BraTS-GLI-1-0", tmp_path)
    no_modality = _record("BraTS-SSA-1-0", tmp_path, drop=("t2w",))
    no_label = _record("BraTS-MEN-1-0", tmp_path, with_label=False)
    plan = plan_conversion([good, no_modality, no_label], tmp_path / "raw")
    assert plan.converted == ["BraTS-GLI-1-0"]
    assert plan.skipped["BraTS-SSA-1-0"] == ["t2w"]
    assert plan.skipped["BraTS-MEN-1-0"] == ["seg"]


def test_apply_conversion_symlinks_and_writes_dataset_json(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    case = "BraTS-PED-00001-000"
    for m in CHANNEL_ORDER:
        (src / f"{case}-{m}.nii.gz").write_bytes(b"vol")
    (src / f"{case}-seg.nii.gz").write_bytes(b"seg")
    rec = _record(case, src)

    plan = plan_conversion([rec], tmp_path / "raw")
    apply_conversion(plan, link=True)

    ds = plan.dataset_dir
    assert (ds / "imagesTr" / f"{case}_0000.nii.gz").is_symlink()
    assert (ds / "labelsTr" / f"{case}.nii.gz").read_bytes() == b"seg"
    dj = json.loads((ds / "dataset.json").read_text())
    assert dj["numTraining"] == 1
