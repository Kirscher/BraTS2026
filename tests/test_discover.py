import json
import tarfile
from io import BytesIO
from pathlib import Path

from brats2026.discover import discover_mri_cases, discover_pathology, write_jsonl
from brats2026.tasks import get_task


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_task1_discovery_uses_corrected_labels_and_tracks_optional_missing(tmp_path):
    spec = get_task("task1")
    case_id = "BraTS-MET-00001-000"
    case_dir = tmp_path / "Task1" / spec.train_roots[0] / case_id

    for suffix in ("t1n", "t1c", "t2f", "seg"):
        touch(case_dir / f"{case_id}-{suffix}.nii.gz")

    corrected = tmp_path / "Task1" / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels" / f"{case_id}-seg.nii.gz"
    touch(corrected)

    records = discover_mri_cases(tmp_path, spec, split="train")

    assert len(records) == 1
    record = records[0]
    assert record.case_id == case_id
    assert record.missing_required == []
    assert record.missing_optional == ["t2w"]
    assert record.target == str(corrected)
    assert record.notes == ["uses_corrected_label"]


def test_mri_discovery_respects_max_records(tmp_path):
    spec = get_task("task2")
    root = tmp_path / "Task2" / spec.train_roots[0]

    for index in range(3):
        case_id = f"BraTS-PED-{index:05d}-000"
        case_dir = root / case_id
        for suffix in ("t1n", "t1c", "t2f", "t2w", "seg"):
            touch(case_dir / f"{case_id}-{suffix}.nii.gz")

    records = discover_mri_cases(tmp_path, spec, split="train", max_records=2)

    assert [record.case_id for record in records] == [
        "BraTS-PED-00000-000",
        "BraTS-PED-00001-000",
    ]


def test_write_jsonl(tmp_path):
    spec = get_task("task4")
    case_id = "BraTS-GLI-00001-000"
    case_dir = tmp_path / "Task4" / spec.validation_roots[0] / case_id
    touch(case_dir / f"{case_id}-t1n-voided.nii.gz")
    touch(case_dir / f"{case_id}-mask.nii.gz")
    records = discover_mri_cases(tmp_path, spec, split="validation")

    output = tmp_path / "manifest.jsonl"
    write_jsonl(records, output)

    lines = output.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["case_id"] == case_id


def test_pathology_discovery_reads_manifest_class_map_and_first_shard(tmp_path):
    task_dir = tmp_path / "Task5"
    task_dir.mkdir()
    shard = task_dir / "shard-000000.tar"

    with tarfile.open(shard, "w") as tar:
        jpg_payload = b"not-a-real-image-but-valid-tar-member"
        jpg_info = tarfile.TarInfo("train_abc.jpg")
        jpg_info.size = len(jpg_payload)
        tar.addfile(jpg_info, BytesIO(jpg_payload))

        cls_payload = b"5"
        cls_info = tarfile.TarInfo("train_abc.cls")
        cls_info.size = len(cls_payload)
        tar.addfile(cls_info, BytesIO(cls_payload))

    (task_dir / "class_map.json").write_text('{"CT": 0, "NC": 5, "NOTA": 9}')
    (task_dir / "SYNAPSE_METADATA_MANIFEST.tsv").write_text(
        "path\tparent\tname\tid\tsynapseStore\tcontentType\n"
        f"{shard}\tsyn-test\tshard-000000.tar\tsyn-shard\tTrue\tapplication/x-tar\n"
    )

    info = discover_pathology(tmp_path, get_task("task5"), inspect_first_shard=True)

    assert info["num_manifest_entries"] == 1
    assert info["num_shards"] == 1
    assert info["class_map"] == {"CT": 0, "NC": 5, "NOTA": 9}
    assert info["first_shard"]["sample_keys"] == ["train_abc"]
    assert info["first_shard"]["sample_labels"] == ["5"]
