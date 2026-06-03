from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .tasks import TaskSpec, task_base


@dataclass
class CaseRecord:
    task: str
    split: str
    case_id: str
    case_dir: str
    inputs: dict[str, str]
    target: str | None
    missing_required: list[str]
    missing_optional: list[str]
    notes: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def iter_case_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def find_case_file(case_dir: Path, case_id: str, suffix: str) -> Path | None:
    direct = case_dir / f"{case_id}-{suffix}.nii.gz"
    if direct.exists():
        return direct
    matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    return matches[0] if matches else None


def find_corrected_label(task_dir: Path, case_id: str) -> Path | None:
    correction_dir = task_dir / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
    flat = correction_dir / f"{case_id}-seg.nii.gz"
    if flat.exists():
        return flat
    nested = correction_dir / case_id / f"{case_id}-seg.nii.gz"
    if nested.exists():
        return nested
    return None


def discover_mri_cases(
    data_root: Path,
    spec: TaskSpec,
    split: str,
    max_records: int | None = None,
) -> list[CaseRecord]:
    roots = spec.train_roots if split == "train" else spec.validation_roots
    records: list[CaseRecord] = []
    base = task_base(data_root, spec)

    for root_name in roots:
        root = base / root_name
        for case_dir in iter_case_dirs(root):
            case_id = case_dir.name
            inputs: dict[str, str] = {}
            missing_required: list[str] = []
            missing_optional: list[str] = []
            notes: list[str] = []

            for suffix in spec.input_suffixes:
                path = find_case_file(case_dir, case_id, suffix)
                if path is None:
                    missing_required.append(suffix)
                else:
                    inputs[suffix] = str(path)

            for suffix in spec.optional_suffixes:
                path = find_case_file(case_dir, case_id, suffix)
                if path is None:
                    missing_optional.append(suffix)
                else:
                    inputs[suffix] = str(path)

            target: str | None = None
            if spec.target_suffix and split == "train":
                target_path = find_case_file(case_dir, case_id, spec.target_suffix)
                if spec.task_id == 1:
                    corrected = find_corrected_label(base, case_id)
                    if corrected is not None:
                        target_path = corrected
                        notes.append("uses_corrected_label")
                target = str(target_path) if target_path else None
                if target_path is None:
                    notes.append("missing_target")

            records.append(
                CaseRecord(
                    task=f"task{spec.task_id}",
                    split=split,
                    case_id=case_id,
                    case_dir=str(case_dir),
                    inputs=inputs,
                    target=target,
                    missing_required=missing_required,
                    missing_optional=missing_optional,
                    notes=notes,
                )
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def read_synapse_manifest(task_dir: Path) -> list[dict[str, str]]:
    manifest = task_dir / "SYNAPSE_METADATA_MANIFEST.tsv"
    if not manifest.exists():
        return []
    with manifest.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def inspect_pathology_shard(path: Path, limit: int = 5) -> dict[str, object]:
    samples: list[str] = []
    labels: list[str] = []
    with tarfile.open(path) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".jpg") and len(samples) < limit:
                samples.append(Path(member.name).stem)
            if member.isfile() and member.name.endswith(".cls") and len(labels) < limit:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    labels.append(extracted.read().decode("utf-8").strip())
            if len(samples) >= limit and len(labels) >= limit:
                break
    return {"path": str(path), "sample_keys": samples, "sample_labels": labels}


def discover_pathology(data_root: Path, spec: TaskSpec, inspect_first_shard: bool = False) -> dict[str, object]:
    base = task_base(data_root, spec)
    manifest = read_synapse_manifest(base)
    shards = [Path(row["path"]) for row in manifest if row.get("name", "").endswith(".tar")]
    class_map_path = base / "class_map.json"
    class_map = json.loads(class_map_path.read_text()) if class_map_path.exists() else spec.label_map
    info: dict[str, object] = {
        "task": f"task{spec.task_id}",
        "kind": spec.kind,
        "class_map": class_map,
        "num_manifest_entries": len(manifest),
        "num_shards": len(shards),
        "shards": [str(path) for path in shards],
    }
    if inspect_first_shard and shards:
        info["first_shard"] = inspect_pathology_shard(shards[0])
    return info


def write_jsonl(records: Iterable[CaseRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
