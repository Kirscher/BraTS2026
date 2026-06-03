from __future__ import annotations

import json
from pathlib import Path


def require_mri_dependencies():
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "MRI preprocessing requires optional dependencies. Install them with:\n"
            "  python -m pip install -e '.[mri]'\n"
            "or:\n"
            "  python -m pip install numpy nibabel"
        ) from exc
    return nib, np


def robust_normalize(volume, np):
    mask = volume != 0
    if not mask.any():
        return volume.astype("float32")
    values = volume[mask]
    low, high = np.percentile(values, [0.5, 99.5])
    clipped = np.clip(volume, low, high)
    mean = clipped[mask].mean()
    std = clipped[mask].std()
    if std < 1e-6:
        std = 1.0
    normalized = (clipped - mean) / std
    normalized[~mask] = 0
    return normalized.astype("float32")


def foreground_bbox(volumes, np, margin: int = 8):
    mask = np.zeros(volumes[0].shape, dtype=bool)
    for volume in volumes:
        mask |= volume != 0
    if not mask.any():
        return tuple(slice(0, size) for size in volumes[0].shape)
    coords = np.where(mask)
    slices = []
    for axis, size in enumerate(mask.shape):
        start = max(int(coords[axis].min()) - margin, 0)
        stop = min(int(coords[axis].max()) + margin + 1, size)
        slices.append(slice(start, stop))
    return tuple(slices)


def load_manifest_records(manifest: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with manifest.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def preprocess_mri_manifest(
    manifest: Path,
    output_dir: Path,
    modalities: list[str] | None = None,
    limit: int | None = None,
    crop_margin: int = 8,
) -> None:
    nib, np = require_mri_dependencies()
    records = load_manifest_records(manifest, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        available = record["inputs"]
        selected = modalities or list(available.keys())
        missing = [modality for modality in selected if modality not in available]
        if missing:
            raise ValueError(f"{record['case_id']} missing modalities requested for preprocessing: {missing}")

        images = [nib.load(available[modality]) for modality in selected]
        arrays = [image.get_fdata(dtype=np.float32) for image in images]
        bbox = foreground_bbox(arrays, np, margin=crop_margin)
        normalized = [robust_normalize(array[bbox], np) for array in arrays]
        stacked = np.stack(normalized, axis=0)

        payload = {
            "image": stacked,
            "modalities": np.array(selected),
            "case_id": record["case_id"],
            "bbox_start": np.array([s.start for s in bbox], dtype=np.int16),
            "bbox_stop": np.array([s.stop for s in bbox], dtype=np.int16),
            "affine": images[0].affine.astype(np.float32),
        }

        if record.get("target"):
            target = nib.load(record["target"]).get_fdata(dtype=np.float32).astype(np.uint8)
            payload["label"] = target[bbox]

        out_file = output_dir / f"{record['task']}_{record['split']}_{record['case_id']}.npz"
        np.savez_compressed(out_file, **payload)
