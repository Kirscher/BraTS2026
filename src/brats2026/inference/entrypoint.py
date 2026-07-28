"""Phase 6 — the submission container's ``/input`` → ``/output`` entrypoint.

This is what runs inside the Docker image the organizers execute. The challenge rules it
must satisfy (``docs/synapse_wiki_raw/instructions.md``) are non-negotiable:

- iterate every folder in ``/input`` and emit **one** ``.nii.gz`` per case,
- write **flat** into ``/output`` — a sub-folder invalidates the submission,
- never write to ``/input``, which is mounted read-only,
- preserve the input NIfTI geometry **exactly**,
- finish the whole test set inside 8 hours, with no network access.

The module is split so the decision logic is testable on a laptop: case discovery, the
staging plan and output naming are pure and unit-tested, while the two steps that need a GPU
(``nnUNetv2_predict``) or an imaging stack (writing NIfTIs) sit behind guarded imports.

``nnUNetv2_predict`` expects its input folder to use nnU-Net's channel-index naming
(``<identifier>_0000.nii.gz`` … ``_0003.nii.gz``) and emits ``<identifier>.nii.gz``. We stage
with ``identifier = case_id`` — the ``/input`` folder name — so nnU-Net's own output filename
is already the one the challenge asks for (it ends with the 5-digit case ID and 3-digit
timepoint), and no rename step can get it wrong.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..discover import find_case_file
from ..nnunet.convert import CHANNEL_ORDER, LinkOp

# The container's fixed mount points (challenge rule, not configurable).
INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")

# Scratch inside the image; /input is read-only so staging must live elsewhere.
DEFAULT_STAGING = Path("/tmp/brats2026_staging")


@dataclass(frozen=True)
class ContainerCase:
    """One discovered ``/input`` case: its id, its folder and its resolved modality paths."""

    case_id: str
    case_dir: Path
    inputs: dict[str, str]

    @property
    def reference_path(self) -> str:
        """The NIfTI whose geometry every prediction for this case must reproduce.

        ``t1n`` by convention — the four modalities are co-registered, so any of them defines
        the same grid, and picking one fixed channel keeps the guard deterministic.
        """
        return self.inputs[CHANNEL_ORDER[0]]


def output_name(case_id: str) -> str:
    """The flat ``/output`` filename for ``case_id``.

    The rule is that the name must end with the 5-digit case ID and 3-digit timepoint followed
    directly by ``.nii.gz``. The ``/input`` folder name already carries both (e.g.
    ``BraTS-MET-12345-100``), so echoing it unchanged satisfies the rule and keeps predictions
    traceable to their source folder.
    """
    return f"{case_id}.nii.gz"


def discover_container_cases(
    input_dir: str | Path = INPUT_DIR,
) -> tuple[list[ContainerCase], dict[str, list[str]]]:
    """Scan ``input_dir`` for case folders and resolve each one's four modalities.

    Returns ``(cases, incomplete)``. ``cases`` holds only folders where all four modalities
    resolved; ``incomplete`` maps a case id to the suffixes it is missing. An incomplete case
    is **not** dropped silently — the caller still has to emit a file for it, because a missing
    output is scored as a failure for that case rather than skipped.

    Uses the same :func:`brats2026.discover.find_case_file` matching as training-time discovery,
    so a naming quirk that works in one path works in both.
    """
    root = Path(input_dir)
    cases: list[ContainerCase] = []
    incomplete: dict[str, list[str]] = {}
    if not root.is_dir():
        return cases, incomplete

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        case_id = case_dir.name
        inputs: dict[str, str] = {}
        missing: list[str] = []
        for suffix in CHANNEL_ORDER:
            path = find_case_file(case_dir, case_id, suffix)
            if path is None:
                missing.append(suffix)
            else:
                inputs[suffix] = str(path)
        if missing:
            incomplete[case_id] = missing
            continue
        cases.append(ContainerCase(case_id=case_id, case_dir=case_dir, inputs=inputs))

    return cases, incomplete


def staging_ops(cases: Sequence[ContainerCase], staging_dir: str | Path) -> list[LinkOp]:
    """Plan the copy/link of every case's modalities into nnU-Net's channel-index naming.

    Emits ``<staging_dir>/<case_id>_0000.nii.gz`` … ``_0003.nii.gz`` in :data:`CHANNEL_ORDER`,
    reusing :class:`brats2026.nnunet.convert.LinkOp` so staging and training-time conversion
    describe filesystem work the same way. Pure: nothing touches disk here.
    """
    staging = Path(staging_dir)
    ops: list[LinkOp] = []
    for case in cases:
        for index, modality in enumerate(CHANNEL_ORDER):
            ops.append(
                LinkOp(
                    src=case.inputs[modality],
                    dst=str(staging / f"{case.case_id}_{index:04d}.nii.gz"),
                )
            )
    return ops


def apply_staging(ops: Sequence[LinkOp], link: bool = True) -> None:
    """Materialise a staging plan.

    Symlinks by default (no pixel copy, so a large test set costs no extra disk against the
    200 GB limit); falls back to a real copy when the filesystem refuses symlinks. ``/input``
    is never written — only the destinations under the staging dir are created.
    """
    for op in ops:
        dst = Path(op.dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            try:
                dst.symlink_to(Path(op.src).resolve())
                continue
            except (OSError, NotImplementedError):
                pass  # filesystem without symlink support — fall through to copy
        shutil.copy2(op.src, dst)


def predict_argv(
    staging_dir: str | Path,
    raw_output_dir: str | Path,
    model_dir: str | Path,
    folds: Sequence[int],
    trainer: str,
    plans: str,
    configuration: str,
    disable_tta: bool,
) -> list[str]:
    """The ``nnUNetv2_predict`` argv this container runs.

    Thin wrapper over :func:`brats2026.inference.predict.predict_command` that pins the staging
    and raw-output directories, so the container has exactly one place where the prediction
    command is defined.
    """
    from .predict import predict_command

    return predict_command(
        model_dir=model_dir,
        input_dir=staging_dir,
        output_dir=raw_output_dir,
        folds=tuple(folds),
        trainer=trainer,
        plans=plans,
        configuration=configuration,
        disable_tta=disable_tta,
    )


def postprocess_labels(label_array, et_suppression_min_voxels: Optional[int], cc_min_voxels: Optional[int]):
    """Apply the configured post-processing to one predicted label map.

    Both thresholds are ``# SPECIALIST:`` values read from ``configs/infer.yaml``; ``None``
    means "not configured" and the corresponding step is skipped rather than guessed. Order is
    deliberate: whole-tumour component filtering first (drops spurious distant blobs entirely),
    then ET-suppression (relabels tiny enhancing components to NCR, keeping the tumour core
    intact). Returns a new array; the input is not mutated.
    """
    import numpy as np

    from .postprocess import connected_component_filter, et_suppression

    out = np.asarray(label_array).copy()

    if cc_min_voxels is not None and cc_min_voxels > 1:
        keep = connected_component_filter(out > 0, cc_min_voxels)
        out[~keep] = 0

    if et_suppression_min_voxels is not None and et_suppression_min_voxels > 1:
        out = et_suppression(out, et_suppression_min_voxels)

    return out


def write_prediction(label_array, reference_path: str | Path, out_path: str | Path) -> None:
    """Write a label map to ``out_path`` carrying ``reference_path``'s geometry exactly.

    The affine and header are taken from the reference input rather than rebuilt, then the
    result is re-read and checked with
    :func:`brats2026.packaging.geometry.assert_same_geometry` at ``atol=0.0``. A geometry
    mismatch raises instead of shipping a file the scorer would reject.
    """
    import nibabel as nib
    import numpy as np

    from ..packaging.geometry import assert_same_geometry, read_geometry

    reference = nib.load(str(reference_path))
    arr = np.asarray(label_array).astype(np.uint8)

    out_img = nib.Nifti1Image(arr, reference.affine, header=reference.header)
    out_img.set_data_dtype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))

    ref_affine, ref_shape = read_geometry(reference_path)
    out_affine, out_shape = read_geometry(out_path)
    assert_same_geometry(ref_affine, ref_shape, out_affine, out_shape)


def write_empty_prediction(reference_path: str | Path, out_path: str | Path) -> None:
    """Write an all-background segmentation with ``reference_path``'s geometry.

    Used for a case the model could not be run on (a missing modality). Emitting an empty
    prediction keeps ``/output`` at one file per case — a *missing* file is scored as a failed
    case and can invalidate the run, whereas an empty one simply scores poorly on that case.
    """
    import numpy as np

    import nibabel as nib

    reference = nib.load(str(reference_path))
    write_prediction(np.zeros(reference.shape, dtype=np.uint8), reference_path, out_path)


def run_container(
    input_dir: str | Path = INPUT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    staging_dir: str | Path = DEFAULT_STAGING,
    infer_config: Optional[str | Path] = None,
    model_dir: str = "501",
    folds: Sequence[int] = (0,),
    trainer: str = "nnUNetTrainerGoAT",
    plans: str = "nnUNetResEncUNetLPlans",
    configuration: str = "3d_fullres",
    disable_tta: bool = False,
) -> int:
    """Run the full container pipeline. Returns the number of cases written to ``/output``.

    Sequence: discover ``/input`` → stage into nnU-Net naming → ``nnUNetv2_predict`` → per-case
    post-processing → geometry-checked flat write to ``/output``. Cases missing a modality get
    an empty prediction so the output stays one-file-per-case.

    Post-processing thresholds come from ``infer_config`` (``configs/infer.yaml``). When it is
    absent or its thresholds are ``null``, post-processing is **skipped** — the raw nnU-Net
    output ships unchanged. That is the safe default: an unconfigured threshold must never be
    invented at submission time.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    staging_dir = Path(staging_dir)
    raw_output = staging_dir.parent / "raw_predictions"

    et_min: Optional[int] = None
    cc_min: Optional[int] = None
    if infer_config is not None and Path(infer_config).is_file():
        from ..config import load_config

        cfg = load_config(Path(infer_config))
        et_min = cfg.get("et_suppression_min_voxels")
        cc_min = cfg.get("cc_min_voxels")

    cases, incomplete = discover_container_cases(input_dir)
    print(f"[entrypoint] discovered {len(cases)} complete case(s) in {input_dir}", flush=True)
    if incomplete:
        print(
            f"[entrypoint] WARNING {len(incomplete)} case(s) missing modalities: "
            f"{sorted(incomplete)[:5]}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if cases:
        staging_dir.mkdir(parents=True, exist_ok=True)
        apply_staging(staging_ops(cases, staging_dir))

        argv = predict_argv(
            staging_dir, raw_output, model_dir, folds, trainer, plans, configuration, disable_tta
        )
        print(f"[entrypoint] {' '.join(argv)}", flush=True)
        subprocess.run(argv, check=True)

    import nibabel as nib
    import numpy as np

    written = 0
    for case in cases:
        raw_path = raw_output / f"{case.case_id}.nii.gz"
        if not raw_path.is_file():
            print(f"[entrypoint] WARNING no prediction for {case.case_id}; writing empty", flush=True)
            write_empty_prediction(case.reference_path, output_dir / output_name(case.case_id))
            written += 1
            continue
        arr = np.asanyarray(nib.load(str(raw_path)).dataobj)
        arr = np.rint(arr).astype(np.uint8)
        arr = postprocess_labels(arr, et_min, cc_min)
        write_prediction(arr, case.reference_path, output_dir / output_name(case.case_id))
        written += 1

    for case_id in incomplete:
        case_dir = input_dir / case_id
        reference = next(iter(sorted(case_dir.glob("*.nii.gz"))), None)
        if reference is None:
            print(f"[entrypoint] WARNING {case_id} has no NIfTI at all; cannot emit", flush=True)
            continue
        write_empty_prediction(reference, output_dir / output_name(case_id))
        written += 1

    print(f"[entrypoint] wrote {written} prediction(s) to {output_dir}", flush=True)
    return written
