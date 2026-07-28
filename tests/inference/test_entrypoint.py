from pathlib import Path

import pytest

from brats2026.inference.entrypoint import (
    ContainerCase,
    apply_staging,
    discover_container_cases,
    output_name,
    postprocess_labels,
    predict_argv,
    staging_ops,
)
from brats2026.nnunet.convert import CHANNEL_ORDER

np = pytest.importorskip("numpy")


def _make_case(root: Path, case_id: str, drop=()):
    """Create a /input-style case folder holding the four modality NIfTIs."""
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    for modality in CHANNEL_ORDER:
        if modality in drop:
            continue
        (case_dir / f"{case_id}-{modality}.nii.gz").write_bytes(b"vol")
    return case_dir


def test_output_name_keeps_case_id_and_timepoint():
    assert output_name("BraTS-MET-12345-100") == "BraTS-MET-12345-100.nii.gz"


def test_discover_finds_complete_cases(tmp_path):
    _make_case(tmp_path, "BraTS-GLI-00001-000")
    _make_case(tmp_path, "BraTS-MEN-00002-000")

    cases, incomplete = discover_container_cases(tmp_path)

    assert [c.case_id for c in cases] == ["BraTS-GLI-00001-000", "BraTS-MEN-00002-000"]
    assert incomplete == {}
    assert cases[0].reference_path.endswith("BraTS-GLI-00001-000-t1n.nii.gz")


def test_discover_reports_missing_modalities_without_dropping_them(tmp_path):
    _make_case(tmp_path, "BraTS-GLI-00001-000")
    _make_case(tmp_path, "BraTS-SSA-00002-000", drop=("t2w",))

    cases, incomplete = discover_container_cases(tmp_path)

    assert [c.case_id for c in cases] == ["BraTS-GLI-00001-000"]
    assert incomplete == {"BraTS-SSA-00002-000": ["t2w"]}


def test_discover_on_missing_input_dir_is_empty(tmp_path):
    cases, incomplete = discover_container_cases(tmp_path / "nope")
    assert cases == [] and incomplete == {}


def test_staging_ops_use_nnunet_channel_indices(tmp_path):
    case = ContainerCase(
        case_id="BraTS-GLI-00001-000",
        case_dir=tmp_path,
        inputs={m: f"/input/BraTS-GLI-00001-000-{m}.nii.gz" for m in CHANNEL_ORDER},
    )
    ops = staging_ops([case], tmp_path / "stage")
    names = [Path(op.dst).name for op in ops]

    assert names == [
        "BraTS-GLI-00001-000_0000.nii.gz",
        "BraTS-GLI-00001-000_0001.nii.gz",
        "BraTS-GLI-00001-000_0002.nii.gz",
        "BraTS-GLI-00001-000_0003.nii.gz",
    ]
    # channel index order must follow CHANNEL_ORDER exactly (t1n, t1c, t2f, t2w)
    assert ops[0].src.endswith("-t1n.nii.gz")
    assert ops[3].src.endswith("-t2w.nii.gz")


def test_apply_staging_materialises_files(tmp_path):
    _make_case(tmp_path / "input", "BraTS-PED-00001-000")
    cases, _ = discover_container_cases(tmp_path / "input")
    staging = tmp_path / "stage"

    apply_staging(staging_ops(cases, staging))

    # symlink or copy — both must resolve to readable content (Windows falls back to copy)
    for index in range(4):
        staged = staging / f"BraTS-PED-00001-000_{index:04d}.nii.gz"
        assert staged.exists()
        assert staged.read_bytes() == b"vol"


def test_apply_staging_never_writes_into_input(tmp_path):
    input_dir = tmp_path / "input"
    _make_case(input_dir, "BraTS-GLI-00001-000")
    before = sorted(p.name for p in (input_dir / "BraTS-GLI-00001-000").iterdir())

    cases, _ = discover_container_cases(input_dir)
    apply_staging(staging_ops(cases, tmp_path / "stage"))

    after = sorted(p.name for p in (input_dir / "BraTS-GLI-00001-000").iterdir())
    assert before == after


def test_predict_argv_pins_staging_and_output(tmp_path):
    argv = predict_argv(
        staging_dir=tmp_path / "stage",
        raw_output_dir=tmp_path / "raw",
        model_dir="501",
        folds=(0, 1),
        trainer="nnUNetTrainerGoAT",
        plans="nnUNetResEncUNetLPlans",
        configuration="3d_fullres",
        disable_tta=False,
    )
    assert argv[0] == "nnUNetv2_predict"
    assert str(tmp_path / "stage") in argv
    assert str(tmp_path / "raw") in argv
    assert "-f" in argv and "0" in argv and "1" in argv
    assert "--disable_tta" not in argv


def test_predict_argv_can_disable_tta_for_the_time_budget(tmp_path):
    argv = predict_argv(
        staging_dir=tmp_path, raw_output_dir=tmp_path, model_dir="501", folds=(0,),
        trainer="t", plans="p", configuration="3d_fullres", disable_tta=True,
    )
    assert "--disable_tta" in argv


def test_postprocess_is_a_noop_when_unconfigured():
    arr = np.zeros((8, 8, 8), dtype=int)
    arr[1, 1, 1] = 3  # a single-voxel ET speck
    out = postprocess_labels(arr, et_suppression_min_voxels=None, cc_min_voxels=None)
    assert np.array_equal(out, arr), "unconfigured thresholds must not invent post-processing"


def test_postprocess_suppresses_small_et_to_ncr():
    arr = np.zeros((10, 10, 10), dtype=int)
    arr[5:9, 5:9, 5:9] = 3           # a large ET block that must survive
    arr[0, 0, 0] = 3                 # an isolated ET speck that must be relabelled
    out = postprocess_labels(arr, et_suppression_min_voxels=8, cc_min_voxels=None)

    assert out[0, 0, 0] == 1, "tiny ET component should become NCR, not vanish"
    assert (out[5:9, 5:9, 5:9] == 3).all(), "large ET component must be preserved"


def test_postprocess_cc_filter_drops_distant_blobs():
    arr = np.zeros((12, 12, 12), dtype=int)
    arr[6:11, 6:11, 6:11] = 2        # main tumour
    arr[0, 0, 0] = 2                 # spurious distant blob
    out = postprocess_labels(arr, et_suppression_min_voxels=None, cc_min_voxels=10)

    assert out[0, 0, 0] == 0
    assert (out[6:11, 6:11, 6:11] == 2).all()


def test_postprocess_does_not_mutate_its_input():
    arr = np.zeros((6, 6, 6), dtype=int)
    arr[0, 0, 0] = 3
    original = arr.copy()
    postprocess_labels(arr, et_suppression_min_voxels=4, cc_min_voxels=4)
    assert np.array_equal(arr, original)
