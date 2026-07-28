import json

import pytest

from brats2026.domains import COHORTS
from brats2026.synthesis.dataset import (
    SYNTH_MANIFEST_NAME,
    apply_synth_dataset,
    discover_synthetic_cases,
    is_synthetic_case_id,
    plan_synth_dataset,
    read_synthetic_ids,
    synth_dataset_name,
    train_only_splits,
)
from brats2026.synthesis.generate import synthetic_case_id

MODALITIES = ("t1n", "t1c", "t2f", "t2w")


def _write_case(root, case_id, suffixes=MODALITIES + ("seg",)):
    """Create a BraTS-layout case dir with placeholder NIfTIs for each suffix."""
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (case_dir / f"{case_id}-{suffix}.nii.gz").write_bytes(b"nifti")
    return case_dir


def _real_record(tmp_path, case_id):
    case_dir = _write_case(tmp_path / "real", case_id)
    return {
        "case_id": case_id,
        "inputs": {m: str(case_dir / f"{case_id}-{m}.nii.gz") for m in MODALITIES},
        "target": str(case_dir / f"{case_id}-seg.nii.gz"),
    }


# --- is_synthetic_case_id ----------------------------------------------------------------

def test_is_synthetic_case_id_flags_minted_ids():
    for cohort in COHORTS:
        assert is_synthetic_case_id(synthetic_case_id(cohort, 7))


@pytest.mark.parametrize(
    "case_id",
    ["BraTS-GLI-00001-000", "BraTS-GoAT-00123", "BraTS-MEN-00042", "", "BraTS-PED-abcdef"],
)
def test_is_synthetic_case_id_rejects_real_ids(case_id):
    assert not is_synthetic_case_id(case_id)


# --- synth_dataset_name ------------------------------------------------------------------

def test_synth_dataset_name_defaults_to_701():
    assert synth_dataset_name() == "Dataset701_BraTSGoATSynth"


def test_synth_dataset_name_honours_explicit_id():
    assert synth_dataset_name(801) == "Dataset801_BraTSGoATSynth"


# --- discover_synthetic_cases ------------------------------------------------------------

def test_discover_synthetic_cases_reads_cohort_subdirs(tmp_path):
    root = tmp_path / "synthetic"
    _write_case(root / "GLI", synthetic_case_id("GLI", 0))
    _write_case(root / "MET", synthetic_case_id("MET", 1))

    records, incomplete = discover_synthetic_cases(root)

    assert len(records) == 2
    assert incomplete == {}
    assert all(set(r["inputs"]) == set(MODALITIES) and r["target"] for r in records)


def test_discover_synthetic_cases_reads_flat_layout(tmp_path):
    root = tmp_path / "synthetic"
    _write_case(root, synthetic_case_id("PED", 3))

    records, incomplete = discover_synthetic_cases(root)

    assert [r["case_id"] for r in records] == [synthetic_case_id("PED", 3)]
    assert incomplete == {}


def test_discover_synthetic_cases_reports_incomplete_case(tmp_path):
    root = tmp_path / "synthetic"
    cid = synthetic_case_id("SSA", 4)
    _write_case(root / "SSA", cid, suffixes=("t1n", "t1c", "seg"))

    records, incomplete = discover_synthetic_cases(root)

    assert records == []
    assert sorted(incomplete[cid]) == ["t2f", "t2w"]


def test_discover_synthetic_cases_on_missing_root_is_empty(tmp_path):
    records, incomplete = discover_synthetic_cases(tmp_path / "nope")
    assert records == [] and incomplete == {}


# --- plan_synth_dataset ------------------------------------------------------------------

def test_plan_synth_dataset_merges_real_and_synthetic(tmp_path):
    real = [_real_record(tmp_path, "BraTS-GLI-00001-000")]
    root = tmp_path / "synthetic"
    _write_case(root / "GLI", synthetic_case_id("GLI", 0))

    plan, synthetic_ids = plan_synth_dataset(real, root, tmp_path / "raw")

    assert plan.num_training == 2
    assert synthetic_ids == [synthetic_case_id("GLI", 0)]
    assert plan.dataset_dir.name == synth_dataset_name()


def test_plan_synth_dataset_rejects_foreign_synthetic_id(tmp_path):
    root = tmp_path / "synthetic"
    _write_case(root / "GLI", "Foreign-Case-000001")

    with pytest.raises(ValueError, match="GoAT cohort"):
        plan_synth_dataset([], root, tmp_path / "raw")


def test_plan_synth_dataset_rejects_collision_with_real_case(tmp_path):
    cid = synthetic_case_id("GLI", 0)
    real = [_real_record(tmp_path, cid)]
    root = tmp_path / "synthetic"
    _write_case(root / "GLI", cid)

    with pytest.raises(ValueError, match="collide"):
        plan_synth_dataset(real, root, tmp_path / "raw")


# --- apply_synth_dataset -----------------------------------------------------------------

def test_apply_synth_dataset_writes_sidecar_and_dataset_json(tmp_path):
    real = [_real_record(tmp_path, "BraTS-GLI-00001-000")]
    root = tmp_path / "synthetic"
    _write_case(root / "MET", synthetic_case_id("MET", 0))

    plan, synthetic_ids = plan_synth_dataset(real, root, tmp_path / "raw")
    sidecar = apply_synth_dataset(plan, synthetic_ids, link=False)

    body = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar.name == SYNTH_MANIFEST_NAME
    assert body["n_synthetic"] == 1 and body["n_real"] == 1 and body["n_total"] == 2
    assert body["per_cohort"] == {"MET": 1}
    assert read_synthetic_ids(plan.dataset_dir) == synthetic_ids

    dataset_json = json.loads((plan.dataset_dir / "dataset.json").read_text())
    assert dataset_json["numTraining"] == 2


def test_read_synthetic_ids_absent_sidecar_is_empty(tmp_path):
    assert read_synthetic_ids(tmp_path) == []


# --- train_only_splits (the leak guard) --------------------------------------------------

def _real_ids(n_per_cohort=4):
    return [f"BraTS-{c}-{i:05d}-000" for c in COHORTS for i in range(n_per_cohort)]


def test_train_only_splits_never_puts_synthetic_in_val():
    real = _real_ids()
    synth = [synthetic_case_id(c, i) for c in COHORTS for i in range(2)]

    folds = train_only_splits(real + synth, synthetic_ids=synth, k=5, seed=1)

    for fold in folds:
        assert not set(fold["val"]).intersection(synth)


def test_train_only_splits_puts_every_synthetic_in_every_train_fold():
    real = _real_ids()
    synth = [synthetic_case_id(c, i) for c in COHORTS for i in range(2)]

    folds = train_only_splits(real + synth, synthetic_ids=synth, k=5, seed=1)

    for fold in folds:
        assert set(synth).issubset(fold["train"])


def test_train_only_splits_val_union_covers_every_real_case():
    real = _real_ids()
    synth = [synthetic_case_id("GLI", i) for i in range(3)]

    folds = train_only_splits(real + synth, synthetic_ids=synth, k=5, seed=1)

    covered = set().union(*(set(f["val"]) for f in folds))
    assert covered == set(real)


def test_train_only_splits_infers_synthetic_ids_from_convention():
    real = _real_ids()
    synth = [synthetic_case_id("PED", i) for i in range(3)]

    folds = train_only_splits(real + synth, k=5, seed=1)

    for fold in folds:
        assert not set(fold["val"]).intersection(synth)
        assert set(synth).issubset(fold["train"])


def test_train_only_splits_matches_plain_kfold_when_no_synthetic():
    from brats2026.nnunet.splits import domain_balanced_kfold

    real = _real_ids()
    assert train_only_splits(real, synthetic_ids=[], k=5, seed=3) == domain_balanced_kfold(
        real, k=5, seed=3
    )
