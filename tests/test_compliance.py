from brats2026.compliance import (
    Status,
    check_data_provenance,
    check_nas_not_committed,
    check_no_prior_weights,
    gate_status,
)


def test_data_provenance_passes_when_all_under_allowed_root(tmp_path):
    root = tmp_path / "Task3"
    cases = [str(root / "BraTS-GLI-1-0"), str(root / "BraTS-PED-2-0")]
    result = check_data_provenance(cases, [str(root)])
    assert result.status is Status.PASS


def test_data_provenance_fails_on_outside_case(tmp_path):
    root = tmp_path / "Task3"
    cases = [str(root / "ok"), "/some/external/dataset/case"]
    result = check_data_provenance(cases, [str(root)])
    assert result.status is Status.FAIL
    assert "outside" in result.detail


def test_data_provenance_unsure_without_allowed_roots():
    assert check_data_provenance(["/x"], []).status is Status.UNSURE


def test_no_prior_weights_flags_prior_brats_and_pretrained():
    refs = ["weights/brats2023_resenc.pth", "from_scratch_fold0.pth"]
    result = check_no_prior_weights(refs)
    assert result.status is Status.FAIL
    assert "brats2023" in result.detail


def test_no_prior_weights_passes_for_from_scratch():
    assert check_no_prior_weights(["fold0_scratch.pth"]).status is Status.PASS


def test_nas_not_committed():
    nas = ["/mnt/NAS2418_RADT/datasets/x/case-t1n.nii.gz"]
    assert check_nas_not_committed(nas, ["/mnt/"]).status is Status.FAIL
    assert check_nas_not_committed(["work/manifests/m.jsonl"], ["/mnt/"]).status is Status.PASS


def test_gate_status_priority():
    ok = check_no_prior_weights(["scratch.pth"])
    bad = check_no_prior_weights(["brats2021.pth"])
    unsure = check_data_provenance(["/x"], [])
    assert gate_status([ok]) is Status.PASS
    assert gate_status([ok, unsure]) is Status.UNSURE
    assert gate_status([ok, unsure, bad]) is Status.FAIL
