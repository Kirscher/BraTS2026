import json

from brats2026.nnunet.splits import (
    domain_balanced_kfold,
    leave_one_domain_out,
    write_splits,
)


def _cohort_ids():
    ids = []
    for cohort, n in (("GLI", 20), ("SSA", 10), ("MEN", 10), ("MET", 5), ("PED", 5)):
        ids += [f"BraTS-{cohort}-{i:05d}-000" for i in range(n)]
    return ids


def test_kfold_partitions_validation_without_overlap():
    ids = _cohort_ids()
    folds = domain_balanced_kfold(ids, k=5, seed=0)
    assert len(folds) == 5
    seen = []
    for fold in folds:
        seen += fold["val"]
        # train and val are disjoint and together cover everything
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert set(fold["train"]) | set(fold["val"]) == set(ids)
    # every case validated exactly once across folds
    assert sorted(seen) == sorted(ids)


def test_kfold_is_domain_balanced():
    ids = _cohort_ids()
    folds = domain_balanced_kfold(ids, k=5, seed=0)
    # 20 GLI across 5 folds -> 4 per fold
    for fold in folds:
        gli = [c for c in fold["val"] if "GLI" in c]
        assert len(gli) == 4


def test_kfold_is_deterministic():
    ids = _cohort_ids()
    assert domain_balanced_kfold(ids, 5, seed=7) == domain_balanced_kfold(ids, 5, seed=7)


def test_leave_one_domain_out_holds_out_each_cohort():
    ids = _cohort_ids()
    folds = leave_one_domain_out(ids)
    assert [f["held_out"] for f in folds] == ["GLI", "SSA", "MEN", "MET", "PED"]
    for fold in folds:
        held = fold["held_out"]
        assert all(held in cid for cid in fold["val"])
        assert all(held not in cid for cid in fold["train"])


def test_write_splits_roundtrip(tmp_path):
    ids = _cohort_ids()
    folds = domain_balanced_kfold(ids, k=5, seed=0)
    out = write_splits(folds, tmp_path / "splits_final.json")
    assert json.loads(out.read_text()) == folds
