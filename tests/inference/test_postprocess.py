import numpy as np
import pytest

from brats2026.inference.postprocess import (
    connected_component_filter,
    et_suppression,
    merge_tta,
)


def test_merge_tta_averages_maps():
    a = np.zeros((2, 2)) + 0.2
    b = np.zeros((2, 2)) + 0.8
    merged = merge_tta([a, b])
    assert np.allclose(merged, 0.5)


def test_merge_tta_single_map_is_identity():
    a = np.array([[0.1, 0.9], [0.3, 0.7]])
    assert np.allclose(merge_tta([a]), a)


def test_merge_tta_rejects_empty():
    with pytest.raises(ValueError):
        merge_tta([])


def test_merge_tta_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        merge_tta([np.zeros((2, 2)), np.zeros((2, 3))])


def test_component_filter_drops_small_keeps_large():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True  # size-1 blob
    mask[5:9, 5:9] = True  # size-16 blob
    cleaned = connected_component_filter(mask, min_voxels=5)
    assert not cleaned[0, 0]
    assert cleaned[5:9, 5:9].all()
    assert cleaned.sum() == 16


def test_component_filter_boundary_size_equal_is_kept():
    mask = np.zeros((10,), dtype=bool)
    mask[0:3] = True  # size exactly 3
    cleaned = connected_component_filter(mask, min_voxels=3)
    assert cleaned[0:3].all()


def test_et_suppression_converts_small_et_to_ncr():
    arr = np.zeros((10, 10), dtype=int)
    arr[0, 0] = 3  # tiny ET blob (size 1)
    out = et_suppression(arr, min_voxels=5)
    assert out[0, 0] == 1  # relabelled to NCR
    assert not (out == 3).any()


def test_et_suppression_keeps_large_et_and_other_labels():
    arr = np.zeros((12, 12), dtype=int)
    arr[0, 0] = 3  # tiny ET -> should be suppressed
    arr[5:9, 5:9] = 3  # large ET (16) -> kept
    arr[0, 5] = 1  # NCR untouched
    arr[5, 0] = 2  # ED untouched
    out = et_suppression(arr, min_voxels=5)
    assert out[0, 0] == 1
    assert (out[5:9, 5:9] == 3).all()
    assert out[0, 5] == 1
    assert out[5, 0] == 2


def test_et_suppression_boundary_size_equal_is_kept():
    arr = np.zeros((10,), dtype=int)
    arr[0:5] = 3  # size exactly 5
    out = et_suppression(arr, min_voxels=5)
    assert (out[0:5] == 3).all()


def test_et_suppression_custom_replace_with():
    arr = np.zeros((6,), dtype=int)
    arr[0] = 3
    out = et_suppression(arr, min_voxels=5, replace_with=2)
    assert out[0] == 2
