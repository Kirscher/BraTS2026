import numpy as np
import pytest

from brats2026.packaging.geometry import (
    GeometryError,
    assert_same_geometry,
    geometry_matches,
)

AFF = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
SHAPE = (5, 6, 7)


def test_identical_geometry_passes():
    assert geometry_matches(AFF, SHAPE, AFF.copy(), SHAPE)
    assert assert_same_geometry(AFF, SHAPE, AFF.copy(), SHAPE) is None


def test_shape_mismatch_fails():
    assert not geometry_matches(AFF, SHAPE, AFF, (5, 6, 8))
    with pytest.raises(GeometryError, match="shape"):
        assert_same_geometry(AFF, SHAPE, AFF, (5, 6, 8))


def test_affine_mismatch_fails():
    bad = AFF.copy()
    bad[0, 3] = 2.0
    assert not geometry_matches(AFF, SHAPE, bad, SHAPE)
    with pytest.raises(GeometryError, match="affine"):
        assert_same_geometry(AFF, SHAPE, bad, SHAPE)


def test_default_atol_is_exact():
    near = AFF.copy()
    near[0, 0] += 1e-9
    assert not geometry_matches(AFF, SHAPE, near, SHAPE)  # atol=0.0 -> exact


def test_tiny_diff_within_atol_passes():
    near = AFF.copy()
    near[0, 0] += 1e-7
    assert geometry_matches(AFF, SHAPE, near, SHAPE, atol=1e-5)
    assert assert_same_geometry(AFF, SHAPE, near, SHAPE, atol=1e-5) is None


def test_diff_outside_atol_raises():
    near = AFF.copy()
    near[0, 0] += 1e-3
    with pytest.raises(GeometryError, match="affine"):
        assert_same_geometry(AFF, SHAPE, near, SHAPE, atol=1e-5)
