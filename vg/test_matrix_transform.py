import numpy as np
import pytest
from .matrix import transform as apply_transform

scale_factor = np.array([3.0, 0.5, 2.0])
transform = np.array(
    [
        [scale_factor[0], 0, 0, 0],
        [0, scale_factor[1], 0, 0],
        [0, 0, scale_factor[2], 0],
        [0, 0, 0, 1],
    ]
)


def test_apply_homogeneous():
    point = np.array([5.0, 0.0, 1.0])
    expected_point = np.array([15.0, 0.0, 2.0])
    np.testing.assert_array_equal(apply_transform(point, transform), expected_point)


def test_apply_homogeneous_stacked():
    points = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, 1.0]])
    expected_points = np.array([[3.0, 1.0, 6.0], [15.0, 0.0, 2.0]])
    np.testing.assert_array_equal(apply_transform(points, transform), expected_points)


def test_apply_homogeneous_error():
    with pytest.raises(ValueError, match="Transformation matrix should be 4x4"):
        apply_transform(np.array([1.0, 2.0, 3.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="Vertices should be 3x1 or Nx3"):
        apply_transform(np.array([1.0, 2.0]), transform)
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        apply_transform(np.array([[[1.0, 2.0, 3.0]]]), transform)
