import numpy as np
import pytest
import vg.compat.v1 as vg


def test_v1_has_functions():
    np.testing.assert_array_equal(
        vg.normalize(np.array([5, 0, 0])), np.array([1, 0, 0])
    )


def test_v1_has_constants():
    np.testing.assert_array_equal(vg.basis.x, np.array([1, 0, 0]))


def test_v1_has_shape_check():
    input_value = np.zeros(3)
    vg.shape.check(locals(), "input_value", (3,))


def test_v1_has_matrix_transform():
    scale_factor = np.array([3.0, 0.5, 2.0])
    transform = np.array(
        [
            [scale_factor[0], 0, 0, 0],
            [0, scale_factor[1], 0, 0],
            [0, 0, scale_factor[2], 0],
            [0, 0, 0, 1],
        ]
    )
    point = np.array([5.0, 0.0, 1.0])
    expected_point = np.array([15.0, 0.0, 2.0])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.matrix.transform(point, transform), expected_point
        )


def test_v1_orient_is_alias_for_aligned_with():
    v1 = np.array([1.0, 2.0, 3.0])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.orient(v1, along=vg.basis.z), vg.aligned_with(v1, along=vg.basis.z)
        )
