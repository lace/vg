import numpy as np
import pytest
import vg


def test_cross():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])

    expected = np.array([-3.0, 6.0, -3.0])

    np.testing.assert_array_almost_equal(vg.cross(v1, v2), expected)
    np.testing.assert_array_almost_equal(vg.cross(v2, v1), -expected)


def test_cross_stacked():
    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([[2.0, 2.0, 2.0], [4.0, 5.0, 6.0]])

    expected = np.array([[2.0, -4.0, 2.0], [-3.0, 6.0, -3.0]])

    np.testing.assert_array_almost_equal(vg.cross(v1, v2), expected)
    np.testing.assert_array_almost_equal(vg.cross(v2, v1), -expected)


def test_cross_mixed():
    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([4.0, 5.0, 6.0])

    expected = np.array([[5.0, -10.0, 5.0], [-3.0, 6.0, -3.0]])
    np.testing.assert_array_almost_equal(vg.cross(v1, v2), expected)
    np.testing.assert_array_almost_equal(vg.cross(v2, v1), -expected)


def test_cross_error():
    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([[4.0, 5.0, 6.0]])

    with pytest.raises(
        ValueError, match="v2 must be an array with shape \\(2, 3\\); got \\(1, 3\\)"
    ):
        vg.cross(v1, v2)

    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([[[4.0, 5.0, 6.0]]])

    with pytest.raises(
        ValueError, match="Not sure what to do with 2 dimensions and 3 dimensions"
    ):
        vg.cross(v1, v2)
