import math
import numpy as np
from . import core as vg


def test_euclidean_distance():
    v1 = np.array([1, 1, 0])
    v2 = np.array([2, -1, 5])
    expected = math.sqrt(1 ** 2 + 2 ** 2 + 5 ** 2)
    result = vg.euclidean_distance(v1, v2)
    np.testing.assert_almost_equal(result, expected)
    assert isinstance(result, float)


def test_euclidean_distance_stacked():
    v1s = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
    v2s = np.array([[2, -1, 5], [3, 4, 0], [-1, 0, 6]])
    expected = np.array(
        [
            math.sqrt(1 ** 2 + 2 ** 2 + 5 ** 2),
            math.sqrt(4 ** 2 + 4 ** 2),
            math.sqrt(1 ** 2 + 1 ** 2),
        ]
    )
    np.testing.assert_array_almost_equal(vg.euclidean_distance(v1s, v2s), expected)


def test_euclidean_distance_mixed():
    v1s = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
    v2 = np.array([2, -1, 5])
    expected = np.array(
        [
            math.sqrt(1 ** 2 + 2 ** 2 + 5 ** 2),
            math.sqrt(3 ** 2 + 1 ** 2 + 5 ** 2),
            math.sqrt(2 ** 2 + 1 ** 2),
        ]
    )
    np.testing.assert_array_almost_equal(vg.euclidean_distance(v1s, v2), expected)
