import math
import numpy as np
import pytest
from . import core as vg


def test_perpendicular():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])

    expected = np.array([-3.0, 6.0, -3.0])
    expected_magnitude = math.sqrt(9.0 + 36.0 + 9.0)

    np.testing.assert_array_almost_equal(
        vg.perpendicular(v1, v2, normalized=False), expected
    )

    np.testing.assert_array_almost_equal(
        vg.perpendicular(v1, v2, normalized=True), expected / expected_magnitude
    )


def test_perpendicular_stacked():
    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([[2.0, 2.0, 2.0], [4.0, 5.0, 6.0]])

    expected = np.array([[2.0, -4.0, 2.0], [-3.0, 6.0, -3.0]])
    expected_magnitude = np.array(
        [math.sqrt(4.0 + 16.0 + 4.0), math.sqrt(9.0 + 36.0 + 9.0)]
    )

    np.testing.assert_array_almost_equal(
        vg.perpendicular(v1, v2, normalized=False), expected
    )

    np.testing.assert_array_almost_equal(
        vg.perpendicular(v1, v2, normalized=True),
        expected / expected_magnitude[:, np.newaxis],
    )


def test_perpendicular_error():
    v1 = np.array([[1.0, 0.0, -1.0], [1.0, 2.0, 3.0]])
    v2 = np.array([4.0, 5.0, 6.0])

    with pytest.raises(
        ValueError, match="Not sure what to do with 2 dimensions and 1 dimension"
    ):
        vg.perpendicular(v1, v2)
