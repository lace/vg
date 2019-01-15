import numpy as np
import math
from . import core as vg


def test_magnitude():
    v = np.array([1, 1, 0])
    expected = math.sqrt(2)
    np.testing.assert_almost_equal(vg.magnitude(v), expected)
    assert isinstance(vg.magnitude(v), float)


def test_magnitude_stacked():
    vs = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
    expected = np.array([math.sqrt(2), 1, 5])
    np.testing.assert_array_almost_equal(vg.magnitude(vs), expected)
