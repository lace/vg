import numpy as np
from . import core as vg


def test_reject():
    v = np.array([2.0, 4.0, 0.0])
    from_v = np.array([0.0, 5.0, 0.0])
    expected = np.array([2.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(vg.reject(v, from_v=from_v), expected)


def test_reject_stacked():
    vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
    from_v = np.array([0.0, 5.0, 0.0])
    expected = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(vg.reject(vs, from_v=from_v), expected)
