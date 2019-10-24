import numpy as np
from . import core as vg


def test_scale_factor():
    v1 = np.array([1.0, 2.0, 3.0])
    neg_v1 = np.array([-1.0, -2.0, -3.0])
    thrice_v1 = np.array([3.0, 6.0, 9.0])

    np.testing.assert_almost_equal(vg.scale_factor(v1, v1), 1.0)
    np.testing.assert_almost_equal(vg.scale_factor(v1, neg_v1), -1.0)
    np.testing.assert_almost_equal(vg.scale_factor(v1, thrice_v1), 3.0)
    np.testing.assert_almost_equal(vg.scale_factor(v1, np.zeros(3)), 0.0)
    np.testing.assert_almost_equal(vg.scale_factor(np.zeros(3), v1), np.nan)
