import numpy as np
from . import core as vg


def test_scale_factor():
    v = np.array([1.0, 2.0, 3.0])
    neg_v = np.array([-1.0, -2.0, -3.0])
    thrice_v = np.array([3.0, 6.0, 9.0])

    np.testing.assert_almost_equal(vg.scale_factor(v, v), 1.0)
    np.testing.assert_almost_equal(vg.scale_factor(v, neg_v), -1.0)
    np.testing.assert_almost_equal(vg.scale_factor(v, thrice_v), 3.0)
    np.testing.assert_almost_equal(vg.scale_factor(v, np.zeros(3)), 0.0)
    np.testing.assert_almost_equal(vg.scale_factor(np.zeros(3), v), np.nan)


def test_scale_factor_vectorized_both():
    v = np.array([1.0, 2.0, 3.0])
    neg_v = np.array([-1.0, -2.0, -3.0])
    thrice_v = np.array([3.0, 6.0, 9.0])
    zero = np.zeros(3)

    v1s = np.array([neg_v, thrice_v, neg_v, v, zero])
    v2s = np.array([neg_v, v, thrice_v, thrice_v, v])

    np.testing.assert_array_equal(
        vg.scale_factor(v1s, v2s), np.array([1.0, 1.0 / 3.0, -3.0, 3.0, np.nan])
    )


def test_scale_factor_vectorized_first():
    v = np.array([1.0, 2.0, 3.0])
    neg_v = np.array([-1.0, -2.0, -3.0])
    thrice_v = np.array([3.0, 6.0, 9.0])
    zero = np.zeros(3)

    v1s = np.array([neg_v, thrice_v, neg_v, v, zero])

    np.testing.assert_array_equal(
        vg.scale_factor(v1s, v), np.array([-1.0, 1.0 / 3.0, -1.0, 1.0, np.nan])
    )


def test_scale_factor_vectorized_second():
    v = np.array([1.0, 2.0, 3.0])
    neg_v = np.array([-1.0, -2.0, -3.0])
    thrice_v = np.array([3.0, 6.0, 9.0])
    zero = np.zeros(3)

    v2s = np.array([neg_v, v, thrice_v, thrice_v, zero])

    np.testing.assert_array_equal(
        vg.scale_factor(v, v2s), np.array([-1.0, 1.0, 3.0, 3.0, 0.0])
    )
