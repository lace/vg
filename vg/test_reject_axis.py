import numpy as np
from . import core as vg


def test_reject_axis():
    v = np.array([2.0, 4.0, 0.0])
    expected = np.array([2.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(vg.reject_axis(v, axis=1), expected)


def test_reject_axis_stacked():
    vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
    expected = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(vg.reject_axis(vs, axis=1), expected)


def test_reject_axis_with_squash():
    v = np.array([2.0, 4.0, 0.0])
    expected = np.array([2.0, 0.0])
    np.testing.assert_array_almost_equal(
        vg.reject_axis(v, axis=1, squash=True), expected
    )


def test_reject_axis_stacked_with_squash():
    vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
    expected = np.array([[2.0, 0.0], [-2.0, 0.0]])
    np.testing.assert_array_almost_equal(
        vg.reject_axis(vs, axis=1, squash=True), expected
    )
