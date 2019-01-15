import numpy as np
import pytest
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


def test_reject_axis_error():
    with pytest.raises(ValueError, match="axis should be 0, 1, or 2"):
        vg.reject_axis(np.array([2.0, 4.0, 0.0]), axis=5)
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.reject_axis(np.array([[[2.0, 4.0, 0.0]]]), axis=1)
    with pytest.raises(ValueError, match="axis should be 0, 1, or 2"):
        vg.reject_axis(np.array([2.0, 4.0, 0.0]), axis=5, squash=True)
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.reject_axis(np.array([[[2.0, 4.0, 0.0]]]), axis=1, squash=True)
