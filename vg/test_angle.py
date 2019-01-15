import math
import numpy as np
import pytest
from . import core as vg


def test_basic():
    perpendicular = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    assert vg.angle(*perpendicular) == 90
    assert vg.angle(*perpendicular[::-1]) == 90
    assert isinstance(vg.angle(*perpendicular), float)

    acute = np.array([[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
    acute_angle = math.acos(3.0 / 5.0)
    assert vg.angle(*acute, units="rad") == acute_angle
    assert vg.angle(*acute, units="rad") == acute_angle


def test_units():
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 0])
    assert vg.angle(v1, v2, units="deg") == 90
    assert vg.angle(v2, v1, units="rad") == math.pi / 2.0
    with pytest.raises(ValueError):
        vg.angle(v2, v1, units="cm")


def test_assume_normalized():
    acute = np.array([[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
    acute_angle = math.acos(3.0 / 5.0)
    np.testing.assert_almost_equal(
        vg.angle(*acute, units="rad", assume_normalized=False), acute_angle
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_almost_equal,
        vg.angle(*acute, units="rad", assume_normalized=True),
        acute_angle,
    )

    acute_norm = vg.normalize(acute)
    np.testing.assert_almost_equal(
        vg.angle(*acute_norm, units="rad", assume_normalized=False), acute_angle
    )
    np.testing.assert_almost_equal(
        vg.angle(*acute_norm, units="rad", assume_normalized=True), acute_angle
    )


def test_look():
    v1 = np.array([1, 1, 1])
    v2 = np.array([-1, 1, 0])
    np.testing.assert_almost_equal(vg.angle(v1, v2, look=vg.basis.z), 90.0)
    np.testing.assert_almost_equal(vg.angle(v1, v2, look=vg.basis.x), 45.0)
    np.testing.assert_almost_equal(vg.angle(v1, v2, look=vg.basis.y), 135.0)


def test_stacked_basic():
    v1 = np.array([[1, 1, 0], [1, 1, 0]])
    v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
    np.testing.assert_array_almost_equal(vg.angle(v2, v1), np.array([90, 180]))


def test_stacked_units():
    v1 = np.array([[1, 1, 0], [1, 1, 0]])
    v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
    np.testing.assert_array_almost_equal(
        vg.angle(v2, v1, units="deg"), np.array([90, 180])
    )
    np.testing.assert_array_almost_equal(
        vg.angle(v2, v1, units="rad"), np.array([math.pi / 2.0, math.pi])
    )


def test_stacked_assume_normalized():
    v1 = np.array([[1.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
    v2 = np.array([[-1.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
    expected = np.array([math.acos(3.0 / 5.0), math.pi])
    np.testing.assert_array_almost_equal(
        vg.angle(v2, v1, assume_normalized=False, units="rad"), expected
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        vg.angle(v2, v1, assume_normalized=True, units="rad"),
        expected,
    )

    v1, v2 = [vg.normalize(v) for v in (v1, v2)]
    np.testing.assert_array_almost_equal(
        vg.angle(v2, v1, assume_normalized=False, units="rad"), expected
    )
    np.testing.assert_array_almost_equal(
        vg.angle(v2, v1, assume_normalized=True, units="rad"), expected
    )


def test_stacked_look():
    v1 = np.array([[1, 1, 1], vg.basis.y])
    v2 = np.array([[-1, 1, 0], vg.basis.z])
    np.testing.assert_almost_equal(
        vg.angle(v1, v2, look=vg.basis.x), np.array([45.0, 90.0])
    )
