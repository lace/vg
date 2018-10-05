import unittest
import math
import numpy as np
from . import core as vx


class TestAngle(unittest.TestCase):
    def test_ndim_1_basic(self):
        perpendicular = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        self.assertEqual(vx.angle(*perpendicular), 90)
        self.assertEqual(vx.angle(*perpendicular[::-1]), 90)
        self.assertIsInstance(vx.angle(*perpendicular), float)

        acute = np.array([[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
        acute_angle = math.acos(3.0 / 5.0)
        self.assertEqual(vx.angle(*acute, units="rad"), acute_angle)
        self.assertEqual(vx.angle(*acute, units="rad"), acute_angle)

    def test_ndim_1_units(self):
        v1 = np.array([1, 1, 0])
        v2 = np.array([-1, 1, 0])
        self.assertEqual(vx.angle(v1, v2, units="deg"), 90)
        self.assertEqual(vx.angle(v2, v1, units="rad"), math.pi / 2.0)
        with self.assertRaises(ValueError):
            vx.angle(v2, v1, units="cm")

    def test_ndim_1_assume_normalized(self):
        acute = np.array([[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
        acute_angle = math.acos(3.0 / 5.0)
        np.testing.assert_almost_equal(
            vx.angle(*acute, units="rad", assume_normalized=False), acute_angle
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_almost_equal,
            vx.angle(*acute, units="rad", assume_normalized=True),
            acute_angle,
        )

        acute_norm = vx.normalize(acute)
        np.testing.assert_almost_equal(
            vx.angle(*acute_norm, units="rad", assume_normalized=False), acute_angle
        )
        np.testing.assert_almost_equal(
            vx.angle(*acute_norm, units="rad", assume_normalized=True), acute_angle
        )

    def test_ndim_1_look(self):
        v1 = np.array([1, 1, 1])
        v2 = np.array([-1, 1, 0])
        np.testing.assert_almost_equal(vx.angle(v1, v2, look=vx.basis.z), 90.0)
        np.testing.assert_almost_equal(vx.angle(v1, v2, look=vx.basis.x), 45.0)
        np.testing.assert_almost_equal(vx.angle(v1, v2, look=vx.basis.y), 135.0)

    def test_ndim_2_basic(self):
        v1 = np.array([[1, 1, 0], [1, 1, 0]])
        v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
        np.testing.assert_array_almost_equal(vx.angle(v2, v1), np.array([90, 180]))

    def test_ndim_2_units(self):
        v1 = np.array([[1, 1, 0], [1, 1, 0]])
        v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
        np.testing.assert_array_almost_equal(
            vx.angle(v2, v1, units="deg"), np.array([90, 180])
        )
        np.testing.assert_array_almost_equal(
            vx.angle(v2, v1, units="rad"), np.array([math.pi / 2.0, math.pi])
        )

    def test_ndim_2_assume_normalized(self):
        v1 = np.array([[1.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
        v2 = np.array([[-1.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
        expected = np.array([math.acos(3.0 / 5.0), math.pi])
        np.testing.assert_array_almost_equal(
            vx.angle(v2, v1, assume_normalized=False, units="rad"), expected
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            vx.angle(v2, v1, assume_normalized=True, units="rad"),
            expected,
        )

        v1, v2 = [vx.normalize(v) for v in (v1, v2)]
        np.testing.assert_array_almost_equal(
            vx.angle(v2, v1, assume_normalized=False, units="rad"), expected
        )
        np.testing.assert_array_almost_equal(
            vx.angle(v2, v1, assume_normalized=True, units="rad"), expected
        )

    def test_ndim_2_look(self):
        v1 = np.array([[1, 1, 1], vx.basis.y])
        v2 = np.array([[-1, 1, 0], vx.basis.z])
        np.testing.assert_almost_equal(
            vx.angle(v1, v2, look=vx.basis.x), np.array([45.0, 90.0])
        )


class TestSignedAngle(unittest.TestCase):
    def test_signed_angle_ndim_1(self):
        v1 = np.array([1, 1, 0])
        v2 = np.array([-1, 1, 0])
        look = vx.basis.z
        self.assertEqual(vx.signed_angle(v1, v2, look), 90)
        self.assertEqual(vx.signed_angle(v2, v1, look), -90)
        self.assertIsInstance(vx.signed_angle(v1, v2, look), float)

    def test_signed_angle_ndim_2(self):
        v1 = np.array([[1, 1, 0], [1, 1, 0]])
        v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
        look = vx.basis.z
        np.testing.assert_array_almost_equal(
            vx.signed_angle(v2, v1, look), np.array([-90, 180])
        )
