import numpy as np
from . import core as vg


def test_constants():
    np.testing.assert_array_equal(vg.basis.x, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(vg.basis.y, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(vg.basis.z, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(vg.basis.neg_x, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(vg.basis.neg_y, np.array([0.0, -1.0, 0.0]))
    np.testing.assert_array_equal(vg.basis.neg_z, np.array([0.0, 0.0, -1.0]))


def test_that_constants_are_copies():
    x1 = vg.basis.x
    x1[1] = 5.0

    x2 = vg.basis.y
    x2[0] = 3.0

    np.testing.assert_array_equal(x1, [1.0, 5.0, 0.0])
    np.testing.assert_array_equal(x2, [3.0, 1.0, 0.0])
