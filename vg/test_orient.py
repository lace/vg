import numpy as np
from . import core as vg


def test_orient():
    v1 = np.array([1.0, 2.0, 3.0])
    neg_v1 = np.array([-1.0, -2.0, -3.0])

    np.testing.assert_array_equal(vg.orient(v1, along=vg.basis.z), v1)
    np.testing.assert_array_equal(vg.orient(neg_v1, along=vg.basis.z), v1)

    np.testing.assert_array_equal(vg.orient(v1, along=vg.basis.z, reverse=True), neg_v1)
    np.testing.assert_array_equal(
        vg.orient(neg_v1, along=vg.basis.z, reverse=True), neg_v1
    )
