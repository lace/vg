import numpy as np
import vg.compat.v1 as vg


def test_v1_has_functions():
    np.testing.assert_array_equal(
        vg.normalize(np.array([5, 0, 0])), np.array([1, 0, 0])
    )


def test_v1_has_constants():
    np.testing.assert_array_equal(vg.basis.x, np.array([1, 0, 0]))
