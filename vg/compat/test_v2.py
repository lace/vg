import numpy as np
import pytest
import vg.compat.v2 as vg


def test_v2_has_functions():
    np.testing.assert_array_equal(
        vg.normalize(np.array([5, 0, 0])), np.array([1, 0, 0])
    )


def test_v2_has_constants():
    np.testing.assert_array_equal(vg.basis.x, np.array([1, 0, 0]))


def test_v2_has_shape_check():
    input_value = np.zeros(3)
    vg.shape.check(locals(), "input_value", (3,))
