import numpy as np
from . import core as vg


def test_basic():
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 0])
    look = vg.basis.z
    assert vg.signed_angle(v1, v2, look) == 90
    assert vg.signed_angle(v2, v1, look) == -90
    assert isinstance(vg.signed_angle(v1, v2, look), float)


def test_stacked_basic():
    v1 = np.array([[1, 1, 0], [1, 1, 0]])
    v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
    look = vg.basis.z
    np.testing.assert_array_almost_equal(
        vg.signed_angle(v2, v1, look), np.array([-90, 180])
    )
