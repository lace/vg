import numpy as np
from . import core as vx


def test_basic():
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, 1, 0])
    look = vx.basis.z
    assert vx.signed_angle(v1, v2, look) == 90
    assert vx.signed_angle(v2, v1, look) == -90
    assert isinstance(vx.signed_angle(v1, v2, look), float)


def test_stacked_basic():
    v1 = np.array([[1, 1, 0], [1, 1, 0]])
    v2 = np.array([[-1, 1, 0], [-1, -1, 0]])
    look = vx.basis.z
    np.testing.assert_array_almost_equal(
        vx.signed_angle(v2, v1, look), np.array([-90, 180])
    )
