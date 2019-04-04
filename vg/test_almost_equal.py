import numpy as np
from . import core as vg


def test_almost_equal():
    assert (
        vg.almost_equal(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.000000000001, 2.000000000001, 3.000000000001]),
        )
        == True
    )
    assert (
        vg.almost_equal(np.array([1.0, 2.0, 3.0]), np.array([1.01, 2.0, 3.0])) == False
    )
    assert vg.almost_equal(np.array([1.0, 2.0, 3.0]), np.array([1.0])) == False
