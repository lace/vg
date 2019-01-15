import numpy as np
from . import core as vg


def test_almost_zero():
    assert vg.almost_zero(np.array([0.0, 0.0, 0.0])) is True
    assert vg.almost_zero(np.array([0.000000000000000001, 0.0, 0.0])) is True
    assert vg.almost_zero(np.array([0.0000001, 0.0, 0.0])) is False
