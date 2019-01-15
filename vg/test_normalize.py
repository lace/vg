import math
import pytest
import numpy as np
from . import core as vg


def test_normalize():
    v = np.array([1, 1, 0])
    expected = np.array([math.sqrt(2) / 2.0, math.sqrt(2) / 2.0, 0])
    np.testing.assert_array_almost_equal(vg.normalize(v), expected)


def test_normalize_stacked():
    vs = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
    expected = np.array(
        [[math.sqrt(2) / 2.0, math.sqrt(2) / 2.0, 0], [-1, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_array_almost_equal(vg.normalize(vs), expected)


def test_normalized_wrong_dim():
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.normalize(np.array([[[1, 1, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0]]]))
