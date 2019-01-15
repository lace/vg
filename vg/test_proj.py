import numpy as np
import pytest
from . import core as vg


def test_proj():
    v = np.array([5.0, -3.0, 1.0])
    onto = np.array([0, -1.0, 0])
    expected_s = 3.0
    expected_v = np.array([0, -3.0, 0])
    np.testing.assert_array_almost_equal(vg.sproj(v, onto=onto), expected_s)
    np.testing.assert_array_almost_equal(vg.proj(v, onto=onto), expected_v)

    with pytest.raises(ValueError, match="onto should be a vector"):
        vg.proj(v, onto=np.array([vg.basis.x, vg.basis.x]))


def test_proj_stacked():
    vs = np.array([[5.0, -3.0, 1.0], [1.0, 0, 1.0], [0.0, 1, 0.0], [0.0, 0, 0.0]])
    onto = np.array([0, -1.0, 0])
    expected_s = np.array([3.0, 0.0, -1.0, 0])
    expected_v = np.array(
        [[0.0, -3.0, 0.0], [0.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0, 0.0]]
    )
    np.testing.assert_array_almost_equal(vg.sproj(vs, onto=onto), expected_s)
    np.testing.assert_array_almost_equal(vg.proj(vs, onto=onto), expected_v)


def test_proj_error():
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.proj(np.array([[[5.0, -3.0, 1.0]]]), onto=np.array([0, -1.0, 0]))
