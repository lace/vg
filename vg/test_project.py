import numpy as np
import pytest
from . import core as vg


def test_project():
    v = np.array([5.0, -3.0, 1.0])
    onto = np.array([0, -1.0, 0])
    expected_s = 3.0
    expected_v = np.array([0, -3.0, 0])
    np.testing.assert_array_almost_equal(vg.scalar_projection(v, onto=onto), expected_s)
    np.testing.assert_array_almost_equal(vg.project(v, onto=onto), expected_v)

    with pytest.raises(ValueError, match="onto should be a vector"):
        vg.project(v, onto=np.array([vg.basis.x, vg.basis.x]))


def test_project_stacked():
    vs = np.array([[5.0, -3.0, 1.0], [1.0, 0, 1.0], [0.0, 1, 0.0], [0.0, 0, 0.0]])
    onto = np.array([0, -1.0, 0])
    expected_s = np.array([3.0, 0.0, -1.0, 0])
    expected_v = np.array(
        [[0.0, -3.0, 0.0], [0.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0, 0.0]]
    )
    np.testing.assert_array_almost_equal(
        vg.scalar_projection(vs, onto=onto), expected_s
    )
    np.testing.assert_array_almost_equal(vg.project(vs, onto=onto), expected_v)


def test_project_error():
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.project(np.array([[[5.0, -3.0, 1.0]]]), onto=np.array([0, -1.0, 0]))
