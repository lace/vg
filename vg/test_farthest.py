import numpy as np
import pytest
from . import core as vg


def test_farthest():
    to_point = np.array([-1.0, 0.0, 0.0])

    from_points = np.array([[1.0, -2.0, -3.0], [-1.0, -20.0, -30.0]])

    point, index = vg.farthest(from_points, to_point, ret_index=True)

    np.testing.assert_array_equal(point, from_points[1])
    np.testing.assert_array_equal(index, 1)

    with pytest.raises(
        ValueError, match="Invalid shape \\(3,\\): farthest expects nx3"
    ):
        vg.farthest(to_point, to_point)

    with pytest.raises(ValueError, match="to_point should be 3x1"):
        vg.farthest(from_points, from_points)
