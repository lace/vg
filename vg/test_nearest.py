import numpy as np
from . import core as vg


def test_nearest():
    to_point = np.array([-1.0, 0.0, 0.0])

    from_points = np.array([[1.0, -2.0, -3.0], [-1.0, -20.0, -30.0]])

    point, index = vg.nearest(from_points, to_point, ret_index=True)

    np.testing.assert_array_equal(point, from_points[0])
    np.testing.assert_array_equal(index, 0)

    np.testing.assert_array_equal(
        vg.nearest(from_points, to_point, ret_index=False), from_points[0]
    )
