import numpy as np
import math
import pytest
from . import core as vg


around_axis = np.array([0.0, 2.0, -1.0])


def test_rotate():
    # Example verified using
    # http://www.nh.cas.cz/people/lazar/celler/online_tools.php?start_vec=1,1,1&rot_ax=0,2,-1&rot_ang=180
    v = np.array([1.0, 1.0, 1.0])

    np.testing.assert_array_almost_equal(
        vg.rotate(v, around_axis=around_axis, angle=90),
        np.array([1.341641, -0.047214, -1.094427]),
    )

    np.testing.assert_array_almost_equal(
        vg.rotate(v, around_axis=around_axis, angle=180), np.array([-1, -0.2, -1.4])
    )

    np.testing.assert_array_almost_equal(
        vg.rotate(v, around_axis=around_axis, angle=math.pi, units="rad"),
        np.array([-1, -0.2, -1.4]),
    )


def test_rotate_stacked():
    vs = np.array([[3.0, -1.0, 5.0], [1.0, 1.0, 1.0]])
    np.testing.assert_array_almost_equal(
        vg.rotate(vs, around_axis=around_axis, angle=90),
        np.array([[4.024922, -4.141641, -1.283282], [1.341641, -0.047214, -1.094427]]),
    )


def test_rotate_error():
    with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
        vg.rotate(np.array([[[5.0, -3.0, 1.0]]]), around_axis=around_axis, angle=90),
    with pytest.raises(ValueError, match='Unknown units "in"; expected "deg" or "rad"'):
        vg.rotate(
            np.array([1.0, 1.0, 1.0]), around_axis=around_axis, angle=90, units="in"
        )
