import numpy as np
from . import core as vg


def test_average():
    np.testing.assert_array_equal(
        vg.average(np.array([[1.0, 2.0, 3.0], [-6.0, -9.0, -15.0]])),
        np.array([-2.5, -3.5, -6.0]),
    )
    np.testing.assert_array_equal(
        vg.average(np.array([[1.0, 2.0, 3.0], [-6.0, -9.0, -15.0]]), weights=(3, 5)),
        np.array([-3.375, -4.875, -8.25]),
    )
    result, sum_of_weights = vg.average(
        np.array([[1.0, 2.0, 3.0], [-6.0, -9.0, -15.0]]),
        weights=(3, 5),
        ret_sum_of_weights=True,
    )
    np.testing.assert_array_equal(
        result, np.array([-3.375, -4.875, -8.25]),
    )
    assert sum_of_weights == 8.0
