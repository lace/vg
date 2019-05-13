import numpy as np
import pytest
from . import core as vg


def test_within():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 1.0, 0.0],
            [4.0, 5.0, 4.0],
            [0.0, 1.0, 3.9],
            [0.0, 1.0, 4.05],
            [0.0, 1.0, 4.099],
            [0.0, 1.0, 4.1],
        ]
    )

    is_within_4 = np.array([True, True, True, True, False, True, True, True, False])
    points_within_4 = points[is_within_4]
    indices_within_4, = is_within_4.nonzero()

    np.testing.assert_array_almost_equal(
        vg.within(points, radius=4.0, of_point=np.array([0.0, 1.0, 0.0]), atol=0.1),
        points_within_4,
    )
    actual_points, actual_indices = vg.within(
        points,
        radius=4.0,
        of_point=np.array([0.0, 1.0, 0.0]),
        atol=0.1,
        ret_indices=True,
    )
    assert isinstance(actual_indices, np.ndarray)
    np.testing.assert_array_equal(actual_indices, indices_within_4)
    np.testing.assert_array_almost_equal(actual_points, points_within_4)

    np.testing.assert_array_almost_equal(
        vg.within(points, radius=4.0, of_point=np.array([0.0, 1.0, 0.0]), atol=1e-4),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [4.0, 1.0, 0.0],
                [0.0, 1.0, 3.9],
            ]
        ),
    )


def test_within_error():
    with pytest.raises(ValueError, match=r"Invalid shape \(3,\): within expects nx3"):
        vg.within(
            np.array([2.0, 4.0, 0.0]), radius=4.0, of_point=np.array([0.0, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match=r"Invalid shape \(1, 2\): within expects nx3"):
        vg.within(
            np.array([[2.0, 4.0]]), radius=4.0, of_point=np.array([0.0, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match="radius should be a float"):
        vg.within(
            np.array([[2.0, 4.0, 0.0]]),
            radius=False,
            of_point=np.array([0.0, 1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="to_point should be 3x1"):
        vg.within(
            np.array([[2.0, 4.0, 0.0]]), radius=4.0, of_point=np.array([0.0, 1.0])
        )
