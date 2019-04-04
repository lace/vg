import numpy as np
import pytest
from . import core as vg


def test_apex():
    points = np.array(
        [
            [-0.97418884, -0.79808404, -0.18545491],
            [0.60675227, 0.32673201, -0.20369793],
            [0.67040405, 0.19267665, -0.56983579],
            [-0.68038753, -0.90011588, 0.4649872],
            [-0.62813991, -0.23947753, 0.07933854],
            [0.26348356, 0.23701114, -0.38230596],
            [0.08302473, 0.2784907, 0.09308946],
            [0.58695587, -0.33253376, -0.33493078],
            [-0.39221704, -0.45240036, 0.25284163],
            [0.46270635, -0.3865265, -0.98106526],
        ]
    )

    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.x), [0.67040405, 0.19267665, -0.56983579]
    )
    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.neg_x), [-0.97418884, -0.79808404, -0.18545491]
    )
    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.y), [0.60675227, 0.32673201, -0.20369793]
    )
    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.neg_y), [-0.68038753, -0.90011588, 0.4649872]
    )
    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.z), [-0.68038753, -0.90011588, 0.4649872]
    )
    np.testing.assert_array_equal(
        vg.apex(points, along=vg.basis.neg_z), [0.46270635, -0.3865265, -0.98106526]
    )

    v = np.full(3, 1 / 3 ** 0.5)
    expected = points[np.argmax(points.sum(axis=1))]
    np.testing.assert_array_equal(vg.apex(points, along=v), expected)

    # Test non-normalized too.
    np.testing.assert_array_equal(vg.apex(points, along=np.array([1, 1, 1])), expected)

    with pytest.raises(ValueError, match="Invalid shape \\(3,\\): apex expects nx3"):
        vg.apex(vg.basis.x, along=vg.basis.x)

    with pytest.raises(ValueError, match="along should be a 3x1 vector"):
        vg.apex(points, along=points)
