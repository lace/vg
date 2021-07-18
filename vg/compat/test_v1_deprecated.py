import numpy as np
import pytest
import vg.compat.v1 as vg


def test_v1_unpad():
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.matrix.unpad(
                np.array(
                    [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]]
                )
            ),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        )

    # NB: on windows, the sizes here will render as 3L, not 3:
    with pytest.deprecated_call():
        with pytest.raises(
            ValueError, match=r"^Invalid shape \(3L?, 3L?\): unpad expects nx4$"
        ):
            vg.matrix.unpad(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))

    with pytest.deprecated_call():
        with pytest.raises(
            ValueError, match=r"^Invalid shape \(4L?,\): unpad expects nx4$"
        ):
            vg.matrix.unpad(np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="Expected a column of ones"):
            vg.matrix.unpad(
                np.array(
                    [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 3.0]]
                )
            )

def test_v1_pad_with_ones():
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.matrix.pad_with_ones(
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
            ),
            np.array(
                [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]]
            ),
        )


def test_v1_pad_with_wrong_dimensions():
    # NB: on windows, the sizes here will render as 3L, not 3:
    with pytest.deprecated_call():
        with pytest.raises(
            ValueError, match=r"^Invalid shape \(3L?, 4L?\): pad expects nx3$"
        ):
            vg.matrix.pad_with_ones(
                np.array(
                    [
                        [1.0, 2.0, 3.0, 42.0],
                        [2.0, 3.0, 4.0, 42.0],
                        [5.0, 6.0, 7.0, 42.0],
                    ]
                )
            )

    with pytest.deprecated_call():
        with pytest.raises(
            ValueError, match=r"^Invalid shape \(3L?,\): pad expects nx3$"
        ):
            vg.matrix.pad_with_ones(np.array([1.0, 2.0, 3.0]))


scale_factor = np.array([3.0, 0.5, 2.0])
transform = np.array(
    [
        [scale_factor[0], 0, 0, 0],
        [0, scale_factor[1], 0, 0],
        [0, 0, scale_factor[2], 0],
        [0, 0, 0, 1],
    ]
)


def test_apply_homogeneous():
    point = np.array([5.0, 0.0, 1.0])
    expected_point = np.array([15.0, 0.0, 2.0])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(vg.matrix.transform(point, transform), expected_point)


def test_apply_homogeneous_stacked():
    points = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, 1.0]])
    expected_points = np.array([[3.0, 1.0, 6.0], [15.0, 0.0, 2.0]])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.matrix.transform(points, transform), expected_points
        )


def test_apply_homogeneous_error():
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="Transformation matrix should be 4x4"):
            vg.matrix.transform(np.array([1.0, 2.0, 3.0]), np.array([1.0]))
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match=r"Vertices should be \(3,\) or Nx3"):
            vg.matrix.transform(np.array([1.0, 2.0]), transform)
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="Not sure what to do with 3 dimensions"):
            vg.matrix.transform(np.array([[[1.0, 2.0, 3.0]]]), transform)
