import numpy as np
import pytest
from .matrix import pad_with_ones


def test_pad_with_ones():
    np.testing.assert_array_equal(
        pad_with_ones(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])),
        np.array([[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]]),
    )


def test_pad_with_wrong_dimensions():
    # NB: on windows, the sizes here will render as 3L, not 3:
    with pytest.raises(
        ValueError, match=r"^Invalid shape \(3L?, 4L?\): pad expects nx3$"
    ):
        pad_with_ones(
            np.array(
                [[1.0, 2.0, 3.0, 42.0], [2.0, 3.0, 4.0, 42.0], [5.0, 6.0, 7.0, 42.0]]
            )
        )

    with pytest.raises(ValueError, match=r"^Invalid shape \(3L?,\): pad expects nx3$"):
        pad_with_ones(np.array([1.0, 2.0, 3.0]))
