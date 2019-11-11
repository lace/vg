import numpy as np
import pytest
from .matrix import convert_33_to_44


def test_convert_33_to_44():
    np.testing.assert_array_equal(
        convert_33_to_44(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])),
        np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [2.0, 3.0, 4.0, 0.0],
                [5.0, 6.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(
        ValueError, match=r"^matrix must be an array with shape \(3, 3\); got \(4,\)$"
    ):
        convert_33_to_44(np.array([1.0, 2.0, 3.0, 4.0]))
