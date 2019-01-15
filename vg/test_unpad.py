import numpy as np
import pytest
from . import core as vg


def test_unpad():
    np.testing.assert_array_equal(
        vg.unpad(
            np.array([[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]])
        ),
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
    )

    # NB: on windows, the sizes here will render as 3L, not 3:
    with pytest.raises(
        ValueError, match=r"^Invalid shape \(3L?, 3L?\): unpad expects nx4$"
    ):
        vg.unpad(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))

    with pytest.raises(
        ValueError, match=r"^Invalid shape \(4L?,\): unpad expects nx4$"
    ):
        vg.unpad(np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(ValueError, match="Expected a column of ones"):
        vg.unpad(
            np.array([[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 3.0]])
        )
