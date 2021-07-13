import numpy as np
import pytest
from vg.matrix import unpad


def test_unpad():
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            unpad(
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
            unpad(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))

    with pytest.deprecated_call():
        with pytest.raises(
            ValueError, match=r"^Invalid shape \(4L?,\): unpad expects nx4$"
        ):
            unpad(np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="Expected a column of ones"):
            unpad(
                np.array(
                    [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 3.0]]
                )
            )
