import numpy as np
import pytest
from ._helpers import _check_value_any
from .core import raise_dimension_error


def test_raise_dimension_error():
    with pytest.raises(ValueError, match="Not sure what to do with those inputs"):
        raise_dimension_error(np.array([]), np.array([]), np.array([]))


def test_check_value_any_valid():
    assert _check_value_any(np.zeros((3,)), (3,), (-1, 3), name="points") is None
    assert _check_value_any(np.zeros((12, 3)), (3,), (-1, 3), name="points") == 12
    assert _check_value_any(np.zeros((0, 3)), (3,), (-1, 3), name="points") == 0
    assert _check_value_any(
        np.zeros((5, 3, 3)), (-1, 3), (-1, -1, 3), name="points"
    ) == (5, 3)


def test_check_value_any_errors():
    with pytest.raises(ValueError, match="At least one shape is required"):
        _check_value_any(np.zeros(9).reshape(-3, 3))


def test_check_value_any_message():
    with pytest.raises(
        ValueError,
        match=r"^Expected an array with shape \(-1, 2\) or \(2,\); got \(3, 3\)$",
    ):
        _check_value_any(np.zeros(9).reshape(-3, 3), (-1, 2), (2,))

    with pytest.raises(
        ValueError,
        match=r"^Expected coords to be an array with shape \(-1, 2\) or \(2,\); got \(3, 3\)$",
    ):
        _check_value_any(np.zeros(9).reshape(-3, 3), (-1, 2), (2,), name="coords")

    with pytest.raises(
        ValueError,
        match=r"^Expected coords to be an array with shape \(-1, 2\) or \(2,\); got None$",
    ):
        _check_value_any(None, (-1, 2), (2,), name="coords")
