import pytest
import numpy as np
from .shape import check_value, check


def test_check_value_valid():
    check_value(np.zeros(3), (3,))


def test_check_value_valid_wildcard():
    assert check_value(np.zeros((5, 3)), (-1, 3)) == 5
    assert check_value(np.zeros((5, 3)), (5, -1)) == 3
    assert check_value(np.zeros((5, 3, 2)), (-1, 3, -1)) == (5, 2)


def test_check_value_wrong_shape():
    with pytest.raises(ValueError) as e:
        check_value(np.zeros(4), (3,))
    assert "Expected an array with shape (3,); got (4,)" in str(e.value)


def test_check_value_wrong_shape_wildcard():
    with pytest.raises(ValueError) as e:
        check_value(np.zeros((5, 4)), (-1, 3))
    assert "Expected an array with shape (-1, 3); got (5, 4)" in str(e.value)


def test_check_value_none():
    with pytest.raises(ValueError) as e:
        check_value(None, (3,))
    assert "Expected an array with shape (3,); got None" in str(e.value)


def test_check_value_wrong_type():
    with pytest.raises(ValueError) as e:
        check_value({}, (3,))
    assert "Expected an array with shape (3,); got dict" in str(e.value)

    class Value:
        def __init__(self):
            self.shape = None

    with pytest.raises(ValueError) as e:
        check_value(Value(), (3,))
    assert "Expected an array with shape (3,); got Value" in str(e.value)


def test_check_value_valid_named():
    check_value(np.zeros(3), (3,), name="input_value")


def test_check_value_valid_wildcard_named():
    assert check_value(np.zeros((5, 3)), (-1, 3), name="input_value") == 5
    assert check_value(np.zeros((5, 3)), (5, -1), name="input_value") == 3


def test_check_value_wrong_shape_named():
    with pytest.raises(ValueError) as e:
        check_value(np.zeros(4), (3,), name="input_value")
    assert "input_value must be an array with shape (3,); got (4,)" in str(e.value)


def test_check_value_wrong_shape_wildcard_named():
    with pytest.raises(ValueError) as e:
        check_value(np.zeros((5, 4)), (-1, 3), name="input_value")
    assert "input_value must be an array with shape (-1, 3); got (5, 4)" in str(e.value)


def test_check_value_none_named():
    with pytest.raises(ValueError) as e:
        check_value(None, (3,), name="input_value")
    assert "input_value must be an array with shape (3,); got None" in str(e.value)


def test_check_value_with_invalid_shape_raises_expected_error():
    with pytest.raises(ValueError) as e:
        check_value(np.zeros(3), (3.0,))
    assert "Expected shape dimensions to be int" in str(e.value)


def test_check():
    input_value = np.zeros(3)
    check(locals(), "input_value", (3,))


def test_check_valid_wildcard():
    input_value = np.zeros((5, 3))
    assert check(locals(), "input_value", (-1, 3)) == 5
    assert check(locals(), "input_value", (5, -1)) == 3
    input_value = np.zeros((5, 3, 2))
    assert check(locals(), "input_value", (-1, 3, -1)) == (5, 2)


def test_check_wrong_shape_named():
    input_value = np.zeros(4)
    with pytest.raises(ValueError) as e:
        check(locals(), "input_value", (3,))
    assert "input_value must be an array with shape (3,); got (4,)" in str(e.value)


def test_check_wrong_shape_wildcard_named():
    input_value = np.zeros((5, 4))
    with pytest.raises(ValueError) as e:
        check(locals(), "input_value", (-1, 3))
    assert "input_value must be an array with shape (-1, 3); got (5, 4)" in str(e.value)


def test_check_none_named():
    input_value = None
    with pytest.raises(ValueError) as e:
        check(locals(), "input_value", (3,))
    assert "input_value must be an array with shape (3,); got None" in str(e.value)
