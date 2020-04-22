import numpy as np
import pytest
from .shape import check, check_value, check_value_any, columnize


def test_check_value_valid():
    check_value(np.zeros(3), (3,))


def test_check_value_valid_scalar():
    check_value(np.int64(3), ())


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


def test_check_value_any_valid():
    assert check_value_any(np.zeros((3,)), (3,), (-1, 3), name="points") is None
    assert check_value_any(np.zeros((12, 3)), (3,), (-1, 3), name="points") == 12
    assert check_value_any(np.zeros((0, 3)), (3,), (-1, 3), name="points") == 0
    assert check_value_any(
        np.zeros((5, 3, 3)), (-1, 3), (-1, -1, 3), name="points"
    ) == (5, 3)


def test_check_value_any_errors():
    with pytest.raises(ValueError, match="At least one shape is required"):
        check_value_any(np.zeros(9).reshape(-3, 3))
    with pytest.raises(
        ValueError, match=r"Expected an array with shape \(3,\) or \(-1, 3\); got list"
    ):
        check_value_any([1, 2, 3], (3,), (-1, 3))
    with pytest.raises(
        ValueError, match=r"Expected an array with shape \(3,\); got list"
    ):
        check_value_any([1, 2, 3], (3,))


def test_check_value_any_message():
    with pytest.raises(
        ValueError,
        match=r"^Expected an array with shape \(-1, 2\) or \(2,\); got \(3, 3\)$",
    ):
        check_value_any(np.zeros(9).reshape(-3, 3), (-1, 2), (2,))

    with pytest.raises(
        ValueError,
        match=r"^Expected coords to be an array with shape \(-1, 2\) or \(2,\); got \(3, 3\)$",
    ):
        check_value_any(np.zeros(9).reshape(-3, 3), (-1, 2), (2,), name="coords")

    with pytest.raises(
        ValueError,
        match=r"^Expected coords to be an array with shape \(-1, 2\) or \(2,\); got None$",
    ):
        check_value_any(None, (-1, 2), (2,), name="coords")


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


def test_columnize_with_2d_shape():
    shape = (-1, 3)

    columnized, is_columnized, transform_result = columnize(
        np.array([1.0, 0.0, 0.0]), shape
    )
    np.testing.assert_array_equal(columnized, np.array([[1.0, 0.0, 0.0]]))
    assert columnized.shape == (1, 3)
    assert is_columnized is False
    assert transform_result([1.0]) == 1.0

    columnized, is_columnized, transform_result = columnize(
        np.array([[1.0, 0.0, 0.0]]), shape
    )
    np.testing.assert_array_equal(columnized, np.array([[1.0, 0.0, 0.0]]))
    assert columnized.shape == (1, 3)
    assert is_columnized is True
    assert transform_result([1.0]) == [1.0]


def test_columnize_with_3d_shape():
    shape = (-1, 3, 3)

    columnized, is_columnized, transform_result = columnize(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), shape
    )
    np.testing.assert_array_equal(
        columnized, np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    )
    assert columnized.shape == (1, 3, 3)
    assert is_columnized is False
    assert transform_result([1.0]) == 1.0

    columnized, is_columnized, transform_result = columnize(
        np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), shape
    )
    np.testing.assert_array_equal(
        columnized, np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    )
    assert columnized.shape == (1, 3, 3)
    assert is_columnized is True
    assert transform_result([1.0]) == [1.0]


def test_columnize_invalid_shape():
    with pytest.raises(ValueError, match="shape should be a tuple"):
        columnize(np.array([1.0, 0.0, 0.0]), "this is not a shape")
    with pytest.raises(ValueError, match="shape should have at least two dimension"):
        columnize(np.array([1.0, 0.0, 0.0]), (3,))
