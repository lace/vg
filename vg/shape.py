def check_value(arr, shape, **kwargs):
    """
    Check that the given argument has the expected shape. Shape dimensions can
    be ints or -1 for a wildcard. The wildcard dimensions are returned, which
    allows them to be used for subsequent validation or elsewhere in the
    function.

    Args:
        arr (np.arraylike): An array-like input.
        shape (list): Shape to validate. To require an array with 3 elements,
            pass `(3,)`. To require n by 3, pass `(-1, 3)`.
        name (str): Variable name to embed in the error message.

    Returns:
        object: The wildcard dimension (if one) or a tuple of wildcard
        dimensions (if more than one).

    Example:
        >>> vg.shape.check_value(np.zeros((4, 3)), (-1, 3))
        >>> # Proceed with confidence that `points` is a k x 3 array.

    Example:
        >>> k = vg.shape.check_value(np.zeros((4, 3)), (-1, 3))
        >>> k
        4
    """

    def is_wildcard(dim):
        return dim == -1

    if any(not isinstance(dim, int) and not is_wildcard(dim) for dim in shape):
        raise ValueError("Expected shape dimensions to be int")

    if "name" in kwargs:
        preamble = "{} must be an array".format(kwargs["name"])
    else:
        preamble = "Expected an array"

    if arr is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape))
    try:
        len(arr.shape)
    except (AttributeError, TypeError):
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape, arr.__class__.__name__)
        )

    # Check non-wildcard dimensions.
    if len(arr.shape) != len(shape) or any(
        actual != expected
        for actual, expected in zip(arr.shape, shape)
        if not is_wildcard(expected)
    ):
        raise ValueError("{} with shape {}; got {}".format(preamble, shape, arr.shape))

    wildcard_dims = [
        actual for actual, expected in zip(arr.shape, shape) if is_wildcard(expected)
    ]
    if len(wildcard_dims) == 0:
        return None
    elif len(wildcard_dims) == 1:
        return wildcard_dims[0]
    else:
        return tuple(wildcard_dims)


# TODO-2.x: Remove kwargs hack when upgrading to Python 3.
def check_value_any(arr, *shapes, **kwargs):
    """
    Check that the given argument has any of the expected shapes. Shape dimensons
    can be ints or -1 for a wildcard.

    Args:
        arr (np.arraylike): An array-like input.
        shape (list): Shape candidates to validate. To require an array with 3
            elements, pass `(3,)`. To require n by 3, pass `(-1, 3)`.
        name (str): Variable name to embed in the error message.

    Returns:
        object: The wildcard dimension of the matched shape (if one) or a tuple
        of wildcard dimensions (if more than one). If the matched shape has no
        wildcard dimensions, returns `None`.

    Example:
        >>> k = check_shape_any(points, (3,), (-1, 3), name="points")
        >>> check_shape_any(
                reference_points_of_lines,
                (3,),
                (-1 if k is None else k, 3),
                name="reference_points_of_lines",
            )
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required")
    for shape in shapes:
        try:
            return check_value(arr, shape, name=kwargs.get("name", "arr"))
        except ValueError:
            pass

    if "name" in kwargs:
        preamble = "Expected {} to be an array".format(kwargs["name"])
    else:
        preamble = "Expected an array"

    if len(shapes) == 1:
        (shape_choices,) = shapes
    else:
        shape_choices = ", ".join(
            shapes[:-2] + (" or ".join([str(shapes[-2]), str(shapes[-1])]),)
        )

    if arr is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape_choices))
    else:
        try:
            len(arr.shape)
        except (AttributeError, TypeError):
            raise ValueError(
                "{} with shape {}; got {}".format(
                    preamble, shape_choices, arr.__class__.__name__
                )
            )
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape_choices, arr.shape)
        )


def check(locals_namespace, name, shape):
    """
    Convenience function for invoking `vg.shape.check_value()` with a
    `locals()` dict.

    Args:
        namespace (dict): A subscriptable object, typically `locals()`.
        name (str): Key to pull from `namespace`.
        shape (list): Shape to validate. To require 3 by 1, pass `(3,)`. To
            require n by 3, pass `(-1, 3)`.

    Returns:
        object: The wildcard dimension (if one) or a tuple of wildcard
        dimensions (if more than one).

    Example:
        >>> def my_fun_function(points):
        ...     vg.shape.check(locals(), 'points', (-1, 3))
        ...     # Proceed with confidence that `points` is a k x 3 array.

    Example:
        >>> def my_fun_function(points):
        ...     k = vg.shape.check(locals(), 'points', (-1, 3))
        ...     print("my_fun_function invoked with {} points".format(k))

    """
    return check_value(locals_namespace[name], shape, name=name)


def columnize(arr, shape=(-1, 3), name=None):
    """
    Helper for functions which may accept a stack of points (`kx3`) returning
    a stack of results, or a single set of three points `(3,)` returning a
    single result.

    For either kind of input, it returns the points as `kx3`, a boolean
    `is_columnized`, and a `maybe_decolumnized` function which can be applied
    to the result before returning it. For a columnized input this function
    does nothing, and for a non-columnized input, it decolumnizes it,
    producing the desired return value.

    This is not limited to `kx3`. It can be used for different dimensional
    shapes like `kx4`, and even higher dimensional shapes like `kx3x3`.
    """
    if not isinstance(shape, tuple):
        raise ValueError("shape should be a tuple")
    if len(shape) < 2:
        raise ValueError("shape should have at least two dimensions")

    check_value_any(arr, shape, shape[1:], name=name or "arr")

    if arr.ndim == len(shape):
        return arr, True, lambda x: x
    else:
        return arr.reshape(*shape), False, lambda x: x[0]
