def check_value(a, shape, **kwargs):
    """
    Check that the given argument has the expected shape. Shape dimensions can
    be ints or -1 for a wildcard. The wildcard dimensions are returned, which
    allows them to be used for subsequent validation or elsewhere in the
    function.

    Args:
        a (np.arraylike): An array-like input.
        shape (list): Shape to validate. To require 3 by 1, pass `(3,)`. To
            require n by 3, pass `(-1, 3)`.
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
    is_wildcard = lambda dim: dim == -1
    if all(not isinstance(dim, int) and not is_wildcard(dim) for dim in shape):
        raise ValueError("Expected shape dimensions to be int")

    if "name" in kwargs:
        preamble = "{} must be an array".format(kwargs["name"])
    else:
        preamble = "Expected an array"

    if a is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape))
    try:
        len(a.shape)
    except (AttributeError, TypeError):
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape, a.__class__.__name__)
        )

    # Check non-wildcard dimensions.
    if len(a.shape) != len(shape) or any(
        actual != expected
        for actual, expected in zip(a.shape, shape)
        if not is_wildcard(expected)
    ):
        raise ValueError("{} with shape {}; got {}".format(preamble, shape, a.shape))

    wildcard_dims = [
        actual for actual, expected in zip(a.shape, shape) if is_wildcard(expected)
    ]
    if len(wildcard_dims) == 0:
        return None
    elif len(wildcard_dims) == 1:
        return wildcard_dims[0]
    else:
        return tuple(wildcard_dims)


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
