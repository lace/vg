import numpy as np
from .shape import check, check_value


def pluralize(noun, count):
    return noun if count == 1 else "{}s".format(noun)


def raise_dimension_error(*input_values):
    messages = [
        "{} {}".format(input_value.ndim, pluralize("dimension", input_value.ndim))
        for input_value in input_values
    ]
    if len(messages) == 1:
        message = messages[0]
    elif len(messages) == 2:
        message = "{} and {}".format(*messages)
    else:
        message = "those inputs"
    raise ValueError("Not sure what to do with {}".format(message))


def broadcast_and_tile(v1, v2):
    if v1.ndim == 1 and v2.ndim == 2:
        check(locals(), "v1", (3,))
        k = check(locals(), "v2", (-1, 3))
        return np.tile(v1, (k, 1)), v2
    elif v1.ndim == 2 and v2.ndim == 1:
        k = check(locals(), "v1", (-1, 3))
        check(locals(), "v2", (3,))
        return v1, np.tile(v2, (k, 1))
    elif v1.ndim == 2 and v2.ndim == 2:
        k = check(locals(), "v1", (-1, 3))
        check(locals(), "v2", (k, 3))
        return v1, v2
    else:
        raise_dimension_error(v1, v2)


# TODO: After dropping Python 2, make `name=None` a regular kwarg.
def _check_value_any(a, *shapes, **kwargs):
    """
    Similar to `check_value()`, but accepts many candidate shapes and checks
    each of them before raising an error.

    Returns:
        object: The wildcard dimension (if one) or a tuple of wildcard
        dimensions (if more than one) of the matched shape
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required")
    name = kwargs.get("name")
    for shape in shapes:
        try:
            return check_value(a, shape, name=name)
        except ValueError:
            pass

    if name is None:
        preamble = "Expected an array"
    else:
        preamble = "Expected {} to be an array".format(name)

    shape_choices = ", ".join(
        shapes[:-2] + (" or ".join([str(shapes[-2]), str(shapes[-1])]),)
    )

    if a is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape_choices))
    else:
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape_choices, a.shape)
        )
