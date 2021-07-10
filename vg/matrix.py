import numpy as np
from ._helpers import raise_dimension_error


def pad_with_ones(matrix):
    """
    Deprecated. Matrix functions have been moved to polliwog. Will be removed in
    vg 2.

    Add a column of ones. Transform from:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])
    to:
        array([[1., 2., 3., 1.],
               [2., 3., 4., 1.],
               [5., 6., 7., 1.]])
    """
    import warnings

    warnings.warn(
        "`vg.matrix.pad_with_ones()` has been deprecated and will be removed in vg 2. "
        + "Matrix functions have been moved to polliwog.",
        DeprecationWarning,
    )

    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Invalid shape %s: pad expects nx3" % (matrix.shape,))
    return np.pad(matrix, ((0, 0), (0, 1)), mode="constant", constant_values=1)


def unpad(matrix):
    """
    Deprecated. Matrix functions have been moved to polliwog. Will be removed in
    vg 2.

    Strip off a column (e.g. of ones). Transform from:
        array([[1., 2., 3., 1.],
               [2., 3., 4., 1.],
               [5., 6., 7., 1.]])
    to:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])
    """
    import warnings

    warnings.warn(
        "`vg.matrix.unpad()` has been deprecated and will be removed in vg 2. "
        + "Matrix functions have been moved to polliwog.",
        DeprecationWarning,
    )

    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("Invalid shape %s: unpad expects nx4" % (matrix.shape,))
    if not all(matrix[:, 3] == 1.0):
        raise ValueError("Expected a column of ones")
    return np.delete(matrix, 3, axis=1)


def transform(vertices, transform):
    """
    Deprecated. Will be removed in vg 2. Use
    `polliwog.transform.apply_affine_transform()` instead.

    Apply the given transformation matrix to the vertices using homogenous
    coordinates.
    """
    import warnings

    warnings.warn(
        "`vg.matrix.transform()` has been deprecated and will be removed in vg 2. "
        + "Use `polliwog.transform.apply_affine_transform()` instead.",
        DeprecationWarning,
    )

    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix should be 4x4")

    if vertices.ndim == 1:
        matrix = vertices[np.newaxis]
    elif vertices.ndim == 2:
        matrix = vertices
    else:
        raise_dimension_error(vertices)

    if matrix.shape[1] != 3:
        raise ValueError("Vertices should be (3,) or Nx3")

    result = unpad(np.dot(transform, pad_with_ones(matrix).T).T)
    return result[0] if vertices.ndim == 1 else result
