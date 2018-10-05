import numpy as np

__all__ = [
    "normalize",
    "sproj",
    "proj",
    "reject",
    "reject_axis",
    "magnitude",
    "angle",
    "signed_angle",
    "almost_zero",
    "almost_collinear",
    "pad_with_ones",
    "unpad",
    "apply_homogeneous",
    "basis",
]


def normalize(vector):
    """
    Return the vector, normalized.

    If vector is 2d, treats it as stacked vectors, and normalizes each one.

    """
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)


def sproj(vector, onto):
    """
    Compute the scalar projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if onto.ndim != 1:
        raise ValueError("onto should be a vector")
    return np.dot(vector, normalize(onto))


def proj(vector, onto):
    """
    Compute the vector projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        return sproj(vector, onto=onto) * normalize(onto)
    elif vector.ndim == 2:
        return sproj(vector, onto=onto)[:, np.newaxis] * normalize(onto)
    else:
        raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)


def reject(vector, from_v):
    """
    Compute the vector rejection of `vector` from `from_v` -- i.e.
    the vector component of `vector` perpendicular to `from_v`.

    `from_v` need not be normalized.

    """
    return vector - proj(vector, onto=from_v)


def reject_axis(vector, axis, squash=False):
    """
    Compute the vector component of `vector` perpendicular to the basis
    vector specified by `axis`. 0 means x, 1 means y, 2 means z.

    In other words, return a copy of vector that zeros the `axis` component.

    When `squash` is True, instead of zeroing the component, it drops it, so
    an input vector (in R3) is mapped to a point in R2.

    (N.B. Don't be misled: this meaning of `axis` is pretty different from
    the typical meaning in numpy.)

    """
    if squash:
        dims_to_keep = [0, 1, 2]
        try:
            dims_to_keep.remove(axis)
        except ValueError:
            raise ValueError("axis should be 0, 1, or 2")

        if vector.ndim == 1:
            return vector[dims_to_keep]
        elif vector.ndim == 2:
            return vector[:, dims_to_keep]
        else:
            raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)
    else:
        result = vector.copy()
        if vector.ndim == 1:
            result[axis] = 0.0
        elif vector.ndim == 2:
            result[:, axis] = 0.0
        else:
            raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)
        return result


def magnitude(vector):
    """
    Compute the magnitude of `vector`. For stacked inputs, compute the magnitude
    of each one.

    Args:
        vector (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.

    Returns:
        object: For `3x1` inputs, a `float` with the magnitude. For `kx1`
            inputs, a `kx1` array.
    """
    if vector.ndim == 1:
        return np.linalg.norm(vector)
    elif vector.ndim == 2:
        return np.linalg.norm(vector, axis=1)
    else:
        raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)


# Alias because angle()'s parameter shadows the name.
_normalize = normalize


def angle(v1, v2, look=None, assume_normalized=False, units="deg"):
    """
    Compute the unsigned angle between two vectors. For stacked inputs, the
    angle is computed pairwise.

    When `look` is provided, the angle is computed in that viewing plane
    (`look` is the normal). Otherwise the angle is computed in 3-space.

    Args:
        v1 (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        look (np.arraylike): A `3x1` vector specifying the normal of a viewing
            plane, or `None` to compute the angle in 3-space.
        assume_normalized (bool): When `True`, assume the input vectors
            are unit length, which improves performance.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Return:
        object: For `3x1` inputs, a `float` with the angle. For `kx1` inputs,
            a `kx1` array.
    """
    if units not in ["deg", "rad"]:
        raise ValueError("Unrecognized units {}; expected deg or rad".format(units))

    if look is not None:
        # This is a simple approach. Since this is working in two dimensions,
        # a smarter approach could reduce the amount of computation needed.
        v1, v2 = [reject(v, from_v=look) for v in (v1, v2)]

    dot_products = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / magnitude(v1) / magnitude(v2)

    # Clip, because the dot product can slip past 1 or -1 due to rounding and
    # we can't compute arccos(-1.00001).
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles


def signed_angle(v1, v2, look, units="deg"):
    """
    Compute the signed angle between two vectors. For stacked inputs, the
    angle is computed pairwise.

    Results are in the range -180 and 180 (or `-math.pi` and `math.pi`). A
    positive number indicates a clockwise sweep from `v1` to `v2`. A negative
    number is counterclockwise.

    Args:
        v1 (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        look (np.arraylike): A `3x1` vector specifying the normal of the
            viewing plane.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Returns:
        object: For `3x1` inputs, a `float` with the angle. For `kx1` inputs,
            a `kx1` array.
    """
    # The sign of (A x B) dot look gives the sign of the angle.
    # > 0 means clockwise, < 0 is counterclockwise.
    sign = np.array(np.sign(np.cross(v1, v2).dot(look)))

    # 0 means collinear: 0 or 180. Let's call that clockwise.
    sign[sign == 0] = 1

    return sign * angle(v1, v2, look, units=units)


def almost_zero(v, atol=1e-08):
    """
    Test if v is almost the zero vector.

    """
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    """
    Test if v1 and v2 are almost collinear.

    Mathematically speaking, the zero vector is collinear to everything.
    Geometrically that doesn't necessarily make sense. If you care, test
    your inputs with vx.almost_zero.

    """
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def pad_with_ones(matrix):
    """
    Add a column of ones. Transform from:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])
    to:
        array([[1., 2., 3., 1.],
               [2., 3., 4., 1.],
               [5., 6., 7., 1.]])

    """
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Invalid shape %s: pad expects nx3" % (matrix.shape,))
    return np.pad(matrix, ((0, 0), (0, 1)), mode="constant", constant_values=1)


def unpad(matrix):
    """
    Strip off a column (e.g. of ones). Transform from:
        array([[1., 2., 3., 1.],
               [2., 3., 4., 1.],
               [5., 6., 7., 1.]])
    to:
        array([[1., 2., 3.],
               [2., 3., 4.],
               [5., 6., 7.]])

    """
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("Invalid shape %s: unpad expects nx4" % (matrix.shape,))
    if not all(matrix[:, 3] == 1.0):
        raise ValueError("Expected a column of ones")
    return np.delete(matrix, 3, axis=1)


def apply_homogeneous(vertices, transform):
    """
    Apply the given transformation matrix to the vertices using homogenous
    coordinates.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices should be N x 3")
    if transform.shape != (4, 4):
        raise ValueError("Transformation matrix should be 4 x 4")

    return unpad(np.dot(transform, pad_with_ones(vertices).T).T)


class _BasisVectors(object):
    """
    The cartesian basis vectors.
    """

    @property
    def x(self):
        return np.array([1.0, 0.0, 0.0])

    @property
    def y(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def z(self):
        return np.array([0.0, 0.0, 1.0])

    @property
    def neg_x(self):
        return np.array([-1.0, 0.0, 0.0])

    @property
    def neg_y(self):
        return np.array([0.0, -1.0, 0.0])

    @property
    def neg_z(self):
        return np.array([0.0, 0.0, -1.0])


basis = _BasisVectors()
