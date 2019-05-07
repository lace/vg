import math
import numpy as np
from ._helpers import raise_dimension_error

__all__ = [
    "normalize",
    "perpendicular",
    "project",
    "scalar_projection",
    "reject",
    "reject_axis",
    "magnitude",
    "angle",
    "signed_angle",
    "rotate",
    "almost_zero",
    "almost_unit_length",
    "almost_collinear",
    "almost_equal",
    "principal_components",
    "major_axis",
    "apex",
    "farthest",
    "basis",
    "within",
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
        raise_dimension_error(vector)


def perpendicular(v1, v2, normalized=True):
    """
    Given two noncollinear vectors, return a vector perpendicular to both. For
    stacked inputs, compute the result vectors pairwise such that `result[k]` is
    perpendicular to `v1[k]` and `v2[k]`.

    Result vectors follow the right-hand rule. When the right index finger
    points along `v1` and the right middle finger along `v2`, the right thumb
    points along the result.

    Args:
        v1 (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        normalized (bool): When `True`, the result vector is guaranteed to be
            unit length.

    Return:
        np.arraylike: An array with the same shape as `v1` and `v2`.

    See also:
        - https://en.wikipedia.org/wiki/Cross_product#Definition
        - https://commons.wikimedia.org/wiki/File:Right_hand_rule_cross_product.svg
    """
    if v1.ndim == 1 and v2.ndim == 1:
        result = np.cross(v1, v2)
        return normalize(result) if normalized else result
    elif v1.ndim == 2 and v2.ndim == 2:
        result = np.cross(v1[:, np.newaxis, :], v2[:, np.newaxis, :])[:, 0, :]
        return normalize(result) if normalized else result
    else:
        raise_dimension_error(v1, v2)


def project(vector, onto):
    """
    Compute the vector projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if vector.ndim == 1:
        return scalar_projection(vector, onto=onto) * normalize(onto)
    elif vector.ndim == 2:
        return scalar_projection(vector, onto=onto)[:, np.newaxis] * normalize(onto)
    else:
        raise_dimension_error(vector)


def scalar_projection(vector, onto):
    """
    Compute the scalar projection of `vector` onto the vector `onto`.

    `onto` need not be normalized.

    """
    if onto.ndim != 1:
        raise ValueError("onto should be a vector")
    return np.dot(vector, normalize(onto))


def reject(vector, from_v):
    """
    Compute the vector rejection of `vector` from `from_v` -- i.e.
    the vector component of `vector` perpendicular to `from_v`.

    `from_v` need not be normalized.

    """
    return vector - project(vector, onto=from_v)


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
            raise_dimension_error(vector)
    else:
        if axis not in [0, 1, 2]:
            raise ValueError("axis should be 0, 1, or 2")
        result = vector.copy()
        if vector.ndim == 1:
            result[axis] = 0.0
        elif vector.ndim == 2:
            result[:, axis] = 0.0
        else:
            raise_dimension_error(vector)
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
        raise_dimension_error(vector)


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
            are unit length. This improves performance, however when the inputs
            are not normalized, setting this will cause an incorrect results.
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


def rotate(vector, around_axis, angle, units="deg", assume_normalized=False):
    """
    Rotate a point or vector around a given axis. The direction of rotation
    around `around_axis` is determined by the right-hand rule.

    Args:
        vector (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.
        around_axis (np.arraylike): A `3x1` vector specifying the axis of rotation.
        assume_normalized (bool): When `True`, assume `around_axis` is unit
            length. This improves performance marginally, however
            when the inputs are not normalized, setting this will cause an
            incorrect results.
        units (str): `'deg'` to specify `angle` in degrees or `'rad'` to specify
            radians.

    Returns:
        np.arraylike: The transformed point or points. This has the same shape as
            `vector`.

    See also:
        - https://en.wikipedia.org/wiki/Cross_product#Definition
        - https://commons.wikimedia.org/wiki/File:Right_hand_rule_cross_product.svg
    """
    if units == "deg":
        angle = math.radians(angle)
    elif units != "rad":
        raise ValueError('Unknown units "{}"; expected "deg" or "rad"'.format(units))

    cosine = math.cos(angle)
    sine = math.sin(angle)

    if not assume_normalized:
        around_axis = normalize(around_axis)

    if vector.ndim == 1:
        dot_products = np.inner(around_axis, vector)
    elif vector.ndim == 2:
        dot_products = np.inner(around_axis, vector)[:, np.newaxis]
    else:
        raise_dimension_error(vector)

    # Rodrigues' rotation formula.
    return (
        cosine * vector
        + sine * np.cross(around_axis, vector)
        + (1 - cosine) * dot_products * around_axis
    )


def almost_zero(v, atol=1e-08):
    """
    Test if v is almost the zero vector.

    """
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_unit_length(vector, atol=1e-08):
    """
    Test if the `vector` has almost unit length. For stacked inputs, test each
    one.

    Args:
        vector (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.

    Returns:
        object: For `3x1` inputs, a `bool`. For `kx1` inputs, a `kx1` array.
    """
    return np.isclose(magnitude(vector), 1.0, rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    """
    Test if v1 and v2 are almost collinear.

    Mathematically speaking, the zero vector is collinear to everything.
    Geometrically that doesn't necessarily make sense. If you care, test
    your inputs with vg.almost_zero.

    """
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def almost_equal(v1, v2, atol=1e-08):
    """
    Test if `v1` and `v2` are equal within the given absolute tolerance.

    See also:
        - https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html

    """
    return np.allclose(v1, v2, rtol=0, atol=atol)


def principal_components(coords):
    """
    Compute the principal components of the input coordinates. These are
    useful for dimensionality reduction and feature modeling.

    Args:
        coords (np.arraylike): A `nxk` stack of coordinates.

    Returns:
        np.ndarray: A `kxk` stack of vectors.

    See also:
        - http://setosa.io/ev/principal-component-analysis/
        - https://en.wikipedia.org/wiki/Principal_component_analysis
        - https://plot.ly/ipython-notebooks/principal-component-analysis/
    """
    mean = np.mean(coords, axis=0)
    _, _, result = np.linalg.svd(coords - mean)
    return result


def major_axis(coords):
    """
    Compute the first principal component of the input coordinates. This is
    the vector which best describes the multidimensional data using a single
    dimension.

    Args:
        coords (np.arraylike): A `nxk` stack of coordinates.

    Returns:
        np.ndarray: A `kx1` vector.

    See also:
        - http://setosa.io/ev/principal-component-analysis/
        - https://en.wikipedia.org/wiki/Principal_component_analysis
        - https://plot.ly/ipython-notebooks/principal-component-analysis/
    """
    return principal_components(coords)[0]


def apex(points, along):
    """
    Find the most extreme point in the direction provided.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        along (np.arraylike): A `3x1` vector specifying the direction of
            interest.

    Returns:
        np.ndarray: A `3x1` point taken from `points`.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid shape %s: apex expects nx3" % (points.shape,))
    if along.shape != (3,):
        raise ValueError("along should be a 3x1 vector")
    coords_on_axis = points.dot(along)
    return points[np.argmax(coords_on_axis)]


def farthest(from_points, to_point, ret_index=False):
    """
    Find the point farthest from the given point.

    Args:
        from_points (np.arraylike): A `kx3` stack of points in R^3.
        to_point (np.arraylike): A `3x1` point of interest.
        ret_index (bool): When `True`, return both the point and its index.

    Returns:
        np.ndarray: A `3x1` vector taken from `from_points`.
    """
    if from_points.ndim != 2 or from_points.shape[1] != 3:
        raise ValueError(
            "Invalid shape %s: farthest expects nx3" % (from_points.shape,)
        )
    if to_point.shape != (3,):
        raise ValueError("to_point should be 3x1")

    absolute_distances = magnitude(from_points - to_point)

    index_of_farthest_point = np.argmax(absolute_distances)
    farthest_point = from_points[index_of_farthest_point]

    return farthest_point, index_of_farthest_point if ret_index else farthest_point


def within(points, radius, of_point, atol=1e-08, ret_indices=False):
    """
    Select points within a given radius of a point.

    Args:
        points (np.arraylike): A `kx3` stack of points in R^3.
        radius (float): The radius of the sphere of interest centered on
            `of_point`.
        of_point (np.arraylike): The `3x1` point of interest.
        atol (float): The distance tolerance. Points within `radius + atol`
            of `of_point` are selected.
        ret_indexes (bool): When `True`, return both the points and their
            indices.

    Returns:
        np.ndarray: A `3x1` vector taken from `points`.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid shape %s: within expects nx3" % (points.shape,))
    if not isinstance(radius, float):
        raise ValueError("radius should be a float")
    if of_point.shape != (3,):
        raise ValueError("to_point should be 3x1")

    absolute_distances = magnitude(points - of_point)
    indices_within_radius, = (absolute_distances < radius + atol).nonzero()
    points_within_radius = points[indices_within_radius]
    if ret_indices:
        return points_within_radius, indices_within_radius
    else:
        return points_within_radius


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
