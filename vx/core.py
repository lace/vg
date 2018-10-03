"""
A namespace of vector shortcuts.

Use the named secondary arguments. They tend to make your code more
readable:

    result = vx.proj(v1, onto=v2)


Design principles
-----------------

These are common linear algebra operations which are easily expressed in
numpy, but which we choose to abstract for a few reasons:

1. If you're not programming linalg every day, you might forget the formula.
   These forms are easier to remember and easily referenced.

2. They tend to be self-documenting in a way that the numpy forms are not.
   If you are not programming linalg every day, this will come in handy to
   you, and certainly will to other programmers in that situation.

3. These implementations are more robust. They automatically inspect `ndim`
   on their arguments, so they work equally well if the argument is a vector
   or a stack of vectors. In the long run, they can be more careful about
   checking edge cases like a zero norm or zero cross product and returning
   a sensible result or raising a sensible error, as appropriate.

"""

import numpy as np


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
    Compute the magnitude of `vector`.

    If vector is 2d, treats it as stacked vectors, and computes the magnitude
    of each one.
    """
    if vector.ndim == 1:
        return np.linalg.norm(vector)
    elif vector.ndim == 2:
        return np.linalg.norm(vector, axis=1)
    else:
        raise ValueError("Not sure what to do with %s dimensions" % vector.ndim)


def angle(v1, v2, look):  # FIXME pylint: disable=unused-argument
    """
    Compute the unsigned angle between two vectors.

    Returns a number between 0 and 180.

    """
    import math

    # TODO https://bodylabs.atlassian.net/projects/GEN/issues/GEN-1
    # As pylint points out, we are not using `look` here. This method is
    # supposed to be giving the angle between two vectors when viewed along a
    # particular look vector, squashed into a plane. The code here is
    # returning the angle in 3-space, which might be a reasonable function to
    # have, but is not workable for computing the angle between planes as
    # we're doing in bodylabs.measurement.anatomy.Angle.

    dot = normalize(v1).dot(normalize(v2))
    # Dot product sometimes slips past 1 or -1 due to rounding.
    # Can't acos(-1.00001).
    dot = max(min(dot, 1), -1)

    return math.degrees(math.acos(dot))


def signed_angle(v1, v2, look):
    """
    Compute the signed angle between two vectors.

    Returns a number between -180 and 180. A positive number indicates a
    clockwise sweep from v1 to v2. A negative number is counterclockwise.

    """
    # The sign of (A x B) dot look gives the sign of the angle.
    # > 0 means clockwise, < 0 is counterclockwise.
    sign = np.sign(np.cross(v1, v2).dot(look))

    # 0 means collinear: 0 or 180. Let's call that clockwise.
    if sign == 0:
        sign = 1

    return sign * angle(v1, v2, look)


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


class BasisVectorsFactory(object):
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


basis = BasisVectorsFactory()
