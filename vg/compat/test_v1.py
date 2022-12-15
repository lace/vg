import numpy as np
import pytest
import vg.compat.v1 as vg


def test_v1_has_functions():
    np.testing.assert_array_equal(
        vg.normalize(np.array([5, 0, 0])), np.array([1, 0, 0])
    )


def test_v1_has_constants():
    np.testing.assert_array_equal(vg.basis.x, np.array([1, 0, 0]))


def test_v1_has_shape_check():
    input_value = np.zeros(3)
    vg.shape.check(locals(), "input_value", (3,))


def test_v1_orient_is_alias_for_aligned_with():
    v1 = np.array([1.0, 2.0, 3.0])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.orient(v1, along=vg.basis.z), vg.aligned_with(v1, along=vg.basis.z)
        )


def test_v1_namespace():
    expected_symbols = [
        "aligned_with",
        "almost_collinear",
        "almost_equal",
        "almost_unit_length",
        "almost_zero",
        "angle",
        "apex",
        "apex_and_opposite",
        "argapex",
        "average",
        "basis",
        "cross",
        "dot",
        "euclidean_distance",
        "farthest",
        "magnitude",
        "major_axis",
        "matrix",
        "nearest",
        "normalize",
        "orient",
        "perpendicular",
        "principal_components",
        "project",
        "reject",
        "reject_axis",
        "rotate",
        "scalar_projection",
        "scale_factor",
        "shape",
        "signed_angle",
        "within",
    ]
    assert sorted(vg.__all__) == expected_symbols
