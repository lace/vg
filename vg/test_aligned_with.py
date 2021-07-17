import numpy as np
import pytest
import vg


def test_aligned_with():
    v1 = np.array([1.0, 2.0, 3.0])
    neg_v1 = np.array([-1.0, -2.0, -3.0])

    np.testing.assert_array_equal(vg.aligned_with(v1, along=vg.basis.z), v1)
    np.testing.assert_array_equal(vg.aligned_with(neg_v1, along=vg.basis.z), v1)

    np.testing.assert_array_equal(
        vg.aligned_with(v1, along=vg.basis.z, reverse=True), neg_v1
    )
    np.testing.assert_array_equal(
        vg.aligned_with(neg_v1, along=vg.basis.z, reverse=True), neg_v1
    )


def test_orient_is_alias_for_aligned_with():
    v1 = np.array([1.0, 2.0, 3.0])
    with pytest.deprecated_call():
        np.testing.assert_array_equal(
            vg.orient(v1, along=vg.basis.z), vg.aligned_with(v1, along=vg.basis.z)
        )
