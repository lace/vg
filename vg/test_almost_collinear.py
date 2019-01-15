import numpy as np
from . import core as vg


def test_almost_collinear():
    collinear_vectors = np.array(
        [[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [2.0, 2.0, 0.0], [1.000000001, 1.0, 0.0]]
    )

    for v1 in collinear_vectors:
        for v2 in collinear_vectors:
            assert vg.almost_collinear(v1, v2) == True

    for v in collinear_vectors:
        zero_v = np.array([0.0, 0.0, 0.0])
        assert vg.almost_collinear(v, zero_v) == True
        assert vg.almost_collinear(zero_v, v) == True
        assert vg.almost_collinear(zero_v, zero_v) == True

    non_collinear_vectors = np.array(
        [[1.0, 1.0, 0.0], [-1.0, -1.3, 0.0], [2.0, 2.0, 1.0], [1.000001, 1.0, 0.0]]
    )

    for index1, v1 in enumerate(non_collinear_vectors):
        for index2, v2 in enumerate(non_collinear_vectors):
            if index1 != index2:
                assert vg.almost_collinear(v1, v2) == False
