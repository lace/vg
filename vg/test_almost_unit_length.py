import math
import numpy as np
from . import core as vg


def test_almost_unit_length():
    unit_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.000000001],
            [1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)],
        ]
    )

    for v in unit_vectors:
        assert vg.almost_unit_length(np.array(v)) == True

    assert vg.almost_unit_length(np.array(unit_vectors)).tolist() == [
        True,
        True,
        True,
        True,
    ]

    non_unit_vectors = np.array(
        [[1.0, 1.0, 0.0], [-1.0, -1.3, 0.0], [2.0, 2.0, 1.0], [1.000001, 1.0, 0.0]]
    )

    for v in non_unit_vectors:
        assert vg.almost_unit_length(np.array(v)) == False

    assert vg.almost_unit_length(np.array(non_unit_vectors)).tolist() == [
        False,
        False,
        False,
        False,
    ]

    assert vg.almost_unit_length(
        np.vstack([unit_vectors, non_unit_vectors])
    ).tolist() == [True, True, True, True, False, False, False, False]
