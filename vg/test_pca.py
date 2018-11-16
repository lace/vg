from __future__ import print_function
import unittest
import numpy as np
from . import core as vx


class TestPCA(unittest.TestCase):
    def test_principal_components(self):
        # Example from http://www.iro.umontreal.ca/~pift6080/H09/documents/papers/pca_tutorial.pdf
        coords = np.array(
            [
                [2.5, 2.4],
                [0.5, 0.7],
                [2.2, 2.9],
                [1.9, 2.2],
                [3.1, 3.0],
                [2.3, 2.7],
                [2, 1.6],
                [1, 1.1],
                [1.5, 1.6],
                [1.1, 0.9],
            ]
        )

        pcs = vx.principal_components(coords)

        self.assertEqual(pcs.shape, (2, 2))
        np.testing.assert_array_almost_equal(pcs[0], np.array([-0.677873, -0.735179]))

        first_pc = vx.major_axis(coords)

        np.testing.assert_array_almost_equal(first_pc, np.array([-0.677873, -0.735179]))
