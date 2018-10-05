import unittest
import math
import numpy as np
from . import core as vx


class TestVector(unittest.TestCase):
    def test_normalize(self):
        import math

        v = np.array([1, 1, 0])
        expected = np.array([math.sqrt(2) / 2.0, math.sqrt(2) / 2.0, 0])
        np.testing.assert_array_almost_equal(vx.normalize(v), expected)

    def test_normalize_stacked(self):
        import math

        vs = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
        expected = np.array(
            [[math.sqrt(2) / 2.0, math.sqrt(2) / 2.0, 0], [-1, 0, 0], [0, 0, 1]]
        )
        np.testing.assert_array_almost_equal(vx.normalize(vs), expected)

    def test_proj(self):
        v = np.array([5.0, -3.0, 1.0])
        onto = np.array([0, -1.0, 0])
        expected_s = 3.0
        expected_v = np.array([0, -3.0, 0])
        np.testing.assert_array_almost_equal(vx.sproj(v, onto=onto), expected_s)
        np.testing.assert_array_almost_equal(vx.proj(v, onto=onto), expected_v)

        with self.assertRaises(ValueError):
            vx.proj(v, onto=np.array([vx.basis.x, vx.basis.x]))

    def test_proj_stacked(self):
        vs = np.array([[5.0, -3.0, 1.0], [1.0, 0, 1.0], [0.0, 1, 0.0], [0.0, 0, 0.0]])
        onto = np.array([0, -1.0, 0])
        expected_s = np.array([3.0, 0.0, -1.0, 0])
        expected_v = np.array(
            [[0.0, -3.0, 0.0], [0.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0, 0.0]]
        )
        np.testing.assert_array_almost_equal(vx.sproj(vs, onto=onto), expected_s)
        np.testing.assert_array_almost_equal(vx.proj(vs, onto=onto), expected_v)

    def test_reject(self):
        v = np.array([2.0, 4.0, 0.0])
        from_v = np.array([0.0, 5.0, 0.0])
        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(vx.reject(v, from_v=from_v), expected)

    def test_reject_stacked(self):
        vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
        from_v = np.array([0.0, 5.0, 0.0])
        expected = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(vx.reject(vs, from_v=from_v), expected)

    def test_reject_axis(self):
        v = np.array([2.0, 4.0, 0.0])
        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(vx.reject_axis(v, axis=1), expected)

    def test_reject_axis_stacked(self):
        vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
        expected = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(vx.reject_axis(vs, axis=1), expected)

    def test_reject_axis_with_squash(self):
        v = np.array([2.0, 4.0, 0.0])
        expected = np.array([2.0, 0.0])
        np.testing.assert_array_almost_equal(
            vx.reject_axis(v, axis=1, squash=True), expected
        )

    def test_reject_axis_stacked_with_squash(self):
        vs = np.array([[2.0, 4.0, 0.0], [-2.0, -1.0, 0.0]])
        expected = np.array([[2.0, 0.0], [-2.0, 0.0]])
        np.testing.assert_array_almost_equal(
            vx.reject_axis(vs, axis=1, squash=True), expected
        )

    def test_magnitude(self):
        v = np.array([1, 1, 0])
        expected = math.sqrt(2)
        np.testing.assert_almost_equal(vx.magnitude(v), expected)
        self.assertIsInstance(vx.magnitude(v), float)

    def test_magnitude_stacked(self):
        vs = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 5]])
        expected = np.array([math.sqrt(2), 1, 5])
        np.testing.assert_array_almost_equal(vx.magnitude(vs), expected)

    def test_almost_zero(self):
        self.assertTrue(vx.almost_zero(np.array([0.0, 0.0, 0.0])))
        self.assertTrue(vx.almost_zero(np.array([0.000000000000000001, 0.0, 0.0])))
        self.assertFalse(vx.almost_zero(np.array([0.0000001, 0.0, 0.0])))

    def test_almost_collinear(self):
        collinear_vectors = np.array(
            [
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [2.0, 2.0, 0.0],
                [1.000000001, 1.0, 0.0],
            ]
        )

        for v1 in collinear_vectors:
            for v2 in collinear_vectors:
                self.assertTrue(vx.almost_collinear(v1, v2))

        for v in collinear_vectors:
            zero_v = np.array([0.0, 0.0, 0.0])
            self.assertTrue(vx.almost_collinear(v, zero_v))
            self.assertTrue(vx.almost_collinear(zero_v, v))
            self.assertTrue(vx.almost_collinear(zero_v, zero_v))

        non_collinear_vectors = np.array(
            [[1.0, 1.0, 0.0], [-1.0, -1.3, 0.0], [2.0, 2.0, 1.0], [1.000001, 1.0, 0.0]]
        )

        for index1, _ in enumerate(non_collinear_vectors):
            for index2, _ in enumerate(non_collinear_vectors):
                if index1 != index2:
                    self.assertFalse(
                        vx.almost_collinear(
                            non_collinear_vectors[index1], non_collinear_vectors[index2]
                        )
                    )

    def test_pad_with_ones(self):
        np.testing.assert_array_equal(
            vx.pad_with_ones(
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
            ),
            np.array(
                [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]]
            ),
        )

    def test_pad_with_wrong_dimensions(self):
        # NB: on windows, the sizes here will render as 3L, not 3:
        with self.assertRaisesRegexp(
            ValueError, "^Invalid shape \(3L?, 4L?\): pad expects nx3$"
        ):
            vx.pad_with_ones(
                np.array(
                    [
                        [1.0, 2.0, 3.0, 42.0],
                        [2.0, 3.0, 4.0, 42.0],
                        [5.0, 6.0, 7.0, 42.0],
                    ]
                )
            )

        with self.assertRaisesRegexp(
            ValueError, "^Invalid shape \(3L?,\): pad expects nx3$"
        ):
            vx.pad_with_ones(np.array([1.0, 2.0, 3.0]))

    def test_unpad(self):
        np.testing.assert_array_equal(
            vx.unpad(
                np.array(
                    [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 1.0]]
                )
            ),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        )

        # NB: on windows, the sizes here will render as 3L, not 3:
        with self.assertRaisesRegexp(
            ValueError, "^Invalid shape \(3L?, 3L?\): unpad expects nx4$"
        ):
            vx.unpad(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))

        with self.assertRaisesRegexp(
            ValueError, "^Invalid shape \(4L?,\): unpad expects nx4$"
        ):
            vx.unpad(np.array([1.0, 2.0, 3.0, 4.0]))

        with self.assertRaisesRegexp(ValueError, "^Expected a column of ones$"):
            vx.unpad(
                np.array(
                    [[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 1.0], [5.0, 6.0, 7.0, 3.0]]
                )
            )
