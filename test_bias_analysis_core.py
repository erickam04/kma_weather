import unittest

import numpy as np

from bias_analysis_core import (
    classify_bias_mechanism,
    compute_bias_metrics,
)


class BiasMechanismTests(unittest.TestCase):
    def test_same_direction_worsens(self):
        self.assertEqual(
            classify_bias_mechanism(1.0, 0.5), "same_direction_worsened"
        )

    def test_opposite_direction_improves_without_flip(self):
        self.assertEqual(
            classify_bias_mechanism(1.0, -0.5), "improved_without_flip"
        )

    def test_opposite_direction_improves_with_flip(self):
        self.assertEqual(
            classify_bias_mechanism(1.0, -1.5), "improved_with_flip"
        )

    def test_overshoot_worsens(self):
        self.assertEqual(
            classify_bias_mechanism(1.0, -2.5), "overshoot_worsened"
        )

    def test_equal_magnitude_boundary(self):
        self.assertEqual(
            classify_bias_mechanism(1.0, -2.0), "equal_magnitude"
        )

    def test_zero_baseline(self):
        self.assertEqual(
            classify_bias_mechanism(0.0, 0.2), "zero_baseline_worsened"
        )

    def test_no_correction(self):
        self.assertEqual(
            classify_bias_mechanism(-0.4, 0.0), "no_correction"
        )

    def test_compute_bias_metrics(self):
        result = compute_bias_metrics(
            np.array([1.0, 3.0]),
            np.array([0.0, 1.0]),
        )
        self.assertEqual(result["n_hours"], 2)
        self.assertAlmostEqual(result["BIAS_none"], 2.0)
        self.assertAlmostEqual(result["BIAS_fixed"], 0.5)
        self.assertAlmostEqual(result["CORRECTION_SHIFT"], -1.5)
        self.assertAlmostEqual(result["DELTA_ABS_BIAS"], -1.5)
        self.assertEqual(result["MECHANISM"], "improved_without_flip")

    def test_rejects_mismatched_or_nonfinite_arrays(self):
        with self.assertRaises(ValueError):
            compute_bias_metrics(np.array([1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            compute_bias_metrics(np.array([np.nan]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
