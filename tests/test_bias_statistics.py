import unittest

import numpy as np
import pandas as pd

from analysis.bias_statistics import (
    build_effect_table,
    fit_standardized_ols,
    simple_r2,
    spearman_effect,
    variance_inflation_factors,
)


class BiasStatisticsTests(unittest.TestCase):
    def test_spearman_monotonic_relation(self):
        rho, n = spearman_effect([1, 2, 3, 4], [10, 20, 30, 40])

        self.assertEqual(n, 4)
        self.assertAlmostEqual(rho, 1.0)

    def test_paired_effects_coerce_and_exclude_invalid_pairs(self):
        x = ["1", "invalid", 3, np.inf, 5, 6]
        y = [2, 4, np.nan, 8, 10, 12]

        rho, n = spearman_effect(x, y)

        self.assertEqual(n, 3)
        self.assertAlmostEqual(rho, 1.0)
        self.assertAlmostEqual(simple_r2(x, y), 1.0)

    def test_paired_effects_reject_shape_or_length_mismatch(self):
        with self.assertRaises(ValueError):
            spearman_effect([1, 2], [1])
        with self.assertRaises(ValueError):
            simple_r2([[1, 2]], [1, 2])

    def test_spearman_uses_average_ranks_for_ties(self):
        rho, n = spearman_effect([1, 1, 2, 3], [10, 10, 20, 30])

        self.assertEqual(n, 4)
        self.assertAlmostEqual(rho, 1.0)

    def test_simple_r2_exact_linear_relation(self):
        self.assertAlmostEqual(simple_r2([1, 2, 3], [3, 5, 7]), 1.0)

    def test_standardized_ols_recovers_dominant_predictor(self):
        x = np.arange(1.0, 21.0)
        df = pd.DataFrame({"y": 3.0 * x, "x": x, "noise": (-1.0) ** x})

        coef, summary = fit_standardized_ols(df, "y", ["x", "noise"])

        beta_x = coef.loc[coef["predictor"] == "x", "std_beta"].iloc[0]
        self.assertGreater(beta_x, 0.99)
        self.assertGreater(summary["adjusted_r2"], 0.99)

    def test_standardized_ols_uses_only_finite_complete_rows(self):
        df = pd.DataFrame(
            {
                "y": [2, 4, "bad", 8, np.inf, 12],
                "x": [1, "2", 3, 4, 5, 6],
            }
        )

        coef, summary = fit_standardized_ols(df, "y", ["x"])

        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(coef.loc[0, "std_beta"], 1.0)

    def test_standardized_ols_rejects_invalid_model_specification(self):
        df = pd.DataFrame({"y": [1, 2, 3, 4], "x": [1, 2, 3, 4]})

        with self.assertRaises(ValueError):
            fit_standardized_ols(df, "missing", ["x"])
        with self.assertRaises(ValueError):
            fit_standardized_ols(df, "y", ["missing"])
        with self.assertRaises(ValueError):
            fit_standardized_ols(df, "y", [])
        with self.assertRaises(ValueError):
            fit_standardized_ols(df, "y", ["x", "x"])

    def test_standardized_ols_rejects_minimum_n_and_constant_variables(self):
        with self.assertRaises(ValueError):
            fit_standardized_ols(pd.DataFrame({"y": [1, 2], "x": [1, 2]}), "y", ["x"])
        with self.assertRaises(ValueError):
            fit_standardized_ols(
                pd.DataFrame({"y": [1, 1, 1], "x": [1, 2, 3]}), "y", ["x"]
            )
        with self.assertRaises(ValueError):
            fit_standardized_ols(
                pd.DataFrame({"y": [1, 2, 3], "x": [1, 1, 1]}), "y", ["x"]
            )

    def test_standardized_ols_rejects_rank_deficient_predictors(self):
        x = np.arange(1.0, 6.0)
        df = pd.DataFrame({"y": 3 * x, "x1": x, "x2": 2 * x})

        with self.assertRaises(ValueError):
            fit_standardized_ols(df, "y", ["x1", "x2"])

    def test_vif_detects_collinearity(self):
        x = np.arange(1.0, 11.0)
        matrix = np.column_stack([x, 2.0 * x, (-1.0) ** x])
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

        vif = variance_inflation_factors(matrix, ["x1", "x2", "x3"])

        self.assertTrue(np.isinf(vif.loc[vif["predictor"] == "x1", "VIF"]).iloc[0])

    def test_vif_returns_one_for_single_predictor(self):
        vif = variance_inflation_factors(np.array([[1.0], [2.0], [3.0]]), ["x"])

        self.assertEqual(vif.loc[0, "VIF"], 1.0)

    def test_vif_rejects_invalid_matrix_and_names(self):
        invalid_cases = [
            (np.array([1.0, 2.0]), ["x"]),
            (np.empty((0, 1)), ["x"]),
            (np.array([[1.0], [2.0]]), []),
            (np.array([[1.0, 2.0], [3.0, 4.0]]), ["x"]),
            (np.array([[1.0, 2.0], [3.0, 4.0]]), ["x", "x"]),
            (np.array([[1.0], [np.inf]]), ["x"]),
            (np.array([[1.0], [1.0]]), ["x"]),
        ]

        for matrix, names in invalid_cases:
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    variance_inflation_factors(matrix, names)

    def test_effect_table_separates_independent_and_diagnostic_evidence(self):
        station_df = pd.DataFrame(
            {
                "BIAS_none": [1.0, 2.0, 3.0, 4.0],
                "DELTA_ABS_BIAS": [4.0, 3.0, 2.0, 1.0],
                "elevation": [100.0, 200.0, 300.0, 400.0],
                "error_spread": [0.1, 0.2, 0.3, 0.4],
            }
        )

        table = build_effect_table(station_df, ["elevation"], ["error_spread"])

        independent = table.loc[table["PREDICTOR"] == "elevation"]
        diagnostic = table.loc[table["PREDICTOR"] == "error_spread"]
        self.assertEqual(len(table), 4)
        self.assertEqual(set(independent["EVIDENCE_LEVEL"]), {"independent_association"})
        self.assertEqual(set(diagnostic["EVIDENCE_LEVEL"]), {"outcome_derived_diagnostic"})
        self.assertTrue(independent["NOTES"].str.contains("Pearson R²; association only").all())
        self.assertTrue(diagnostic["NOTES"].str.contains("Pearson R²; derived").all())
        self.assertTrue(table["R2_TYPE"].eq("pearson_r_squared").all())
        self.assertNotIn("p_value", table.columns)

    def test_effect_table_filters_invalid_pairs_and_rejects_missing_columns(self):
        station_df = pd.DataFrame(
            {
                "BIAS_none": [1.0, 2.0, 3.0, np.inf, 5.0],
                "DELTA_ABS_BIAS": [5.0, 4.0, 3.0, 2.0, 1.0],
                "independent": ["1", 2.0, "bad", 4.0, 5.0],
                "diagnostic": [10.0, 20.0, 30.0, np.nan, 50.0],
            }
        )

        table = build_effect_table(station_df, ["independent"], ["diagnostic"])

        self.assertEqual(
            table.loc[
                (table["OUTCOME"] == "BIAS_none") & (table["PREDICTOR"] == "independent"),
                "N",
            ].iloc[0],
            3,
        )
        with self.assertRaises(ValueError):
            build_effect_table(station_df.drop(columns="BIAS_none"), ["independent"], [])
        with self.assertRaises(ValueError):
            build_effect_table(station_df, ["missing"], [])


if __name__ == "__main__":
    unittest.main()
