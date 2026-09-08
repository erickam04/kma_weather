"""BIAS 실질·통계적 유의성 분석 core 테스트."""

import unittest

import numpy as np
import pandas as pd

from analysis.bias_significance import (
    benjamini_hochberg,
    bootstrap_bias_effects,
    bootstrap_boundary_pvalues,
    bootstrap_stationwise_bias_effects,
    classify_station_effects,
    make_month_stratified_block_weights,
)


class MonthStratifiedBlockWeightsTests(unittest.TestCase):
    def test_preserves_each_month_sample_size_and_seed(self):
        """월별 표본 수가 달라지거나 같은 seed가 다른 재표집을 만드는 회귀를 잡는다."""
        dates = pd.date_range("2024-01-01", "2024-02-29", freq="D")

        first = make_month_stratified_block_weights(
            dates, n_boot=8, block_days=3, seed=17
        )
        second = make_month_stratified_block_weights(
            dates, n_boot=8, block_days=3, seed=17
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 60))
        np.testing.assert_array_equal(first[:, :31].sum(axis=1), np.full(8, 31))
        np.testing.assert_array_equal(first[:, 31:].sum(axis=1), np.full(8, 29))

    def test_rejects_duplicate_or_unsorted_dates(self):
        """날짜 열이 달력축과 일대일이 아니어 bootstrap 위치가 뒤섞이는 회귀를 잡는다."""
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_month_stratified_block_weights(
                pd.to_datetime(["2024-01-02", "2024-01-01"]), 3, 1, 1
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            make_month_stratified_block_weights(
                pd.to_datetime(["2024-01-01", "2024-01-01"]), 3, 1, 1
            )


class BootstrapBiasEffectsTests(unittest.TestCase):
    def test_uses_paired_weights_and_hour_counts_for_bias(self):
        """보정 전후에 다른 날짜를 뽑거나 일평균을 단순평균하는 회귀를 잡는다."""
        daily = {
            "sum_none": np.array([[2.0, 12.0], [-2.0, -12.0]]),
            "sum_fixed": np.array([[0.0, 4.0], [0.0, -4.0]]),
            "count": np.array([[2.0, 4.0], [2.0, 4.0]]),
        }
        weights = np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int16)

        result = bootstrap_bias_effects(daily, weights)

        np.testing.assert_allclose(
            result["bias_none"],
            np.array([[14 / 6, -14 / 6], [1.0, -1.0], [3.0, -3.0]]),
        )
        np.testing.assert_allclose(
            result["bias_fixed"],
            np.array([[4 / 6, -4 / 6], [0.0, 0.0], [1.0, -1.0]]),
        )
        np.testing.assert_allclose(
            result["delta_abs_bias"],
            np.array([[-10 / 6, -10 / 6], [-1.0, -1.0], [-2.0, -2.0]]),
        )

    def test_rejects_bootstrap_replica_without_valid_hours(self):
        """결측일만 재표집해 분모가 0인 복제본을 조용히 통과시키는 회귀를 잡는다."""
        daily = {
            "sum_none": np.zeros((1, 2)),
            "sum_fixed": np.zeros((1, 2)),
            "count": np.array([[1.0, 0.0]]),
        }
        weights = np.array([[0, 2]], dtype=np.int16)
        with self.assertRaisesRegex(ValueError, "zero valid paired hours"):
            bootstrap_bias_effects(daily, weights)

    def test_stationwise_bootstrap_samples_only_each_stations_valid_days(self):
        """단기 운영 station의 유효일을 하나도 못 뽑아 분모가 0이 되는 회귀를 잡는다."""
        dates = pd.date_range("2024-12-01", "2024-12-31", freq="D")
        daily = {
            "sum_none": np.zeros((1, 31)),
            "sum_fixed": np.zeros((1, 31)),
            "count": np.zeros((1, 31)),
        }
        daily["sum_none"][0, -3:] = [2.0, 4.0, 6.0]
        daily["sum_fixed"][0, -3:] = [1.0, 2.0, 3.0]
        daily["count"][0, -3:] = [2.0, 2.0, 2.0]

        result = bootstrap_stationwise_bias_effects(
            daily, dates, n_boot=100, block_days=2, seed=9
        )

        self.assertEqual(result["delta_abs_bias"].shape, (100, 1))
        self.assertTrue(np.isfinite(result["delta_abs_bias"]).all())
        self.assertEqual(result["coverage_days"].tolist(), [3])
        self.assertEqual(result["coverage_months"].tolist(), [1])


class BoundaryInferenceTests(unittest.TestCase):
    def test_detects_effect_beyond_practical_improvement_boundary(self):
        """0이 아닌 -delta 경계를 검정하지 않아 작은 개선을 과대분류하는 회귀를 잡는다."""
        theta = -0.30
        samples = theta + np.linspace(-0.02, 0.02, 100)

        p = bootstrap_boundary_pvalues(theta, samples, delta=0.10)

        self.assertLess(p["p_improve"], 0.02)
        self.assertGreater(p["p_worsen"], 0.90)
        self.assertGreater(p["p_equivalent"], 0.90)

    def test_detects_practical_equivalence_around_zero(self):
        """통계적 비유의성을 곧바로 '차이 없음'으로 부르는 회귀를 잡는다."""
        theta = 0.0
        samples = np.linspace(-0.01, 0.01, 100)

        p = bootstrap_boundary_pvalues(theta, samples, delta=0.10)

        self.assertLess(p["p_equivalent"], 0.02)
        self.assertGreater(p["p_improve"], 0.90)
        self.assertGreater(p["p_worsen"], 0.90)

    def test_benjamini_hochberg_matches_hand_calculation(self):
        """정렬 복원이나 누적최솟값 방향이 뒤집혀 FDR q값이 틀리는 회귀를 잡는다."""
        p = np.array([0.01, 0.04, 0.03, 0.002])
        expected = np.array([0.02, 0.04, 0.04, 0.008])
        np.testing.assert_allclose(benjamini_hochberg(p), expected)

    def test_classifies_four_mutually_exclusive_outcomes(self):
        """유의미한 개선·악화·동등·불확실이 겹치거나 누락되는 회귀를 잡는다."""
        frame = pd.DataFrame(
            {
                "q_improve": [0.01, 0.8, 0.8, 0.8],
                "q_worsen": [0.8, 0.01, 0.8, 0.8],
                "q_equivalent": [0.8, 0.8, 0.01, 0.8],
            }
        )

        result = classify_station_effects(frame, alpha=0.05)

        self.assertEqual(
            result["SIGNIFICANCE_CLASS"].tolist(),
            [
                "meaningful_improvement",
                "meaningful_worsening",
                "practically_equivalent",
                "uncertain",
            ],
        )
        self.assertEqual(result["SIGNIFICANCE_CLASS"].nunique(), 4)

    def test_rejects_conflicting_significance_claims(self):
        """한 station을 둘 이상의 결론으로 동시에 판정하는 회귀를 잡는다."""
        frame = pd.DataFrame(
            {"q_improve": [0.01], "q_worsen": [0.01], "q_equivalent": [0.8]}
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            classify_station_effects(frame)


if __name__ == "__main__":
    unittest.main()
