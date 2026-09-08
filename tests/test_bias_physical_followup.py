"""BIAS 물리요인 후속 분석의 순수 계산 계약 테스트.

모든 기대값은 손계산 또는 기존 정본의 승인된 불변식에서 고정했다.
runner·파일 쓰기·그래프 생성은 이 테스트의 범위가 아니다.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


try:
    physical = importlib.import_module('analysis.bias_physical_followup')
    MODULE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # RED 단계에서도 각 테스트를 assertion failure로 보이게 한다.
    physical = None
    MODULE_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_STATION_RESULTS = (
    ROOT
    / "WeatherData_AWS"
    / "interpolation_batch"
    / "fullnet"
    / "bias_feedback_analysis"
    / "bias_feedback_station_results.csv"
)


class PhysicalFunctionTestCase(unittest.TestCase):
    def function(self, name):
        self.assertIsNotNone(
            physical,
            msg=f"bias_physical_followup module is missing: {MODULE_IMPORT_ERROR}",
        )
        function = getattr(physical, name, None)
        self.assertTrue(callable(function), msg=f"required function is missing: {name}")
        return function


def _station_frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "STN",
            "BIAS_none",
            "BIAS_fixed",
            "SIGNIFICANCE_CLASS",
            "MECHANISM",
            "mean_absdz_geom_km",
        ],
    )


class EquivalentDiagnosticsTests(PhysicalFunctionTestCase):
    def test_builds_complete_three_by_three_transition_with_exact_level_edges(self):
        """0.1·0.5°C 경계 또는 빈 전이칸 처리가 바뀌는 회귀를 잡는다."""
        build = self.function("build_equivalent_diagnostics")
        rows = []
        values = {
            "low": 0.10,
            "medium": 0.50,
            "high": 0.5001,
        }
        stn = 1
        for before_name, before in values.items():
            for after_name, after in values.items():
                rows.append(
                    (
                        stn,
                        before,
                        after,
                        "practically_equivalent",
                        "small_change",
                        stn / 1000,
                    )
                )
                stn += 1
        rows.append((999, 0.2, 0.8, "uncertain", "worsened_same_direction", 0.2))

        result = build(_station_frame(rows))
        stations = result["stations"]
        transition = result["transition"]

        self.assertEqual(len(stations), 9)
        self.assertEqual(len(transition), 9)
        self.assertEqual(transition["N"].tolist(), [1] * 9)
        self.assertAlmostEqual(transition["FRACTION_OF_EQUIVALENT"].sum(), 1.0)
        self.assertEqual(
            stations.loc[stations["STN"].eq("1"), "BIAS_NONE_LEVEL"].item(), "low"
        )
        self.assertEqual(
            stations.loc[stations["STN"].eq("5"), "BIAS_NONE_LEVEL"].item(), "medium"
        )
        self.assertEqual(
            stations.loc[stations["STN"].eq("9"), "BIAS_NONE_LEVEL"].item(), "high"
        )

    def test_keeps_before_and_after_group_summaries_separate(self):
        """보정 전 그룹과 보정 후 그룹을 같은 33개로 오인하는 회귀를 잡는다."""
        build = self.function("build_equivalent_diagnostics")
        frame = _station_frame(
            [
                (1, 0.05, 0.08, "practically_equivalent", "small_change", 0.01),
                (2, 0.09, 0.30, "practically_equivalent", "small_change", 0.02),
                (3, 0.20, 0.05, "practically_equivalent", "small_change", 0.03),
                (4, 0.40, 0.60, "practically_equivalent", "small_change", 0.04),
                (5, 0.70, 0.80, "practically_equivalent", "small_change", 0.05),
            ]
        )

        summary = build(frame)["group_summary"].set_index(["GROUPING", "LEVEL"])

        self.assertEqual(summary.loc[("before", "low"), "N"], 2)
        self.assertEqual(summary.loc[("before", "medium"), "N"], 2)
        self.assertEqual(summary.loc[("before", "high"), "N"], 1)
        self.assertEqual(summary.loc[("after", "low"), "N"], 2)
        self.assertEqual(summary.loc[("after", "medium"), "N"], 1)
        self.assertEqual(summary.loc[("after", "high"), "N"], 2)

    def test_rejects_duplicate_station_or_nonfinite_bias(self):
        """한 station 중복이나 NaN이 조용히 분모를 바꾸는 회귀를 잡는다."""
        build = self.function("build_equivalent_diagnostics")
        duplicate = _station_frame(
            [
                (1, 0.1, 0.1, "practically_equivalent", "small_change", 0.01),
                (1, 0.2, 0.2, "practically_equivalent", "small_change", 0.02),
            ]
        )
        nonfinite = duplicate.iloc[:1].copy()
        nonfinite.loc[:, "BIAS_fixed"] = np.nan

        with self.assertRaises(ValueError):
            build(duplicate)
        with self.assertRaises(ValueError):
            build(nonfinite)

    def test_normalizes_station_ids_before_duplicate_check_and_rejects_unknown_class(self):
        """숫자·문자 STN 중복 또는 비공식 분류가 정본 분모에 들어가는 회귀를 잡는다."""
        build = self.function("build_equivalent_diagnostics")
        duplicate_after_normalization = _station_frame(
            [
                (1, 0.1, 0.1, "practically_equivalent", "small_change", 0.01),
                ("1", 0.2, 0.2, "practically_equivalent", "small_change", 0.02),
            ]
        )
        unknown_class = _station_frame(
            [(2, 0.1, 0.1, "almost_equal", "small_change", 0.01)]
        )

        with self.assertRaises(ValueError):
            build(duplicate_after_normalization)
        with self.assertRaises(ValueError):
            build(unknown_class)


class WorseningDiagnosticsTests(PhysicalFunctionTestCase):
    def test_delta_z_bins_use_closed_upper_edges_at_50_100_200_300m(self):
        """정확히 50·100·200·300m인 station이 다음 구간으로 밀리는 회귀를 잡는다."""
        build = self.function("build_worsening_diagnostics")
        dz_m = [50.0, 50.0001, 100.0, 100.0001, 200.0, 200.0001, 300.0, 300.0001]
        frame = _station_frame(
            [
                (
                    index + 1,
                    0.2,
                    0.3,
                    "meaningful_worsening",
                    "same_direction_worsened",
                    value / 1000,
                )
                for index, value in enumerate(dz_m)
            ]
        )

        stations = build(frame)["stations"]

        self.assertEqual(
            stations["DZ_BIN"].tolist(),
            [
                "≤50m",
                "50–100m",
                "50–100m",
                "100–200m",
                "100–200m",
                "200–300m",
                "200–300m",
                ">300m",
            ],
        )

    def test_summarizes_mechanisms_and_continuous_worsening_magnitude(self):
        """악화 여부만 세고 악화 크기와 두 대수적 기전을 잃는 회귀를 잡는다."""
        build = self.function("build_worsening_diagnostics")
        frame = _station_frame(
            [
                (1, 0.1, 0.2, "meaningful_worsening", "same_direction_worsened", 0.02),
                (2, 0.2, 0.5, "meaningful_worsening", "same_direction_worsened", 0.03),
                (3, 0.2, -0.6, "meaningful_worsening", "overshoot_worsened", 0.04),
            ]
        )

        row = build(frame)["by_dz"].set_index("DZ_BIN").loc["≤50m"]

        self.assertEqual(row["N"], 3)
        self.assertEqual(row["SAME_DIRECTION_N"], 2)
        self.assertEqual(row["OVERSHOOT_N"], 1)
        self.assertAlmostEqual(row["MEDIAN_DELTA_ABS_BIAS"], 0.3)
        self.assertAlmostEqual(row["MAX_DELTA_ABS_BIAS"], 0.4)

    def test_rejects_unknown_worsening_mechanism(self):
        """승인되지 않은 기전 라벨이 기타로 조용히 합쳐지는 회귀를 잡는다."""
        build = self.function("build_worsening_diagnostics")
        frame = _station_frame(
            [(1, 0.1, 0.2, "meaningful_worsening", "mystery", 0.02)]
        )
        with self.assertRaises(ValueError):
            build(frame)


class RegressionAndClassificationTests(PhysicalFunctionTestCase):
    ELEVATION_M = np.array([0, 100, 200, 300, 400, 500], dtype=float)
    TEMPERATURE_C = np.array([1.0, 1.4, 1.9, 2.3, 2.8, 3.1], dtype=float)
    NEIGHBOR_IDS = ("101", "102", "103", "104", "105", "106")

    def test_ols_matches_hand_checked_slope_ci_r2_and_hc3(self):
        """고도 단위·t 자유도·HC3 sandwich 계산이 바뀌는 회귀를 잡는다."""
        calculate = self.function("ols_temperature_elevation_slope")

        result = calculate(
            self.ELEVATION_M,
            self.TEMPERATURE_C,
            neighbor_ids=self.NEIGHBOR_IDS,
        )

        self.assertEqual(result["n"], 6)
        self.assertAlmostEqual(result["slope_c_per_km"], 4.314285714285714, places=12)
        self.assertAlmostEqual(result["intercept_c"], 1.0047619047619047, places=12)
        self.assertAlmostEqual(result["ci95_low_c_per_km"], 3.965487368752654, places=10)
        self.assertAlmostEqual(result["ci95_high_c_per_km"], 4.663084059818773, places=10)
        self.assertAlmostEqual(result["r_squared"], 0.9966198003933853, places=12)
        self.assertAlmostEqual(
            result["hc3_standard_error_c_per_km"], 0.20954391943424765, places=12
        )
        self.assertAlmostEqual(result["hc3_ci95_low_c_per_km"], 3.7324985248485367, places=9)
        self.assertAlmostEqual(result["hc3_ci95_high_c_per_km"], 4.89607290372289, places=9)

    def test_ols_rejects_target_leakage_nonfinite_or_unidentified_slope(self):
        """target 포함·NaN·고도분산 0이 회귀에 들어가는 회귀를 잡는다."""
        calculate = self.function("ols_temperature_elevation_slope")
        ids = ["1", "2", "3", "4", "5", "6"]
        with self.assertRaises(ValueError):
            calculate(
                self.ELEVATION_M,
                self.TEMPERATURE_C,
                neighbor_ids=ids,
                target_id="3",
            )
        bad_temperature = self.TEMPERATURE_C.copy()
        bad_temperature[2] = np.nan
        with self.assertRaises(ValueError):
            calculate(
                self.ELEVATION_M,
                bad_temperature,
                neighbor_ids=self.NEIGHBOR_IDS,
            )
        with self.assertRaises(ValueError):
            calculate(
                np.ones(6) * 100,
                self.TEMPERATURE_C,
                neighbor_ids=self.NEIGHBOR_IDS,
            )

    def test_distance_weighted_slope_reports_effective_n_and_weighted_sd(self):
        """거리제곱 가중치·n_eff·가중 고도 SD 공식이 바뀌는 회귀를 잡는다."""
        calculate = self.function("weighted_temperature_elevation_slope")
        distances = np.array([1, 1, 2, 2, 4, 4], dtype=float)
        exact_linear = 10 + 2 * (self.ELEVATION_M / 1000)

        result = calculate(
            self.ELEVATION_M,
            exact_linear,
            distances_km=distances,
            neighbor_ids=self.NEIGHBOR_IDS,
        )

        self.assertAlmostEqual(result["slope_c_per_km"], 2.0, places=12)
        self.assertAlmostEqual(result["intercept_c"], 10.0, places=12)
        self.assertAlmostEqual(result["n_eff"], 3.230769230769231, places=12)
        self.assertAlmostEqual(result["weighted_elevation_sd_m"], 120.3029056824741, places=10)
        np.testing.assert_allclose(
            result["weights_normalized"],
            [0.380952380952381, 0.380952380952381, 0.0952380952380952,
             0.0952380952380952, 0.0238095238095238, 0.0238095238095238],
        )

    def test_weighted_slope_rejects_zero_distance_and_target_leakage(self):
        """무한 가중치나 target 자체가 공간 기울기에 섞이는 회귀를 잡는다."""
        calculate = self.function("weighted_temperature_elevation_slope")
        with self.assertRaises(ValueError):
            calculate(
                self.ELEVATION_M,
                self.TEMPERATURE_C,
                distances_km=[0, 1, 2, 3, 4, 5],
                neighbor_ids=self.NEIGHBOR_IDS,
            )
        with self.assertRaises(ValueError):
            calculate(
                self.ELEVATION_M,
                self.TEMPERATURE_C,
                distances_km=[1, 2, 3, 4, 5, 6],
                neighbor_ids=[1, 2, 3, 4, 5, 6],
                target_id=6,
            )

    def test_four_way_classification_uses_exact_quality_edges_and_both_signs(self):
        """n·span·weighted SD·n_eff의 exact 경계 또는 부호 합의가 완화되는 회귀를 잡는다."""
        classify = self.function("classify_spatial_temperature_structure")
        ols = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "ci95_low_c_per_km": 0.1,
            "ci95_high_c_per_km": 3.9,
        }
        weighted = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 0.01,
            "n_eff": 3.0,
            "weighted_elevation_sd_m": 50.0,
        }

        inversion = classify(ols, weighted)
        self.assertTrue(inversion["eligible"])
        self.assertEqual(inversion["classification"], "inversion_like")

        normal_ols = dict(ols, slope_c_per_km=-2.0, ci95_low_c_per_km=-3.9,
                          ci95_high_c_per_km=-0.1)
        normal_weighted = dict(weighted, slope_c_per_km=-0.01)
        self.assertEqual(
            classify(normal_ols, normal_weighted)["classification"], "normal_negative"
        )

        crossing = dict(ols, ci95_low_c_per_km=-0.1)
        self.assertEqual(classify(crossing, weighted)["classification"], "uncertain")
        disagreeing = dict(weighted, slope_c_per_km=-0.01)
        self.assertEqual(classify(ols, disagreeing)["classification"], "uncertain")

    def test_each_quality_value_below_threshold_is_ineligible(self):
        """품질조건 하나라도 미달인데 uncertain으로 섞이는 회귀를 잡는다."""
        classify = self.function("classify_spatial_temperature_structure")
        ols = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "ci95_low_c_per_km": 0.1,
            "ci95_high_c_per_km": 3.9,
        }
        weighted = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "n_eff": 3.0,
            "weighted_elevation_sd_m": 50.0,
        }
        cases = [
            (dict(ols, n=5), dict(weighted, n=5)),
            (
                dict(ols, elevation_span_m=99.999),
                dict(
                    weighted,
                    elevation_span_m=99.999,
                    weighted_elevation_sd_m=49.9995,
                ),
            ),
            (ols, dict(weighted, n_eff=2.999)),
            (ols, dict(weighted, weighted_elevation_sd_m=49.999)),
        ]
        for case_ols, case_weighted in cases:
            with self.subTest(case=(case_ols, case_weighted)):
                result = classify(case_ols, case_weighted)
                self.assertFalse(result["eligible"])
                self.assertEqual(result["classification"], "ineligible")

    def test_classification_rejects_results_from_different_neighbor_sets(self):
        """OLS와 가중회귀가 다른 이웃 수·고도범위인데 합쳐지는 회귀를 잡는다."""
        classify = self.function("classify_spatial_temperature_structure")
        ols = {
            "n": 6,
            "neighbor_fingerprint": "ols-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "ci95_low_c_per_km": 0.1,
            "ci95_high_c_per_km": 3.9,
        }
        weighted = {
            "n": 7,
            "neighbor_fingerprint": "weighted-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "n_eff": 3.0,
            "weighted_elevation_sd_m": 50.0,
        }
        with self.assertRaises(ValueError):
            classify(ols, weighted)
        with self.assertRaises(ValueError):
            classify(ols, dict(weighted, n=6, elevation_span_m=101.0))

    def test_regressions_emit_order_independent_neighbor_fingerprint(self):
        """같은 ID 집합의 순서 차이는 허용하되 다른 이웃 집합을 같은 것으로 보는 회귀를 잡는다."""
        ols = self.function("ols_temperature_elevation_slope")
        weighted = self.function("weighted_temperature_elevation_slope")
        distances = np.arange(1.0, 7.0)

        ols_result = ols(
            self.ELEVATION_M,
            self.TEMPERATURE_C,
            neighbor_ids=self.NEIGHBOR_IDS,
        )
        reversed_result = weighted(
            self.ELEVATION_M[::-1],
            self.TEMPERATURE_C[::-1],
            distances_km=distances[::-1],
            neighbor_ids=self.NEIGHBOR_IDS[::-1],
        )
        different_result = weighted(
            self.ELEVATION_M,
            self.TEMPERATURE_C,
            distances_km=distances,
            neighbor_ids=("101", "102", "103", "104", "105", "999"),
        )

        self.assertEqual(
            ols_result["neighbor_fingerprint"],
            reversed_result["neighbor_fingerprint"],
        )
        self.assertNotEqual(
            ols_result["neighbor_fingerprint"],
            different_result["neighbor_fingerprint"],
        )

    def test_classification_rejects_same_n_span_but_different_neighbor_fingerprint(self):
        """n·span이 같아도 실제 이웃 ID 집합이 다르면 두 회귀 결합을 거부한다."""
        classify = self.function("classify_spatial_temperature_structure")
        ols = {
            "n": 6,
            "neighbor_fingerprint": "set-A",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "ci95_low_c_per_km": 0.1,
            "ci95_high_c_per_km": 3.9,
        }
        weighted = {
            "n": 6,
            "neighbor_fingerprint": "set-B",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "n_eff": 3.0,
            "weighted_elevation_sd_m": 50.0,
        }

        with self.assertRaises(ValueError):
            classify(ols, weighted)

    def test_classification_rejects_impossible_weighted_geometry_metrics(self):
        """가중치에서 나올 수 없는 n_eff·가중 고도 SD가 상태판정에 들어가는 회귀를 잡는다."""
        classify = self.function("classify_spatial_temperature_structure")
        ols = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "ci95_low_c_per_km": 0.1,
            "ci95_high_c_per_km": 3.9,
        }
        weighted = {
            "n": 6,
            "neighbor_fingerprint": "same-neighbor-set",
            "elevation_span_m": 100.0,
            "slope_c_per_km": 2.0,
            "n_eff": 3.0,
            "weighted_elevation_sd_m": 50.0,
        }
        invalid_weighted = [
            dict(weighted, n_eff=0.999),
            dict(weighted, n_eff=6.001),
            dict(weighted, weighted_elevation_sd_m=-0.001),
            dict(weighted, weighted_elevation_sd_m=50.001),
        ]

        for result in invalid_weighted:
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    classify(ols, result)


class IdwAndConditionBiasTests(PhysicalFunctionTestCase):
    def test_exact_idw_none_uses_distance_power_two_and_zero_distance_shortcut(self):
        """none 보간의 거리제곱 역가중 또는 거리 0 불변식이 바뀌는 회귀를 잡는다."""
        calculate = self.function("exact_idw_none_weighted_mean")

        self.assertAlmostEqual(calculate([0.0, 10.0], [1.0, 2.0]), 2.0)
        self.assertAlmostEqual(calculate([7.0, 100.0], [0.0, 2.0]), 7.0)

    def test_exact_idw_stably_sorts_distance_and_accumulates_in_scalar_order(self):
        """signature ID 순서의 dot product가 원본 거리순 scalar 합산을 바꾸는 회귀를 잡는다."""
        calculate = self.function("exact_idw_none_weighted_mean")
        temperatures = [
            -100000000.0,
            -1.0,
            -1e-10,
            -10000000000.0,
            -100.0,
            -1000.0,
            -1000000.0,
            -100000000000000.0,
            100000.0,
            -10.0,
        ]
        distances = [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0, 9.0, 10.0]

        self.assertEqual(
            calculate(temperatures, distances),
            -4034476570675.554,
        )

    def test_exact_idw_zero_distance_tie_is_stable_and_tiny_distance_is_finite(self):
        """거리 0 tie 규칙이 비결정적이거나 극소거리 가중치가 overflow하는 회귀를 잡는다."""
        calculate = self.function("exact_idw_none_weighted_mean")

        self.assertEqual(calculate([7.0, 9.0], [0.0, 0.0]), 7.0)
        self.assertEqual(calculate([9.0, 7.0], [0.0, 0.0]), 9.0)
        self.assertEqual(calculate([5.0, 10.0], [1e-300, 1.0]), 5.0)

    def test_exact_idw_rejects_target_leakage_or_invalid_vectors(self):
        """target 포함·음의 거리·길이 불일치가 조용히 예측을 만드는 회귀를 잡는다."""
        calculate = self.function("exact_idw_none_weighted_mean")
        with self.assertRaises(ValueError):
            calculate([1.0, 2.0], [1.0, 2.0], neighbor_ids=[10, 11], target_id=11)
        with self.assertRaises(ValueError):
            calculate([1.0, 2.0], [1.0, -2.0])
        with self.assertRaises(ValueError):
            calculate([1.0], [1.0, 2.0])

    def test_condition_summary_averages_signed_errors_before_absolute_bias(self):
        """상태별 BIAS 효과가 시간별 절대오차 차이(MAE)로 바뀌는 회귀를 잡는다."""
        summarize = self.function("summarize_condition_bias")
        frame = pd.DataFrame(
            {
                "state": ["A", "A", "A", "B", "B"],
                "error_none": [-1.0, 1.0, 1.0, -1.0, -1.0],
                "error_fixed": [0.0, 2.0, 2.0, -0.5, -0.5],
            }
        )

        result = summarize(frame, group_by="state").set_index("state")

        self.assertAlmostEqual(result.loc["A", "BIAS_none"], 1 / 3)
        self.assertAlmostEqual(result.loc["A", "BIAS_fixed"], 4 / 3)
        self.assertAlmostEqual(result.loc["A", "DELTA_ABS_BIAS"], 1.0)
        self.assertAlmostEqual(result.loc["A", "DELTA_MAE"], 1 / 3)
        self.assertNotAlmostEqual(
            result.loc["A", "DELTA_ABS_BIAS"], result.loc["A", "DELTA_MAE"]
        )
        self.assertAlmostEqual(result.loc["B", "DELTA_ABS_BIAS"], -0.5)

    def test_condition_summary_rejects_missing_groups_or_nonfinite_errors(self):
        """결측 상태·오차가 분모에서 묵시적으로 빠지는 회귀를 잡는다."""
        summarize = self.function("summarize_condition_bias")
        missing_group = pd.DataFrame(
            {"state": ["A", None], "error_none": [0.0, 1.0], "error_fixed": [0.0, 1.0]}
        )
        nonfinite = pd.DataFrame(
            {"state": ["A"], "error_none": [np.nan], "error_fixed": [0.0]}
        )
        with self.assertRaises(ValueError):
            summarize(missing_group, group_by="state")
        with self.assertRaises(ValueError):
            summarize(nonfinite, group_by="state")


class EffectSizeAndMultipleTestingTests(PhysicalFunctionTestCase):
    def test_rank_biserial_has_declared_positive_orientation_and_tie_handling(self):
        """그룹 방향이 뒤집히거나 tie가 승패로 계산되는 회귀를 잡는다."""
        calculate = self.function("rank_biserial_effect")

        self.assertAlmostEqual(calculate([3, 4], [1, 2]), 1.0)
        self.assertAlmostEqual(calculate([1, 2], [3, 4]), -1.0)
        self.assertAlmostEqual(calculate([1, 2], [1, 2]), 0.0)

    def test_bh_matches_hand_calculation_and_rejects_invalid_pvalues(self):
        """BH 정렬 복원·역누적 최솟값·범위 검증이 바뀌는 회귀를 잡는다."""
        adjust = self.function("benjamini_hochberg")

        np.testing.assert_allclose(
            adjust([0.01, 0.04, 0.03, 0.002]),
            [0.02, 0.04, 0.04, 0.008],
        )
        with self.assertRaises(ValueError):
            adjust([])
        with self.assertRaises(ValueError):
            adjust([0.1, np.nan])
        with self.assertRaises(ValueError):
            adjust([0.1, 1.1])


class ProductionInvariantTests(PhysicalFunctionTestCase):
    @classmethod
    def setUpClass(cls):
        if not PRODUCTION_STATION_RESULTS.exists():
            raise unittest.SkipTest(f"production truth missing: {PRODUCTION_STATION_RESULTS}")
        cls.station_results = pd.read_csv(PRODUCTION_STATION_RESULTS, encoding="utf-8-sig")

    def test_production_official_counts_and_equivalent_transitions_are_exact(self):
        """정본 529/221/109/115/84와 115 전이 불변식이 달라지는 회귀를 잡는다."""
        build = self.function("build_equivalent_diagnostics")
        counts = self.station_results["SIGNIFICANCE_CLASS"].value_counts().to_dict()
        self.assertEqual(
            counts,
            {
                "meaningful_improvement": 221,
                "practically_equivalent": 115,
                "meaningful_worsening": 109,
                "uncertain": 84,
            },
        )
        self.assertEqual(len(self.station_results), 529)

        result = build(self.station_results)
        stations = result["stations"]
        transition = result["transition"].pivot(
            index="BIAS_NONE_LEVEL", columns="BIAS_FIXED_LEVEL", values="N"
        )
        expected = pd.DataFrame(
            [[12, 5, 0], [8, 54, 3], [0, 3, 30]],
            index=["low", "medium", "high"],
            columns=["low", "medium", "high"],
        )
        pd.testing.assert_frame_equal(
            transition, expected, check_names=False, check_like=True
        )

        summary = result["group_summary"].set_index(["GROUPING", "LEVEL"])
        self.assertEqual(
            [summary.loc[("before", level), "N"] for level in ["low", "medium", "high"]],
            [17, 65, 33],
        )
        self.assertEqual(
            [summary.loc[("after", level), "N"] for level in ["low", "medium", "high"]],
            [20, 62, 33],
        )
        self.assertEqual(
            int(
                (
                    stations["BIAS_NONE_LEVEL"].eq("high")
                    & stations["BIAS_FIXED_LEVEL"].eq("high")
                ).sum()
            ),
            30,
        )
        after_high = stations.loc[stations["BIAS_FIXED_LEVEL"].eq("high"), "BIAS_fixed"]
        self.assertEqual(int((after_high > 0).sum()), 19)
        self.assertEqual(int((after_high < 0).sum()), 14)

    def test_production_worsening_mechanisms_and_delta_z_bins_are_exact(self):
        """악화 109개의 91/18 기전과 ΔZ 구간 정본이 달라지는 회귀를 잡는다."""
        build = self.function("build_worsening_diagnostics")

        summary = build(self.station_results)["by_dz"].set_index("DZ_BIN")

        self.assertEqual(summary["N"].tolist(), [72, 27, 7, 2, 1])
        self.assertEqual(summary["SAME_DIRECTION_N"].tolist(), [68, 19, 3, 0, 1])
        self.assertEqual(summary["OVERSHOOT_N"].tolist(), [4, 8, 4, 2, 0])
        self.assertEqual(int(summary["N"].sum()), 109)
        self.assertEqual(int(summary["SAME_DIRECTION_N"].sum()), 91)
        self.assertEqual(int(summary["OVERSHOOT_N"].sum()), 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
