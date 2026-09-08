"""발표 피드백 반영 BIAS 분석의 합성자료 통합 테스트."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_bias_feedback import (
    aggregate_pairs_to_daily,
    build_temporal_summary,
    build_physical_checks,
    build_daily_matrices,
    create_temporal_physical_figure,
    create_dz_before_after_figure,
    select_cases,
    run_sensitivity,
    run_station_inference,
    summarize_station_bootstrap,
    validate_annual_against_existing,
    write_feedback_report,
)


class DailyInputTests(unittest.TestCase):
    def test_aggregates_paired_hours_without_averaging_daily_means(self):
        """유효 시간 수가 다른 날을 같은 가중치로 평균하는 회귀를 잡는다."""
        pairs = pd.DataFrame(
            {
                "TM": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-02 00:00"]
                ),
                "STN": ["101", "101", "101"],
                "error_none": [1.0, 3.0, 10.0],
                "error_fixed": [0.0, 2.0, 4.0],
            }
        )

        daily = aggregate_pairs_to_daily(pairs)

        self.assertEqual(daily["count"].tolist(), [2, 1])
        self.assertEqual(daily["sum_none"].tolist(), [4.0, 10.0])
        self.assertEqual(daily["sum_fixed"].tolist(), [2.0, 4.0])

    def test_builds_station_by_calendar_day_matrices(self):
        """station 또는 날짜 인덱스가 바뀌어 오차가 다른 칸에 들어가는 회귀를 잡는다."""
        daily = pd.DataFrame(
            {
                "STN": ["102", "101", "101"],
                "DATE": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]),
                "sum_none": [8.0, 2.0, 6.0],
                "sum_fixed": [4.0, 0.0, 2.0],
                "count": [2, 1, 3],
            }
        )
        dates = pd.date_range("2024-01-01", "2024-01-03", freq="D")

        matrices = build_daily_matrices(daily, ["101", "102"], dates)

        np.testing.assert_array_equal(
            matrices["sum_none"], np.array([[2.0, 0.0, 6.0], [0.0, 8.0, 0.0]])
        )
        np.testing.assert_array_equal(
            matrices["count"], np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
        )

    def test_recomputes_existing_annual_bias_exactly(self):
        """일별 축약이 기존 연간 BIAS 정본을 바꾸는 회귀를 잡는다."""
        matrices = {
            "sum_none": np.array([[14.0, 0.0], [-8.0, -4.0]]),
            "sum_fixed": np.array([[4.0, 0.0], [-2.0, -2.0]]),
            "count": np.array([[6.0, 0.0], [2.0, 2.0]]),
        }
        existing = pd.DataFrame(
            {
                "STN": ["101", "102"],
                "BIAS_none": [14 / 6, -3.0],
                "BIAS_fixed": [4 / 6, -1.0],
            }
        )

        residuals = validate_annual_against_existing(matrices, existing)

        self.assertLessEqual(residuals["max_abs_bias_none"], 1e-12)
        self.assertLessEqual(residuals["max_abs_bias_fixed"], 1e-12)


class InferenceAndPresentationTests(unittest.TestCase):
    @staticmethod
    def _four_station_results():
        stations = pd.DataFrame(
            {
                "STN": ["101", "102", "103", "104"],
                "지점명": ["개선", "악화", "동등", "불확실"],
                "BIAS_none": [0.5, 0.0, 0.01, 0.2],
                "BIAS_fixed": [0.2, 0.3, 0.01, 0.2],
                "DELTA_ABS_BIAS": [-0.3, 0.3, 0.0, 0.0],
                "mean_dz_geom_km": [0.2, -0.2, 0.01, 0.05],
            }
        )
        narrow = np.linspace(-0.01, 0.01, 400)
        wide = np.linspace(-0.35, 0.35, 400)
        bootstrap = np.column_stack(
            [-0.3 + narrow, 0.3 + narrow, narrow, wide]
        )
        return summarize_station_bootstrap(stations, bootstrap, delta=0.1)

    def test_inference_produces_four_exclusive_classes(self):
        """실질 경계와 불확실성을 결합한 네 범주 중 하나가 누락되는 회귀를 잡는다."""
        result = self._four_station_results()
        self.assertEqual(
            result["SIGNIFICANCE_CLASS"].tolist(),
            [
                "meaningful_improvement",
                "meaningful_worsening",
                "practically_equivalent",
                "uncertain",
            ],
        )
        self.assertTrue((result["CI95_LOW"] <= result["CI95_HIGH"]).all())

    def test_insufficient_annual_coverage_is_kept_but_marked_uncertain(self):
        """한 달 자료만 있는 station을 연간 유의미한 변화로 과대해석하는 회귀를 잡는다."""
        stations = pd.DataFrame(
            {
                "STN": ["101"],
                "DELTA_ABS_BIAS": [-0.5],
                "COVERAGE_ELIGIBLE": [False],
            }
        )
        bootstrap = -0.5 + np.linspace(-0.01, 0.01, 400)[:, None]

        result = summarize_station_bootstrap(stations, bootstrap, delta=0.1)

        self.assertEqual(result.loc[0, "SIGNIFICANCE_CLASS"], "uncertain")
        self.assertEqual(result.loc[0, "q_improve"], 1.0)

    def test_dz_before_after_figure_uses_identical_axes(self):
        """보정 전후 패널의 축이 달라 시각적으로 효과가 과장되는 회귀를 잡는다."""
        result = self._four_station_results()
        fig, axes = create_dz_before_after_figure(result)
        try:
            self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())
            self.assertEqual(axes[0].get_ylim(), axes[1].get_ylim())
            self.assertEqual(axes[0].get_xlabel(), "평균 ΔZ (m)")
            self.assertEqual(axes[1].get_xlabel(), "평균 ΔZ (m)")
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_temporal_summary_keeps_median_and_adds_mean_for_sensitivity_check(self):
        """일부 극단 station 때문에 평균이 달라져도 중앙값과 평균을 함께 확인한다."""
        monthly = pd.DataFrame(
            {
                "MONTH": [1, 1, 1],
                "BIAS_none": [0.0, 0.0, 9.0],
                "BIAS_fixed": [0.0, 0.0, 6.0],
                "DELTA_ABS_BIAS": [0.0, 0.0, -3.0],
            }
        )
        diurnal = monthly.rename(columns={"MONTH": "HOUR"})

        summary = build_temporal_summary(monthly, diurnal)
        month = summary.loc[summary["PERIOD_TYPE"].eq("month")].iloc[0]

        self.assertEqual(month["BIAS_none_median"], 0.0)
        self.assertEqual(month["BIAS_none_mean"], 3.0)
        self.assertEqual(month["DELTA_ABS_BIAS_median"], 0.0)
        self.assertEqual(month["DELTA_ABS_BIAS_mean"], -1.0)

    def test_temporal_figure_has_no_day_night_shading_or_interpretive_labels(self):
        """시간대 그래프에 원인 해석 문구와 배경 음영이 다시 들어오는 회귀를 잡는다."""
        rows = []
        for period_type, periods in [("month", [1, 2]), ("hour", [0, 12, 23])]:
            for period in periods:
                rows.append(
                    {
                        "PERIOD_TYPE": period_type,
                        "PERIOD": period,
                        "BIAS_none_median": -0.1,
                        "BIAS_fixed_median": 0.0,
                        "DELTA_ABS_BIAS_median": -0.1,
                    }
                )
        fig, axes = create_temporal_physical_figure(pd.DataFrame(rows))
        try:
            hour_axes = axes[:, 1]
            self.assertTrue(all(len(ax.patches) == 0 for ax in hour_axes))
            labels = {text.get_text() for ax in hour_axes for text in ax.texts}
            self.assertNotIn("야간 안정층 가능", labels)
            self.assertNotIn("주간 혼합", labels)
            ylabels = [axis.get_ylabel() for axis in axes.flat]
            self.assertTrue(all("station" not in label for label in ylabels))
            self.assertTrue(all("관측소" in label for label in ylabels))
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_primary_sensitivity_cell_reuses_primary_bootstrap(self):
        """동일한 0.1°C·7일 설정이 seed 차이로 기본 분류와 달라지는 회귀를 잡는다."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        matrices = {
            "sum_none": np.vstack([np.full(366, 0.5), np.zeros(366)]),
            "sum_fixed": np.vstack([np.full(366, 0.2), np.full(366, 0.3)]),
            "count": np.ones((2, 366)),
        }
        stations = pd.DataFrame(
            {
                "STN": ["101", "102"],
                "BIAS_none": [0.5, 0.0],
                "BIAS_fixed": [0.2, 0.3],
                "DELTA_ABS_BIAS": [-0.3, 0.3],
            }
        )
        primary, primary_boot = run_station_inference(
            stations, matrices, dates, block_days=7, n_boot=50,
            delta=0.1, seed=20260812
        )

        sensitivity, _ = run_sensitivity(
            stations,
            matrices,
            dates,
            n_boot=50,
            primary_labels=primary["SIGNIFICANCE_CLASS"].to_numpy(object),
            primary_bootstrap=primary_boot,
        )

        primary_row = sensitivity.loc[
            sensitivity["delta_c"].eq(0.1) & sensitivity["block_days"].eq(7)
        ].iloc[0]
        self.assertEqual(primary_row["agreement_with_primary"], 1.0)

    def test_physical_checks_compare_terrain_inside_delta_z_bins(self):
        """ΔZ가 다른 지형군을 그대로 비교해 지형 효과로 오해하는 회귀를 잡는다."""
        result = self._four_station_results()
        result["mean_absdz_geom_km"] = [0.02, 0.02, 0.15, 0.15]
        result["terrain_group"] = ["평지", "산악", "평지", "산악"]
        checks = build_physical_checks(result)
        self.assertIn("terrain_within_delta_z", set(checks["CHECK"]))

    def test_case_selection_follows_prespecified_rules(self):
        """눈에 띄는 station을 임의로 골라 사례 선택 편향이 생기는 회귀를 잡는다."""
        result = self._four_station_results()
        cases = select_cases(result)
        self.assertEqual(cases["STN"].tolist(), ["101", "102", "103", "104"])

    def test_report_gives_physical_meaning_and_evidence_limit_for_each_analysis(self):
        """그래프 수치만 나열하고 물리 과정·한계를 빠뜨리는 회귀를 잡는다."""
        result = self._four_station_results()
        temporal = pd.DataFrame(
            {
                "PERIOD_TYPE": ["hour"],
                "PERIOD": [12],
                "BIAS_none_median": [0.2],
                "BIAS_fixed_median": [0.1],
                "DELTA_ABS_BIAS_median": [-0.1],
            }
        )
        sensitivity = pd.DataFrame(
            {
                "delta_c": [0.1],
                "block_days": [7],
                "agreement_with_primary": [1.0],
                "meaningful_improvement": [1],
                "meaningful_worsening": [1],
                "practically_equivalent": [1],
                "uncertain": [1],
            }
        )
        physical = pd.DataFrame(
            {
                "CHECK": ["absolute_delta_z"],
                "GROUP": ["100–300m"],
                "N": [2],
                "MEDIAN_DELTA_ABS_BIAS": [-0.2],
                "FRAC_meaningful_improvement": [0.5],
                "FRAC_meaningful_worsening": [0.0],
                "FRAC_practically_equivalent": [0.5],
                "FRAC_uncertain": [0.0],
            }
        )
        cases = select_cases(result)
        output = Path(".omc/test_bias_feedback_output")
        output.mkdir(parents=True, exist_ok=True)
        path = write_feedback_report(
            result, temporal, sensitivity, physical, cases, output
        )
        text = Path(path).read_text(encoding="utf-8")
        Path(path).unlink()
        output.rmdir()
        for heading in ["관찰", "수치 근거", "물리적 의미", "검증 범위와 한계"]:
            self.assertIn(heading, text)
        self.assertNotIn("dz_geom", text)
        self.assertIn("보정 전 BIAS", text)
        self.assertIn("유의미한 개선 비율", text)
        self.assertNotIn("SIGNIFICANCE_CLASS", text)
        self.assertNotIn("MEDIAN_DELTA_ABS_BIAS", text)


if __name__ == "__main__":
    unittest.main()
