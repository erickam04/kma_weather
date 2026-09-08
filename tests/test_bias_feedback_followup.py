"""BIAS 후속 분석의 전이표와 ΔZ 분류표 계약 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
from unittest.mock import patch

from analysis import analyze_bias_feedback_followup as followup
from analysis import bias_data
from analysis.analyze_bias_feedback_followup import (
    CLASS_ORDER,
    EQUIVALENT_LEVEL_ORDER,
    SOURCE_STATION_RESULTS,
    assign_calendar_quarter,
    bootstrap_quarterly_effects,
    build_delta_z_category_rates,
    build_equivalent_bias_transition,
    build_leave_one_quarter_out,
    build_quarterly_block_sensitivity,
    build_quarterly_point_effects,
    build_quarterly_reversals,
    classify_ci_position,
    plot_quarterly_effect,
    plot_quarterly_heatmap,
    plot_equivalent_before_after,
    plot_leave_one_quarter_out,
    summarize_quarterly_network,
)
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.api.types import is_numeric_dtype


# 새 checkout에서도 로컬 테스트용 임시 폴더를 준비한다.
(Path(".omc") / "test_tmp").mkdir(parents=True, exist_ok=True)


class BiasFeedbackFollowupTableTests(unittest.TestCase):
    def test_bias_level_transition_uses_absolute_before_and_after(self):
        """절대 BIAS의 0.1·0.5 경계와 zero cell이 바뀌는 회귀를 잡는다."""
        station_results = pd.DataFrame(
            {
                "BIAS_none": [0.1, -0.1, -0.100001, 0.5, 0.500001, 0.2, 0.8],
                "BIAS_fixed": [0.1, 0.5, 0.1, -0.5, 0.500001, 0.1, 0.4],
                "SIGNIFICANCE_CLASS": [
                    "practically_equivalent",
                    "practically_equivalent",
                    "practically_equivalent",
                    "practically_equivalent",
                    "practically_equivalent",
                    "meaningful_improvement",
                    "uncertain",
                ],
            }
        )

        table = build_equivalent_bias_transition(station_results)

        self.assertEqual(len(table), 9)
        self.assertEqual(table["N"].sum(), 5)
        self.assertEqual(table["FRACTION_OF_EQUIVALENT"].sum(), 1.0)
        equivalent_count = 5
        self.assertTrue(
            (table["FRACTION_OF_EQUIVALENT"] == table["N"] / equivalent_count).all()
        )
        wrong_whole_input_denominator = len(station_results)
        self.assertTrue(
            (
                table.loc[table["N"].gt(0), "FRACTION_OF_EQUIVALENT"]
                != table.loc[table["N"].gt(0), "N"] / wrong_whole_input_denominator
            ).all()
        )
        self.assertTrue(is_numeric_dtype(table["N"]))
        self.assertTrue(is_numeric_dtype(table["FRACTION_OF_EQUIVALENT"]))
        expected = {
            ("≤0.1°C", "≤0.1°C"): 1,
            ("≤0.1°C", "(0.1, 0.5]°C"): 1,
            ("≤0.1°C", ">0.5°C"): 0,
            ("(0.1, 0.5]°C", "≤0.1°C"): 1,
            ("(0.1, 0.5]°C", "(0.1, 0.5]°C"): 1,
            ("(0.1, 0.5]°C", ">0.5°C"): 0,
            (">0.5°C", "≤0.1°C"): 0,
            (">0.5°C", "(0.1, 0.5]°C"): 0,
            (">0.5°C", ">0.5°C"): 1,
        }
        actual = {
            (row.BEFORE_ABS_BIAS_LEVEL, row.AFTER_ABS_BIAS_LEVEL): row.N
            for row in table.itertuples(index=False)
        }
        self.assertEqual(actual, expected)
        self.assertEqual(EQUIVALENT_LEVEL_ORDER, ["≤0.1°C", "(0.1, 0.5]°C", ">0.5°C"])

    def test_delta_z_category_rates_include_all_four_classes(self):
        """ΔZ 50·100·200·300m 경계와 20개 zero-preserving 행을 검증한다."""
        station_results = pd.DataFrame(
            {
                "mean_absdz_geom_km": [0.040, 0.050, 0.100, 0.200, 0.300, 0.300001],
                "SIGNIFICANCE_CLASS": [
                    "meaningful_worsening",
                    "meaningful_improvement",
                    "meaningful_worsening",
                    "practically_equivalent",
                    "uncertain",
                    "meaningful_improvement",
                ],
            }
        )

        table = build_delta_z_category_rates(station_results)

        self.assertEqual(len(table), 20)
        self.assertEqual(table["N"].sum(), 6)
        self.assertEqual(set(table["SIGNIFICANCE_CLASS"]), set(CLASS_ORDER))
        self.assertTrue(is_numeric_dtype(table["N"]))
        self.assertTrue(is_numeric_dtype(table["BIN_TOTAL"]))
        self.assertTrue(is_numeric_dtype(table["FRACTION_WITHIN_BIN"]))
        expected_bin_totals = {"≤50m": 2, "50–100m": 1, "100–200m": 1, "200–300m": 1, ">300m": 1}
        self.assertEqual(table.groupby("DELTA_Z_BIN", observed=True)["N"].sum().to_dict(), expected_bin_totals)
        self.assertEqual(table.groupby("DELTA_Z_BIN", observed=True)["BIN_TOTAL"].first().to_dict(), expected_bin_totals)
        self.assertTrue(
            (table["FRACTION_WITHIN_BIN"] == table["N"] / table["BIN_TOTAL"]).all()
        )
        nonempty = table.loc[table["BIN_TOTAL"].gt(0)]
        self.assertTrue(
            (nonempty.groupby("DELTA_Z_BIN", observed=True)["FRACTION_WITHIN_BIN"].sum() == 1.0).all()
        )
        expected_nonzero = {
            ("≤50m", "meaningful_improvement"),
            ("≤50m", "meaningful_worsening"),
            ("50–100m", "meaningful_worsening"),
            ("100–200m", "practically_equivalent"),
            ("200–300m", "uncertain"),
            (">300m", "meaningful_improvement"),
        }
        observed_nonzero = {
            (row.DELTA_Z_BIN, row.SIGNIFICANCE_CLASS)
            for row in table.itertuples(index=False)
            if row.N == 1
        }
        self.assertEqual(observed_nonzero, expected_nonzero)
        first_bin = table.loc[table["DELTA_Z_BIN"].eq("≤50m")]
        self.assertEqual(first_bin.loc[first_bin["SIGNIFICANCE_CLASS"].eq("meaningful_improvement"), "FRACTION_WITHIN_BIN"].iloc[0], 0.5)
        self.assertEqual(first_bin.loc[first_bin["SIGNIFICANCE_CLASS"].eq("meaningful_worsening"), "FRACTION_WITHIN_BIN"].iloc[0], 0.5)
        self.assertTrue((table.loc[table["N"].eq(0), "FRACTION_WITHIN_BIN"] == 0.0).all())

    def test_null_significance_class_is_rejected_before_bin_totals_are_built(self):
        """공식 네 분류 밖의 null을 bin 분모에만 남기는 회귀를 잡는다."""
        station_results = pd.DataFrame(
            {
                "BIAS_none": [0.1],
                "BIAS_fixed": [0.1],
                "mean_absdz_geom_km": [0.05],
                "SIGNIFICANCE_CLASS": [None],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing.*SIGNIFICANCE_CLASS"):
            build_equivalent_bias_transition(station_results)
        with self.assertRaisesRegex(ValueError, "missing.*SIGNIFICANCE_CLASS"):
            build_delta_z_category_rates(station_results)

    @unittest.skipUnless(SOURCE_STATION_RESULTS.is_file(), "로컬 BIAS 정본 CSV가 필요합니다")
    def test_actual_feedback_counts_are_reproduced(self):
        """정본 529개 결과의 승인된 전이·ΔZ 셀 수가 달라지는 회귀를 잡는다."""
        source = pd.read_csv(SOURCE_STATION_RESULTS, encoding="utf-8-sig", dtype={"STN": str})

        transition = build_equivalent_bias_transition(source)
        rates = build_delta_z_category_rates(source)

        self.assertEqual(len(source), 529)
        self.assertEqual(int(source["SIGNIFICANCE_CLASS"].eq("practically_equivalent").sum()), 115)
        expected_transition = [12, 5, 0, 8, 54, 3, 0, 3, 30]
        self.assertEqual(transition["N"].tolist(), expected_transition)
        expected_rates = [
            70, 72, 115, 34,
            47, 27, 0, 25,
            41, 7, 0, 16,
            25, 2, 0, 5,
            38, 1, 0, 4,
        ]
        self.assertEqual(rates["N"].tolist(), expected_rates)
        self.assertEqual(int(rates["N"].sum()), 529)

    def test_equivalent_plot_writes_only_the_requested_staging_path(self):
        """전이 그림이 최종 게시 폴더나 부수 산출물을 만드는 회귀를 잡는다."""
        station_results = pd.DataFrame(
            {
                "BIAS_none": [0.1, 0.2, 0.7],
                "BIAS_fixed": [0.1, 0.3, 0.6],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"] * 3,
            }
        )
        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            output_path = Path(temp_dir) / "bias_equivalent_before_after.png"

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = plot_equivalent_before_after(station_results, output_path)

            self.assertEqual(Path(result), output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual({path.name for path in Path(temp_dir).iterdir()}, {output_path.name})
            self.assertFalse(any("Glyph" in str(item.message) for item in caught))

    def test_plot_creates_a_missing_parent_directory(self):
        """호출자가 staging 하위 폴더를 미리 만들지 않아도 PNG를 저장해야 한다."""
        import matplotlib.pyplot as plt

        station_results = pd.DataFrame(
            {
                "BIAS_none": [0.1],
                "BIAS_fixed": [0.1],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
            }
        )
        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            output_path = Path(temp_dir) / "nested" / "bias_equivalent_before_after.png"
            try:
                result = plot_equivalent_before_after(station_results, output_path)

                self.assertEqual(Path(result), output_path)
                self.assertTrue(output_path.is_file())
            finally:
                plt.close("all")

    def test_plot_closes_figure_when_savefig_raises(self):
        """파일 저장 예외가 열린 matplotlib figure를 남기는 회귀를 잡는다."""
        import matplotlib.pyplot as plt

        station_results = pd.DataFrame(
            {
                "BIAS_none": [0.1],
                "BIAS_fixed": [0.1],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
            }
        )
        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            output_path = Path(temp_dir) / "bias_equivalent_before_after.png"
            try:
                with patch("matplotlib.figure.Figure.savefig", side_effect=RuntimeError("save failed")):
                    with self.assertRaisesRegex(RuntimeError, "save failed"):
                        plot_equivalent_before_after(station_results, output_path)
                self.assertEqual(plt.get_fignums(), [])
            finally:
                plt.close("all")


class BiasFeedbackFollowupQuarterTests(unittest.TestCase):
    @staticmethod
    def _daily_for_dates(stn, dates, sum_none=1.0, sum_fixed=0.5, count=1):
        dates = pd.DatetimeIndex(dates)
        return pd.DataFrame(
            {
                "STN": [str(stn)] * len(dates),
                "DATE": dates,
                "sum_none": np.broadcast_to(sum_none, len(dates)),
                "sum_fixed": np.broadcast_to(sum_fixed, len(dates)),
                "count": np.broadcast_to(count, len(dates)),
            }
        )

    def test_quarter_assignment_is_contiguous_complete_and_calendar_labeled(self):
        """2024년 366일이 빈틈·중복 없이 달력상 Q1–Q4에 배정되어야 한다."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")

        assigned = pd.Series(assign_calendar_quarter(dates), index=dates)

        self.assertEqual(
            assigned.value_counts(sort=False).to_dict(),
            {"Q1": 91, "Q2": 91, "Q3": 92, "Q4": 92},
        )
        expected = {
            "2024-03-31": "Q1",
            "2024-04-01": "Q2",
            "2024-06-30": "Q2",
            "2024-07-01": "Q3",
            "2024-09-30": "Q3",
            "2024-10-01": "Q4",
            "2024-12-31": "Q4",
        }
        for date, quarter in expected.items():
            self.assertEqual(assigned.loc[date], quarter)
        with self.assertRaisesRegex(ValueError, "2024"):
            assign_calendar_quarter(["2023-12-31"])

    def test_canonical_station_universe_keeps_station_missing_from_daily(self):
        """daily에 행이 전혀 없는 canonical station을 골격에서 삭제하는 회귀를 잡는다."""
        daily = self._daily_for_dates("101", ["2024-01-01"], count=0)

        result = build_quarterly_point_effects(daily, stations=["101", "999"])

        self.assertEqual(len(result), 8)
        missing = result.loc[result["STN"].eq("999")]
        self.assertEqual(missing["QUARTER"].tolist(), ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(set(missing["COVERAGE_REASON"]), {"no_data"})
        self.assertTrue(
            missing[["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]].isna().all().all()
        )

    def test_empty_daily_with_explicit_stations_builds_no_data_point_and_bootstrap_skeleton(self):
        """전체 empty daily에서도 명시 station×4 골격과 NA CI를 만들어야 한다."""
        empty = pd.DataFrame(columns=["STN", "DATE", "sum_none", "sum_fixed", "count"])

        point = build_quarterly_point_effects(empty, stations=[1, "002"])
        boot = bootstrap_quarterly_effects(
            empty, n_boot=20, block_days=7, seed=3, stations=[1, "002"]
        )

        expected_point_columns = [
            "STN",
            "QUARTER",
            "CALENDAR_DAYS",
            "MIN_VALID_DAYS",
            "VALID_DAYS",
            "COVERAGE",
            "MONTHS_PRESENT",
            "PAIRED_HOURS",
            "COVERAGE_ELIGIBLE",
            "COVERAGE_REASON",
            "BIAS_none",
            "BIAS_fixed",
            "DELTA_ABS_BIAS",
            "POINT_CLASS",
        ]
        expected_boot_columns = expected_point_columns + [
            "CI95_LOW",
            "CI95_HIGH",
            "CI_POSITION",
            "BOOT_BLOCK_DAYS",
            "BOOT_REPLICATES",
            "BOOT_SEED",
        ]
        self.assertEqual(point.columns.tolist(), expected_point_columns)
        self.assertEqual(boot.columns.tolist(), expected_boot_columns)
        self.assertEqual(len(point), 8)
        self.assertEqual(point["STN"].unique().tolist(), ["1", "002"])
        self.assertEqual(set(point["COVERAGE_REASON"]), {"no_data"})
        self.assertTrue(point["DELTA_ABS_BIAS"].isna().all())
        self.assertEqual(len(boot), 8)
        self.assertTrue(boot[["DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"]].isna().all().all())
        with self.assertRaisesRegex(ValueError, "stations.*daily"):
            build_quarterly_point_effects(empty)

    def test_bootstrap_parameters_are_validated_even_when_no_quarter_is_eligible(self):
        """eligible 0개일 때 n_boot·block_days 0을 조용히 통과시키는 회귀를 잡는다."""
        empty = pd.DataFrame(columns=["STN", "DATE", "sum_none", "sum_fixed", "count"])

        with self.assertRaisesRegex(ValueError, "n_boot.*positive integer"):
            bootstrap_quarterly_effects(
                empty, n_boot=0, block_days=7, seed=1, stations=["101"]
            )
        with self.assertRaisesRegex(ValueError, "block_days.*positive integer"):
            bootstrap_quarterly_effects(
                empty, n_boot=20, block_days=0, seed=1, stations=["101"]
            )
        with self.assertRaisesRegex(ValueError, "n_boot.*positive integer"):
            build_quarterly_block_sensitivity(
                n_boot=0,
                results_by_block={
                    3: pd.DataFrame(),
                    7: pd.DataFrame(),
                    14: pd.DataFrame(),
                },
            )
        with self.assertRaisesRegex(ValueError, "n_boot.*positive integer"):
            bootstrap_quarterly_effects(
                empty, n_boot=True, block_days=7, seed=1, stations=["101"]
            )
        with self.assertRaisesRegex(ValueError, "block_days.*positive integer"):
            bootstrap_quarterly_effects(
                empty, n_boot=20, block_days=True, seed=1, stations=["101"]
            )

    def test_empty_quarterly_reversals_have_fixed_schema_and_zero_eligible_annual_rows(self):
        """empty input이 열 없는 표나 KeyError로 바뀌는 회귀를 잡는다."""
        empty_quarterly = pd.DataFrame(
            columns=["STN", "QUARTER", "COVERAGE_ELIGIBLE", "DELTA_ABS_BIAS"]
        )
        empty_annual = pd.DataFrame(columns=["STN", "DELTA_ABS_BIAS", "SIGNIFICANCE_CLASS"])
        expected_columns = [
            "STN",
            "N_ELIGIBLE_QUARTERS",
            "MIN_ELIGIBLE_QUARTERS",
            "REVERSAL_STATUS",
            "RAW_SIGN_CROSSING",
            "PRACTICAL_CROSSING",
            "ANNUAL_QUARTER_POINT_CLASS_MISMATCH",
            "ANNUAL_DELTA_ABS_BIAS",
            "ANNUAL_POINT_CLASS",
            "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE",
        ]

        empty_result = build_quarterly_reversals(empty_quarterly, empty_annual)
        annual_result = build_quarterly_reversals(
            empty_quarterly,
            pd.DataFrame(
                {
                    "STN": ["101"],
                    "DELTA_ABS_BIAS": [0.0],
                    "SIGNIFICANCE_CLASS": ["uncertain"],
                }
            ),
        )

        self.assertEqual(empty_result.columns.tolist(), expected_columns)
        self.assertEqual(len(empty_result), 0)
        self.assertEqual(annual_result.loc[0, "N_ELIGIBLE_QUARTERS"], 0)
        self.assertEqual(
            annual_result.loc[0, "REVERSAL_STATUS"], "insufficient_eligible_quarters"
        )
        with self.assertRaisesRegex(ValueError, "annual results missing columns"):
            build_quarterly_reversals(empty_quarterly, pd.DataFrame({"STN": ["101"]}))

    def test_quarter_effect_uses_raw_sums_not_monthly_delta_mean_and_keeps_skeleton(self):
        """월별 Δ|BIAS| 평균으로 비선형 집계하거나 빈 분기를 삭제하는 회귀를 잡는다."""
        q1 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        daily = self._daily_for_dates("001", q1, sum_none=0.0, sum_fixed=1.0)
        january = daily["DATE"].dt.month.eq(1)
        daily.loc[january, "sum_none"] = 100.0
        daily.loc[january, "sum_fixed"] = 0.0
        daily.loc[january, "count"] = 100
        daily = pd.concat(
            [daily, self._daily_for_dates("2", ["2024-07-01"], count=0)],
            ignore_index=True,
        )

        result = build_quarterly_point_effects(daily)

        self.assertEqual(len(result), 8)
        self.assertEqual(result["STN"].map(type).unique().tolist(), [str])
        self.assertEqual(
            result.groupby("STN", sort=False)["QUARTER"].apply(list).to_dict(),
            {"001": ["Q1", "Q2", "Q3", "Q4"], "2": ["Q1", "Q2", "Q3", "Q4"]},
        )
        row = result.loc[result["STN"].eq("001") & result["QUARTER"].eq("Q1")].iloc[0]
        expected_none = 3100.0 / 3160.0
        expected_fixed = 60.0 / 3160.0
        expected_effect = abs(expected_fixed) - abs(expected_none)
        self.assertAlmostEqual(row["BIAS_none"], expected_none)
        self.assertAlmostEqual(row["BIAS_fixed"], expected_fixed)
        self.assertAlmostEqual(row["DELTA_ABS_BIAS"], expected_effect)
        self.assertNotAlmostEqual(row["DELTA_ABS_BIAS"], (-1.0 + 1.0 + 1.0) / 3.0)
        empty = result.loc[result["STN"].eq("001") & result["QUARTER"].eq("Q2")].iloc[0]
        self.assertEqual(empty["COVERAGE_REASON"], "no_data")
        self.assertTrue(pd.isna(empty["DELTA_ABS_BIAS"]))

    def test_quarter_coverage_uses_ceil_85_percent_all_months_and_exact_reasons(self):
        """Q1 91일의 최소 78일과 3개월 presence·무자료 사유가 바뀌는 회귀를 잡는다."""
        q1 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        exact = q1[:78]
        below = q1[:77]
        missing = q1[q1.month.isin([1, 3])]
        daily = pd.concat(
            [
                self._daily_for_dates("exact", exact),
                self._daily_for_dates("below", below),
                self._daily_for_dates("missing", missing),
                self._daily_for_dates("none", ["2024-01-01"], count=0),
            ],
            ignore_index=True,
        )

        q1_rows = build_quarterly_point_effects(daily).query("QUARTER == 'Q1'").set_index("STN")

        self.assertEqual(q1_rows.loc["exact", "CALENDAR_DAYS"], 91)
        self.assertEqual(q1_rows.loc["exact", "MIN_VALID_DAYS"], 78)
        self.assertEqual(q1_rows.loc["exact", "VALID_DAYS"], 78)
        self.assertAlmostEqual(q1_rows.loc["exact", "COVERAGE"], 78 / 91)
        self.assertEqual(q1_rows.loc["exact", "MONTHS_PRESENT"], 3)
        self.assertTrue(q1_rows.loc["exact", "COVERAGE_ELIGIBLE"])
        self.assertEqual(q1_rows.loc["exact", "COVERAGE_REASON"], "eligible")
        self.assertEqual(q1_rows.loc["below", "COVERAGE_REASON"], "below_85pct")
        self.assertEqual(
            q1_rows.loc["missing", "COVERAGE_REASON"],
            "missing_months_and_below_85pct",
        )
        self.assertEqual(q1_rows.loc["none", "COVERAGE_REASON"], "no_data")
        ineligible = q1_rows.loc[["below", "missing", "none"]]
        self.assertTrue(ineligible[["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]].isna().all().all())
        self.assertTrue((ineligible["VALID_DAYS"] >= 0).all())

    def test_quarter_bootstrap_passes_one_paired_weight_matrix_to_shared_helper(self):
        """none/fixed에 서로 다른 날짜 가중치를 적용하는 회귀를 잡는다."""
        from analysis import analyze_bias_feedback_followup as module
        from analysis.bias_significance import bootstrap_bias_effects as real_bootstrap

        daily = self._daily_for_dates(
            "101", pd.date_range("2024-01-01", "2024-03-31", freq="D")
        )
        captured = []

        def recording_bootstrap(station_daily, weights):
            captured.append((station_daily, weights))
            return real_bootstrap(station_daily, weights)

        with patch.object(module, "bootstrap_bias_effects", side_effect=recording_bootstrap):
            result = bootstrap_quarterly_effects(daily, n_boot=40, block_days=7, seed=19)

        self.assertEqual(len(captured), 1)
        station_daily, weights = captured[0]
        self.assertEqual(set(station_daily), {"sum_none", "sum_fixed", "count"})
        self.assertEqual(station_daily["sum_none"].shape, station_daily["sum_fixed"].shape)
        self.assertEqual(weights.shape, (40, 91))
        row = result.query("STN == '101' and QUARTER == 'Q1'").iloc[0]
        self.assertTrue(np.isfinite(row[["CI95_LOW", "CI95_HIGH"]].to_numpy(float)).all())
        self.assertEqual(row["BOOT_BLOCK_DAYS"], 7)

    def test_quarter_bootstrap_seed_is_stable_under_input_order_changes(self):
        """STN·분기 seed가 행 순서와 Python hash에 의존하는 회귀를 잡는다."""
        q1 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        daily = pd.concat(
            [
                self._daily_for_dates("2", q1, sum_none=1.0, sum_fixed=0.3),
                self._daily_for_dates("010", q1, sum_none=-0.8, sum_fixed=-0.1),
            ],
            ignore_index=True,
        )
        first = bootstrap_quarterly_effects(daily, n_boot=60, block_days=7, seed=2024)
        second = bootstrap_quarterly_effects(
            daily.sample(frac=1.0, random_state=4).reset_index(drop=True),
            n_boot=60,
            block_days=7,
            seed=2024,
        )

        assert_frame_equal(first, second, check_exact=True)
        eligible = first.loc[first["COVERAGE_ELIGIBLE"]]
        self.assertEqual(eligible["BOOT_SEED"].nunique(), 2)

    def test_stable_quarter_seed_is_identical_across_python_hash_seed_processes(self):
        """stable seed가 현재 process의 Python hash salt에 숨은 의존성을 두는 회귀를 잡는다."""
        import os
        import subprocess
        import sys

        command = [
            sys.executable,
            "-c",
            (
                "from analysis.analyze_bias_feedback_followup import _stable_quarter_seed; "
                "print(_stable_quarter_seed(20260818, '001', 'Q4'))"
            ),
        ]
        outputs = []
        for hash_seed in ["1", "987654"]:
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.check_output(command, env=environment, text=True).strip()
            )

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], "342579988")

    def test_quarter_bootstrap_percentile_ci_matches_hand_calculated_fixed_weights(self):
        """95% CI가 다른 quantile 방식이나 다른 효과 표본을 쓰는 회귀를 잡는다."""
        from analysis import analyze_bias_feedback_followup as module

        dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        daily = self._daily_for_dates("101", dates, sum_none=0.0, sum_fixed=0.0)
        daily["sum_none"] = np.arange(91, dtype=float)
        fixed_weights = np.zeros((40, 91), dtype=float)
        fixed_weights[np.arange(40), np.arange(40)] = 1.0

        with patch.object(
            module,
            "make_month_stratified_block_weights",
            return_value=fixed_weights,
        ):
            result = bootstrap_quarterly_effects(
                daily, n_boot=40, block_days=7, seed=11
            )

        q1 = result.query("STN == '101' and QUARTER == 'Q1'").iloc[0]
        self.assertAlmostEqual(q1["CI95_LOW"], -38.025)
        self.assertAlmostEqual(q1["CI95_HIGH"], -0.975)
        self.assertEqual(q1["CI_POSITION"], "below_minus_delta")

    def test_ci_position_uses_open_outer_bounds_at_exact_practical_threshold(self):
        """95% CI가 정확히 ±0.1에 닿을 때 경계 밖으로 잘못 분류하는 회귀를 잡는다."""
        self.assertEqual(classify_ci_position(-0.3, -0.100001), "below_minus_delta")
        self.assertEqual(classify_ci_position(-0.3, -0.1), "overlaps_practical_band")
        self.assertEqual(classify_ci_position(0.1, 0.3), "overlaps_practical_band")
        self.assertEqual(classify_ci_position(0.100001, 0.3), "above_plus_delta")

    def test_network_summary_reports_quarter_sets_and_four_quarter_common_cohort(self):
        """분기별 가용 station과 4분기 공통 cohort를 섞어 요약하는 회귀를 잡는다."""
        rows = []
        for quarter, a_value in zip(["Q1", "Q2", "Q3", "Q4"], [-0.2, -0.05, 0.0, 0.2]):
            rows.append(
                {
                    "STN": "A",
                    "QUARTER": quarter,
                    "COVERAGE_ELIGIBLE": True,
                    "DELTA_ABS_BIAS": a_value,
                    "CI95_LOW": a_value - 0.05,
                    "CI95_HIGH": a_value + 0.05,
                }
            )
        rows.extend(
            [
                {
                    "STN": "B",
                    "QUARTER": "Q1",
                    "COVERAGE_ELIGIBLE": True,
                    "DELTA_ABS_BIAS": 0.4,
                    "CI95_LOW": 0.3,
                    "CI95_HIGH": 0.5,
                },
                *[
                    {
                        "STN": "B",
                        "QUARTER": q,
                        "COVERAGE_ELIGIBLE": False,
                        "DELTA_ABS_BIAS": np.nan,
                        "CI95_LOW": np.nan,
                        "CI95_HIGH": np.nan,
                    }
                    for q in ["Q2", "Q3", "Q4"]
                ],
            ]
        )

        summary = summarize_quarterly_network(pd.DataFrame(rows))

        self.assertEqual(len(summary), 4)
        q1 = summary.set_index("QUARTER").loc["Q1"]
        self.assertEqual(q1["N_ELIGIBLE"], 2)
        self.assertAlmostEqual(q1["MEDIAN_DELTA_ABS_BIAS"], 0.1)
        self.assertEqual(q1["N_POINT_IMPROVEMENT"], 1)
        self.assertEqual(q1["N_POINT_WORSENING"], 1)
        self.assertEqual(q1["COMMON_COHORT_N"], 1)
        self.assertAlmostEqual(q1["COMMON_MEDIAN_DELTA_ABS_BIAS"], -0.2)
        self.assertEqual(q1["COMMON_N_POINT_IMPROVEMENT"], 1)
        self.assertAlmostEqual(q1["COMMON_FRAC_POINT_IMPROVEMENT"], 1.0)
        self.assertFalse(any("CI" in column for column in summary.columns))

    def test_network_summary_rejects_malformed_quarter_domain_skeleton_and_na_contract(self):
        """Q1–Q4 골격과 eligible/ineligible effect·CI NA 계약을 생략하는 회귀를 잡는다."""
        base = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "QUARTER": ["Q1", "Q2", "Q3", "Q4"],
                "COVERAGE_ELIGIBLE": [True, True, False, False],
                "DELTA_ABS_BIAS": [-0.2, 0.2, np.nan, np.nan],
                "CI95_LOW": [-0.3, 0.1, np.nan, np.nan],
                "CI95_HIGH": [-0.1, 0.3, np.nan, np.nan],
            }
        )
        valid = summarize_quarterly_network(base)
        self.assertEqual(len(valid), 4)

        malformed = []
        wrong_domain = base.copy()
        wrong_domain.loc[3, "QUARTER"] = "Q5"
        malformed.append((wrong_domain, "Q1.*Q4"))
        malformed.append((base.iloc[:-1].copy(), "complete.*station.*quarter"))
        duplicate = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        malformed.append((duplicate, "duplicate"))
        eligible_missing_effect = base.copy()
        eligible_missing_effect.loc[0, "DELTA_ABS_BIAS"] = np.nan
        malformed.append((eligible_missing_effect, "eligible.*finite"))
        eligible_missing_ci = base.copy()
        eligible_missing_ci.loc[0, "CI95_LOW"] = np.nan
        malformed.append((eligible_missing_ci, "eligible.*finite"))
        ineligible_has_effect = base.copy()
        ineligible_has_effect.loc[2, "DELTA_ABS_BIAS"] = 0.0
        malformed.append((ineligible_has_effect, "ineligible.*NA"))
        ineligible_has_ci = base.copy()
        ineligible_has_ci.loc[2, "CI95_HIGH"] = 0.0
        malformed.append((ineligible_has_ci, "ineligible.*NA"))

        for frame, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_quarterly_network(frame)

    def test_coverage_eligible_requires_actual_boolean_values_in_network_and_reversals(self):
        """문자열 'False'나 정수 2를 True로 강제 변환하는 회귀를 잡는다."""
        network = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "QUARTER": ["Q1", "Q2", "Q3", "Q4"],
                "COVERAGE_ELIGIBLE": [True, True, False, "False"],
                "DELTA_ABS_BIAS": [-0.2, 0.2, np.nan, np.nan],
                "CI95_LOW": [-0.3, 0.1, np.nan, np.nan],
                "CI95_HIGH": [-0.1, 0.3, np.nan, np.nan],
            }
        )
        reversals = pd.DataFrame(
            {
                "STN": ["101"],
                "QUARTER": ["Q1"],
                "COVERAGE_ELIGIBLE": [2],
                "DELTA_ABS_BIAS": [-0.2],
            }
        )
        annual = pd.DataFrame(
            {
                "STN": ["101"],
                "DELTA_ABS_BIAS": [-0.2],
                "SIGNIFICANCE_CLASS": ["meaningful_improvement"],
            }
        )

        with self.assertRaisesRegex(ValueError, "COVERAGE_ELIGIBLE.*boolean"):
            summarize_quarterly_network(network)
        with self.assertRaisesRegex(ValueError, "COVERAGE_ELIGIBLE.*boolean"):
            build_quarterly_reversals(reversals, annual)

    def test_reversal_flags_are_separate_and_insufficient_quarters_are_explicit(self):
        """원부호·실질 반전·연간 class 불일치와 부족 상태가 합쳐지는 회귀를 잡는다."""
        values = {
            "A": [-0.2, 0.2, -0.05, 0.0],
            "B": [-0.2, -0.05, np.nan, np.nan],
            "C": [0.2, np.nan, np.nan, np.nan],
            "D": [np.nan, np.nan, np.nan, np.nan],
        }
        rows = []
        for stn, station_values in values.items():
            for quarter, value in zip(["Q1", "Q2", "Q3", "Q4"], station_values):
                rows.append(
                    {
                        "STN": stn,
                        "QUARTER": quarter,
                        "COVERAGE_ELIGIBLE": np.isfinite(value),
                        "DELTA_ABS_BIAS": value,
                    }
                )
        annual = pd.DataFrame(
            {
                "STN": ["A", "B", "C", "D"],
                "DELTA_ABS_BIAS": [-0.1, -0.2, 0.1, 0.3],
                "SIGNIFICANCE_CLASS": [
                    "uncertain",
                    "meaningful_improvement",
                    "practically_equivalent",
                    "meaningful_worsening",
                ],
            }
        )

        result = build_quarterly_reversals(pd.DataFrame(rows), annual).set_index("STN")

        self.assertEqual(len(result), 4)
        self.assertEqual(result.loc["A", "N_ELIGIBLE_QUARTERS"], 4)
        self.assertTrue(result.loc["A", "RAW_SIGN_CROSSING"])
        self.assertTrue(result.loc["A", "PRACTICAL_CROSSING"])
        self.assertTrue(result.loc["A", "ANNUAL_QUARTER_POINT_CLASS_MISMATCH"])
        self.assertEqual(result.loc["A", "ANNUAL_POINT_CLASS"], "small_change")
        self.assertEqual(result.loc["A", "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE"], "uncertain")
        self.assertFalse(result.loc["B", "RAW_SIGN_CROSSING"])
        self.assertEqual(result.loc["C", "REVERSAL_STATUS"], "insufficient_eligible_quarters")
        self.assertTrue(pd.isna(result.loc["C", "RAW_SIGN_CROSSING"]))
        self.assertEqual(result.loc["D", "N_ELIGIBLE_QUARTERS"], 0)

    def test_sensitivity_summary_compares_ci_position_not_point_estimates(self):
        """블록 민감도가 불변인 점추정 agreement를 대신 쓰는 회귀를 잡는다."""
        results = {}
        for block in [3, 7, 14]:
            rows = []
            for quarter in ["Q1", "Q2", "Q3", "Q4"]:
                positions = (
                    ["below_minus_delta", "above_plus_delta"]
                    if block == 7
                    else (["below_minus_delta", "overlaps_practical_band"] if block == 3 else ["overlaps_practical_band", "overlaps_practical_band"])
                )
                for index, position in enumerate(positions):
                    rows.append(
                        {
                            "STN": str(index),
                            "QUARTER": quarter,
                            "COVERAGE_ELIGIBLE": True,
                            "DELTA_ABS_BIAS": 0.0,
                            "CI95_LOW": [-0.3, 0.2][index],
                            "CI95_HIGH": [-0.2, 0.3][index],
                            "CI_POSITION": position,
                        }
                    )
            results[block] = pd.DataFrame(rows)

        summary = build_quarterly_block_sensitivity(results_by_block=results, n_boot=50)

        self.assertEqual(len(summary), 12)
        self.assertNotIn("POINT_ESTIMATE_AGREEMENT", summary.columns)
        q1 = summary.query("QUARTER == 'Q1'").set_index("BLOCK_DAYS")
        self.assertEqual(q1.loc[7, "CI_POSITION_AGREE_N"], 2)
        self.assertEqual(q1.loc[7, "CI_POSITION_AGREE_FRACTION"], 1.0)
        self.assertEqual(q1.loc[3, "CI_POSITION_AGREE_N"], 1)
        self.assertEqual(q1.loc[3, "CI_POSITION_AGREE_FRACTION"], 0.5)
        self.assertEqual(q1.loc[14, "CI_POSITION_AGREE_N"], 0)
        self.assertAlmostEqual(q1.loc[7, "MEDIAN_CI95_WIDTH"], 0.1)

    def test_sensitivity_reuses_injected_primary_7_day_details_without_recalling_bootstrap(self):
        """Task 5가 기존 7일 상세 결과를 두고 같은 bootstrap을 다시 돌리는 회귀를 잡는다."""
        from analysis import analyze_bias_feedback_followup as module

        daily = self._daily_for_dates(
            "101", pd.date_range("2024-01-01", "2024-12-31", freq="D")
        )
        primary = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "QUARTER": ["Q1", "Q2", "Q3", "Q4"],
                "COVERAGE_ELIGIBLE": [True] * 4,
                "DELTA_ABS_BIAS": [-0.2, -0.1, 0.0, 0.2],
                "CI95_LOW": [-0.31, -0.21, -0.11, 0.09],
                "CI95_HIGH": [-0.19, 0.01, 0.11, 0.31],
                "CI_POSITION": [
                    "below_minus_delta",
                    "overlaps_practical_band",
                    "overlaps_practical_band",
                    "overlaps_practical_band",
                ],
                "BOOT_BLOCK_DAYS": [7] * 4,
                "BOOT_REPLICATES": [40] * 4,
                "BOOT_SEED": [
                    module._stable_quarter_seed(9, "101", quarter)
                    for quarter in ["Q1", "Q2", "Q3", "Q4"]
                ],
            }
        )
        primary_before = primary.copy(deep=True)
        called_blocks = []

        def fake_bootstrap(daily, n_boot, block_days, seed, stations=None):
            called_blocks.append(block_days)
            result = primary.copy(deep=True)
            result["CI95_LOW"] = result["CI95_LOW"] - block_days / 1000.0
            result["CI95_HIGH"] = result["CI95_HIGH"] + block_days / 1000.0
            return result

        with patch.object(module, "bootstrap_quarterly_effects", side_effect=fake_bootstrap):
            summary = build_quarterly_block_sensitivity(
                daily=daily,
                n_boot=40,
                seed=9,
                primary_result=primary,
                stations=["101"],
            )

        self.assertEqual(called_blocks, [3, 14])
        q1_primary = summary.query("QUARTER == 'Q1' and BLOCK_DAYS == 7").iloc[0]
        self.assertEqual(q1_primary["MEDIAN_CI95_LOW"], -0.31)
        self.assertEqual(q1_primary["MEDIAN_CI95_HIGH"], -0.19)
        self.assertEqual(q1_primary["MEDIAN_CI95_WIDTH"], 0.12)
        self.assertEqual(q1_primary["CI_POSITION_AGREE_FRACTION"], 1.0)
        assert_frame_equal(primary, primary_before, check_exact=True)

    def test_injected_primary_7_day_details_reject_metadata_key_and_eligibility_mismatch(self):
        """N_BOOT 표시와 다른 primary snapshot을 주입값으로 신뢰하는 회귀를 잡는다."""
        from analysis import analyze_bias_feedback_followup as module

        daily = self._daily_for_dates(
            "101", pd.date_range("2024-01-01", "2024-12-31", freq="D")
        )
        base = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "QUARTER": ["Q1", "Q2", "Q3", "Q4"],
                "COVERAGE_ELIGIBLE": [True] * 4,
                "DELTA_ABS_BIAS": [-0.2, -0.1, 0.0, 0.2],
                "CI95_LOW": [-0.3, -0.2, -0.1, 0.1],
                "CI95_HIGH": [-0.1, 0.0, 0.1, 0.3],
                "CI_POSITION": ["overlaps_practical_band"] * 4,
                "BOOT_BLOCK_DAYS": [7] * 4,
                "BOOT_REPLICATES": [40] * 4,
                "BOOT_SEED": [
                    module._stable_quarter_seed(9, "101", quarter)
                    for quarter in ["Q1", "Q2", "Q3", "Q4"]
                ],
            }
        )
        malformed = []
        wrong_block = base.copy()
        wrong_block.loc[0, "BOOT_BLOCK_DAYS"] = 3
        malformed.append((wrong_block, "BOOT_BLOCK_DAYS"))
        wrong_replicates = base.copy()
        wrong_replicates.loc[0, "BOOT_REPLICATES"] = 39
        malformed.append((wrong_replicates, "BOOT_REPLICATES"))
        wrong_seed = base.copy()
        wrong_seed.loc[0, "BOOT_SEED"] += 1
        malformed.append((wrong_seed, "BOOT_SEED"))
        malformed.append((base.iloc[:-1].copy(), "station-quarter"))
        duplicate = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        malformed.append((duplicate, "duplicate"))
        wrong_quarter = base.copy()
        wrong_quarter.loc[3, "QUARTER"] = "Q5"
        malformed.append((wrong_quarter, "Q1.*Q4"))
        wrong_eligibility = base.copy()
        wrong_eligibility.loc[0, "COVERAGE_ELIGIBLE"] = False
        wrong_eligibility.loc[0, ["BOOT_BLOCK_DAYS", "BOOT_REPLICATES", "BOOT_SEED"]] = np.nan
        malformed.append((wrong_eligibility, "eligibility"))

        with patch.object(module, "bootstrap_quarterly_effects", return_value=base):
            for primary, message in malformed:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        build_quarterly_block_sensitivity(
                            daily=daily,
                            n_boot=40,
                            seed=9,
                            primary_result=primary,
                            stations=["101"],
                        )

        ineligible_rows = pd.DataFrame(
            {
                "STN": ["999"] * 4,
                "QUARTER": ["Q1", "Q2", "Q3", "Q4"],
                "COVERAGE_ELIGIBLE": [False] * 4,
                "DELTA_ABS_BIAS": [np.nan] * 4,
                "CI95_LOW": [np.nan] * 4,
                "CI95_HIGH": [np.nan] * 4,
                "CI_POSITION": [pd.NA] * 4,
                "BOOT_BLOCK_DAYS": [pd.NA] * 4,
                "BOOT_REPLICATES": [pd.NA] * 4,
                "BOOT_SEED": [pd.NA] * 4,
            }
        )
        canonical_primary = pd.concat([base, ineligible_rows], ignore_index=True)
        with patch.object(
            module, "bootstrap_quarterly_effects", return_value=canonical_primary
        ):
            valid_summary = build_quarterly_block_sensitivity(
                daily=daily,
                n_boot=40,
                seed=9,
                primary_result=canonical_primary,
                stations=["101", "999"],
            )
        self.assertEqual(len(valid_summary), 12)

        bad_ineligible_metadata = canonical_primary.copy()
        bad_ineligible_metadata.loc[4, "BOOT_BLOCK_DAYS"] = 7
        with patch.object(
            module, "bootstrap_quarterly_effects", return_value=canonical_primary
        ):
            with self.assertRaisesRegex(ValueError, "ineligible.*metadata.*NA"):
                build_quarterly_block_sensitivity(
                    daily=daily,
                    n_boot=40,
                    seed=9,
                    primary_result=bad_ineligible_metadata,
                    stations=["101", "999"],
                )

    def test_quarter_plots_use_only_requested_color_paths_and_close_on_save_error(self):
        """두 그림이 지정 staging 경로 밖을 쓰거나 저장 예외 후 figure를 남기는 회귀를 잡는다."""
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        quarterly = pd.DataFrame(
            {
                "STN": ["1", "2", "1", "2"],
                "QUARTER": ["Q1", "Q1", "Q2", "Q2"],
                "COVERAGE_ELIGIBLE": [True] * 4,
                "DELTA_ABS_BIAS": [-0.3, 0.2, -0.1, 0.4],
                "CI95_LOW": [-0.4, 0.1, -0.2, 0.3],
                "CI95_HIGH": [-0.2, 0.3, 0.0, 0.5],
            }
        )
        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            paths = [
                Path(temp_dir) / "nested" / "bias_quarterly_effect.png",
                Path(temp_dir) / "nested" / "bias_quarterly_heatmap.png",
            ]
            functions = [plot_quarterly_effect, plot_quarterly_heatmap]
            try:
                for function, path in zip(functions, paths):
                    self.assertEqual(Path(function(quarterly, path)), path)
                    image = mpimg.imread(path)
                    self.assertGreater(image.shape[-1], 2)
                    self.assertTrue(np.any(np.abs(image[..., 0] - image[..., 1]) > 1e-3))
                self.assertEqual(
                    {path.name for path in (Path(temp_dir) / "nested").iterdir()},
                    {"bias_quarterly_effect.png", "bias_quarterly_heatmap.png"},
                )
                self.assertEqual(plt.get_fignums(), [])
                for function, path in zip(functions, paths):
                    with patch("matplotlib.figure.Figure.savefig", side_effect=RuntimeError("save failed")):
                        with self.assertRaisesRegex(RuntimeError, "save failed"):
                            function(quarterly, path)
                    self.assertEqual(plt.get_fignums(), [])
            finally:
                plt.close("all")

    def test_529_station_heatmap_uses_sparse_ticks_and_names_csv_as_value_source(self):
        """overview에 529개 y-label을 강제하거나 개별값 정본을 숨기는 회귀를 잡는다."""
        import matplotlib.axes
        import matplotlib.figure
        import matplotlib.pyplot as plt

        rows = []
        for station_index in range(529):
            for quarter_index, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                rows.append(
                    {
                        "STN": str(station_index),
                        "QUARTER": quarter,
                        "COVERAGE_ELIGIBLE": True,
                        "DELTA_ABS_BIAS": (station_index % 7 - 3) * 0.05 + quarter_index * 0.01,
                    }
                )
        captured_tick_counts = []
        captured_figure_text = []
        real_set_yticks = matplotlib.axes.Axes.set_yticks
        real_figure_text = matplotlib.figure.Figure.text

        def recording_set_yticks(axis, ticks, *args, **kwargs):
            captured_tick_counts.append(len(list(ticks)))
            return real_set_yticks(axis, ticks, *args, **kwargs)

        def recording_figure_text(figure, x, y, text, *args, **kwargs):
            captured_figure_text.append(str(text))
            return real_figure_text(figure, x, y, text, *args, **kwargs)

        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            output_path = Path(temp_dir) / "nested" / "bias_quarterly_heatmap.png"
            try:
                with patch.object(
                    matplotlib.axes.Axes, "set_yticks", new=recording_set_yticks
                ), patch.object(
                    matplotlib.figure.Figure, "text", new=recording_figure_text
                ):
                    result = plot_quarterly_heatmap(pd.DataFrame(rows), output_path)

                self.assertEqual(Path(result), output_path)
                self.assertTrue(output_path.is_file())
                self.assertTrue(captured_tick_counts)
                self.assertLessEqual(max(captured_tick_counts), 20)
                self.assertTrue(
                    any("quarterly_station_effects.csv" in text for text in captured_figure_text)
                )
                self.assertEqual(plt.get_fignums(), [])
            finally:
                plt.close("all")


class BiasFeedbackFollowupLeaveOutTests(unittest.TestCase):
    EXPECTED_COLUMNS = [
        "STN",
        "EXCLUDED_QUARTER",
        "LOQ_STATUS",
        "REMAINING_PAIRED_HOURS",
        "ANNUAL_DELTA_ABS_BIAS",
        "ANNUAL_POINT_CLASS",
        "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE",
        "REMAINING_BIAS_none",
        "REMAINING_BIAS_fixed",
        "REMAINING_DELTA_ABS_BIAS",
        "REMAINING_POINT_CLASS",
        "SHIFT_FROM_ANNUAL",
        "ABS_SHIFT_FROM_ANNUAL",
        "RAW_SIGN_CHANGED",
        "POINT_CLASS_CHANGED",
        "IS_MAX_ABS_INFLUENCE_QUARTER",
    ]

    @staticmethod
    def _empty_daily():
        return pd.DataFrame(columns=["STN", "DATE", "sum_none", "sum_fixed", "count"])

    @staticmethod
    def _annual(stations, effects=None, classes=None):
        station_ids = [str(stn) for stn in stations]
        if effects is None:
            effects = [0.0] * len(station_ids)
        if classes is None:
            classes = ["uncertain"] * len(station_ids)
        return pd.DataFrame(
            {
                "STN": station_ids,
                "DELTA_ABS_BIAS": effects,
                "SIGNIFICANCE_CLASS": classes,
            }
        )

    def test_leave_out_exact_schema_and_canonical_529_by_four_skeleton(self):
        """daily 미출현 station을 삭제하거나 16열 long schema를 흔드는 회귀를 잡는다."""
        annual = self._annual(range(529))

        result = build_leave_one_quarter_out(self._empty_daily(), annual)

        self.assertEqual(result.columns.tolist(), self.EXPECTED_COLUMNS)
        self.assertEqual(len(result), 2116)
        self.assertEqual(result["STN"].nunique(), 529)
        self.assertEqual(result["STN"].map(type).unique().tolist(), [str])
        self.assertEqual(
            result["EXCLUDED_QUARTER"].value_counts(sort=False).to_dict(),
            {"Q1": 529, "Q2": 529, "Q3": 529, "Q4": 529},
        )
        self.assertEqual(set(result["LOQ_STATUS"]), {"no_remaining_data"})
        self.assertTrue((result["REMAINING_PAIRED_HOURS"] == 0).all())
        self.assertTrue(
            result[
                [
                    "REMAINING_BIAS_none",
                    "REMAINING_BIAS_fixed",
                    "REMAINING_DELTA_ABS_BIAS",
                    "SHIFT_FROM_ANNUAL",
                    "ABS_SHIFT_FROM_ANNUAL",
                    "RAW_SIGN_CHANGED",
                    "POINT_CLASS_CHANGED",
                    "IS_MAX_ABS_INFLUENCE_QUARTER",
                ]
            ].isna().all().all()
        )
        duplicate = pd.concat([annual, annual.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_leave_one_quarter_out(self._empty_daily(), duplicate)
        with self.assertRaisesRegex(ValueError, "stations.*annual"):
            build_leave_one_quarter_out(
                self._empty_daily(), annual, stations=[str(i) for i in range(528)]
            )

        ordered = build_leave_one_quarter_out(
            self._empty_daily(), self._annual(["20", "3", "100"])
        )
        self.assertEqual(
            list(ordered[["STN", "EXCLUDED_QUARTER"]].itertuples(index=False, name=None)),
            [
                ("20", "Q1"),
                ("20", "Q2"),
                ("20", "Q3"),
                ("20", "Q4"),
                ("3", "Q1"),
                ("3", "Q2"),
                ("3", "Q3"),
                ("3", "Q4"),
                ("100", "Q1"),
                ("100", "Q2"),
                ("100", "Q3"),
                ("100", "Q4"),
            ],
        )

    def test_leave_out_excludes_exact_2024_dates_and_recomputes_from_remaining_raw_sums(self):
        """분기 효과 평균이나 annual−quarter로 9개월 BIAS를 대체하는 회귀를 잡는다."""
        daily = pd.DataFrame(
            {
                "STN": ["101"] * 9,
                "DATE": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-02-29",
                        "2024-03-31",
                        "2024-04-01",
                        "2024-06-30",
                        "2024-07-01",
                        "2024-09-30",
                        "2024-10-01",
                        "2024-12-31",
                    ]
                ),
                "sum_none": [100.0, 70.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "sum_fixed": [0.0, 7.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "count": [100, 7, 100, 1, 1, 1, 1, 1, 1],
            }
        )
        annual_effect = abs(13 / 213) - abs(270 / 213)
        annual = self._annual(["101"], effects=[annual_effect])

        result = build_leave_one_quarter_out(daily, annual).set_index("EXCLUDED_QUARTER")

        self.assertEqual(result.loc["Q1", "REMAINING_PAIRED_HOURS"], 6)
        self.assertEqual(result.loc["Q1", "REMAINING_BIAS_none"], 0.0)
        self.assertEqual(result.loc["Q1", "REMAINING_BIAS_fixed"], 1.0)
        self.assertEqual(result.loc["Q1", "REMAINING_DELTA_ABS_BIAS"], 1.0)
        self.assertEqual(result.loc["Q2", "REMAINING_PAIRED_HOURS"], 211)
        self.assertAlmostEqual(result.loc["Q2", "REMAINING_BIAS_none"], 270 / 211)
        self.assertAlmostEqual(result.loc["Q2", "REMAINING_BIAS_fixed"], 11 / 211)
        self.assertAlmostEqual(result.loc["Q2", "REMAINING_DELTA_ABS_BIAS"], -259 / 211)
        self.assertAlmostEqual(
            result.loc["Q1", "SHIFT_FROM_ANNUAL"], 1.0 - annual_effect
        )
        self.assertNotAlmostEqual(
            result.loc["Q1", "REMAINING_DELTA_ABS_BIAS"],
            (-1.0 + 1.0 + 1.0) / 3.0,
        )

    def test_leave_out_point_boundaries_do_not_use_official_four_class(self):
        """정확한 ±0.1 경계와 annual 3-class를 공식 4-class로 대체하는 회귀를 잡는다."""
        effects = [-0.1, 0.1, -0.100001, 0.100001]
        stations = ["minus_edge", "plus_edge", "improve", "worsen"]
        dates = pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"])
        rows = []
        for stn, effect in zip(stations, effects):
            for date in dates:
                rows.append(
                    {
                        "STN": stn,
                        "DATE": date,
                        "sum_none": abs(effect) if effect < 0 else 0.0,
                        "sum_fixed": effect if effect >= 0 else 0.0,
                        "count": 1,
                    }
                )
        official_a = [
            "meaningful_worsening",
            "meaningful_improvement",
            "uncertain",
            "practically_equivalent",
        ]
        official_b = list(reversed(official_a))
        annual_a = self._annual(stations, effects=effects, classes=official_a)
        annual_b = self._annual(stations, effects=effects, classes=official_b)

        first = build_leave_one_quarter_out(pd.DataFrame(rows), annual_a)
        second = build_leave_one_quarter_out(pd.DataFrame(rows), annual_b)

        expected_classes = {
            "minus_edge": "small_change",
            "plus_edge": "small_change",
            "improve": "improvement",
            "worsen": "worsening",
        }
        self.assertEqual(
            first.groupby("STN")["ANNUAL_POINT_CLASS"].first().to_dict(),
            expected_classes,
        )
        self.assertEqual(
            first.groupby("STN")["REMAINING_POINT_CLASS"].first().to_dict(),
            expected_classes,
        )
        self.assertFalse(first["POINT_CLASS_CHANGED"].any())
        compare_columns = [
            column
            for column in self.EXPECTED_COLUMNS
            if column != "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE"
        ]
        assert_frame_equal(first[compare_columns], second[compare_columns], check_exact=True)
        self.assertNotEqual(
            first["ANNUAL_SIGNIFICANCE_CLASS_REFERENCE"].tolist(),
            second["ANNUAL_SIGNIFICANCE_CLASS_REFERENCE"].tolist(),
        )

    def test_leave_out_no_remaining_data_and_neutral_sign_are_explicit(self):
        """0을 중립 부호로 보지 않거나 무자료 flag를 False로 두는 회귀를 잡는다."""
        dates = pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"])
        rows = []
        for stn, fixed in [("neutral_change", 0.2), ("neutral_same", 0.0)]:
            for date in dates:
                rows.append(
                    {
                        "STN": stn,
                        "DATE": date,
                        "sum_none": 0.0,
                        "sum_fixed": fixed,
                        "count": 1,
                    }
                )
        annual = self._annual(["empty", "neutral_change", "neutral_same"])

        result = build_leave_one_quarter_out(pd.DataFrame(rows), annual)

        empty = result.loc[result["STN"].eq("empty")]
        self.assertEqual(set(empty["LOQ_STATUS"]), {"no_remaining_data"})
        self.assertTrue((empty["REMAINING_PAIRED_HOURS"] == 0).all())
        self.assertTrue(
            empty[
                [
                    "REMAINING_DELTA_ABS_BIAS",
                    "REMAINING_POINT_CLASS",
                    "SHIFT_FROM_ANNUAL",
                    "RAW_SIGN_CHANGED",
                    "POINT_CLASS_CHANGED",
                    "IS_MAX_ABS_INFLUENCE_QUARTER",
                ]
            ].isna().all().all()
        )
        changed = result.loc[result["STN"].eq("neutral_change")]
        same = result.loc[result["STN"].eq("neutral_same")]
        self.assertTrue(changed["RAW_SIGN_CHANGED"].all())
        self.assertTrue(changed["POINT_CLASS_CHANGED"].all())
        self.assertFalse(same["RAW_SIGN_CHANGED"].any())
        self.assertFalse(same["POINT_CLASS_CHANGED"].any())

    def test_leave_out_marks_all_tied_maximum_absolute_influence_quarters(self):
        """station별 최대 ABS_SHIFT 동률을 하나만 선택하는 회귀를 잡는다."""
        daily = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "DATE": pd.to_datetime(
                    ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
                ),
                "sum_none": [0.0] * 4,
                "sum_fixed": [0.0, 0.0, 1.0, 1.0],
                "count": [1] * 4,
            }
        )

        result = build_leave_one_quarter_out(
            daily, self._annual(["101"], effects=[0.0])
        ).set_index("EXCLUDED_QUARTER")

        self.assertEqual(
            result["IS_MAX_ABS_INFLUENCE_QUARTER"].to_dict(),
            {"Q1": True, "Q2": True, "Q3": False, "Q4": False},
        )
        self.assertAlmostEqual(result.loc["Q1", "ABS_SHIFT_FROM_ANNUAL"], 2 / 3)
        self.assertAlmostEqual(result.loc["Q3", "ABS_SHIFT_FROM_ANNUAL"], 1 / 3)

    def test_leave_out_rejects_dates_outside_2024(self):
        """2024 전용 분석에 다른 해 날짜를 조용히 섞는 회귀를 잡는다."""
        daily = pd.DataFrame(
            {
                "STN": ["101"],
                "DATE": ["2025-01-01"],
                "sum_none": [0.0],
                "sum_fixed": [0.0],
                "count": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "2024"):
            build_leave_one_quarter_out(daily, self._annual(["101"]))

    def test_leave_out_plot_is_color_readable_at_529_scale_and_closes_figures(self):
        """529 station을 y-label로 나열하거나 저장 예외 후 figure를 남기는 회귀를 잡는다."""
        import matplotlib.figure
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        rows = []
        for station_index in range(529):
            for quarter_index, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                shift = ((station_index % 11) - 5) * 0.01 + quarter_index * 0.005
                rows.append(
                    {
                        "STN": str(station_index),
                        "EXCLUDED_QUARTER": quarter,
                        "LOQ_STATUS": "eligible",
                        "SHIFT_FROM_ANNUAL": shift,
                        "ABS_SHIFT_FROM_ANNUAL": abs(shift),
                    }
                )
        loq = pd.DataFrame(rows)
        real_savefig = matplotlib.figure.Figure.savefig
        max_tick_labels = []
        visible_text = []

        def inspecting_savefig(figure, *args, **kwargs):
            for axis in figure.axes:
                max_tick_labels.append(
                    max(len(axis.get_xticklabels()), len(axis.get_yticklabels()))
                )
                visible_text.extend(
                    [axis.get_title(), axis.get_xlabel(), axis.get_ylabel()]
                )
            return real_savefig(figure, *args, **kwargs)

        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            output_path = Path(temp_dir) / "nested" / "bias_leave_one_quarter_out.png"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with patch.object(
                        matplotlib.figure.Figure, "savefig", new=inspecting_savefig
                    ):
                        result = plot_leave_one_quarter_out(loq, output_path)
                self.assertEqual(Path(result), output_path)
                self.assertTrue(output_path.is_file())
                image = mpimg.imread(output_path)
                self.assertTrue(np.any(np.abs(image[..., 0] - image[..., 1]) > 1e-3))
                self.assertLessEqual(max(max_tick_labels), 20)
                self.assertFalse(any("dz_geom" in text or "ΔZ" in text for text in visible_text))
                self.assertFalse(any("Glyph" in str(item.message) for item in caught))
                self.assertEqual(plt.get_fignums(), [])

                with patch(
                    "matplotlib.figure.Figure.savefig",
                    side_effect=RuntimeError("save failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "save failed"):
                        plot_leave_one_quarter_out(loq, output_path)
                self.assertEqual(plt.get_fignums(), [])
            finally:
                plt.close("all")


class BiasFeedbackFollowupRunnerTests(unittest.TestCase):
    @staticmethod
    def _injected_inputs():
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        definitions = [
            ("101", 0.05, 0.05, "practically_equivalent", 0.04, dates),
            ("102", 0.50, 0.20, "meaningful_improvement", 0.075, dates),
            ("103", 0.00, 0.30, "meaningful_worsening", 0.15, dates),
            ("104", 0.20, 0.20, "uncertain", 0.25, dates),
            ("105", -0.40, -0.20, "uncertain", 0.35, dates[dates.month <= 9]),
        ]
        annual_rows = []
        daily_parts = []
        for stn, bias_none, bias_fixed, label, delta_z_km, station_dates in definitions:
            annual_rows.append(
                {
                    "STN": stn,
                    "BIAS_none": bias_none,
                    "BIAS_fixed": bias_fixed,
                    "DELTA_ABS_BIAS": abs(bias_fixed) - abs(bias_none),
                    "SIGNIFICANCE_CLASS": label,
                    "mean_absdz_geom_km": delta_z_km,
                }
            )
            daily_parts.append(
                pd.DataFrame(
                    {
                        "STN": stn,
                        "DATE": station_dates,
                        "sum_none": bias_none,
                        "sum_fixed": bias_fixed,
                        "count": 1,
                    }
                )
            )
        return pd.DataFrame(annual_rows), pd.concat(daily_parts, ignore_index=True)

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def _in_memory_tables(cls):
        annual, daily = cls._injected_inputs()
        annual, prepared, _, residuals = followup._canonical_runner_inputs(
            annual, daily, expected_station_count=len(annual)
        )
        tables = followup._build_followup_tables(annual, prepared, n_boot=20, seed=17)
        return (
            annual.copy(deep=True),
            prepared.copy(deep=True),
            {name: frame.copy(deep=True) for name, frame in tables.items()},
            dict(residuals),
        )

    @classmethod
    def _zero_quarter_tables(cls):
        annual, prepared, tables, residuals = cls._in_memory_tables()
        quarterly = tables["quarterly_station_effects.csv"].copy(deep=True)
        quarterly["COVERAGE_ELIGIBLE"] = False
        quarterly["COVERAGE_REASON"] = "no_data"
        for column in [
            "BIAS_none",
            "BIAS_fixed",
            "DELTA_ABS_BIAS",
            "CI95_LOW",
            "CI95_HIGH",
        ]:
            quarterly[column] = np.nan
        for column in ["POINT_CLASS", "CI_POSITION"]:
            quarterly[column] = pd.NA
        for column in ["BOOT_BLOCK_DAYS", "BOOT_REPLICATES", "BOOT_SEED"]:
            quarterly[column] = pd.array([pd.NA] * len(quarterly), dtype="Int64")
        tables["quarterly_station_effects.csv"] = quarterly
        tables["quarterly_network_summary.csv"] = summarize_quarterly_network(quarterly)
        tables["quarterly_station_reversals.csv"] = build_quarterly_reversals(
            quarterly, annual
        )
        tables["quarterly_block_sensitivity.csv"] = build_quarterly_block_sensitivity(
            results_by_block={block: quarterly.copy(deep=True) for block in (3, 7, 14)},
            n_boot=20,
        )
        tables = followup._order_followup_tables(tables, annual["STN"].tolist())
        return annual, prepared, tables, residuals

    def test_exact_constants_cover_fourteen_artifacts_and_seven_csv_schemas(self):
        self.assertEqual(len(followup.ANALYSIS_ARTIFACTS), 14)
        self.assertEqual(len(set(followup.ANALYSIS_ARTIFACTS)), 14)
        self.assertNotIn("고도보정_BIAS_후속진행보고_20260818.docx", followup.ANALYSIS_ARTIFACTS)
        self.assertEqual(set(followup.CSV_COLUMN_ORDERS), {
            "equivalent_bias_transition.csv",
            "delta_z_category_rates.csv",
            "quarterly_station_effects.csv",
            "quarterly_network_summary.csv",
            "quarterly_station_reversals.csv",
            "leave_one_quarter_out.csv",
            "quarterly_block_sensitivity.csv",
        })
        self.assertEqual(
            list(followup.CSV_COLUMN_ORDERS["leave_one_quarter_out.csv"]),
            followup.LEAVE_ONE_QUARTER_COLUMNS,
        )
        quarterly_columns = followup.CSV_COLUMN_ORDERS["quarterly_station_effects.csv"]
        self.assertIn("COVERAGE_REASON", quarterly_columns)
        self.assertNotIn("COVERAGE_STATUS", quarterly_columns)
        self.assertIn(
            "FRACTION_OF_EQUIVALENT",
            followup.CSV_COLUMN_ORDERS["equivalent_bias_transition.csv"],
        )
        self.assertIn(
            "FRACTION_WITHIN_BIN",
            followup.CSV_COLUMN_ORDERS["delta_z_category_rates.csv"],
        )
        self.assertNotIn(
            "PERCENT_OF_EQUIVALENT",
            followup.CSV_COLUMN_ORDERS["equivalent_bias_transition.csv"],
        )
        self.assertNotIn(
            "PERCENT_WITHIN_BIN",
            followup.CSV_COLUMN_ORDERS["delta_z_category_rates.csv"],
        )

    def test_production_path_guard_rejects_unsafe_targets_before_loading(self):
        safe_targets = [
            followup.FULLNET / "bias_feedback_followup",
            followup.FULLNET / "bias_feedback_followup_accept_review1",
            followup.FULLNET / "bias_feedback_followup_accept_a-b_2",
        ]
        for target in safe_targets:
            self.assertEqual(
                followup._validate_production_out_dir(target), target.resolve()
            )

        unsafe_targets = [
            Path("."),
            followup.FULLNET,
            followup.FULLNET.parent,
            followup.SOURCE_STATION_RESULTS,
            Path(bias_data.DB_PATH),
            Path(bias_data.DB_PATH).parent,
            followup.FULLNET / "bias_analysis",
            followup.FULLNET / "bias_feedback_analysis",
            followup.FULLNET / "arbitrary_output",
            followup.FULLNET / "bias_feedback_followup_accept_bad.token",
            followup.FULLNET / "nested" / "bias_feedback_followup",
            Path(".omc") / "bias_feedback_followup",
        ]
        with patch.object(
            followup,
            "load_production_inputs",
            side_effect=AssertionError("unsafe target must fail before loader"),
        ) as loader:
            for target in unsafe_targets:
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "production out_dir"):
                        followup.run_production_analysis(out_dir=target, n_boot=2000)
            loader.assert_not_called()

    def test_production_entrypoint_guards_bootstrap_and_forwards_canonical_contract(self):
        station_ids = [str(index) for index in range(529)]
        annual = pd.DataFrame(
            {
                "STN": station_ids,
                "BIAS_none": [0.4] * 529,
                "BIAS_fixed": [0.2] * 529,
                "DELTA_ABS_BIAS": [-0.2] * 529,
                "SIGNIFICANCE_CLASS": ["uncertain"] * 529,
                "mean_absdz_geom_km": [0.01] * 529,
            }
        )
        np.testing.assert_allclose(
            annual["DELTA_ABS_BIAS"],
            annual["BIAS_fixed"].abs() - annual["BIAS_none"].abs(),
        )
        daily_sentinel = object()
        safe = followup.FULLNET / "bias_feedback_followup_accept_entrypoint"
        literature = Path(".omc/sdd/bias-feedback-followup/literature_claim_map.md")

        with patch.object(
            followup,
            "load_production_inputs",
            return_value=(annual, daily_sentinel),
        ) as loader, patch.object(
            followup,
            "run_followup_analysis",
            return_value={"status": "complete"},
        ) as runner:
            result = followup.run_production_analysis(
                out_dir=safe,
                literature_source=literature,
                n_boot=2000,
                seed=19,
            )
        self.assertEqual(result, {"status": "complete"})
        loader.assert_called_once_with()
        kwargs = runner.call_args.kwargs
        self.assertIs(kwargs["annual_results"], annual)
        self.assertIs(kwargs["daily"], daily_sentinel)
        self.assertEqual(kwargs["out_dir"], safe.resolve())
        self.assertEqual(kwargs["expected_station_count"], 529)
        self.assertEqual(kwargs["n_boot"], 2000)
        self.assertEqual(kwargs["seed"], 19)
        self.assertTrue(kwargs["enforce_production_acceptance"])

        with patch.object(
            followup,
            "load_production_inputs",
            side_effect=AssertionError("invalid n_boot must fail before loader"),
        ) as invalid_loader:
            with self.assertRaisesRegex(ValueError, "2000"):
                followup.run_production_analysis(out_dir=safe, n_boot=1999)
            invalid_loader.assert_not_called()

    def test_network_validator_recomputes_every_summary_value_and_zero_na_contract(self):
        annual, _, tables, _ = self._in_memory_tables()
        followup._validate_followup_tables(annual, tables, n_boot=20)

        def expect_network_failure(mutator):
            changed = {name: frame.copy(deep=True) for name, frame in tables.items()}
            mutator(changed["quarterly_network_summary.csv"])
            with self.assertRaisesRegex(AssertionError, "network"):
                followup._validate_followup_tables(annual, changed, n_boot=20)

        def swap_values(frame, first, second):
            values = frame.loc[0, [first, second]].tolist()
            frame.loc[0, [first, second]] = values[::-1]

        expect_network_failure(
            lambda frame: swap_values(
                frame, "N_POINT_IMPROVEMENT", "N_POINT_WORSENING"
            )
        )
        expect_network_failure(
            lambda frame: swap_values(
                frame, "FRAC_POINT_IMPROVEMENT", "FRAC_POINT_WORSENING"
            )
        )
        expect_network_failure(
            lambda frame: frame.__setitem__(
                "COMMON_Q25_DELTA_ABS_BIAS",
                frame["COMMON_Q25_DELTA_ABS_BIAS"] + 0.01,
            )
        )
        expect_network_failure(
            lambda frame: frame.__setitem__(
                "MEDIAN_DELTA_ABS_BIAS", frame["MEDIAN_DELTA_ABS_BIAS"] + 0.01
            )
        )

        zero_annual, _, zero_tables, _ = self._zero_quarter_tables()
        followup._validate_followup_tables(zero_annual, zero_tables, n_boot=20)
        bad_zero = {name: frame.copy(deep=True) for name, frame in zero_tables.items()}
        bad_zero["quarterly_network_summary.csv"].loc[
            0, "COMMON_FRAC_POINT_IMPROVEMENT"
        ] = 0.0
        with self.assertRaisesRegex(AssertionError, "network"):
            followup._validate_followup_tables(zero_annual, bad_zero, n_boot=20)

    def test_sensitivity_validator_checks_all_three_blocks_and_zero_na_contract(self):
        annual, _, tables, _ = self._in_memory_tables()
        followup._validate_followup_tables(annual, tables, n_boot=20)

        def expect_sensitivity_failure(mutator):
            changed = {name: frame.copy(deep=True) for name, frame in tables.items()}
            frame = changed["quarterly_block_sensitivity.csv"]
            row = frame.index[(frame["QUARTER"].eq("Q1")) & frame["BLOCK_DAYS"].eq(3)][0]
            mutator(frame, row)
            with self.assertRaisesRegex(AssertionError, "sensitivity"):
                followup._validate_followup_tables(annual, changed, n_boot=20)

        expect_sensitivity_failure(
            lambda frame, row: frame.__setitem__(
                "CI_POSITION_AGREE_N",
                frame["CI_POSITION_AGREE_N"].mask(
                    frame.index == row, frame.loc[row, "N_ELIGIBLE"] + 1
                ),
            )
        )
        expect_sensitivity_failure(
            lambda frame, row: frame.__setitem__(
                "CI_POSITION_AGREE_FRACTION",
                frame["CI_POSITION_AGREE_FRACTION"].mask(frame.index == row, 0.123),
            )
        )
        expect_sensitivity_failure(
            lambda frame, row: frame.__setitem__(
                "MEDIAN_CI95_LOW",
                frame["MEDIAN_CI95_LOW"].mask(frame.index == row, np.nan),
            )
        )
        expect_sensitivity_failure(
            lambda frame, row: frame.__setitem__(
                "MEDIAN_CI95_LOW",
                frame["MEDIAN_CI95_LOW"].mask(
                    frame.index == row, frame.loc[row, "MEDIAN_CI95_HIGH"] + 1.0
                ),
            )
        )
        expect_sensitivity_failure(
            lambda frame, row: frame.__setitem__(
                "MEDIAN_CI95_WIDTH",
                frame["MEDIAN_CI95_WIDTH"].mask(frame.index == row, -0.1),
            )
        )

        zero_annual, _, zero_tables, _ = self._zero_quarter_tables()
        followup._validate_followup_tables(zero_annual, zero_tables, n_boot=20)
        bad_zero = {name: frame.copy(deep=True) for name, frame in zero_tables.items()}
        bad_zero["quarterly_block_sensitivity.csv"].loc[0, "CI_POSITION_AGREE_N"] = 1
        with self.assertRaisesRegex(AssertionError, "sensitivity"):
            followup._validate_followup_tables(zero_annual, bad_zero, n_boot=20)

    @unittest.skipUnless(SOURCE_STATION_RESULTS.is_file(), "로컬 BIAS 정본 CSV가 필요합니다")
    def test_markdown_reports_full_transition_delta_z_and_common_quarter_tables(self):
        annual, _, tables, _ = self._in_memory_tables()
        actual = pd.read_csv(
            SOURCE_STATION_RESULTS, encoding="utf-8-sig", dtype={"STN": str}
        )
        tables["equivalent_bias_transition.csv"] = build_equivalent_bias_transition(actual)
        tables["delta_z_category_rates.csv"] = build_delta_z_category_rates(actual)
        transition = tables["equivalent_bias_transition.csv"]
        delta_z = tables["delta_z_category_rates.csv"]
        network = tables["quarterly_network_summary.csv"]

        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            path = followup.write_followup_report(
                actual, tables, Path(temp_dir) / "report.md"
            )
            text = Path(path).read_text(encoding="utf-8")

        self.assertIn("3×3 전체 전이표", text)
        for before in followup.EQUIVALENT_LEVEL_ORDER:
            group = transition.loc[transition["BEFORE_ABS_BIAS_LEVEL"].eq(before)]
            counts = [
                int(group.loc[group["AFTER_ABS_BIAS_LEVEL"].eq(after), "N"].iloc[0])
                for after in followup.EQUIVALENT_LEVEL_ORDER
            ]
            self.assertIn(f"| {before} | {counts[0]} | {counts[1]} | {counts[2]} |", text)
        self.assertIn("낮음→낮음 12개", text)
        self.assertIn("높음→높음 30개", text)
        self.assertIn("같은 구간에 남은 경우는 96/115개", text)

        for label in ["유의미한 개선", "유의미한 악화", "차이 거의 없음", "판단 불확실"]:
            self.assertIn(label, text)
        for delta_bin in followup.DELTA_Z_BIN_ORDER:
            group = delta_z.loc[delta_z["DELTA_Z_BIN"].eq(delta_bin)]
            cells = []
            for label in followup.CLASS_ORDER:
                row = group.loc[group["SIGNIFICANCE_CLASS"].eq(label)].iloc[0]
                cells.append(
                    f"{int(row['N'])}개 ({100 * float(row['FRACTION_WITHIN_BIN']):.1f}%)"
                )
            self.assertIn(f"| {delta_bin} | " + " | ".join(cells) + " |", text)

        self.assertIn("점추정 세 구간", text)
        for point_label in ["개선", "작은 변화", "악화"]:
            self.assertIn(point_label, text)
        for row in network.itertuples(index=False):
            cells = []
            for point_class in followup.POINT_CLASS_ORDER:
                suffix = point_class.upper()
                cells.append(
                    f"{int(getattr(row, f'COMMON_N_POINT_{suffix}'))}개 "
                    f"({100 * float(getattr(row, f'COMMON_FRAC_POINT_{suffix}')):.1f}%)"
                )
            self.assertIn(f"| {row.QUARTER} | {int(row.COMMON_COHORT_N)} |", text)
            for cell in cells:
                self.assertIn(cell, text)

    def test_injected_runner_is_exact_reproducible_and_reuses_primary_seven_day_result(self):
        import matplotlib.image as mpimg

        annual, daily = self._injected_inputs()
        real_bootstrap = followup.bootstrap_quarterly_effects
        block_calls = []

        def recording_bootstrap(*args, **kwargs):
            block_days = kwargs.get("block_days", args[2] if len(args) > 2 else None)
            block_calls.append(block_days)
            return real_bootstrap(*args, **kwargs)

        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            root = Path(temp_dir)
            literature = root / "approved_literature.md"
            literature.write_text(
                "# 검토된 문헌 지도\n\n전문 검토가 끝난 근거 문서입니다.\n",
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            with patch.object(
                followup,
                "load_daily_paired_errors",
                side_effect=AssertionError("injected path must not load DB"),
            ), patch.object(
                followup,
                "bootstrap_quarterly_effects",
                side_effect=recording_bootstrap,
            ):
                first_summary = followup.run_followup_analysis(
                    annual_results=annual,
                    daily=daily,
                    literature_source=literature,
                    out_dir=first,
                    expected_station_count=5,
                    n_boot=20,
                    seed=17,
                )
            second_summary = followup.run_followup_analysis(
                annual_results=annual,
                daily=daily,
                literature_source=literature,
                out_dir=second,
                expected_station_count=5,
                n_boot=20,
                seed=17,
            )

            self.assertEqual(block_calls, [7, 3, 14])
            self.assertEqual(set(path.name for path in first.iterdir()), set(followup.ANALYSIS_ARTIFACTS))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in first.iterdir()))
            for filename in followup.PNG_ARTIFACTS:
                image = mpimg.imread(first / filename)
                self.assertGreaterEqual(image.ndim, 2)
                self.assertGreater(image.size, 0)

            tables = {
                filename: pd.read_csv(first / filename, encoding="utf-8-sig", dtype={"STN": str})
                for filename in followup.CSV_COLUMN_ORDERS
            }
            for filename, columns in followup.CSV_COLUMN_ORDERS.items():
                self.assertEqual(tables[filename].columns.tolist(), list(columns))
            self.assertEqual(len(tables["equivalent_bias_transition.csv"]), 9)
            self.assertTrue((tables["equivalent_bias_transition.csv"]["N"] == 0).any())
            self.assertEqual(tables["equivalent_bias_transition.csv"]["N"].sum(), 1)
            self.assertAlmostEqual(
                tables["equivalent_bias_transition.csv"]["FRACTION_OF_EQUIVALENT"].sum(), 1.0
            )
            self.assertEqual(len(tables["delta_z_category_rates.csv"]), 20)
            self.assertEqual(tables["delta_z_category_rates.csv"]["N"].sum(), 5)
            self.assertEqual(len(tables["quarterly_station_effects.csv"]), 20)
            self.assertEqual(len(tables["quarterly_network_summary.csv"]), 4)
            self.assertEqual(len(tables["quarterly_station_reversals.csv"]), 5)
            self.assertEqual(len(tables["leave_one_quarter_out.csv"]), 20)
            self.assertEqual(len(tables["quarterly_block_sensitivity.csv"]), 12)
            q = tables["quarterly_station_effects.csv"]
            self.assertEqual(
                set(map(tuple, q[["STN", "QUARTER"]].to_numpy())),
                {(stn, quarter) for stn in annual["STN"] for quarter in followup.QUARTER_ORDER},
            )
            missing = q.loc[q["STN"].eq("105") & q["QUARTER"].eq("Q4")].iloc[0]
            self.assertEqual(missing["COVERAGE_REASON"], "no_data")
            self.assertTrue(
                missing[["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"]]
                .isna()
                .all()
            )
            eligible = q.loc[q["COVERAGE_ELIGIBLE"]]
            self.assertTrue(eligible["BOOT_BLOCK_DAYS"].eq(7).all())
            self.assertTrue(eligible["BOOT_REPLICATES"].eq(20).all())
            sensitivity = tables["quarterly_block_sensitivity.csv"]
            seven = sensitivity.loc[sensitivity["BLOCK_DAYS"].eq(7)]
            self.assertTrue(seven["CI_POSITION_AGREE_N"].eq(seven["N_ELIGIBLE"]).all())
            self.assertTrue(seven["CI_POSITION_AGREE_FRACTION"].eq(1.0).all())

            raw_json = (first / "bias_feedback_followup_summary.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw_json)
            summary = json.loads(raw_json, parse_constant=lambda value: self.fail(value))
            self.assertEqual(set(summary), set(followup.SUMMARY_JSON_KEYS))
            self.assertEqual(summary, first_summary)
            self.assertEqual(first_summary, second_summary)
            report = (first / "bias_feedback_followup_report.md").read_text(encoding="utf-8")
            for required in [
                "2024년 달력상 분기",
                "85%",
                "최적이라고 확인된 값은 아니다",
                "고정된 시계 시간",
                "인과를 증명하지는 않는다",
                "공식 네 분류",
                "점추정 세 구간",
                "가운데 50% 범위",
            ]:
                self.assertIn(required, report)
            report_lower = report.lower()
            self.assertNotIn("dz_geom", report_lower)
            self.assertNotIn("stationary bootstrap", report_lower)
            self.assertNotIn("iqr", report_lower)

            core = list(followup.CSV_COLUMN_ORDERS) + ["bias_feedback_followup_summary.json"]
            first_hashes = {name: self._sha256(first / name) for name in core}
            second_hashes = {name: self._sha256(second / name) for name in core}
            self.assertEqual(first_hashes, second_hashes)

    def test_all_user_facing_plot_axes_use_plain_korean_labels(self):
        import matplotlib.figure

        annual, _ = self._injected_inputs()
        quarterly = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "QUARTER": followup.QUARTER_ORDER,
                "COVERAGE_ELIGIBLE": [True] * 4,
                "DELTA_ABS_BIAS": [-0.2, -0.1, 0.0, 0.1],
                "CI95_LOW": [-0.3, -0.2, -0.1, 0.0],
                "CI95_HIGH": [-0.1, 0.0, 0.1, 0.2],
            }
        )
        loq = pd.DataFrame(
            {
                "STN": ["101"] * 4,
                "EXCLUDED_QUARTER": followup.QUARTER_ORDER,
                "LOQ_STATUS": ["eligible"] * 4,
                "SHIFT_FROM_ANNUAL": [-0.1, 0.0, 0.1, 0.2],
                "ABS_SHIFT_FROM_ANNUAL": [0.1, 0.0, 0.1, 0.2],
            }
        )
        visible_axes = []

        def inspecting_savefig(figure, *args, **kwargs):
            for axis in figure.axes:
                visible_axes.extend([axis.get_title(), axis.get_xlabel(), axis.get_ylabel()])

        with patch.object(matplotlib.figure.Figure, "savefig", new=inspecting_savefig):
            followup.plot_equivalent_before_after(annual, Path(".omc") / "ignored-1.png")
            followup.plot_quarterly_effect(quarterly, Path(".omc") / "ignored-2.png")
            followup.plot_quarterly_heatmap(quarterly, Path(".omc") / "ignored-3.png")
            followup.plot_leave_one_quarter_out(loq, Path(".omc") / "ignored-4.png")
        joined = " ".join(visible_axes)
        lowered = joined.lower()
        for forbidden in ["station", "annual", "iqr", "dz_geom"]:
            self.assertNotIn(forbidden, lowered)
        self.assertIn("관측소", joined)
        self.assertIn("연간", joined)

    def test_directory_replacement_rolls_back_and_cleanup_failure_only_warns(self):
        import os

        with TemporaryDirectory(dir=Path(".omc") / "test_tmp") as temp_dir:
            root = Path(temp_dir)
            final = root / "final"
            staging = root / ".final.staging-test"
            final.mkdir()
            staging.mkdir()
            (final / "old.txt").write_text("old", encoding="utf-8")
            (staging / "new.txt").write_text("new", encoding="utf-8")
            real_replace = os.replace
            call_count = 0

            def fail_publish(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("publish failed")
                return real_replace(source, destination)

            with patch.object(followup.os, "replace", side_effect=fail_publish):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    followup._publish_staging(staging, final)
            self.assertTrue((final / "old.txt").is_file())
            self.assertTrue((staging / "new.txt").is_file())
            self.assertFalse(any("backup-" in path.name for path in root.iterdir()))

            final2 = root / "final2"
            staging2 = root / ".final2.staging-test"
            final2.mkdir()
            staging2.mkdir()
            (final2 / "old.txt").write_text("old", encoding="utf-8")
            (staging2 / "new.txt").write_text("new", encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch.object(followup.shutil, "rmtree", side_effect=OSError("cleanup failed")):
                    followup._publish_staging(staging2, final2)
            self.assertTrue((final2 / "new.txt").is_file())
            self.assertTrue(any("backup" in str(item.message).lower() for item in caught))

    def test_production_acceptance_contract_checks_all_canonical_counts(self):
        labels = (
            ["meaningful_improvement"] * 221
            + ["meaningful_worsening"] * 109
            + ["practically_equivalent"] * 115
            + ["uncertain"] * 84
        )
        effects = [-0.2] * 221 + [0.2] * 109 + [0.0] * 199
        annual = pd.DataFrame(
            {
                "STN": [str(index) for index in range(529)],
                "BIAS_none": [0.2] * 529,
                "BIAS_fixed": [0.2] * 529,
                "DELTA_ABS_BIAS": effects,
                "SIGNIFICANCE_CLASS": labels,
                "mean_absdz_geom_km": [0.01] * 529,
            }
        )
        empty_daily = pd.DataFrame(columns=["STN", "DATE", "sum_none", "sum_fixed", "count"])
        stations = annual["STN"].tolist()
        quarterly = bootstrap_quarterly_effects(
            empty_daily, n_boot=2000, block_days=7, seed=17, stations=stations
        )
        tables = {
            "equivalent_bias_transition.csv": build_equivalent_bias_transition(annual),
            "delta_z_category_rates.csv": build_delta_z_category_rates(annual),
            "quarterly_station_effects.csv": quarterly,
            "quarterly_network_summary.csv": summarize_quarterly_network(quarterly),
            "quarterly_station_reversals.csv": build_quarterly_reversals(quarterly, annual),
            "leave_one_quarter_out.csv": build_leave_one_quarter_out(empty_daily, annual),
            "quarterly_block_sensitivity.csv": build_quarterly_block_sensitivity(
                daily=empty_daily,
                n_boot=2000,
                seed=17,
                primary_result=quarterly,
                stations=stations,
            ),
        }
        followup.validate_production_acceptance(
            annual_results=annual,
            tables=tables,
            annual_validation={"max_abs_bias_none": 0.0, "max_abs_bias_fixed": 0.0},
        )
        bad = annual.copy()
        bad.loc[0, "SIGNIFICANCE_CLASS"] = "uncertain"
        with self.assertRaisesRegex(AssertionError, "official class counts"):
            followup.validate_production_acceptance(
                annual_results=bad,
                tables=tables,
                annual_validation={"max_abs_bias_none": 0.0, "max_abs_bias_fixed": 0.0},
            )

if __name__ == "__main__":
    unittest.main()
