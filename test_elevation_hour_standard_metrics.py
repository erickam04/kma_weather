"""고도×시간대 표준지표 분석의 계산·집계 계약을 검증한다."""

import importlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


try:
    analysis = importlib.import_module("analyze_elevation_hour_standard_metrics")
except ModuleNotFoundError as exc:
    if exc.name != "analyze_elevation_hour_standard_metrics":
        raise
    analysis = None


def cohort_fixture(stns=("1", "2"), bands=("100m 미만", "100m 미만"), main=None):
    if main is None:
        main = [True] * len(stns)
    return pd.DataFrame({
        "STN": list(stns),
        "지점명": [f"예시-{stn}" for stn in stns],
        "elev_m": [50.0 if band == "100m 미만" else 600.0 for band in bands],
        "ELEVATION_BAND": list(bands),
        "COHORT_MAIN": list(main),
    })


def agg_row(stn, hour, method, n, se, se2, sa):
    return {
        "STN": str(stn), "bucket": hour, "METHOD": method,
        "n": n, "se": se, "se2": se2, "sa": sa,
    }


def full_day_agg(stn, none_error, fixed_error, n=2):
    rows = []
    for hour in range(24):
        for method, error in (("none", none_error), ("fixed_lapse", fixed_error)):
            rows.append(agg_row(
                stn, hour, method, n,
                se=n * error, se2=n * error ** 2, sa=n * abs(error),
            ))
    return rows


class ElevationHourStandardMetricsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            analysis,
            "승인된 station×hour 표준지표 분석 모듈이 아직 구현되지 않았음",
        )

    def test_station_hour_metrics_use_saved_sums_and_only_main_cohort(self):
        # 합계를 잘못 나누거나 COHORT_MAIN=False 관측소를 포함하면 실패한다.
        agg = pd.DataFrame([
            agg_row("1", 0, "none", 4, 4, 10, 6),
            agg_row("1", 0, "fixed_lapse", 4, 2, 4, 4),
            agg_row("9", 0, "none", 4, 40, 400, 40),
            agg_row("9", 0, "fixed_lapse", 4, 20, 100, 20),
        ])
        cohort = cohort_fixture(
            stns=("1", "9"), bands=("100m 미만", "500m 이상"), main=("true", "false"),
        )

        out = analysis.prepare_station_hour_metrics(agg, cohort, require_complete=False)

        self.assertEqual(set(out.STN), {"1"})
        none = out.loc[out.METHOD.eq("none")].iloc[0]
        fixed = out.loc[out.METHOD.eq("fixed_lapse")].iloc[0]
        self.assertEqual(none.BIAS_C, 1.0)
        self.assertEqual(none.MAE_C, 1.5)
        self.assertAlmostEqual(none.RMSE_C, np.sqrt(2.5))
        self.assertEqual(fixed.BIAS_C, 0.5)
        self.assertEqual(fixed.MAE_C, 1.0)
        self.assertEqual(fixed.RMSE_C, 1.0)

    def test_cohort_main_parser_accepts_only_explicit_boolean_encodings(self):
        # 문자열 "false"를 bool("false")처럼 True로 취급하거나 임의 값을 허용하면 실패한다.
        parsed = analysis.parse_cohort_main(pd.Series([
            True, False, "true", " FALSE ", "1", "0", 1, 0,
        ]))
        self.assertEqual(parsed.tolist(), [True, False, True, False, True, False, True, False])
        for invalid in (None, np.nan, "", "yes", "2", 2, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "COHORT_MAIN"):
                    analysis.parse_cohort_main(pd.Series([invalid]))

    def test_station_hour_metrics_reject_unpaired_method_counts(self):
        # none/fixed의 표본 수가 다른데도 공정 비교로 통과시키면 실패한다.
        agg = pd.DataFrame([
            agg_row("1", 0, "none", 4, 4, 10, 6),
            agg_row("1", 0, "fixed_lapse", 3, 2, 4, 4),
        ])
        with self.assertRaisesRegex(ValueError, "paired counts"):
            analysis.prepare_station_hour_metrics(
                agg, cohort_fixture(stns=("1",), bands=("100m 미만",)),
                require_complete=False,
            )

    def test_internal_error_identities_are_enforced(self):
        # 표준 오차 부등식 또는 그 sufficient-statistics 형태를 위반해도 통과시키면 실패한다.
        base = analysis.prepare_station_hour_metrics(
            pd.DataFrame([
                agg_row("1", 0, "none", 4, 4, 10, 6),
                agg_row("1", 0, "fixed_lapse", 4, 2, 4, 4),
            ]),
            cohort_fixture(stns=("1",), bands=("100m 미만",)),
            require_complete=False,
        )
        analysis.validate_error_identities(base)
        mutations = (
            ("BIAS_C", 2.0, "BIAS.*MAE"),
            ("RMSE_C", 0.1, "MAE.*RMSE"),
            ("ABS_ERROR_SUM_C", 1, "se.*sa"),
            ("ERROR_SQ_SUM_C2", 0, r"se\^2.*n\*se2"),
        )
        for column, value, message in mutations:
            with self.subTest(column=column):
                broken = base.copy()
                broken.loc[broken.index[0], column] = value
                with self.assertRaisesRegex(ValueError, message):
                    analysis.validate_error_identities(broken)

    def test_reference_bias_alignment_checks_every_main_key_bias_count_and_total(self):
        # 일부 키만 대조하거나 BIAS/n/방법별 합계 중 하나를 생략하면 실패한다.
        station = analysis.prepare_station_hour_metrics(
            pd.DataFrame([
                agg_row("1", 0, "none", 4, 4, 10, 6),
                agg_row("1", 0, "fixed_lapse", 4, 2, 4, 4),
            ]),
            cohort_fixture(stns=("1",), bands=("100m 미만",)),
            require_complete=False,
        )
        reference = pd.DataFrame({
            "STN": ["1", "9"], "HOUR": [0, 0], "n_hours": [4, 99],
            "BIAS_BEFORE_C": [1.0, 9.0], "BIAS_AFTER_C": [0.5, 9.0],
            "COHORT_MAIN": ["true", "false"],
        })
        result = analysis.validate_reference_bias_alignment(
            station, reference, expected_method_total=4,
        )
        self.assertEqual(result["n_main_station_hours"], 1)
        self.assertEqual(result["paired_n_by_method"], {"none": 4, "fixed_lapse": 4})
        self.assertEqual(result["max_abs_bias_diff_c"], {"none": 0.0, "fixed_lapse": 0.0})
        self.assertEqual(result["max_abs_n_diff"], {"none": 0, "fixed_lapse": 0})

        bad_bias = reference.copy()
        bad_bias.loc[0, "BIAS_AFTER_C"] = 0.6
        with self.assertRaisesRegex(ValueError, "BIAS mismatch"):
            analysis.validate_reference_bias_alignment(
                station, bad_bias, expected_method_total=4,
            )
        with self.assertRaisesRegex(ValueError, "paired n total"):
            analysis.validate_reference_bias_alignment(
                station, reference, expected_method_total=5,
            )

    def test_hourly_summary_is_station_equal_weight_and_has_no_delta_columns(self):
        # n=1000인 관측소가 n=1인 관측소보다 큰 집단 가중치를 받으면 실패한다.
        agg = pd.DataFrame([
            agg_row("1", 0, "none", 1, 2, 4, 2),
            agg_row("1", 0, "fixed_lapse", 1, 1, 1, 1),
            agg_row("2", 0, "none", 1000, 8000, 64000, 8000),
            agg_row("2", 0, "fixed_lapse", 1000, 4000, 16000, 4000),
        ])
        station = analysis.prepare_station_hour_metrics(
            agg, cohort_fixture(), require_complete=False,
        )

        row = analysis.summarize_hourly_by_elevation(station).iloc[0]

        self.assertEqual(row.N_STATIONS, 2)
        self.assertEqual(row.NONE_BIAS_MEDIAN_C, 5.0)
        self.assertEqual(row.NONE_BIAS_Q25_C, 3.5)
        self.assertEqual(row.NONE_BIAS_Q75_C, 6.5)
        self.assertEqual(row.NONE_BIAS_IQR_C, 3.0)
        forbidden = ("DELTA", "IMPROVEMENT", "개선")
        self.assertFalse(any(token in str(col).upper() for col in row.index for token in forbidden[:2]))
        self.assertFalse(any(forbidden[2] in str(col) for col in row.index))

    def test_daynight_metrics_pool_hourly_sums_within_station(self):
        # 시간별 RMSE를 단순 평균하지 않고 선택 시간의 제곱오차 합으로 다시 계산해야 한다.
        rows = []
        for hour in range(24):
            if 9 <= hour <= 17:
                n, none = 1, 2.0
            elif hour >= 21 or hour <= 5:
                n, none = 3, 4.0
            else:
                n, none = 7, 99.0
            rows.extend([
                agg_row("1", hour, "none", n, n * none, n * none ** 2, n * abs(none)),
                agg_row("1", hour, "fixed_lapse", n, n, n, n),
            ])
        station = analysis.prepare_station_hour_metrics(
            pd.DataFrame(rows), cohort_fixture(stns=("1",), bands=("100m 미만",)),
        )

        periods = analysis.build_station_daynight_metrics(station).set_index(["PERIOD", "METHOD"])

        self.assertEqual(periods.loc[("주간(09~17시)", "none"), "BIAS_C"], 2.0)
        self.assertEqual(periods.loc[("주간(09~17시)", "none"), "RMSE_C"], 2.0)
        self.assertEqual(periods.loc[("야간(21~05시)", "none"), "MAE_C"], 4.0)
        self.assertEqual(periods.loc[("야간(21~05시)", "fixed_lapse"), "RMSE_C"], 1.0)
        summary = analysis.summarize_daynight_by_elevation(periods.reset_index())
        self.assertEqual(set(summary.PERIOD), {"주간(09~17시)", "야간(21~05시)"})

    def test_plotting_creates_separate_standard_metric_files(self):
        rows = []
        cohort_rows = []
        for stn, band, none, fixed in (
            ("1", "100m 미만", 2.0, 1.0),
            ("2", "100~500m 미만", 3.0, 2.0),
            ("3", "500m 이상", 4.0, 3.0),
        ):
            rows.extend(full_day_agg(stn, none, fixed))
            cohort_rows.append({
                "STN": stn, "지점명": f"예시-{stn}", "elev_m": 50.0,
                "ELEVATION_BAND": band, "COHORT_MAIN": True,
            })
        station = analysis.prepare_station_hour_metrics(
            pd.DataFrame(rows), pd.DataFrame(cohort_rows),
        )
        summary = analysis.summarize_hourly_by_elevation(station)

        plots = Path("WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect/plots")
        # 실제 주513 그래프를 테스트 자료로 덮어쓰지 않도록 저장 호출만 가로챈다.
        with patch("PIL.Image.Image.save") as save_mock:
            paths = analysis.plot_hourly_standard_metrics(summary, plots)
        self.assertEqual({path.name for path in paths}, {
            "04_hourly_mae_standard_metrics.png",
            "05_hourly_rmse_standard_metrics.png",
            "06_hourly_bias_standard_metrics.png",
        })
        self.assertEqual(save_mock.call_count, 3)
        self.assertEqual(
            {Path(call.args[0]).name for call in save_mock.call_args_list},
            {path.name for path in paths},
        )

    def test_plot_pixels_do_not_depend_on_iqr_values(self):
        # IQR 음영을 다시 그리면 중앙값이 같아도 두 렌더가 달라져 실패한다.
        rows, cohort_rows = [], []
        for stn, band in (("1", "100m 미만"), ("2", "100~500m 미만"), ("3", "500m 이상")):
            rows.extend(full_day_agg(stn, 2.0, 1.0))
            cohort_rows.append({
                "STN": stn, "지점명": stn, "elev_m": 50.0,
                "ELEVATION_BAND": band, "COHORT_MAIN": True,
            })
        station = analysis.prepare_station_hour_metrics(pd.DataFrame(rows), pd.DataFrame(cohort_rows))
        summary = analysis.summarize_hourly_by_elevation(station)
        altered = summary.copy()
        for column in altered.columns:
            if "_Q25_C" in column:
                altered[column] = 0.0
            elif "_Q75_C" in column:
                altered[column] = 99.0

        first, second = io.BytesIO(), io.BytesIO()
        analysis._draw_metric_plot(summary, "BIAS", first)
        analysis._draw_metric_plot(altered, "BIAS", second)
        self.assertEqual(first.getvalue(), second.getvalue())

    def test_bias_uses_shared_y_limits_and_draws_zero_baseline(self):
        # 패널별 BIAS 축을 따로 잡거나 0 기준선을 생략하면 실패한다.
        rows, cohort_rows = [], []
        for stn, band, none, fixed in (
            ("1", "100m 미만", -2.0, 1.0),
            ("2", "100~500m 미만", 3.0, 2.0),
            ("3", "500m 이상", -0.5, 0.5),
        ):
            rows.extend(full_day_agg(stn, none, fixed))
            cohort_rows.append({
                "STN": stn, "지점명": stn, "elev_m": 50.0,
                "ELEVATION_BAND": band, "COHORT_MAIN": True,
            })
        station = analysis.prepare_station_hour_metrics(pd.DataFrame(rows), pd.DataFrame(cohort_rows))
        summary = analysis.summarize_hourly_by_elevation(station)

        limits = analysis._panel_y_limits(summary, "BIAS")

        self.assertEqual(len(set(limits.values())), 1)
        low, high = limits["100m 미만"]
        self.assertLess(low, -2.0)
        self.assertGreater(high, 3.0)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

        from PIL import Image
        rendered = io.BytesIO()
        analysis._draw_metric_plot(summary, "BIAS", rendered)
        rendered.seek(0)
        image = Image.open(rendered).convert("RGB")
        y_zero = round(610 - 18 - (0.0 - low) / (high - low) * (610 - 115 - 36))
        self.assertIn((119, 119, 119), {
            image.getpixel((120, y)) for y in range(y_zero - 2, y_zero + 3)
        })

    def test_manifest_declares_pooled_period_and_independent_panel_scales(self):
        # 주야 집계 정의나 패널별 y축 계약이 manifest에서 빠지면 실패한다.
        manifest = analysis.build_run_summary(
            n_main_stations=513,
            n_station_hours=12312,
            band_counts={"100m 미만": 315, "100~500m 미만": 158, "500m 이상": 40},
            outputs=["example.csv", "plots/06_hourly_bias_standard_metrics.png"],
            validation={"paired_n_by_method": {"none": 4454204, "fixed_lapse": 4454204}},
            input_hashes={"station_hour_bias_csv_sha256": "abc"},
        )
        self.assertEqual(manifest["panel_y_scales"], "independent")
        self.assertEqual(manifest["bias_panel_y_scales"], "shared")
        self.assertIn("plots/06_hourly_bias_standard_metrics.png", manifest["outputs"])
        self.assertEqual(
            manifest["period_aggregation"],
            "관측소 내부 9시간의 n·se·se2·sa를 pooled해 표준 period 지표를 재계산한 후, 관측소 간 동일가중 중앙값·IQR로 요약",
        )
        self.assertEqual(manifest["validation"]["paired_n_by_method"]["none"], 4454204)
        self.assertEqual(manifest["input_hashes"]["station_hour_bias_csv_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
