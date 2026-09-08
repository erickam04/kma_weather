"""고도별 시간대 보정효과: 실제 수치·짝 비교·집계 순서를 검증한다."""

import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


try:
    analysis = importlib.import_module('analysis.analyze_elevation_hour_effect')
except ModuleNotFoundError as exc:
    if exc.name != 'analysis.analyze_elevation_hour_effect':
        raise
    analysis = None


def hourly_fixture(stn="1", day_gain=2.0, night_gain=1.0):
    rows = []
    for hour in range(24):
        gain = day_gain if 9 <= hour <= 17 else night_gain
        rows.append({
            "STN": stn, "HOUR": hour, "n_hours": 20,
            "BIAS_none": 10.0, "BIAS_fixed": 10.0 - gain,
        })
    return pd.DataFrame(rows)


def cohort_fixture(stns=("1",)):
    return pd.DataFrame({
        "STN": list(stns), "지점명": ["예시"] * len(stns),
        "elev_m": [50.0] * len(stns),
        "ELEVATION_BAND": ["100m 미만"] * len(stns),
        "COHORT_MAIN": [True] * len(stns),
    })


class ElevationHourEffectTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(analysis, "승인된 보정효과 집계 모듈이 아직 구현되지 않았음")

    def test_runner_uses_configured_metadata_path(self):
        external = analysis.ROOT.parent / "external_metadata.csv"
        with patch.object(analysis, "META_PATH", external, create=True):
            with patch.object(analysis.pd, "read_csv", return_value=pd.DataFrame()):
                with patch.object(Path, "mkdir"):
                    with patch.object(analysis, "load_station_meta", side_effect=RuntimeError("captured")) as loader:
                        with self.assertRaisesRegex(RuntimeError, "captured"):
                            analysis.run(output=Path("unused_test_output"))
        loader.assert_called_once_with(external)

    def test_source_label_supports_external_metadata_without_absolute_path(self):
        label = getattr(analysis, "source_label", None)
        self.assertTrue(callable(label), "외부 입력 경로를 기록할 수 있어야 합니다")
        self.assertEqual(label(analysis.ROOT / "meta.csv"), "meta.csv")
        self.assertEqual(label(analysis.ROOT.parent / "external.csv"), "external/external.csv")

    def test_gain_is_closeness_to_zero_not_signed_shift(self):
        # 부호 반전이나 같은 방향 이동을 단순 온도 감소로 개선 판정하면 실패한다.
        source = pd.DataFrame({
            "STN": ["1"] * 4, "HOUR": [0, 1, 2, 3], "n_hours": [10] * 4,
            "BIAS_none": [0.8, -0.8, 0.3, 0.0],
            "BIAS_fixed": [-0.2, -0.2, -0.5, 0.2],
        })
        out = analysis.prepare_hourly(source, cohort_fixture(), require_complete=False)
        np.testing.assert_allclose(out.IMPROVEMENT_C, [0.6, 0.6, -0.2, -0.2])
        np.testing.assert_allclose(out.BIAS_BEFORE_C, [0.8, -0.8, 0.3, 0.0])

    def test_median_is_taken_after_within_station_comparison(self):
        # 집단 중앙값의 차이(0)를 개선량 중앙값(-1)으로 바꾸어 쓰면 실패한다.
        source = pd.DataFrame({
            "STN": ["1", "2", "3"], "HOUR": [0] * 3, "n_hours": [10] * 3,
            "BIAS_none": [0, 1, 10], "BIAS_fixed": [1, 2, 0],
        })
        prepared = analysis.prepare_hourly(source, cohort_fixture(("1", "2", "3")), False)
        row = analysis.group_hourly_summary(prepared).iloc[0]
        self.assertEqual(row.BIAS_BEFORE_MEDIAN_C, 1.0)
        self.assertEqual(row.BIAS_AFTER_MEDIAN_C, 1.0)
        self.assertEqual(row.IMPROVEMENT_MEDIAN_C, -1.0)

    def test_each_station_has_equal_weight_in_group_mean(self):
        source = pd.DataFrame({
            "STN": ["1", "2"], "HOUR": [0, 0], "n_hours": [1, 1000],
            "BIAS_none": [2, 8], "BIAS_fixed": [1, 4],
        })
        prepared = analysis.prepare_hourly(source, cohort_fixture(("1", "2")), False)
        row = analysis.group_hourly_summary(prepared).iloc[0]
        self.assertEqual(row.N_STATIONS, 2)
        self.assertEqual(row.IMPROVEMENT_MEAN_C, 2.5)

    def test_constant_effect_offset_does_not_change_day_night_contrast(self):
        # 하루 내내 효과가 2도 더 큰 것과 주야 차이가 더 큰 것을 구분한다.
        source = pd.concat([
            hourly_fixture("1", 2, 1), hourly_fixture("2", 4, 3),
            hourly_fixture("3", 4, 1),
        ], ignore_index=True)
        prepared = analysis.prepare_hourly(source, cohort_fixture(("1", "2", "3")))
        contrasts = analysis.station_daynight(prepared).set_index("STN")
        self.assertEqual(contrasts.loc["1", "DAY_NIGHT_ADVANTAGE_C"], 1)
        self.assertEqual(contrasts.loc["2", "DAY_NIGHT_ADVANTAGE_C"], 1)
        self.assertEqual(contrasts.loc["3", "DAY_NIGHT_ADVANTAGE_C"], 3)

    def test_day_night_group_contrast_is_not_difference_of_group_medians(self):
        source = pd.concat([
            hourly_fixture("1", 0, 1), hourly_fixture("2", 1, 2),
            hourly_fixture("3", 10, 0),
        ], ignore_index=True)
        hourly = analysis.prepare_hourly(source, cohort_fixture(("1", "2", "3")))
        row = analysis.group_daynight_summary(analysis.station_daynight(hourly)).iloc[0]
        self.assertEqual(row.DAY_IMPROVEMENT_MEDIAN_C, 1)
        self.assertEqual(row.NIGHT_IMPROVEMENT_MEDIAN_C, 1)
        self.assertEqual(row.DAY_NIGHT_ADVANTAGE_MEDIAN_C, -1)

    def test_day_night_uses_nine_hours_each_and_excludes_transition_only_from_summary(self):
        source = hourly_fixture()
        source.loc[source.HOUR.isin([6, 7, 8, 18, 19, 20]), "BIAS_fixed"] = 100
        prepared = analysis.prepare_hourly(source, cohort_fixture())
        row = analysis.station_daynight(prepared).iloc[0]
        self.assertEqual(len(prepared), 24)
        self.assertEqual(row.DAY_IMPROVEMENT_C, 2.0)
        self.assertEqual(row.NIGHT_IMPROVEMENT_C, 1.0)
        self.assertEqual(row.DAY_HOURS, 9)
        self.assertEqual(row.NIGHT_HOURS, 9)

    def test_period_averages_hours_equally_not_by_observation_count(self):
        source = hourly_fixture(day_gain=0, night_gain=0)
        source.loc[source.HOUR.eq(9), ["BIAS_fixed", "n_hours"]] = [1, 1000]
        row = analysis.station_daynight(analysis.prepare_hourly(source, cohort_fixture())).iloc[0]
        self.assertEqual(row.DAY_IMPROVEMENT_C, 1.0)

    def test_duplicate_station_hour_is_rejected(self):
        source = hourly_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.prepare_hourly(pd.concat([source, source.iloc[[0]]]), cohort_fixture())

    def test_missing_hour_is_not_silently_pooled(self):
        with self.assertRaisesRegex(ValueError, "24"):
            analysis.prepare_hourly(hourly_fixture().iloc[:-1], cohort_fixture())

    def test_missing_metadata_station_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cohort"):
            analysis.prepare_hourly(hourly_fixture("99"), cohort_fixture())

    def test_nonfinite_bias_and_nonpositive_count_are_rejected(self):
        source = hourly_fixture()
        source.loc[0, "BIAS_none"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.prepare_hourly(source, cohort_fixture())
        source = hourly_fixture()
        source.loc[0, "n_hours"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            analysis.prepare_hourly(source, cohort_fixture())

    def test_elevation_boundaries_are_nonoverlapping(self):
        want = ["100m 미만", "100m 미만", "100~500m 미만", "100~500m 미만", "500m 이상"]
        got = [analysis.elevation_band(z) for z in [0.7, 99.9, 100, 499.9, 500]]
        self.assertEqual(got, want)

    def test_history_ignores_future_metadata_and_identifies_actual_crossing(self):
        meta = pd.DataFrame({
            "지점": ["1", "2", "2", "1"],
            "시작일": pd.to_datetime(["2010-01-01", "2010-01-01", "2024-09-11", "2025-01-01"]),
            "종료일": pd.to_datetime(["2025-01-01", "2024-09-11", None, None]),
            "노장해발고도(m)": [50., 111., 68., 800.],
        })
        out = analysis.metadata_elevation_history(meta, ["1", "2"]).set_index("STN")
        self.assertFalse(out.loc["1", "BAND_CHANGED"])
        self.assertEqual(out.loc["1", "ELEV_MAX_M"], 50)
        self.assertTrue(out.loc["2", "BAND_CHANGED"])
        self.assertEqual(out.loc["2", "ELEV_MIN_M"], 68)

    def test_main_cohort_excludes_low_coverage_and_band_crossing_separately(self):
        factors = pd.DataFrame({"STN": ["1", "2", "3"], "지점명": ["가", "나", "다"], "elev_m": [50., 111., 600.]})
        coverage = pd.DataFrame({"STN": ["1", "2", "3"], "COVERAGE_DAYS": [200, 365, 20], "COVERAGE_MONTHS": [10, 12, 2]})
        meta = pd.DataFrame({
            "지점": ["1", "2", "2", "3"],
            "시작일": pd.to_datetime(["2010-01-01", "2010-01-01", "2024-09-11", "2010-01-01"]),
            "종료일": pd.to_datetime([None, "2024-09-11", None, None]),
            "노장해발고도(m)": [50., 111., 68., 600.],
        })
        out = analysis.build_cohort(factors, coverage, meta).set_index("STN")
        self.assertEqual(out.COHORT_MAIN.to_dict(), {"1": True, "2": False, "3": False})
        self.assertEqual(out.loc["2", "MAIN_EXCLUDE_REASON"], "고도 구간 변경")
        self.assertEqual(out.loc["3", "MAIN_EXCLUDE_REASON"], "연간 자료 부족")

    def test_reference_date_inactive_station_uses_only_2024_valid_metadata(self):
        # 866 회귀 반례: 파일상 마지막 과거 행이나 미래 행을 대표고도로 쓰면 실패한다.
        factors = pd.DataFrame({"STN": ["866"], "지점명": ["과거 지점"], "elev_m": [105.5]})
        coverage = pd.DataFrame({"STN": ["866"], "COVERAGE_DAYS": [93], "COVERAGE_MONTHS": [4]})
        meta = pd.DataFrame({
            "지점": ["866"] * 3, "지점명": ["2024 지점", "미래 지점", "과거 지점"],
            "시작일": pd.to_datetime(["2021-10-15", "2025-01-01", "2002-01-01"]),
            "종료일": pd.to_datetime(["2024-04-04", None, "2003-01-01"]),
            "노장해발고도(m)": [588.19, 900., 105.5],
        })
        row = analysis.build_cohort(factors, coverage, meta).iloc[0]
        self.assertEqual(row.ELEVATION_BAND, "500m 이상")
        self.assertEqual(row.elev_m, 588.19)
        self.assertEqual(row["지점명"], "2024 지점")
        self.assertEqual(row.FACTOR_ELEV_M, 105.5)
        self.assertEqual(row.FACTOR_STATION_NAME, "과거 지점")
        self.assertEqual(row.ELEV_REFERENCE_BASIS, "2024년 내 마지막 유효 정보")
        self.assertTrue(row.FACTOR_METADATA_MISMATCH)
        self.assertFalse(row.COHORT_MAIN)

    def test_raw_aggregation_means_signed_errors_before_taking_absolute_value(self):
        pairs = pd.DataFrame({
            "STN": ["1", "1", "1"],
            "TM": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-07-01"]),
            "error_none": [2., -2., 4.], "error_fixed": [1., -3., 1.],
        })
        seasonal = analysis.aggregate_pairs_by_hour(pairs, seasonal=True).set_index("SEASON")
        self.assertEqual(seasonal.loc["겨울(1·2·12월)", "BIAS_none"], 0.)
        self.assertEqual(seasonal.loc["겨울(1·2·12월)", "BIAS_fixed"], -1.)
        self.assertEqual(seasonal.loc["여름(6~8월)", "BIAS_none"], 4.)
        self.assertEqual(seasonal.loc["겨울(1·2·12월)", "n_hours"], 2)

    def test_raw_aggregation_rejects_duplicate_time(self):
        pairs = pd.DataFrame({
            "STN": ["1", "1"], "TM": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "error_none": [2., 2.], "error_fixed": [1., 1.],
        })
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.aggregate_pairs_by_hour(pairs, seasonal=True)

    def test_coverage_recheck_rejects_day_or_month_mismatch(self):
        cohort = pd.DataFrame({"STN": ["1"], "COVERAGE_DAYS": [200], "COVERAGE_MONTHS": [9]})
        audit = pd.DataFrame({"STN": ["1"], "PAIRED_DAYS": [200], "PAIRED_MONTHS": [9]})
        analysis.verify_coverage(cohort, audit)
        for column in ("PAIRED_DAYS", "PAIRED_MONTHS"):
            bad = audit.copy()
            bad.loc[0, column] -= 1
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                analysis.verify_coverage(cohort, bad)

    def test_seasonal_cohort_requires_all_hours_in_all_four_seasons(self):
        rows = [{"STN": stn, "SEASON": season, "HOUR": hour}
                for stn in ("1", "2", "3", "4")
                for season in analysis.SEASONS for hour in range(24)]
        hourly = pd.DataFrame(rows)
        # 2번은 겨울 1시각 부족, 3번은 겨울 전체 부족, 4번은 주분석 표본 밖이다.
        hourly = hourly.loc[~(hourly.STN.eq("2") & hourly.SEASON.eq(analysis.SEASONS[3]) & hourly.HOUR.eq(23))]
        hourly = hourly.loc[~(hourly.STN.eq("3") & hourly.SEASON.eq(analysis.SEASONS[3]))]
        self.assertEqual(analysis.common_seasonal_ids(hourly, {"1", "2", "3"}), {"1"})


if __name__ == "__main__":
    unittest.main()
