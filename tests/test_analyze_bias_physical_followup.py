"""BIAS 물리요인 후속분석 production runner 계약 테스트.

작은 주입형 fixture로 원자료 정합성·시간별 구조·조건별 BIAS·artifact 게시를
검증한다. 실제 529개 production runner는 이 테스트에서 실행하지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import gc
import tracemalloc
import weakref
from contextlib import contextmanager
import shutil
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from weather.find_stations import get_distance_km
from analysis.bias_physical_followup import (
    classify_spatial_temperature_structure,
    exact_idw_none_weighted_mean,
    ols_temperature_elevation_slope,
    weighted_temperature_elevation_slope,
)


try:
    runner = importlib.import_module('analysis.analyze_bias_physical_followup')
    MODULE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    runner = None
    MODULE_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = ROOT / ".omc" / "tmp" / "task3-tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def _workspace_temp_dir():
    """Windows runtime의 tempfile ACL 문제를 피한 workspace 임시폴더."""
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir(parents=False, exist_ok=False)
    try:
        yield str(path)
    finally:
        if path.exists():
            shutil.rmtree(path)


def _create_delete_mode_sqlite(path: Path) -> None:
    """production source identity test용 sidecar 없는 실제 DELETE SQLite 파일."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE IF NOT EXISTS probe (value INTEGER)")
        connection.commit()
    finally:
        connection.close()


def _attach_fixture_snapshot_metadata(
    sources: dict[str, object], db_path: Path
) -> dict[str, object]:
    """Direct metadata-unit fixtures get the public identity of a verified snapshot."""

    enriched = dict(sources)
    enriched.update(
        {
            "db_snapshot_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
            "db_snapshot_bytes": int(db_path.stat().st_size),
            "db_snapshot_creation_elapsed_seconds": 0.0,
            "db_snapshot_hash_elapsed_seconds": 0.0,
        }
    )
    return enriched


PRODUCTION_STATION_RESULTS = (
    ROOT
    / "WeatherData_AWS"
    / "interpolation_batch"
    / "fullnet"
    / "bias_feedback_analysis"
    / "bias_feedback_station_results.csv"
)


EXPECTED_CSV_SCHEMAS = {
    "station_structure_coverage.csv": [
        "STN",
        "TOTAL_PAIRED_HOURS",
        "REPRODUCED_HOURS",
        "REPRODUCTION_RATE",
        "MAX_NONE_ABS_DIFF_C",
        "PRIMARY_ELIGIBLE_HOURS",
        "PRIMARY_INVERSION_LIKE_HOURS",
        "PRIMARY_NORMAL_NEGATIVE_HOURS",
        "PRIMARY_UNCERTAIN_HOURS",
        "PRIMARY_INELIGIBLE_HOURS",
        "SPAN200_ELIGIBLE_HOURS",
        "SPAN200_INVERSION_LIKE_HOURS",
        "NEFF4_ELIGIBLE_HOURS",
        "NEFF4_INVERSION_LIKE_HOURS",
        "OLS_ONLY_ELIGIBLE_HOURS",
        "OLS_ONLY_INVERSION_LIKE_HOURS",
        "WIND_COVERAGE",
        "RN_COVERAGE",
        "HM_COVERAGE",
        "TD_COVERAGE",
        "WEATHER_PRIMARY_ELIGIBLE",
        "WEATHER_DROP_REASON",
        "EXCLUSION_COUNTS_JSON",
        "PRIMARY_INELIGIBLE_REASONS_JSON",
        "MEDIAN_DELTA_COAST_KM",
        "MEDIAN_WEIGHTED_COASTAL_FRACTION",
        "MEDIAN_WEIGHTED_ISLAND_FRACTION",
        "MEDIAN_RELIEF_MISMATCH_M",
        "MEDIAN_TERRAIN_MISMATCH_FRACTION",
        "TPI_STATUS",
        "ANALYSIS_POPULATION",
    ],
    "state_condition_bias.csv": [
        "STN",
        "CONFIG",
        "STRUCTURE_STATE",
        "CONDITION_VARIABLE",
        "CONDITION_LEVEL",
        "N_HOURS",
        "N_DAYS",
        "N_MONTHS",
        "N_STRATA",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "MAE_none",
        "MAE_fixed",
        "DELTA_MAE",
        "BLOCK7_CI_LOW",
        "BLOCK7_CI_HIGH",
        "BLOCK7_REPLICATES",
        "COVERAGE_ELIGIBLE",
        "DROP_REASON",
        "EVIDENCE_GRADE",
        "LIMITATION",
        "ANALYSIS_POPULATION",
        "SUBSET_MEANINGFUL_WORSENING",
        "SUBSET_HIGH_AFTER",
    ],
    "coast_time_signed_bias.csv": [
        "STN",
        "MONTH",
        "SOLAR_STATE",
        "N_HOURS",
        "N_DAYS",
        "N_MONTHS",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "MAE_none",
        "MAE_fixed",
        "DELTA_MAE",
        "DELTA_COAST_KM",
        "H2_EXPECTED_BIAS_SIGN",
        "H2_PLACEMENT_SCOPE",
        "H2_EVIDENCE_SCOPE",
        "EVIDENCE_GRADE",
        "LIMITATION",
        "ANALYSIS_POPULATION",
        "SUBSET_MEANINGFUL_WORSENING",
        "SUBSET_HIGH_AFTER",
    ],
    "equivalent_diagnostics.csv": [
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "ABS_BIAS_NONE",
        "ABS_BIAS_FIXED",
        "CORRECTION_SHIFT",
        "ABS_CORRECTION_SHIFT",
        "DELTA_ABS_BIAS",
        "ABS_DZ_M",
        "BIAS_NONE_LEVEL",
        "BIAS_FIXED_LEVEL",
    ],
    "worsening_diagnostics.csv": [
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "MECHANISM",
        "CORRECTION_SHIFT",
        "DELTA_ABS_BIAS",
        "ABS_DZ_M",
        "DZ_BIN",
    ],
    "high_residual_group_diagnostic.csv": [
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "BIAS_FIXED_SIGN",
        "BIAS_NONE_LEVEL",
        "BIAS_FIXED_LEVEL",
        "HIGH_BEFORE",
        "HIGH_AFTER",
        "HIGH_COMMON",
        "IS_PRIMARY_HIGH_RESIDUAL",
        "PRIMARY_STRUCTURE_ELIGIBLE",
        "MEDIAN_DELTA_COAST_KM",
        "MEDIAN_RELIEF_MISMATCH_M",
        "TPI_STATUS",
        "ANALYSIS_POPULATION",
    ],
    "station_885_diagnostic.csv": [
        "STN",
        "MONTH",
        "SOLAR_STATE",
        "STRUCTURE_STATE",
        "N_HOURS",
        "N_DAYS",
        "N_MONTHS",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "MAE_none",
        "MAE_fixed",
        "DELTA_MAE",
        "DELTA_COAST_KM",
        "H2_EXPECTED_BIAS_SIGN",
        "CASE_SCOPE",
        "EVIDENCE_GRADE",
        "LIMITATION",
    ],
    "matched_state_contrast.csv": [
        "STN",
        "CONFIG",
        "N_STRATA",
        "N_DAYS",
        "N_MONTHS",
        "DELTA_ABS_BIAS_INVERSION",
        "DELTA_ABS_BIAS_NORMAL",
        "CONTRAST_INV_MINUS_NORMAL",
        "BLOCK7_CI_LOW",
        "BLOCK7_CI_HIGH",
        "BLOCK7_REQUESTED_REPLICATES",
        "BLOCK7_VALID_REPLICATES",
        "BLOCK7_VALID_FRACTION",
        "BLOCK7_SEED",
        "BLOCK7_BLOCK_DAYS",
        "STRATA_WEIGHTING",
        "ESTIMAND",
        "SAMPLING_SCHEME",
        "ANALYSIS_SUPPORT_CHECKED",
        "ANALYSIS_DATE_COUNT",
        "SUPPORTED_ANALYSIS_DATE_COUNT",
        "UNCOVERED_ANALYSIS_DATE_COUNT",
        "ANALYSIS_DATE_SUPPORT_FRACTION",
        "UNCOVERED_ANALYSIS_DATES_JSON",
        "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON",
        "COMPARISON_ELIGIBLE",
        "DROP_REASON",
        "Q_VALUE",
        "EVIDENCE_GRADE",
        "LIMITATION",
        "ANALYSIS_POPULATION",
        "SUBSET_MEANINGFUL_WORSENING",
        "SUBSET_HIGH_AFTER",
    ],
}

# Integration round-1 adds three fixed tables and extends the published schemas.
# The semantic/mutation tests below remain independent of this exact-column registry.
if runner is not None:
    EXPECTED_CSV_SCHEMAS = {
        name: list(columns) for name, columns in runner.CSV_SCHEMAS.items()
    }

EXPECTED_PNGS = {
    "equivalent_before_after.png",
    "worsening_by_dz.png",
    "high_residual_factors.png",
    "structure_condition_bias.png",
    "station_885_case.png",
}


class RunnerFunctionTestCase(unittest.TestCase):
    def function(self, name):
        self.assertIsNotNone(
            runner,
            msg=f"analyze_bias_physical_followup module is missing: {MODULE_IMPORT_ERROR}",
        )
        function = getattr(runner, name, None)
        self.assertTrue(callable(function), msg=f"required function is missing: {name}")
        return function


def _metadata_fixture(*, zero_distance=False):
    rows = [
        ("100", "target", 37.0, 127.0, 50.0),
        ("101", "n1", 37.0, 127.0 if zero_distance else 127.02, 0.0),
        ("102", "n2", 37.0, 126.98, 100.0),
        ("103", "n3", 37.02, 127.0, 200.0),
        ("104", "n4", 36.97, 127.0, 300.0),
        ("105", "n5", 37.0, 127.04, 400.0),
        ("106", "n6", 37.05, 127.0, 500.0),
        ("107", "n7_missing_elev", 37.0, 127.06, np.nan),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["지점", "지점명", "위도", "경도", "노장해발고도(m)"],
    )
    frame["시작일"] = pd.Timestamp("2020-01-01")
    frame["종료일"] = pd.NaT
    return frame[
        [
            "지점",
            "지점명",
            "시작일",
            "종료일",
            "위도",
            "경도",
            "노장해발고도(m)",
        ]
    ]


def _terrain_fixture():
    ids = [str(value) for value in range(100, 108)]
    return pd.DataFrame(
        {
            "STN": ids,
            "coast_dist_km": [5.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0],
            "is_coastal": [True, False, False, False, False, False, False, False],
            "is_island": [False, False, False, False, False, False, False, True],
            "relief_1km_m": [50.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
            "terrain_group": ["해안", "내륙", "내륙", "산악", "산악", "내륙", "산악", "섬"],
        }
    )


def _stable_geometry(meta, neighbor_ids):
    target = meta.loc[meta["지점"].eq("100")].iloc[0]
    order_lookup = {str(sid): order for order, sid in enumerate(meta["지점"])}
    values = []
    for sid in neighbor_ids:
        row = meta.loc[meta["지점"].eq(str(sid))].iloc[0]
        distance = get_distance_km(
            target["위도"], target["경도"], row["위도"], row["경도"]
        )
        values.append((distance, order_lookup[str(sid)], str(sid), row))
    return sorted(values, key=lambda item: (item[0], item[1]))


def _manual_idw(temperatures_by_id, geometry):
    if geometry[0][0] == 0:
        return float(temperatures_by_id[geometry[0][2]])
    minimum = geometry[0][0]
    numerator = 0.0
    denominator = 0.0
    for distance, _, sid, _ in geometry:
        weight = (minimum / distance) ** 2
        numerator += weight * float(temperatures_by_id[sid])
        denominator += weight
    return numerator / denominator


def _station_fixture(*, zero_distance=False):
    meta = _metadata_fixture(zero_distance=zero_distance)
    terrain = _terrain_fixture()
    neighbor_ids = ["107", "106", "105", "104", "103", "102", "101"]
    geometry = _stable_geometry(meta, neighbor_ids)
    times = pd.DatetimeIndex(["2024-06-21 12:00", "2024-12-21 00:00"])

    positive = {
        "101": 10.0,
        "102": 10.5,
        "103": 11.0,
        "104": 11.5,
        "105": 12.0,
        "106": 12.5,
        "107": 9.3,
    }
    negative = {
        "101": 15.0,
        "102": 14.5,
        "103": 14.0,
        "104": 13.5,
        "105": 13.0,
        "106": 12.5,
        "107": 9.0,
    }
    temperature_rows = [positive, negative]
    ta_data = pd.DataFrame(temperature_rows, index=times)
    predictions = [_manual_idw(row, geometry) for row in temperature_rows]
    error_none = np.array([0.2, -0.3])
    error_fixed = np.array([0.4, -0.1])
    pairs = pd.DataFrame(
        {
            "TM": times,
            "STN": ["100", "100"],
            "OBSERVED": np.asarray(predictions) - error_none,
            "error_none": error_none,
            "error_fixed": error_fixed,
            "NEIGHBOR_SET_ID": ["sig-A", "sig-A"],
            "COORD_EPOCH": ["2020-01-01", "2020-01-01"],
            "NEIGHBOR_COUNT": [7, 7],
        }
    )
    weather = pd.DataFrame(
        {
            "STN": ["100", "100"],
            "TM": times,
            "WS10": [np.nan, 1.0],
            "WS1": [2.0, 3.0],
            "RN_60m": [0.0, 1.0],
            "HM": [50.0, 60.0],
            "TD": [-5.0, -6.0],
        }
    )
    return {
        "pairs": pairs,
        "neighbor_map": {"sig-A": neighbor_ids},
        "meta": meta,
        "ta_data": ta_data,
        "weather": weather,
        "terrain": terrain,
    }


class ExactNeighborContractTests(RunnerFunctionTestCase):
    def test_exact_ids_are_trimmed_finite_unique_target_free_and_at_most_ten(self):
        """비공식·중복·target 포함·11개 ID가 exact signature로 들어가는 회귀를 잡는다."""
        validate = self.function("validate_exact_neighbor_ids")

        self.assertEqual(
            validate(" 100 ", [" 101 ", 102], expected_count=2),
            ("101", "102"),
        )
        invalid = [
            ("100", ["101", "101"], 2),
            (" 100 ", ["101", "100"], 2),
            ("100", ["101", np.nan], 2),
            ("100", [str(value) for value in range(101, 112)], 11),
            ("100", ["101", "102"], 3),
        ]
        for target, neighbors, expected in invalid:
            with self.subTest(neighbors=neighbors):
                with self.assertRaises(ValueError):
                    validate(target, neighbors, expected_count=expected)

    def test_signature_geometry_ignores_signature_order_and_uses_metadata_distance_order(self):
        """neighbor_set_map의 ID 정렬순서를 IDW 계산순서로 신뢰하는 회귀를 잡는다."""
        reconstruct = self.function("reconstruct_signature_geometry")
        fixture = _station_fixture()

        geometry = reconstruct(
            target_id="100",
            neighbor_ids=fixture["neighbor_map"]["sig-A"],
            timestamp=pd.Timestamp("2024-06-21 12:00"),
            coord_epoch="2020-01-01",
            expected_neighbor_count=7,
            meta=fixture["meta"],
            terrain=fixture["terrain"],
        )
        expected = _stable_geometry(
            fixture["meta"], fixture["neighbor_map"]["sig-A"]
        )

        self.assertEqual(geometry["neighbor_ids"], tuple(item[2] for item in expected))
        np.testing.assert_allclose(
            geometry["distances_km"], [item[0] for item in expected], rtol=0, atol=1e-12
        )
        self.assertNotEqual(
            geometry["neighbor_ids"], tuple(fixture["neighbor_map"]["sig-A"])
        )
        self.assertEqual(geometry["regression_neighbor_count"], 6)
        self.assertEqual(geometry["tpi_status"], "NOT_MEASURED")

    def test_equal_distance_and_two_zero_distance_ties_follow_metadata_source_order(self):
        """동일거리·거리 0 동률에서 signature ID순으로 첫 이웃이 바뀌는 회귀를 잡는다."""
        reconstruct = self.function("reconstruct_signature_geometry")
        exact_idw = exact_idw_none_weighted_mean

        equal_meta = _metadata_fixture()
        equal_meta.loc[equal_meta["지점"].eq("101"), ["위도", "경도"]] = [37.0, 127.02]
        equal_meta.loc[equal_meta["지점"].eq("102"), ["위도", "경도"]] = [37.0, 127.02]
        equal = reconstruct(
            target_id="100",
            neighbor_ids=["102", "101"],
            timestamp=pd.Timestamp("2024-06-21 12:00"),
            coord_epoch="2020-01-01",
            expected_neighbor_count=2,
            meta=equal_meta,
        )
        self.assertEqual(equal["neighbor_ids"], ("101", "102"))
        self.assertAlmostEqual(equal["distances_km"][0], equal["distances_km"][1], 12)

        zero_meta = _metadata_fixture()
        zero_meta.loc[
            zero_meta["지점"].isin(["101", "102"]), ["위도", "경도"]
        ] = [37.0, 127.0]
        zero = reconstruct(
            target_id="100",
            neighbor_ids=["102", "101"],
            timestamp=pd.Timestamp("2024-06-21 12:00"),
            coord_epoch="2020-01-01",
            expected_neighbor_count=2,
            meta=zero_meta,
        )
        self.assertEqual(zero["neighbor_ids"], ("101", "102"))
        self.assertEqual(zero["distances_km"].tolist(), [0.0, 0.0])
        self.assertEqual(
            exact_idw(
                [1.0, 99.0],
                zero["distances_km"],
                neighbor_ids=zero["neighbor_ids"],
                target_id="100",
            ),
            1.0,
        )

    def test_vectorized_slopes_match_task2_scalar_contract_for_every_hour(self):
        """batch 최적화가 Task2 OLS/WLS·4분류 의미론에서 벗어나는 회귀를 잡는다."""
        calculate = self.function("vectorized_structure_diagnostics")
        fixture = _station_fixture()
        geometry = self.function("reconstruct_signature_geometry")(
            target_id="100",
            neighbor_ids=fixture["neighbor_map"]["sig-A"],
            timestamp=fixture["pairs"].iloc[0]["TM"],
            coord_epoch="2020-01-01",
            expected_neighbor_count=7,
            meta=fixture["meta"],
            terrain=fixture["terrain"],
        )
        ids = geometry["regression_neighbor_ids"]
        matrix = fixture["ta_data"].loc[:, list(ids)].to_numpy(float)
        batch = calculate(
            elevations_m=geometry["regression_elevations_m"],
            temperature_matrix_c=matrix,
            distances_km=geometry["regression_distances_km"],
            neighbor_ids=ids,
            target_id="100",
        )

        for index in range(len(matrix)):
            ols = ols_temperature_elevation_slope(
                geometry["regression_elevations_m"],
                matrix[index],
                neighbor_ids=ids,
                target_id="100",
            )
            weighted = weighted_temperature_elevation_slope(
                geometry["regression_elevations_m"],
                matrix[index],
                distances_km=geometry["regression_distances_km"],
                neighbor_ids=ids,
                target_id="100",
            )
            state = classify_spatial_temperature_structure(ols, weighted)
            self.assertAlmostEqual(
                batch.iloc[index]["OLS_SLOPE_C_PER_KM"], ols["slope_c_per_km"], places=11
            )
            self.assertAlmostEqual(
                batch.iloc[index]["WLS_SLOPE_C_PER_KM"], weighted["slope_c_per_km"], places=11
            )
            self.assertEqual(batch.iloc[index]["STATE_PRIMARY"], state["classification"])

    def test_zero_distance_reproduces_none_but_marks_structure_ineligible(self):
        """거리 0 IDW shortcut을 회귀에 무한가중치로 넣거나 시각 전체를 버리는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture(zero_distance=True)
        result = analyze(**fixture, strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"]

        self.assertTrue(hourly["REPRODUCED"].all())
        self.assertLessEqual(hourly["NONE_ABS_DIFF_C"].max(), 1e-10)
        self.assertTrue(hourly["STATE_PRIMARY"].eq("ineligible").all())
        self.assertTrue(
            hourly["STATE_PRIMARY_REASON"].str.contains("zero_neighbor_distance").all()
        )


class SolarAndWeatherTests(RunnerFunctionTestCase):
    def test_noaa_solar_elevation_matches_hand_checked_solstice_noon_midnight(self):
        """KST·경도·위도 보정이 빠져 고정 시각을 자연 주야로 오인하는 회귀를 잡는다."""
        calculate = self.function("solar_elevation_noaa")
        times = pd.DatetimeIndex(
            [
                "2024-06-21 12:00",
                "2024-06-21 00:00",
                "2024-12-21 12:00",
                "2024-12-21 00:00",
            ]
        )
        result = calculate(times, latitude_deg=37.5, longitude_deg=127.0)

        np.testing.assert_allclose(
            result,
            [74.22245, -28.53978, 28.65847, -74.51922],
            rtol=0,
            atol=0.02,
        )
        low_latitude = calculate(
            pd.DatetimeIndex(["2024-06-21 12:00"]),
            latitude_deg=33.0,
            longitude_deg=127.0,
        )[0]
        high_latitude = calculate(
            pd.DatetimeIndex(["2024-06-21 12:00"]),
            latitude_deg=38.0,
            longitude_deg=127.0,
        )[0]
        self.assertGreater(low_latitude, high_latitude)

    def test_solar_state_uses_bright_dark_transition_technical_thresholds(self):
        """전환대를 임의로 낮·밤에 합치거나 고정 시계시간을 상태로 쓰는 회귀를 잡는다."""
        classify = self.function("classify_solar_state")
        self.assertEqual(
            classify(np.array([-6.1, -6.0, 0.0, 0.0001])).tolist(),
            ["dark", "transition", "transition", "bright"],
        )

    def test_solar_time_treats_naive_as_kst_and_handles_aware_instant_leap_day_longitude(self):
        """naive KST를 UTC로 localize하거나 윤일·경도·timezone을 잃는 회귀를 잡는다."""
        calculate = self.function("solar_elevation_noaa")
        naive_kst = pd.DatetimeIndex(["2024-06-21 12:00"])
        aware_utc = pd.DatetimeIndex([pd.Timestamp("2024-06-21 03:00", tz="UTC")])
        naive_value = calculate(naive_kst, latitude_deg=37.5, longitude_deg=127.0)[0]
        aware_value = calculate(aware_utc, latitude_deg=37.5, longitude_deg=127.0)[0]
        self.assertAlmostEqual(naive_value, aware_value, places=10)

        leap = calculate(
            pd.DatetimeIndex(["2024-02-29 12:00"]),
            latitude_deg=37.5,
            longitude_deg=127.0,
        )[0]
        west = calculate(naive_kst, latitude_deg=37.5, longitude_deg=125.0)[0]
        east = calculate(naive_kst, latitude_deg=37.5, longitude_deg=129.0)[0]
        self.assertTrue(np.isfinite(leap))
        self.assertNotAlmostEqual(west, east)

    def test_weather_fallback_coverage_quartiles_and_non_precipitation_label(self):
        """WS fallback·80% paired gate·RN=0 의미가 바뀌는 회귀를 잡는다."""
        merge = self.function("merge_weather_conditions")
        times = pd.date_range("2024-01-01", periods=5, freq="h")
        pairs = pd.DataFrame({"TM": times})
        pairs["STN"] = "100"
        weather = pd.DataFrame(
            {
                "STN": ["100"] * 5,
                "TM": times,
                "WS10": [1.0, np.nan, 3.0, 4.0, np.nan],
                "WS1": [9.0, 2.0, 9.0, 9.0, np.nan],
                "RN_60m": [0.0, 0.0, 1.0, 2.0, np.nan],
                "HM": [50.0, 60.0, 70.0, 80.0, np.nan],
                "TD": [-5.0, -4.0, -3.0, -2.0, np.nan],
            }
        )
        weather = pd.concat(
            [
                weather,
                pd.DataFrame(
                    {
                        "STN": ["100"],
                        "TM": [pd.Timestamp("2024-02-01")],
                        "WS10": [9.0],
                        "WS1": [9.0],
                        "RN_60m": [0.0],
                        "HM": [99.0],
                        "TD": [1.0],
                    }
                ),
            ],
            ignore_index=True,
        )

        result = merge(pairs, weather, min_coverage=0.8)
        hourly = result["hourly"]
        coverage = result["coverage"].iloc[0]

        np.testing.assert_allclose(hourly["WS_EFFECTIVE"].iloc[:4], [1, 2, 3, 4])
        self.assertEqual(hourly["WS_SOURCE"].tolist(), ["WS10", "WS1", "WS10", "WS10", "missing"])
        self.assertEqual(hourly["WIND_QUARTILE"].iloc[:4].tolist(), ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(hourly.iloc[0]["PRECIPITATION_STATE"], "non_precipitation")
        self.assertNotIn("clear", set(hourly["PRECIPITATION_STATE"].dropna()))
        self.assertAlmostEqual(coverage["WIND_COVERAGE"], 0.8)
        self.assertAlmostEqual(coverage["RN_COVERAGE"], 0.8)
        self.assertEqual(int(coverage["PAIRED_HOURS"]), 5)
        self.assertTrue(bool(coverage["WEATHER_PRIMARY_ELIGIBLE"]))

        duplicate = pd.concat([weather, weather.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            merge(pairs, duplicate, min_coverage=0.8)

    def test_weather_rejects_negative_ws_rn_hm_but_allows_negative_td(self):
        """물리적으로 가능한 음의 이슬점과 불가능한 WS/RN/HM을 같은 규칙으로 처리하는 회귀를 잡는다."""
        merge = self.function("merge_weather_conditions")
        pairs = pd.DataFrame({"STN": ["100"], "TM": [pd.Timestamp("2024-01-01")]})
        base = {
            "STN": ["100"],
            "TM": [pd.Timestamp("2024-01-01")],
            "WS10": [1.0],
            "WS1": [1.0],
            "RN_60m": [0.0],
            "HM": [50.0],
            "TD": [-20.0],
        }
        self.assertEqual(len(merge(pairs, pd.DataFrame(base))["hourly"]), 1)
        for column in ["WS10", "WS1", "RN_60m", "HM"]:
            invalid = dict(base)
            invalid[column] = [-0.1]
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    merge(pairs, pd.DataFrame(invalid))

    def test_weather_join_never_nearest_fills_and_variable_gates_are_separate(self):
        """+30분 자료를 exact 시각에 붙이거나 RN 결측으로 wind 분석까지 버리는 회귀를 잡는다."""
        merge = self.function("merge_weather_conditions")
        pair = pd.DataFrame({"STN": ["100"], "TM": [pd.Timestamp("2024-01-01 00:00")]})
        shifted = pd.DataFrame(
            {
                "STN": ["100"],
                "TM": [pd.Timestamp("2024-01-01 00:30")],
                "WS10": [1.0],
                "WS1": [1.0],
                "RN_60m": [0.0],
                "HM": [50.0],
                "TD": [-5.0],
            }
        )
        exact = merge(pair, shifted)
        self.assertTrue(pd.isna(exact["hourly"].iloc[0]["WS_EFFECTIVE"]))
        self.assertEqual(exact["coverage"].iloc[0]["WIND_COVERAGE"], 0.0)

        build = self.function("build_state_condition_bias")
        dates = []
        for month in range(1, 7):
            dates.extend(pd.date_range(f"2024-{month:02d}-01", periods=5, freq="D"))
        hourly = pd.DataFrame(
            {
                "STN": "100",
                "TM": dates,
                "REPRODUCED": True,
                "STATE_PRIMARY": "inversion_like",
                "STATE_SPAN200": "inversion_like",
                "STATE_NEFF4": "inversion_like",
                "STATE_OLS_ONLY": "inversion_like",
                "SOLAR_STATE": "bright",
                "WIND_QUARTILE": "Q1",
                "PRECIPITATION_STATE": pd.NA,
                "error_none": 0.2,
                "error_fixed": 0.1,
            }
        )
        coverage = pd.DataFrame(
            [{"WIND_COVERAGE": 1.0, "RN_COVERAGE": 0.0, "HM_COVERAGE": 1.0}]
        )
        result = build(hourly, coverage)
        wind = result.loc[
            result["CONFIG"].eq("primary")
            & result["STRUCTURE_STATE"].eq("inversion_like")
            & result["CONDITION_VARIABLE"].eq("WIND_QUARTILE")
            & result["CONDITION_LEVEL"].eq("Q1")
        ].iloc[0]
        rain = result.loc[
            result["CONFIG"].eq("primary")
            & result["STRUCTURE_STATE"].eq("inversion_like")
            & result["CONDITION_VARIABLE"].eq("PRECIPITATION_STATE")
            & result["CONDITION_LEVEL"].eq("non_precipitation")
        ].iloc[0]
        self.assertTrue(bool(wind["COVERAGE_ELIGIBLE"]))
        self.assertFalse(bool(rain["COVERAGE_ELIGIBLE"]))


class StationAnalysisTests(RunnerFunctionTestCase):
    def test_neighbor_coordinate_epoch_change_rebuilds_geometry_with_same_signature(self):
        """target epoch·이웃 ID가 같아도 이웃 이설 뒤 옛 거리를 재사용하는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        boundary = pd.Timestamp("2024-10-15")
        neighbor_ids = fixture["neighbor_map"]["sig-A"]

        meta = fixture["meta"].copy()
        old_neighbor = meta["지점"].eq("101")
        meta.loc[old_neighbor, "종료일"] = boundary
        new_neighbor = meta.loc[old_neighbor].iloc[0].copy()
        new_neighbor["시작일"] = boundary
        new_neighbor["종료일"] = pd.NaT
        new_neighbor["경도"] = 127.08
        meta = pd.concat([meta, pd.DataFrame([new_neighbor])], ignore_index=True)

        before_meta = meta.loc[meta.index != meta.index[-1]].copy()
        after_meta = meta.loc[~old_neighbor.reindex(meta.index, fill_value=False)].copy()
        temperature_rows = fixture["ta_data"].to_dict(orient="records")
        expected_predictions = np.asarray(
            [
                _manual_idw(
                    temperature_rows[0], _stable_geometry(before_meta, neighbor_ids)
                ),
                _manual_idw(
                    temperature_rows[1], _stable_geometry(after_meta, neighbor_ids)
                ),
            ]
        )
        pairs = fixture["pairs"].copy()
        pairs["OBSERVED"] = expected_predictions - pairs["error_none"].to_numpy(float)
        fixture["meta"] = meta
        fixture["pairs"] = pairs

        result = analyze(**fixture, strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"].sort_values("TM")

        np.testing.assert_allclose(
            hourly["PRED_NONE_RECONSTRUCTED"], expected_predictions, rtol=0, atol=1e-12
        )
        self.assertLessEqual(float(hourly["NONE_ABS_DIFF_C"].max()), 1e-10)
        self.assertEqual(result["metrics"]["geometry_build_count"], 2)

    def test_neighbor_noon_metadata_boundary_takes_effect_on_next_calendar_day(self):
        """Stage1의 date_key 계약과 달리 metadata 시각부터 즉시 새 좌표를 쓰는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        boundary = pd.Timestamp("2024-10-15 12:00")
        times = pd.DatetimeIndex(["2024-10-15 18:00", "2024-10-16 00:00"])
        neighbor_ids = fixture["neighbor_map"]["sig-A"]

        meta = fixture["meta"].copy()
        old_neighbor = meta["지점"].eq("101")
        meta.loc[old_neighbor, "종료일"] = boundary
        new_neighbor = meta.loc[old_neighbor].iloc[0].copy()
        new_neighbor["시작일"] = boundary
        new_neighbor["종료일"] = pd.NaT
        new_neighbor["경도"] = 127.08
        meta = pd.concat([meta, pd.DataFrame([new_neighbor])], ignore_index=True)
        before_meta = meta.loc[meta.index != meta.index[-1]].copy()
        after_meta = meta.loc[~old_neighbor.reindex(meta.index, fill_value=False)].copy()

        ta_data = fixture["ta_data"].copy()
        ta_data.index = times
        temperature_rows = ta_data.to_dict(orient="records")
        expected_predictions = np.asarray(
            [
                _manual_idw(
                    temperature_rows[0], _stable_geometry(before_meta, neighbor_ids)
                ),
                _manual_idw(
                    temperature_rows[1], _stable_geometry(after_meta, neighbor_ids)
                ),
            ]
        )
        pairs = fixture["pairs"].copy()
        pairs["TM"] = times
        pairs["OBSERVED"] = expected_predictions - pairs["error_none"].to_numpy(float)
        weather = fixture["weather"].copy()
        weather["TM"] = times
        fixture.update(meta=meta, ta_data=ta_data, pairs=pairs, weather=weather)

        result = analyze(**fixture, strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"].sort_values("TM")

        np.testing.assert_allclose(
            hourly["PRED_NONE_RECONSTRUCTED"], expected_predictions, rtol=0, atol=1e-12
        )
        self.assertLessEqual(float(hourly["NONE_ABS_DIFF_C"].max()), 1e-10)
        self.assertEqual(result["metrics"]["geometry_build_count"], 2)

    def test_neighbor_finite_end_without_successor_never_reuses_expired_geometry(self):
        """이웃 종료일 뒤에도 이전 geometry를 재사용해 유효하지 않은 signature를 숨기는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        meta = fixture["meta"].copy()
        meta.loc[meta["지점"].eq("101"), "종료일"] = pd.Timestamp("2024-10-15")
        fixture["meta"] = meta

        with self.assertRaisesRegex(
            ValueError, "neighbor 101 metadata is not uniquely valid"
        ):
            analyze(**fixture, strict_reproduction=True, return_hourly=True)

    def test_neighbor_elevation_only_epoch_change_rebuilds_structure_diagnostics(self):
        """좌표가 같다는 이유로 이웃 고도 변경 뒤 회귀 geometry를 재사용하는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        boundary = pd.Timestamp("2024-10-15")
        meta = fixture["meta"].copy()
        old_neighbor = meta["지점"].eq("101")
        meta.loc[old_neighbor, "종료일"] = boundary
        new_neighbor = meta.loc[old_neighbor].iloc[0].copy()
        new_neighbor["시작일"] = boundary
        new_neighbor["종료일"] = pd.NaT
        new_neighbor["노장해발고도(m)"] = 1000.0
        fixture["meta"] = pd.concat(
            [meta, pd.DataFrame([new_neighbor])], ignore_index=True
        )

        result = analyze(**fixture, strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"].sort_values("TM")

        np.testing.assert_allclose(
            hourly["ELEVATION_SPAN_M"], [500.0, 900.0], rtol=0, atol=1e-12
        )
        self.assertEqual(result["metrics"]["geometry_build_count"], 2)
        self.assertEqual(result["metrics"]["processed_paired_hours"], 2)
        self.assertEqual(result["metrics"]["processed_neighbor_values"], 14)

    def test_station_analysis_reproduces_none_and_preserves_all_denominators(self):
        """exact 검산·고도결측 부분집합·민감도·제외분모 중 하나가 사라지는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        result = analyze(**_station_fixture(), strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"]
        summary = result["station_summary"].iloc[0]

        self.assertEqual(len(hourly), 2)
        self.assertTrue(hourly["REPRODUCED"].all())
        self.assertLessEqual(float(hourly["NONE_ABS_DIFF_C"].max()), 1e-10)
        self.assertEqual(hourly["REGRESSION_N"].tolist(), [6, 6])
        self.assertEqual(hourly["STATE_PRIMARY"].tolist(), ["inversion_like", "normal_negative"])
        self.assertEqual(hourly["SOLAR_STATE"].tolist(), ["bright", "dark"])
        self.assertEqual(hourly["WS_EFFECTIVE"].tolist(), [2.0, 1.0])
        self.assertTrue((hourly["DELTA_COAST_KM"] < 0).all())
        self.assertTrue(hourly["TPI_STATUS"].eq("NOT_MEASURED").all())
        self.assertEqual(int(summary["TOTAL_PAIRED_HOURS"]), 2)
        self.assertEqual(int(summary["REPRODUCED_HOURS"]), 2)
        self.assertEqual(int(summary["PRIMARY_ELIGIBLE_HOURS"]), 2)
        self.assertEqual(int(summary["PRIMARY_INVERSION_LIKE_HOURS"]), 1)
        self.assertEqual(int(summary["PRIMARY_NORMAL_NEGATIVE_HOURS"]), 1)
        self.assertEqual(result["metrics"]["geometry_build_count"], 1)
        self.assertEqual(result["metrics"]["processed_paired_hours"], 2)
        self.assertEqual(result["metrics"]["processed_neighbor_values"], 14)
        all_state = result["state_condition_bias"].loc[
            result["state_condition_bias"]["CONDITION_VARIABLE"].eq("ALL")
        ]
        self.assertEqual(
            all_state.groupby("CONFIG")["N_HOURS"].sum().to_dict(),
            {"neff4": 2, "ols_only": 2, "primary": 2, "span200": 2},
        )

    def test_reproduction_mismatch_is_explicit_exclusion_or_strict_failure(self):
        """DB none 불일치를 조용히 회귀에 넣거나 이유 없이 누락하는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        fixture["pairs"] = fixture["pairs"].copy()
        fixture["pairs"].loc[0, "error_none"] += 0.01

        permissive = analyze(**fixture, strict_reproduction=False, return_hourly=True)
        self.assertFalse(bool(permissive["hourly"].iloc[0]["REPRODUCED"]))
        self.assertEqual(
            permissive["hourly"].iloc[0]["EXCLUSION_REASON"],
            "none_reproduction_mismatch",
        )
        with self.assertRaises(ValueError):
            analyze(**fixture, strict_reproduction=True, return_hourly=True)

    def test_strict_station_analysis_rejects_all_exact_data_corruptions(self):
        """unknown signature·TA 결측·count/epoch/paired-key 오류가 publish까지 가는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")

        cases = []
        unknown = _station_fixture()
        unknown["pairs"] = unknown["pairs"].copy()
        unknown["pairs"].loc[:, "NEIGHBOR_SET_ID"] = "unknown"
        cases.append(("unknown_signature", unknown))

        missing_ta = _station_fixture()
        missing_ta["ta_data"] = missing_ta["ta_data"].copy()
        missing_ta["ta_data"].loc[missing_ta["ta_data"].index[0], "101"] = np.nan
        cases.append(("neighbor_ta_missing", missing_ta))

        count_mismatch = _station_fixture()
        count_mismatch["pairs"] = count_mismatch["pairs"].copy()
        count_mismatch["pairs"].loc[:, "NEIGHBOR_COUNT"] = 6
        cases.append(("neighbor_count_mismatch", count_mismatch))

        epoch_mismatch = _station_fixture()
        epoch_mismatch["pairs"] = epoch_mismatch["pairs"].copy()
        epoch_mismatch["pairs"].loc[:, "COORD_EPOCH"] = "2021-01-01"
        cases.append(("epoch_mismatch", epoch_mismatch))

        target_leakage = _station_fixture()
        target_leakage["neighbor_map"] = {
            "sig-A": ["100", "101", "102", "103", "104", "105", "106"]
        }
        cases.append(("target_leakage", target_leakage))

        duplicate_pair = _station_fixture()
        duplicate_pair["pairs"] = pd.concat(
            [duplicate_pair["pairs"], duplicate_pair["pairs"].iloc[[0]]],
            ignore_index=True,
        )
        cases.append(("duplicate_paired_hour", duplicate_pair))

        for name, fixture in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    analyze(**fixture, strict_reproduction=True, return_hourly=True)

    def test_state_condition_bias_averages_signed_errors_before_absolute_value(self):
        """상태별 BIAS 효과를 시간별 절대오차 변화로 바꾸는 회귀를 잡는다."""
        build = self.function("build_state_condition_bias")
        hourly = pd.DataFrame(
            {
                "STN": ["100", "100", "100"],
                "TM": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"]
                ),
                "MONTH": [1, 1, 1],
                "REPRODUCED": [True, True, True],
                "STATE_PRIMARY": ["inversion_like"] * 3,
                "STATE_SPAN200": ["inversion_like"] * 3,
                "STATE_NEFF4": ["inversion_like"] * 3,
                "STATE_OLS_ONLY": ["inversion_like"] * 3,
                "SOLAR_STATE": ["dark"] * 3,
                "WIND_QUARTILE": ["Q1"] * 3,
                "PRECIPITATION_STATE": ["non_precipitation"] * 3,
                "error_none": [-1.0, 1.0, 1.0],
                "error_fixed": [0.0, 2.0, 2.0],
            }
        )
        coverage = pd.DataFrame(
            [{"WIND_COVERAGE": 1.0, "RN_COVERAGE": 1.0, "HM_COVERAGE": 1.0}]
        )

        result = build(hourly, coverage)
        row = result.loc[
            result["CONFIG"].eq("primary")
            & result["CONDITION_VARIABLE"].eq("ALL")
            & result["STRUCTURE_STATE"].eq("inversion_like")
        ].iloc[0]
        self.assertAlmostEqual(row["BIAS_none"], 1 / 3)
        self.assertAlmostEqual(row["BIAS_fixed"], 4 / 3)
        self.assertAlmostEqual(row["DELTA_ABS_BIAS"], 1.0)
        self.assertAlmostEqual(row["DELTA_MAE"], 1 / 3)
        self.assertEqual(int(row["N_HOURS"]), 3)
        # 한 날짜뿐인 자료는 연간 상태효과의 불확실성을 추론할 수 없다.
        self.assertEqual(int(row["BLOCK7_REPLICATES"]), 0)
        self.assertTrue(pd.isna(row["BLOCK7_CI_LOW"]))
        self.assertTrue(pd.isna(row["BLOCK7_CI_HIGH"]))
        primary_all = result.loc[
            result["CONFIG"].eq("primary")
            & result["CONDITION_VARIABLE"].eq("ALL")
        ]
        self.assertEqual(
            set(primary_all["STRUCTURE_STATE"]),
            {"inversion_like", "normal_negative", "uncertain", "ineligible"},
        )
        zero = primary_all.loc[primary_all["STRUCTURE_STATE"].eq("normal_negative")].iloc[0]
        self.assertEqual(int(zero["N_HOURS"]), 0)
        self.assertTrue(pd.isna(zero["DELTA_ABS_BIAS"]))

    def test_state_coverage_keeps_29_day_5_month_fail_and_30_day_6_month_pass(self):
        """30일·6개월 주 coverage 경계를 결과행 삭제나 시간수로 대체하는 회귀를 잡는다."""
        build = self.function("build_state_condition_bias")
        dates = []
        for month in range(1, 7):
            dates.extend(pd.date_range(f"2024-{month:02d}-01", periods=5, freq="D"))
        hourly = pd.DataFrame(
            {
                "STN": "100",
                "TM": dates,
                "MONTH": [value.month for value in dates],
                "REPRODUCED": True,
                "STATE_PRIMARY": "inversion_like",
                "STATE_SPAN200": "inversion_like",
                "STATE_NEFF4": "inversion_like",
                "STATE_OLS_ONLY": "inversion_like",
                "SOLAR_STATE": "dark",
                "WIND_QUARTILE": "Q1",
                "PRECIPITATION_STATE": "non_precipitation",
                "error_none": 0.2,
                "error_fixed": 0.1,
            }
        )
        coverage = pd.DataFrame(
            [{"WIND_COVERAGE": 1.0, "RN_COVERAGE": 1.0, "HM_COVERAGE": 1.0}]
        )

        passing = build(hourly, coverage)
        pass_row = passing.loc[
            passing["CONFIG"].eq("primary")
            & passing["CONDITION_VARIABLE"].eq("ALL")
            & passing["STRUCTURE_STATE"].eq("inversion_like")
        ].iloc[0]
        self.assertEqual((pass_row["N_DAYS"], pass_row["N_MONTHS"]), (30, 6))
        self.assertTrue(bool(pass_row["COVERAGE_ELIGIBLE"]))

        failing = build(
            hourly.loc[hourly["TM"].ne(dates[-1])],
            coverage,
        )
        fail_row = failing.loc[
            failing["CONFIG"].eq("primary")
            & failing["CONDITION_VARIABLE"].eq("ALL")
            & failing["STRUCTURE_STATE"].eq("inversion_like")
        ].iloc[0]
        self.assertEqual(int(fail_row["N_DAYS"]), 29)
        self.assertFalse(bool(fail_row["COVERAGE_ELIGIBLE"]))
        self.assertEqual(fail_row["DROP_REASON"], "insufficient_coverage")

    def test_h2_table_uses_prespecified_signed_coast_direction_and_declares_extrapolation(self):
        """해안거리 절댓값 상관이나 Kim의 Tmax/Tmin을 시간별 직접증명으로 쓰는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        result = analyze(**_station_fixture(), strict_reproduction=True, return_hourly=False)
        coast = result["coast_time_signed_bias"].set_index(["MONTH", "SOLAR_STATE"])

        self.assertEqual(coast.loc[(6, "bright"), "H2_EXPECTED_BIAS_SIGN"], "positive")
        self.assertEqual(coast.loc[(12, "dark"), "H2_EXPECTED_BIAS_SIGN"], "negative")
        self.assertEqual(
            coast.iloc[0]["H2_EVIDENCE_SCOPE"],
            "hourly_extrapolation_from_Kim_Tmax_Tmin",
        )


class CanonicalAndArtifactTests(RunnerFunctionTestCase):
    def test_production_canonical_counts_and_high_groups_are_exact(self):
        """H1/H2 모집단을 선택된 109·33개로 축소하거나 high 전후 집단을 섞는 회귀를 잡는다."""
        if not PRODUCTION_STATION_RESULTS.exists():
            self.skipTest(f"production truth missing: {PRODUCTION_STATION_RESULTS}")
        validate = self.function("validate_canonical_station_results")
        frame = pd.read_csv(PRODUCTION_STATION_RESULTS, encoding="utf-8-sig")

        result = validate(frame)

        self.assertEqual(result["TOTAL_STATIONS"], 529)
        self.assertEqual(result["CLASS_COUNTS"], {
            "meaningful_improvement": 221,
            "meaningful_worsening": 109,
            "practically_equivalent": 115,
            "uncertain": 84,
        })
        self.assertEqual(result["HIGH_BEFORE_N"], 33)
        self.assertEqual(result["HIGH_AFTER_N"], 33)
        self.assertEqual(result["HIGH_COMMON_N"], 30)
        self.assertEqual(result["HIGH_AFTER_POSITIVE_N"], 19)
        self.assertEqual(result["HIGH_AFTER_NEGATIVE_N"], 14)
        self.assertEqual(
            result["EQUIVALENT_AFTER_LEVEL_COUNTS"],
            {"low": 20, "medium": 62, "high": 33},
        )
        self.assertEqual(
            result["WORSENING_MECHANISM_COUNTS"],
            {"same_direction_worsened": 91, "overshoot_worsened": 18},
        )
        self.assertEqual(
            result["WORSENING_DZ_BIN_COUNTS"],
            {"≤50m": 72, "50–100m": 27, "100–200m": 7, "200–300m": 2, ">300m": 1},
        )
        self.assertEqual(result["PRIMARY_HYPOTHESIS_POPULATION"], "all_529_stations")

    def test_injected_runner_publishes_exact_validated_manifest_and_no_hourly_network_dump(self):
        """staging 검증 없이 게시하거나 schema·hash·행수·최종 leaf 계약이 흔들리는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        fixture = _station_fixture()
        station_results = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.mkdir(parents=True)
            (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

            result = run(
                station_results=station_results,
                station_payloads={"100": fixture},
                out_dir=out_dir,
                allowed_fullnet_root=out_dir.parent,
                enforce_canonical=False,
            )

            self.assertEqual(Path(result["output_dir"]), out_dir.resolve())
            self.assertFalse((out_dir / "stale.txt").exists())
            self.assertFalse((out_dir / "hourly.csv").exists())
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            listed_paths = [entry["path"] for entry in manifest["artifacts"]]
            self.assertEqual(len(listed_paths), len(set(listed_paths)))
            listed = set(listed_paths)
            self.assertEqual(
                listed,
                set(EXPECTED_CSV_SCHEMAS)
                | EXPECTED_PNGS
                | {"analysis_metadata.json", "analysis_summary.md"},
            )
            for filename, columns in EXPECTED_CSV_SCHEMAS.items():
                frame = pd.read_csv(out_dir / filename, encoding="utf-8-sig")
                self.assertEqual(frame.columns.tolist(), columns)
                entry = next(item for item in manifest["artifacts"] if item["path"] == filename)
                self.assertEqual(entry["rows"], len(frame))
                digest = hashlib.sha256((out_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(entry["sha256"], digest)
            for entry in manifest["artifacts"]:
                path = out_dir / entry["path"]
                self.assertEqual(entry["bytes"], path.stat().st_size)
                self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(
                {path.name for path in out_dir.iterdir()},
                listed | {"manifest.json"},
            )
            for filename in EXPECTED_PNGS:
                path = out_dir / filename
                self.assertGreater(path.stat().st_size, 0)
                image = mpimg.imread(path)
                self.assertGreater(image.shape[0], 10)
                self.assertGreater(image.shape[1], 10)
            metadata = json.loads(
                (out_dir / "analysis_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["time_interpretation"], "naive_TM_is_KST")
            self.assertEqual(metadata["spatial_state_term"], "regional_inversion_like")
            self.assertEqual(metadata["tpi_status"], "NOT_MEASURED")
            uncertainty = metadata["state_contrast_uncertainty"]
            self.assertIn("calendar_contiguous_7day_blocks", uncertainty)
            self.assertIn("without_month_wrap", uncertainty)
            self.assertNotIn("circular", uncertainty)
            self.assertEqual(metadata["processed_paired_hours"], 2)
            self.assertEqual(metadata["processed_neighbor_values"], 14)
            self.assertEqual(metadata["geometry_build_count"], 1)
            self.assertEqual(metadata["db_query_count"], 0)
            self.assertEqual(metadata["aws_file_read_count"], 0)
            self.assertIn("peak_chunk_memory_bytes", metadata)
            self.assertIsInstance(metadata["peak_chunk_memory_bytes"], int)
            self.assertGreater(metadata["peak_chunk_memory_bytes"], 0)
            self.assertLess(metadata["peak_chunk_memory_bytes"], 20_000_000)

    def test_figure_save_failure_leaves_existing_final_directory_bitwise_unchanged(self):
        """PNG 생성 중 실패가 final leaf를 부분 교체하거나 기존 결과를 지우는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        fixture = _station_fixture()
        station_results = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.mkdir(parents=True)
            marker = out_dir / "previous.txt"
            marker.write_bytes(b"previous-final")

            with self.assertRaises(RuntimeError):
                run(
                    station_results=station_results,
                    station_payloads={"100": fixture},
                    out_dir=out_dir,
                    allowed_fullnet_root=out_dir.parent,
                    enforce_canonical=False,
                    failure_injection="figure_save",
                )

            self.assertEqual(marker.read_bytes(), b"previous-final")
            self.assertEqual({path.name for path in out_dir.iterdir()}, {"previous.txt"})

    def test_atomic_publish_rolls_back_first_publish_and_retries_transient_permission_error(self):
        """첫 게시 promote 뒤 실패와 OneDrive 일시 잠금이 부분 final을 남기는 회귀를 잡는다."""
        publish = self.function("_atomic_publish")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            staging = root / ".bias_physical_followup.staging-first"
            final = root / "bias_physical_followup"
            staging.mkdir()
            (staging / "artifact.bin").write_bytes(b"new")
            with self.assertRaises(RuntimeError):
                publish(
                    staging,
                    final,
                    failure_injection="after_promote_before_cleanup",
                )
            self.assertFalse(final.exists())

        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            staging = root / ".bias_physical_followup.staging-retry"
            final = root / "bias_physical_followup"
            staging.mkdir()
            (staging / "artifact.bin").write_bytes(b"new")
            original_replace = Path.replace
            attempts = {"promote": 0}

            def transient_once(source, target):
                if Path(source) == staging and attempts["promote"] == 0:
                    attempts["promote"] += 1
                    raise PermissionError(5, "transient OneDrive lock")
                return original_replace(source, target)

            with patch.object(Path, "replace", autospec=True, side_effect=transient_once):
                publish(staging, final, failure_injection=None)
            self.assertEqual(attempts["promote"], 1)
            self.assertEqual((final / "artifact.bin").read_bytes(), b"new")

        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            staging = root / ".bias_physical_followup.staging-existing"
            final = root / "bias_physical_followup"
            staging.mkdir()
            final.mkdir()
            (staging / "artifact.bin").write_bytes(b"new")
            (final / "artifact.bin").write_bytes(b"old")
            with self.assertRaises(RuntimeError):
                publish(
                    staging,
                    final,
                    failure_injection="after_promote_before_cleanup",
                )
            self.assertEqual((final / "artifact.bin").read_bytes(), b"old")

    def test_output_guard_and_production_api_fail_before_any_unsafe_write_or_db_load(self):
        """production API가 임의 폴더를 지우거나 guard 뒤에 DB를 먼저 여는 회귀를 잡는다."""
        guard = self.function("guard_output_directory")
        production = self.function("run_production_analysis")
        with _workspace_temp_dir() as temp_dir:
            unsafe = Path(temp_dir) / "bias_physical_followup"
            with self.assertRaises(ValueError):
                guard(unsafe)
            with patch('analysis.bias_data.open_readonly_db') as db_open:
                with self.assertRaises(ValueError):
                    production(out_dir=unsafe)
                db_open.assert_not_called()
            traversal = Path(temp_dir) / "fullnet" / "x" / ".." / "bias_physical_followup"
            with self.assertRaises(ValueError):
                guard(traversal)

    def test_selection_label_is_joined_after_full_station_state_calculation(self):
        """109·33 label을 먼저 적용해 동일 hourly 자료의 full-family 상태값이 바뀌는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        fixture = _station_fixture()
        equivalent = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.2],
                "BIAS_fixed": [0.21],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        worsening = equivalent.copy()
        worsening.loc[:, "BIAS_none"] = 0.1
        worsening.loc[:, "BIAS_fixed"] = 0.2
        worsening.loc[:, "SIGNIFICANCE_CLASS"] = "meaningful_worsening"
        worsening.loc[:, "MECHANISM"] = "same_direction_worsened"

        with _workspace_temp_dir() as temp_dir:
            first = Path(temp_dir) / "a" / "fullnet" / "bias_physical_followup"
            second = Path(temp_dir) / "b" / "fullnet" / "bias_physical_followup"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            run(equivalent, {"100": fixture}, first, allowed_fullnet_root=first.parent, enforce_canonical=False)
            run(worsening, {"100": fixture}, second, allowed_fullnet_root=second.parent, enforce_canonical=False)
            first_state = pd.read_csv(first / "state_condition_bias.csv", encoding="utf-8-sig")
            second_state = pd.read_csv(second / "state_condition_bias.csv", encoding="utf-8-sig")
            value_columns = [
                column
                for column in first_state.columns
                if not column.startswith("SUBSET_")
            ]
            pd.testing.assert_frame_equal(
                first_state[value_columns],
                second_state[value_columns],
                check_dtype=False,
            )
            self.assertFalse(first_state["SUBSET_MEANINGFUL_WORSENING"].any())
            self.assertTrue(second_state["SUBSET_MEANINGFUL_WORSENING"].all())

    def test_vectorized_path_handles_529_by_24_representative_hours(self):
        """production 경로가 529×시간을 Python 행루프로만 처리하도록 퇴행하는 회귀를 잡는다."""
        calculate = self.function("vectorized_structure_diagnostics")
        elevations = np.array([0, 100, 200, 300, 400, 500], dtype=float)
        distances = np.ones(6, dtype=float)
        base = 10.0 + elevations / 1000.0
        matrix = np.tile(base, (529 * 24, 1))

        result = calculate(
            elevations_m=elevations,
            temperature_matrix_c=matrix,
            distances_km=distances,
            neighbor_ids=["101", "102", "103", "104", "105", "106"],
            target_id="100",
        )

        self.assertEqual(len(result), 529 * 24)
        self.assertTrue(result["STATE_PRIMARY"].eq("inversion_like").all())


class CriticalPreflightRegressionTests(RunnerFunctionTestCase):
    def test_matched_contrast_uses_sufficient_month_strata_shared_blocks_and_seed(self):
        """상태 단독 CI나 1일 표본 대신 공통 월×태양층 7일 block 대비를 검증한다."""
        build = self.function("build_matched_state_contrast")
        rows = []
        for month in range(1, 7):
            dates = pd.date_range(f"2024-{month:02d}-01", periods=10, freq="D")
            for offset, date in enumerate(dates):
                inversion = offset < 5
                rows.append(
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=12),
                        "SOLAR_STATE": "bright",
                        "REPRODUCED": True,
                        "STATE_PRIMARY": "inversion_like" if inversion else "normal_negative",
                        "STATE_SPAN200": "inversion_like" if inversion else "normal_negative",
                        "STATE_NEFF4": "inversion_like" if inversion else "normal_negative",
                        "STATE_OLS_ONLY": "inversion_like" if inversion else "normal_negative",
                        "error_none": 1.0 if inversion else 0.4,
                        "error_fixed": 0.2 if inversion else 0.3,
                    }
                )
        hourly = pd.DataFrame(rows)
        first = build(hourly, n_boot=100, seed=11)
        second = build(hourly, n_boot=100, seed=11)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["COMPARISON_ELIGIBLE"].all())
        self.assertTrue(first["DROP_REASON"].eq("eligible").all())
        self.assertTrue(first["N_STRATA"].eq(6).all())
        self.assertTrue(first["N_DAYS"].eq(30).all())
        self.assertTrue(first["N_MONTHS"].eq(6).all())
        np.testing.assert_allclose(first["DELTA_ABS_BIAS_INVERSION"], -0.8)
        np.testing.assert_allclose(first["DELTA_ABS_BIAS_NORMAL"], -0.1)
        np.testing.assert_allclose(first["CONTRAST_INV_MINUS_NORMAL"], -0.7)
        np.testing.assert_allclose(first["BLOCK7_CI_LOW"], -0.7)
        np.testing.assert_allclose(first["BLOCK7_CI_HIGH"], -0.7)
        self.assertTrue(first["BLOCK7_VALID_REPLICATES"].eq(100).all())
        self.assertTrue(first["BLOCK7_REQUESTED_REPLICATES"].eq(100).all())
        self.assertTrue(first["BLOCK7_VALID_FRACTION"].eq(1.0).all())
        self.assertTrue(first["STRATA_WEIGHTING"].eq("equal_common_month_solar_strata").all())
        self.assertTrue(first["ANALYSIS_SUPPORT_CHECKED"].all())
        self.assertTrue(first["ANALYSIS_DATE_COUNT"].eq(60).all())
        self.assertTrue(first["SUPPORTED_ANALYSIS_DATE_COUNT"].eq(60).all())
        self.assertTrue(first["UNCOVERED_ANALYSIS_DATE_COUNT"].eq(0).all())
        self.assertTrue(first["ANALYSIS_DATE_SUPPORT_FRACTION"].eq(1.0).all())
        np.testing.assert_allclose(first["P_VALUE"], 1 / 101)

    def test_matched_contrast_rejects_dates_outside_every_calendar_block_support(self):
        """point에는 있으나 어떤 7일 block에도 없는 날짜에서 거짓 CI가 생기는 회귀를 잡는다."""
        build = self.function("build_matched_state_contrast")
        rows = []
        for month in range(1, 7):
            for day in [1, 2, 3, 4, 5, 6, 7, 20]:
                date = pd.Timestamp(2024, month, day)
                for hour, state in [(0, "inversion_like"), (1, "normal_negative")]:
                    rows.append(
                        {
                            "STN": "100",
                            "TM": date + pd.Timedelta(hours=hour),
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": state,
                            "STATE_SPAN200": state,
                            "STATE_NEFF4": state,
                            "STATE_OLS_ONLY": state,
                            "error_none": 1.0,
                            "error_fixed": 0.5,
                        }
                    )

        result = build(pd.DataFrame(rows), n_boot=100, seed=31)
        self.assertFalse(result["COMPARISON_ELIGIBLE"].any())
        self.assertTrue(result["DROP_REASON"].eq("uncovered_analysis_dates").all())
        self.assertTrue(result["ANALYSIS_SUPPORT_CHECKED"].all())
        self.assertTrue(result["ANALYSIS_DATE_COUNT"].eq(48).all())
        self.assertTrue(result["SUPPORTED_ANALYSIS_DATE_COUNT"].eq(42).all())
        self.assertTrue(result["UNCOVERED_ANALYSIS_DATE_COUNT"].eq(6).all())
        np.testing.assert_allclose(result["ANALYSIS_DATE_SUPPORT_FRACTION"], 42 / 48)
        self.assertTrue(result[["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]].isna().all().all())
        uncovered = json.loads(result.iloc[0]["UNCOVERED_ANALYSIS_DATES_JSON"])
        self.assertEqual(
            uncovered,
            [f"2024-{month:02d}-20" for month in range(1, 7)],
        )
        diagnostics = json.loads(result.iloc[0]["ANALYSIS_SUPPORT_DIAGNOSTICS_JSON"])
        self.assertEqual(len(diagnostics), 12)

    def test_runner_matched_contrast_equal_strata_blocks_simpson_reversal(self):
        """층별 차가 0인데 표본수 불균형 pooled 대비가 커지는 Simpson 회귀를 잡는다."""
        build = self.function("build_matched_state_contrast")
        rows = []
        for month in range(1, 7):
            for day in range(1, 9):
                date = pd.Timestamp(2024, month, day)
                patterns = [
                    ("bright", "inversion_like", range(0, 10), 10.0),
                    ("bright", "normal_negative", range(10, 11), 10.0),
                    ("dark", "inversion_like", range(11, 12), 0.0),
                    ("dark", "normal_negative", range(12, 22), 0.0),
                ]
                for solar, state, hours, fixed_error in patterns:
                    for hour in hours:
                        rows.append(
                            {
                                "STN": "100",
                                "TM": date + pd.Timedelta(hours=hour),
                                "SOLAR_STATE": solar,
                                "REPRODUCED": True,
                                "STATE_PRIMARY": state,
                                "STATE_SPAN200": state,
                                "STATE_NEFF4": state,
                                "STATE_OLS_ONLY": state,
                                "error_none": 0.0,
                                "error_fixed": fixed_error,
                            }
                        )
        result = build(pd.DataFrame(rows), n_boot=100, seed=27)
        np.testing.assert_allclose(result["CONTRAST_INV_MINUS_NORMAL"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["BLOCK7_CI_LOW"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["BLOCK7_CI_HIGH"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["P_VALUE"], 1.0, atol=0)

    def test_epoch_specific_target_coordinates_drive_solar_and_exact_geometry_metrics(self):
        """연중 이설 뒤에도 첫 epoch 좌표로 태양고도를 고정하는 회귀를 잡는다."""
        analyze = self.function("analyze_station_injected")
        fixture = _station_fixture()
        meta = fixture["meta"].copy()
        old_target = meta["지점"].eq("100")
        meta.loc[old_target, "종료일"] = pd.Timestamp("2024-07-01")
        new_target = meta.loc[old_target].iloc[0].copy()
        new_target["시작일"] = pd.Timestamp("2024-07-01")
        new_target["종료일"] = pd.NaT
        new_target["경도"] = 129.0
        meta = pd.concat([meta, pd.DataFrame([new_target])], ignore_index=True)
        pairs = fixture["pairs"].copy()
        pairs.loc[1, "COORD_EPOCH"] = "2024-07-01"

        predictions = []
        for row in pairs.itertuples(index=False):
            geometry = self.function("reconstruct_signature_geometry")(
                target_id="100",
                neighbor_ids=fixture["neighbor_map"]["sig-A"],
                timestamp=row.TM,
                coord_epoch=row.COORD_EPOCH,
                expected_neighbor_count=7,
                meta=meta,
                terrain=fixture["terrain"],
            )
            values = fixture["ta_data"].loc[row.TM, list(geometry["neighbor_ids"])].to_numpy(float)
            if geometry["zero_neighbor_distance"]:
                prediction = values[np.flatnonzero(geometry["distances_km"] == 0)[0]]
            else:
                prediction = float(values @ geometry["idw_weights"])
            predictions.append(prediction)
        pairs["OBSERVED"] = np.asarray(predictions) - pairs["error_none"].to_numpy(float)
        fixture["meta"] = meta
        fixture["pairs"] = pairs
        result = analyze(**fixture, strict_reproduction=True, return_hourly=True)
        hourly = result["hourly"].sort_values("TM")

        expected_second = self.function("solar_elevation_noaa")(
            pd.DatetimeIndex([hourly.iloc[1]["TM"]]),
            latitude_deg=37.0,
            longitude_deg=129.0,
        )[0]
        self.assertAlmostEqual(hourly.iloc[1]["SOLAR_ELEVATION_DEG"], expected_second, places=10)
        self.assertEqual(result["metrics"]["geometry_build_count"], 2)
        self.assertEqual(result["metrics"]["processed_neighbor_values"], 14)

    def test_multistation_weather_coverage_preserves_exact_80_percent_boundary(self):
        """paired 분모를 전체 파일행으로 바꾸거나 79.99%를 80%로 반올림하는 회귀를 잡는다."""
        merge = self.function("merge_weather_conditions")
        times = pd.date_range("2024-01-01", periods=10_000, freq="h")
        pairs = pd.concat(
            [
                pd.DataFrame({"STN": station, "TM": times})
                for station in ["100", "101"]
            ],
            ignore_index=True,
        )
        weather_parts = []
        for station, valid_count in [("100", 8000), ("101", 7999)]:
            observed = np.full(len(times), np.nan)
            observed[:valid_count] = 1.0
            weather_parts.append(
                pd.DataFrame(
                    {
                        "STN": station,
                        "TM": times,
                        "WS10": observed,
                        "WS1": np.nan,
                        "RN_60m": np.where(np.isfinite(observed), 0.0, np.nan),
                        "HM": np.where(np.isfinite(observed), 50.0, np.nan),
                        "TD": np.where(np.isfinite(observed), -5.0, np.nan),
                    }
                )
            )
        result = merge(pairs, pd.concat(weather_parts, ignore_index=True), min_coverage=0.8)
        coverage = result["coverage"].set_index("STN")
        self.assertEqual(coverage.loc["100", "WIND_COVERAGE"], 0.8)
        self.assertTrue(bool(coverage.loc["100", "WEATHER_PRIMARY_ELIGIBLE"]))
        self.assertEqual(coverage.loc["101", "WIND_COVERAGE"], 0.7999)
        self.assertFalse(bool(coverage.loc["101", "WEATHER_PRIMARY_ELIGIBLE"]))

    def test_full_529_skeleton_survives_missing_payloads_before_subset_join(self):
        """원자료가 없는 관측소나 109·33 밖 관측소를 선필터해 분모를 줄이는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        ids = [str(value) for value in range(100, 629)]
        truth = pd.DataFrame(
            {
                "STN": ids,
                "BIAS_none": 0.2,
                "BIAS_fixed": 0.2,
                "SIGNIFICANCE_CLASS": "uncertain",
                "MECHANISM": "",
                "mean_absdz_geom_km": 0.01,
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out_dir, allowed_fullnet_root=out_dir.parent, enforce_canonical=False)
            station = pd.read_csv(out_dir / "station_structure_coverage.csv", encoding="utf-8-sig")
            state = pd.read_csv(out_dir / "state_condition_bias.csv", encoding="utf-8-sig")
            matched = pd.read_csv(out_dir / "matched_state_contrast.csv", encoding="utf-8-sig")
            spatial = pd.read_csv(
                out_dir / "spatial_structure_sensitivity_summary.csv",
                encoding="utf-8-sig",
            )
            coverage = pd.read_csv(
                out_dir / "matched_state_coverage_sensitivity.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(len(station), 529)
            self.assertEqual(station["STN"].astype(str).nunique(), 529)
            self.assertEqual(state["STN"].astype(str).nunique(), 529)
            self.assertEqual(matched["STN"].astype(str).nunique(), 529)
            self.assertEqual(len(spatial), 529 * 81 * 4)
            self.assertEqual(len(coverage), 529 * 81 * 9)
            missing = station.loc[station["STN"].astype(str).eq("101")].iloc[0]
            self.assertEqual(missing["WEATHER_DROP_REASON"], "source_payload_missing")

    def test_manifest_validator_rejects_duplicates_tampering_and_backup_failure_rolls_back(self):
        """전 artifact hash와 backup rename 뒤 rollback이 실제로 작동하는지 검증한다."""
        run = self.function("run_injected_analysis")
        validate = self.function("validate_published_artifacts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out_dir, allowed_fullnet_root=out_dir.parent, enforce_canonical=False)
            original_manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
            parsed = json.loads(original_manifest)
            parsed["artifacts"].append(dict(parsed["artifacts"][0]))
            (out_dir / "manifest.json").write_text(json.dumps(parsed), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)
            (out_dir / "manifest.json").write_text(original_manifest, encoding="utf-8")
            (out_dir / "analysis_summary.md").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.mkdir(parents=True)
            (out_dir / "previous.bin").write_bytes(b"bitwise-stable")
            with self.assertRaises(RuntimeError):
                run(
                    truth,
                    {"100": _station_fixture()},
                    out_dir,
                    allowed_fullnet_root=out_dir.parent,
                    enforce_canonical=False,
                    failure_injection="after_backup_rename",
                )
            self.assertEqual({path.name for path in out_dir.iterdir()}, {"previous.bin"})
            self.assertEqual((out_dir / "previous.bin").read_bytes(), b"bitwise-stable")

    def test_published_validator_rejects_semantic_key_numeric_category_and_count_mutations(self):
        """hash를 함께 위조해도 의미상 모순인 CSV가 validator를 통과하지 못하게 한다."""
        run = self.function("run_injected_analysis")
        validate = self.function("validate_published_artifacts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out_dir = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out_dir.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out_dir, allowed_fullnet_root=out_dir.parent, enforce_canonical=False)
            original_manifest = (out_dir / "manifest.json").read_bytes()
            original_csv = {
                name: (out_dir / name).read_bytes() for name in EXPECTED_CSV_SCHEMAS
            }

            def restore():
                (out_dir / "manifest.json").write_bytes(original_manifest)
                for name, payload in original_csv.items():
                    (out_dir / name).write_bytes(payload)

            def rewrite(name, frame):
                frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
                manifest = json.loads(original_manifest.decode("utf-8"))
                entry = next(item for item in manifest["artifacts"] if item["path"] == name)
                entry["rows"] = len(frame)
                entry["bytes"] = (out_dir / name).stat().st_size
                entry["sha256"] = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
                (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            state_name = "state_condition_bias.csv"
            state = pd.read_csv(out_dir / state_name, encoding="utf-8-sig")
            rewrite(state_name, pd.concat([state, state.iloc[[0]]], ignore_index=True))
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

            restore()
            state = pd.read_csv(out_dir / state_name, encoding="utf-8-sig")
            state.loc[state["N_HOURS"].gt(0).idxmax(), "BIAS_none"] = np.inf
            rewrite(state_name, state)
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

            restore()
            state = pd.read_csv(out_dir / state_name, encoding="utf-8-sig")
            state.loc[0, "STRUCTURE_STATE"] = "actual_vertical_inversion"
            rewrite(state_name, state)
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

            restore()
            station_name = "station_structure_coverage.csv"
            station = pd.read_csv(out_dir / station_name, encoding="utf-8-sig")
            station.loc[0, "PRIMARY_UNCERTAIN_HOURS"] += 1
            rewrite(station_name, station)
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

            restore()
            state = pd.read_csv(out_dir / state_name, encoding="utf-8-sig")
            zero_index = state["N_HOURS"].eq(0).idxmax()
            state.loc[zero_index, "BIAS_none"] = 0.1
            rewrite(state_name, state)
            with self.assertRaises(ValueError):
                validate(out_dir, allowed_fullnet_root=out_dir.parent)

    def test_production_source_path_streams_one_station_and_counts_actual_queries_reads(self):
        """production 경로가 paired를 중복조회하거나 같은 TA station-year를 재독하는 회귀를 잡는다."""
        production = self.function("run_production_analysis")
        fixture = _station_fixture()
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            truth_path = root / "truth.csv"
            terrain_path = root / "terrain.csv"
            meta_path = root / "meta.csv"
            db_path = root / "readonly.db"
            truth.to_csv(truth_path, index=False, encoding="utf-8-sig")
            fixture["terrain"].to_csv(terrain_path, index=False, encoding="utf-8-sig")
            meta_path.write_text("patched loader", encoding="utf-8")
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE raw (STN TEXT, TM TEXT, interpolable INTEGER, METHOD TEXT)"
            )
            for timestamp in fixture["pairs"]["TM"]:
                for method in ["none", "fixed_lapse"]:
                    connection.execute(
                        "INSERT INTO raw VALUES (?, ?, 1, ?)",
                        ("100", str(timestamp), method),
                    )
            connection.commit()
            aws_root = root / "aws"
            for station_id in [str(value) for value in range(100, 108)]:
                folder = aws_root / f"Station_{station_id}"
                folder.mkdir(parents=True)
                for row_index, (timestamp, month) in enumerate(
                    zip(fixture["pairs"]["TM"], ["202406", "202412"])
                ):
                    target_weather = fixture["weather"].iloc[row_index]
                    ta_value = (
                        fixture["ta_data"].loc[timestamp, station_id]
                        if station_id in fixture["ta_data"].columns
                        else 0.0
                    )
                    pd.DataFrame(
                        {
                            "TM": [timestamp],
                            "TA": [ta_value],
                            "WS10": [target_weather["WS10"]],
                            "WS1": [target_weather["WS1"]],
                            "RN_60m": [target_weather["RN_60m"]],
                            "HM": [target_weather["HM"]],
                            "TD": [target_weather["TD"]],
                        }
                    ).to_csv(folder / f"AWS_{month}.csv", index=False)
            out_dir = root / "fullnet" / "bias_physical_followup"
            out_dir.parent.mkdir(parents=True)
            receipt_path = root / "bias_physical_followup.acceptance.json"

            with (
                patch('analysis.bias_data.open_readonly_db', return_value=connection),
                patch('analysis.bias_data.load_neighbor_map', return_value=fixture["neighbor_map"]),
                patch('analysis.bias_data.load_station_pairs', return_value=fixture["pairs"]) as load_pairs,
                patch('weather.find_stations.load_station_meta', return_value=fixture["meta"]),
                patch.object(
                    runner,
                    "validate_canonical_station_results",
                    return_value=runner._expected_canonical_invariants(),
                ),
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
            ):
                production(
                    out_dir=out_dir,
                    db_path=db_path,
                    station_results_path=truth_path,
                    metadata_path=meta_path,
                    terrain_path=terrain_path,
                    aws_base_folder=aws_root,
                    acceptance_receipt_path=receipt_path,
                )

            self.assertEqual(load_pairs.call_count, 1)
            metadata = json.loads((out_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["db_query_count"], 2)
            self.assertEqual(metadata["db_full_hash_verification_count"], 4)
            self.assertNotIn("db_hash_verification_count", metadata)
            self.assertNotIn("db_hash_verification_elapsed_seconds", metadata)
            self.assertGreaterEqual(
                metadata["input_sources"]["db_snapshot_creation_elapsed_seconds"],
                0.0,
            )
            self.assertGreaterEqual(
                metadata["input_sources"]["db_snapshot_hash_elapsed_seconds"],
                0.0,
            )
            self.assertRegex(
                metadata["input_sources"]["db_snapshot_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertGreater(metadata["input_sources"]["db_snapshot_bytes"], 0)
            self.assertEqual(metadata["aws_file_read_count"], 16)  # 8개 관측소×월 파일 2개, 재독 없음
            self.assertEqual(metadata["geometry_build_count"], 1)
            self.assertEqual(metadata["processed_neighbor_values"], 14)
            self.assertFalse(
                any(out_dir.parent.glob(".bias-db-snapshot-*")),
                msg="private DB snapshot must be deleted after publication",
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["source_snapshot_sha256"],
                metadata["input_sources"]["db_snapshot_sha256"],
            )
            self.assertEqual(
                receipt["output_manifest_sha256"],
                hashlib.sha256((out_dir / "manifest.json").read_bytes()).hexdigest(),
            )
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
                patch.object(
                    runner,
                    "validate_canonical_station_results",
                    return_value=runner._expected_canonical_invariants(),
                ),
            ):
                runner.validate_acceptance_receipt(out_dir, receipt_path)

            failed_connection = sqlite3.connect(db_path)
            with (
                patch('analysis.bias_data.open_readonly_db', return_value=failed_connection),
                patch('analysis.bias_data.load_neighbor_map', return_value=fixture["neighbor_map"]),
                patch('analysis.bias_data.load_station_pairs', return_value=fixture["pairs"]),
                patch('weather.find_stations.load_station_meta', return_value=fixture["meta"]),
                patch.object(
                    runner,
                    "validate_canonical_station_results",
                    return_value=runner._expected_canonical_invariants(),
                ),
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
                patch.object(
                    runner,
                    "_publish_precomputed_analysis",
                    side_effect=RuntimeError("injected publish failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "publish failure"),
            ):
                production(
                    out_dir=out_dir,
                    db_path=db_path,
                    station_results_path=truth_path,
                    metadata_path=meta_path,
                    terrain_path=terrain_path,
                    aws_base_folder=aws_root,
                    acceptance_receipt_path=root / "failed.acceptance.json",
                )
            self.assertFalse(
                any(out_dir.parent.glob(".bias-db-snapshot-*")),
                msg="private DB snapshot must also be deleted after publication failure",
            )

            # 재시작 receipt 검증은 당시 원본 경로가 이동·보관된 뒤에도
            # 외부 receipt와 final bundle만으로 완료되어야 한다.
            for source_path in [truth_path, meta_path, terrain_path, db_path]:
                source_path.unlink()
            for aws_path in aws_root.rglob("*.csv"):
                aws_path.unlink()
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
                self.assertRaises(FileNotFoundError),
            ):
                runner.validate_published_artifacts(
                    out_dir,
                    trusted_primary_eligibility_binding_sha256=receipt[
                        "primary_eligibility_source_binding_sha256"
                    ],
                    sources_prevalidated=True,
                )
            with patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir):
                runner.validate_acceptance_receipt(out_dir, receipt_path)

            original_receipt = receipt_path.read_bytes()
            mutated_receipt = json.loads(original_receipt.decode("utf-8"))
            mutated_receipt["primary_eligibility_source_binding_sha256"] = "0" * 64
            receipt_path.write_text(
                json.dumps(mutated_receipt, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
                self.assertRaises(ValueError),
            ):
                runner.validate_acceptance_receipt(out_dir, receipt_path)
            receipt_path.write_bytes(original_receipt)

            ledger_path = out_dir / "primary_eligibility_ledger.csv"
            ledger_path.write_bytes(ledger_path.read_bytes() + b"\n")
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item
                for item in manifest["artifacts"]
                if item["path"] == "primary_eligibility_ledger.csv"
            )
            entry["bytes"] = ledger_path.stat().st_size
            entry["sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
                self.assertRaisesRegex(ValueError, "receipt identity|manifest"),
            ):
                runner.validate_acceptance_receipt(out_dir, receipt_path)


class IntegrationReviewFixRound1Tests(RunnerFunctionTestCase):
    def test_review_primary_state_uses_the_same_leave_one_gate_as_sensitivity(self):
        """한 이웃 제거 시 부호가 깨지는 시각을 H1/885만 inversion으로 세는 회귀를 잡는다."""
        calculate = self.function("vectorized_structure_diagnostics")
        classify = self.function("_sensitivity_state_for_setting")
        elevations = np.arange(0.0, 700.0, 100.0)
        distances = np.array(
            [
                0.6174001644172612,
                0.8186797077627904,
                2.1419834004806186,
                0.7349522281848071,
                0.5040476513680775,
                2.0393892832930427,
                0.604585018533224,
            ]
        )
        temperatures = np.array(
            [
                [
                    10.85593162101752,
                    11.03362138596307,
                    11.100487556084179,
                    11.592956768662441,
                    10.382319114002044,
                    14.67199757651267,
                    14.870283735139695,
                ]
            ]
        )
        diagnostics = calculate(
            elevations_m=elevations,
            temperature_matrix_c=temperatures,
            distances_km=distances,
            neighbor_ids=[str(value) for value in range(101, 108)],
            target_id="100",
        )
        self.assertFalse(bool(diagnostics.loc[0, "LOO_STABLE_D2"]))
        hourly = diagnostics.assign(REPRODUCED=True)
        sensitivity_state, _ = classify(
            hourly,
            {
                "SETTING_ID": "n6_span100_wsd50_wd2",
                "MIN_N": 6,
                "MIN_SPAN_M": 100,
                "MIN_WEIGHTED_SD_M": 50,
                "AUX_WEIGHT": "d2",
            },
        )
        self.assertEqual(sensitivity_state[0], "uncertain")
        self.assertEqual(diagnostics.loc[0, "STATE_PRIMARY"], sensitivity_state[0])

    def test_c1_published_p_eligible_only_bh_and_q_gated_evidence(self):
        """ineligible p가 BH family에 들어가거나 CI만으로 자료 지지가 되는 회귀를 잡는다."""
        apply_bh = self.function("_apply_bh_within_config")
        frame = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "CONFIG": ["primary", "primary", "span200", "span200"],
                "COMPARISON_ELIGIBLE": [True, False, True, True],
                "P_VALUE": [0.04, 0.90, 0.04, 0.90],
                "Q_VALUE": np.nan,
                "BLOCK7_CI_LOW": [-1.0, -1.0, -1.0, -1.0],
                "BLOCK7_CI_HIGH": [-0.1, -0.1, -0.1, -0.1],
                "EVIDENCE_GRADE": "자료로 지지",
                "SUBSET_MEANINGFUL_WORSENING": [False, True, False, True],
                "SUBSET_HIGH_AFTER": [False, True, True, False],
            }
        )
        apply_bh(frame)

        self.assertAlmostEqual(frame.loc[0, "Q_VALUE"], 0.04)
        self.assertTrue(pd.isna(frame.loc[1, "P_VALUE"]))
        self.assertTrue(pd.isna(frame.loc[1, "Q_VALUE"]))
        self.assertEqual(frame.loc[0, "EVIDENCE_GRADE"], "자료로 지지")
        self.assertAlmostEqual(frame.loc[2, "Q_VALUE"], 0.08)
        self.assertNotEqual(frame.loc[2, "EVIDENCE_GRADE"], "자료로 지지")

        mutated = frame.copy()
        mutated["SUBSET_MEANINGFUL_WORSENING"] = ~mutated[
            "SUBSET_MEANINGFUL_WORSENING"
        ]
        mutated["SUBSET_HIGH_AFTER"] = ~mutated["SUBSET_HIGH_AFTER"]
        mutated["Q_VALUE"] = np.nan
        apply_bh(mutated)
        np.testing.assert_allclose(
            mutated["P_VALUE"], frame["P_VALUE"], equal_nan=True
        )
        np.testing.assert_allclose(
            mutated["Q_VALUE"], frame["Q_VALUE"], equal_nan=True
        )

    def test_c1_validator_recomputes_published_q_and_rejects_ineligible_p(self):
        """manifest hash를 맞춰도 published p→q와 eligible family 모순을 통과시키지 않는다."""
        validate = self.function("_validate_matched_inference_contract")
        frame = pd.DataFrame(
            {
                "STN": ["100", "101"],
                "CONFIG": ["primary", "primary"],
                "COMPARISON_ELIGIBLE": [True, True],
                "DROP_REASON": ["eligible", "eligible"],
                "ANALYSIS_SUPPORT_CHECKED": [True, True],
                "UNCOVERED_ANALYSIS_DATE_COUNT": [0, 0],
                "ANALYSIS_DATE_SUPPORT_FRACTION": [1.0, 1.0],
                "BLOCK7_VALID_FRACTION": [1.0, 1.0],
                "BLOCK7_VALID_REPLICATES": [2000, 2000],
                "P_VALUE": [0.04, 0.90],
                "Q_VALUE": [0.08, 0.90],
                "BLOCK7_CI_LOW": [-1.0, -1.0],
                "BLOCK7_CI_HIGH": [-0.1, -0.1],
                "EVIDENCE_GRADE": ["현재 자료에서 지지되지 않음"] * 2,
            }
        )
        validate(frame)
        wrong_q = frame.copy()
        wrong_q.loc[0, "Q_VALUE"] = 0.04
        with self.assertRaises(ValueError):
            validate(wrong_q)
        ineligible_p = frame.copy()
        ineligible_p.loc[0, "COMPARISON_ELIGIBLE"] = False
        with self.assertRaises(ValueError):
            validate(ineligible_p)
        shrunk_family = frame.copy()
        shrunk_family.loc[1, ["P_VALUE", "Q_VALUE"]] = np.nan
        shrunk_family.loc[0, "Q_VALUE"] = 0.04
        shrunk_family.loc[0, "EVIDENCE_GRADE"] = "자료로 지지"
        with self.assertRaisesRegex(ValueError, "eligible.*P_VALUE|full-family"):
            validate(shrunk_family)
        missing_boolean = frame.copy()
        missing_boolean["COMPARISON_ELIGIBLE"] = missing_boolean[
            "COMPARISON_ELIGIBLE"
        ].astype(object)
        missing_boolean.loc[0, "COMPARISON_ELIGIBLE"] = np.nan
        with self.assertRaisesRegex(ValueError, "boolean|eligible"):
            validate(missing_boolean)
        flipped_eligibility = frame.copy()
        flipped_eligibility.loc[1, "COMPARISON_ELIGIBLE"] = False
        flipped_eligibility.loc[
            1, ["P_VALUE", "Q_VALUE", "BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]
        ] = np.nan
        flipped_eligibility.loc[1, "EVIDENCE_GRADE"] = "현재 자료에서 지지되지 않음"
        flipped_eligibility.loc[1, "DROP_REASON"] = "insufficient_valid_replicates"
        flipped_eligibility.loc[0, "Q_VALUE"] = 0.04
        flipped_eligibility.loc[0, "EVIDENCE_GRADE"] = "자료로 지지"
        with self.assertRaisesRegex(ValueError, "DROP_REASON|eligibility"):
            validate(flipped_eligibility)

    def test_c2_h2_observed_sign_flip_and_885_ineligible_case_are_not_supported(self):
        """사전 예상 문자열만으로 반대 signed BIAS와 885 기술사례를 지지하는 회귀를 잡는다."""
        coast_builder = self.function("_build_coast_time_bias")
        hourly = pd.DataFrame(
            {
                "STN": ["100"] * 30,
                "TM": pd.date_range("2024-06-01 12:00", periods=30, freq="D"),
                "MONTH": 6,
                "SOLAR_STATE": "bright",
                "REPRODUCED": True,
                "error_none": -0.8,
                "error_fixed": -1.0,
                "DELTA_COAST_KM": -10.0,
            }
        )
        coast = coast_builder(hourly).iloc[0]
        self.assertEqual(coast["H2_EXPECTED_BIAS_SIGN"], "positive")
        self.assertEqual(coast["H2_OBSERVED_BIAS_SIGN"], "negative")
        self.assertFalse(bool(coast["H2_DIRECTION_MATCH"]))
        self.assertEqual(coast["EVIDENCE_GRADE"], "현재 자료에서 지지되지 않음")

        case_builder = self.function("_station_885_table")
        case_hourly = hourly.assign(
            STN="885",
            STATE_PRIMARY="ineligible",
            STATE_PRIMARY_REASON="n_below_threshold",
        )
        case = case_builder(case_hourly)
        self.assertTrue(case["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all())
        self.assertTrue(case["LIMITATION"].str.contains("descriptive").all())

    def test_c2_physical_hypothesis_table_separates_high_control_and_signed_high(self):
        """high33/control82와 high 양·음 부호를 한 raw 표로 합치는 회귀를 잡는다."""
        build = self.function("build_physical_hypothesis_comparison")
        equivalent = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "BIAS_none": [0.7, -0.7, 0.2, -0.2],
                "BIAS_fixed": [0.8, -0.8, 0.2, -0.2],
                "ABS_BIAS_NONE": [0.7, 0.7, 0.2, 0.2],
                "ABS_BIAS_FIXED": [0.8, 0.8, 0.2, 0.2],
            }
        )
        structure = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "MEDIAN_RELIEF_MISMATCH_M": [100.0, -100.0, 10.0, -10.0],
                "MEDIAN_DELTA_COAST_KM": [-20.0, 20.0, -5.0, 5.0],
                "MEDIAN_ABS_DELTA_COAST_KM": [20.0, 20.0, 5.0, 5.0],
                "TPI_STATUS": "NOT_MEASURED",
            }
        )
        coast = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "MONTH": 6,
                "SOLAR_STATE": "bright",
                "N_HOURS": 30,
                "N_DAYS": 30,
                "N_MONTHS": 1,
                "BIAS_fixed": [0.8, -0.8, 0.2, -0.2],
                "DELTA_COAST_KM": [-20.0, 20.0, -5.0, 5.0],
                "H2_EXPECTED_BIAS_SIGN": ["positive", "negative", "positive", "negative"],
            }
        )
        result = build(equivalent, structure, coast)
        self.assertEqual(len(result), 6)
        self.assertEqual(
            set(result["CONTRAST_ID"]),
            {
                "H1_ALL_HIGH_V_CONTROL",
                "H1_POS_HIGH_V_CONTROL",
                "H1_NEG_HIGH_V_CONTROL",
                "H2_ALL_HIGH_V_CONTROL",
                "H2_POS_HIGH_V_CONTROL",
                "H2_NEG_HIGH_V_CONTROL",
            },
        )
        all_rows = result.loc[result["SIGN_STRATUM"].eq("all")]
        self.assertTrue((all_rows["TARGET_CANONICAL_N"] == 2).all())
        self.assertTrue((all_rows["REFERENCE_CANONICAL_N"] == 2).all())
        positive = result.loc[result["SIGN_STRATUM"].eq("positive")]
        negative = result.loc[result["SIGN_STRATUM"].eq("negative")]
        self.assertTrue((positive["TARGET_CANONICAL_N"] == 1).all())
        self.assertTrue((negative["TARGET_CANONICAL_N"] == 1).all())
        h1 = result.loc[result["HYPOTHESIS_ID"].eq("H1_LOCAL_TOPOGRAPHY")]
        self.assertTrue(h1["LIMITATION"].str.contains("TPI").all())
        self.assertFalse(h1["EVIDENCE_GRADE"].eq("자료로 지지").any())

        relabeled = equivalent.copy()
        relabeled.loc[relabeled["STN"].eq("101"), "ABS_BIAS_FIXED"] = 0.2
        changed = build(relabeled, structure, coast)
        changed_all = changed.loc[changed["SIGN_STRATUM"].eq("all")]
        self.assertTrue((changed_all["TARGET_CANONICAL_N"] == 1).all())
        self.assertTrue((changed_all["REFERENCE_CANONICAL_N"] == 3).all())

    def test_c3_production_guard_rejects_lookalike_case_missing_parent_and_symlink_before_load(self):
        """canonical suffix만 같은 임의 폴더가 production 게시 대상으로 승인되는 회귀를 잡는다."""
        guard = self.function("guard_production_output_directory")
        production = self.function("run_production_analysis")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            canonical = root / "WeatherData_AWS" / "interpolation_batch" / "fullnet" / "bias_physical_followup"
            canonical.parent.mkdir(parents=True)
            with patch.object(runner, "CANONICAL_OUTPUT_DIR", canonical):
                self.assertEqual(guard(canonical), canonical.resolve())
                bad_paths = [
                    root / "unrelated" / "fullnet" / "bias_physical_followup",
                    canonical.parent.parent / "sibling" / "fullnet" / "bias_physical_followup",
                    root / "missing" / "fullnet" / "bias_physical_followup",
                    Path(str(canonical).upper()),
                ]
                for bad in bad_paths:
                    with self.subTest(path=bad):
                        with patch('analysis.bias_data.open_readonly_db') as db_open:
                            with self.assertRaises(ValueError):
                                production(out_dir=bad)
                            db_open.assert_not_called()

                link_parent = root / "linked_fullnet"
                try:
                    link_parent.symlink_to(canonical.parent, target_is_directory=True)
                except OSError:
                    link_parent = None
                if link_parent is not None:
                    with self.assertRaises(ValueError):
                        guard(link_parent / "bias_physical_followup")

    def test_c3_injected_guard_requires_explicit_allowed_fullnet_root(self):
        """test용 임시 root 허용이 production canonical guard를 완화하는 회귀를 잡는다."""
        guard = self.function("guard_injected_output_directory")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            fullnet = root / "fullnet"
            fullnet.mkdir()
            intended = fullnet / "bias_physical_followup"
            self.assertEqual(
                guard(intended, allowed_fullnet_root=fullnet), intended.resolve()
            )
            with self.assertRaises(ValueError):
                guard(
                    root / "other" / "fullnet" / "bias_physical_followup",
                    allowed_fullnet_root=fullnet,
                )

    def test_i1_aws_loader_rejects_infile_and_crossfile_duplicate_tm_with_provenance(self):
        """production thin loader가 동일·충돌 duplicate를 keep-last로 숨기는 회귀를 잡는다."""
        load = self.function("_load_aws_station_year_thin")
        with _workspace_temp_dir() as temp_dir:
            base = Path(temp_dir)
            folder = base / "Station_100"
            folder.mkdir()
            for values in ([1.0, 1.0], [1.0, 9.0]):
                pd.DataFrame(
                    {
                        "TM": ["2024-01-01 00:00", "2024-01-01 00:00"],
                        "TA": values,
                    }
                ).to_csv(folder / "AWS_202401.csv", index=False)
                with self.subTest(values=values):
                    with self.assertRaisesRegex(ValueError, "100.*AWS_202401"):
                        load(
                            station_id="100",
                            months=["202401"],
                            start=pd.Timestamp("2024-01-01"),
                            end=pd.Timestamp("2024-03-01"),
                            base_folder=base,
                        )

            pd.DataFrame({"TM": ["2024-02-01 00:00"], "TA": [1.0]}).to_csv(
                folder / "AWS_202401.csv", index=False
            )
            pd.DataFrame({"TM": ["2024-02-01 00:00"], "TA": [9.0]}).to_csv(
                folder / "AWS_202402.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "AWS_202401.*AWS_202402"):
                load(
                    station_id="100",
                    months=["202401", "202402"],
                    start=pd.Timestamp("2024-01-01"),
                    end=pd.Timestamp("2024-03-01"),
                    base_folder=base,
                )

    def test_i2_structure_sensitivity_has_exact_81_grid_and_leave_one_contract(self):
        """현재 네 CONFIG를 81-setting 설계 이행으로 오인하는 회귀를 잡는다."""
        calculate = self.function("vectorized_structure_diagnostics")
        summarize = self.function("build_structure_sensitivity_summary")
        elevations = np.arange(0.0, 700.0, 100.0)
        distances = np.ones(7)
        positive = 10.0 + elevations / 1000.0
        matrix = np.vstack([positive, 20.0 - elevations / 1000.0])
        diagnostics = calculate(
            elevations_m=elevations,
            temperature_matrix_c=matrix,
            distances_km=distances,
            neighbor_ids=[str(value) for value in range(101, 108)],
            target_id="100",
        )
        diagnostics["STN"] = "100"
        diagnostics["REPRODUCED"] = True
        result = summarize(diagnostics)

        self.assertEqual(len(result), 81 * 4)
        self.assertEqual(result["SETTING_ID"].nunique(), 81)
        self.assertEqual(set(result["MIN_N"]), {5, 6, 7})
        self.assertEqual(set(result["MIN_SPAN_M"]), {50, 100, 200})
        self.assertEqual(set(result["MIN_WEIGHTED_SD_M"]), {30, 50, 75})
        self.assertEqual(set(result["AUX_WEIGHT"]), {"none", "d1", "d2"})
        self.assertEqual(
            set(result["STRUCTURE_STATE"]),
            {"inversion_like", "normal_negative", "uncertain", "ineligible"},
        )
        per_setting = result.groupby("SETTING_ID", sort=False)
        self.assertTrue((per_setting.size() == 4).all())
        self.assertTrue(
            (
                per_setting["N_HOURS"].sum()
                == per_setting["REPRODUCED_HOURS"].first()
            ).all()
        )
        inversion = result.loc[result["STRUCTURE_STATE"].eq("inversion_like")]
        normal = result.loc[result["STRUCTURE_STATE"].eq("normal_negative")]
        self.assertTrue((inversion["N_HOURS"] == 1).all())
        self.assertTrue((normal["N_HOURS"] == 1).all())
        self.assertTrue((result["N_LOO_SIGN_UNSTABLE"] == 0).all())

        unstable = diagnostics.copy()
        unstable.loc[0, "LOO_STABLE_D2"] = False
        unstable.loc[0, "LOO_SIGN_STABILITY_D2"] = 6 / 7
        mutated = summarize(unstable)
        primary_rows = mutated.loc[
            mutated["SETTING_ID"].eq("n6_span100_wsd50_wd2")
        ]
        uncertain = primary_rows.loc[
            primary_rows["STRUCTURE_STATE"].eq("uncertain"), "N_HOURS"
        ].iloc[0]
        ineligible = primary_rows.loc[
            primary_rows["STRUCTURE_STATE"].eq("ineligible"), "N_HOURS"
        ].iloc[0]
        self.assertEqual(int(uncertain), 1)
        self.assertEqual(int(ineligible), 0)
        self.assertEqual(int(primary_rows["N_LOO_SIGN_UNSTABLE"].iloc[0]), 1)

        reproduction_failure = diagnostics.copy()
        reproduction_failure.loc[0, "REPRODUCED"] = False
        failed = summarize(reproduction_failure)
        failed_primary = failed.loc[
            failed["SETTING_ID"].eq("n6_span100_wsd50_wd2")
        ]
        self.assertEqual(int(failed_primary["TOTAL_PAIRED_HOURS"].iloc[0]), 2)
        self.assertEqual(int(failed_primary["REPRODUCED_HOURS"].iloc[0]), 1)
        self.assertEqual(int(failed_primary["N_HOURS"].sum()), 1)
        self.assertEqual(
            int(failed_primary["N_EXACT_REPRODUCTION_FAILURE"].iloc[0]), 1
        )

    def test_i2_matched_coverage_has_exact_nine_grid_and_primary_30_6_row(self):
        """30일·6개월 한 기준만으로 coverage 민감도를 대체하는 회귀를 잡는다."""
        build = self.function("build_matched_state_outputs")
        rows = []
        for month in range(1, 5):
            for day in range(1, 9):
                date = pd.Timestamp(2024, month, day)
                for hour, state in [(0, "inversion_like"), (1, "normal_negative")]:
                    rows.append(
                        {
                            "STN": "100",
                            "TM": date + pd.Timedelta(hours=hour),
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "REGRESSION_N": 7,
                            "ELEVATION_SPAN_M": 600.0,
                            "WEIGHTED_ELEVATION_SD_NONE_M": 200.0,
                            "WEIGHTED_ELEVATION_SD_D1_M": 200.0,
                            "WEIGHTED_ELEVATION_SD_D2_M": 200.0,
                            "N_EFF_NONE": 7.0,
                            "N_EFF_D1": 7.0,
                            "N_EFF_D2": 7.0,
                            "OLS_SLOPE_C_PER_KM": 1.0 if state == "inversion_like" else -1.0,
                            "OLS_CI95_LOW_C_PER_KM": 0.5 if state == "inversion_like" else -1.5,
                            "OLS_CI95_HIGH_C_PER_KM": 1.5 if state == "inversion_like" else -0.5,
                            "AUX_SLOPE_NONE_C_PER_KM": 1.0 if state == "inversion_like" else -1.0,
                            "AUX_SLOPE_D1_C_PER_KM": 1.0 if state == "inversion_like" else -1.0,
                            "AUX_SLOPE_D2_C_PER_KM": 1.0 if state == "inversion_like" else -1.0,
                            "LOO_STABLE_NONE": True,
                            "LOO_STABLE_D1": True,
                            "LOO_STABLE_D2": True,
                            "error_none": 1.0,
                            "error_fixed": 0.5,
                        }
                    )
        outputs = build(pd.DataFrame(rows), n_boot=40, seed=17)
        sensitivity = outputs["sensitivity"]
        primary = outputs["primary"]
        self.assertEqual(len(sensitivity), 81 * 9)
        self.assertEqual(sensitivity["SETTING_ID"].nunique(), 81)
        self.assertEqual(
            set(
                zip(
                    sensitivity["MIN_DAYS_PER_STATE"],
                    sensitivity["MIN_MONTHS_PER_STATE"],
                )
            ),
            {(days, months) for days in [20, 30, 60] for months in [3, 6, 9]},
        )
        flexible = sensitivity.loc[
            sensitivity["SETTING_ID"].eq("n6_span100_wsd50_wd2")
            & sensitivity["MIN_DAYS_PER_STATE"].eq(20)
            & sensitivity["MIN_MONTHS_PER_STATE"].eq(3)
        ].iloc[0]
        strict = sensitivity.loc[
            sensitivity["SETTING_ID"].eq("n6_span100_wsd50_wd2")
            & sensitivity["MIN_DAYS_PER_STATE"].eq(60)
            & sensitivity["MIN_MONTHS_PER_STATE"].eq(9)
        ].iloc[0]
        self.assertTrue(bool(flexible["COVERAGE_ELIGIBLE"]))
        self.assertFalse(bool(strict["COVERAGE_ELIGIBLE"]))
        self.assertEqual(flexible["DROP_REASON"], "eligible")
        self.assertEqual(strict["DROP_REASON"], "insufficient_unique_days")
        primary_grid = sensitivity.loc[
            sensitivity["SETTING_ID"].eq("n6_span100_wsd50_wd2")
            & sensitivity["MIN_DAYS_PER_STATE"].eq(30)
            & sensitivity["MIN_MONTHS_PER_STATE"].eq(6)
        ].reset_index(drop=True)
        self.assertEqual(len(primary_grid), len(primary))
        np.testing.assert_allclose(
            primary_grid["CONTRAST_INV_MINUS_NORMAL"],
            primary["CONTRAST_INV_MINUS_NORMAL"],
            equal_nan=True,
        )

    def test_review_validator_rejects_physical_probability_and_descriptive_grade_mutations(self):
        """음수 p/q 또는 설명용 H2 행의 직접 지지 등급을 hash만 맞춰 승인하는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(
                truth,
                {"100": _station_fixture()},
                out,
                allowed_fullnet_root=out.parent,
                enforce_canonical=False,
            )
            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }

            negative_probability = {
                name: frame.copy() for name, frame in tables.items()
            }
            physical = negative_probability["physical_hypothesis_comparison.csv"]
            index = physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL").idxmax()
            for block in (30, 50, 100):
                physical.loc[index, f"N_BLOCKS_{block}KM"] = 8
                physical.loc[index, f"TARGET_BLOCKS_{block}KM"] = 4
                physical.loc[index, f"REFERENCE_BLOCKS_{block}KM"] = 4
                physical.loc[index, f"VALID_REPLICATES_{block}KM"] = 1900
                physical.loc[index, f"CI_LOW_{block}KM"] = 0.1
                physical.loc[index, f"CI_HIGH_{block}KM"] = 0.5
                physical.loc[
                    index, "P_VALUE" if block == 50 else f"P_VALUE_{block}KM"
                ] = -1.0
            physical.loc[index, "Q_VALUE"] = -1.0
            with self.assertRaisesRegex(ValueError, "physical probability"):
                validate_tables(negative_probability)

            descriptive_support = {
                name: frame.copy() for name, frame in tables.items()
            }
            descriptive = descriptive_support["physical_hypothesis_comparison.csv"]
            descriptive.loc[
                descriptive["CONTRAST_ID"].eq("H2_POS_HIGH_V_CONTROL"),
                "EVIDENCE_GRADE",
            ] = "자료로 지지"
            with self.assertRaisesRegex(ValueError, "descriptive H2 evidence grade"):
                validate_tables(descriptive_support)

            fabricated_inference = {
                name: frame.copy() for name, frame in tables.items()
            }
            physical = fabricated_inference["physical_hypothesis_comparison.csv"]
            index = physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL").idxmax()
            for block in (30, 50, 100):
                physical.loc[index, f"N_BLOCKS_{block}KM"] = 8
                physical.loc[index, f"TARGET_BLOCKS_{block}KM"] = 4
                physical.loc[index, f"REFERENCE_BLOCKS_{block}KM"] = 4
                physical.loc[index, f"VALID_REPLICATES_{block}KM"] = 1900
                physical.loc[index, f"CI_LOW_{block}KM"] = 0.1
                physical.loc[index, f"CI_HIGH_{block}KM"] = 0.5
                physical.loc[
                    index, "P_VALUE" if block == 50 else f"P_VALUE_{block}KM"
                ] = 0.04
            physical.loc[index, "Q_VALUE"] = 0.04
            with self.assertRaisesRegex(ValueError, "physical spatial inference"):
                validate_tables(fabricated_inference)

            coast_mutation = {name: frame.copy() for name, frame in tables.items()}
            coast = coast_mutation["coast_time_signed_bias.csv"]
            coast.loc[0, "H2_OBSERVED_BIAS_SIGN"] = (
                "negative"
                if coast.loc[0, "H2_OBSERVED_BIAS_SIGN"] != "negative"
                else "positive"
            )
            with self.assertRaisesRegex(ValueError, "coast H2 direction"):
                validate_tables(coast_mutation)

    def test_review_validator_requires_stationwise_exact_sensitivity_cartesian_products(self):
        """다른 관측소가 setting을 보유한다는 이유로 한 관측소의 4행·9행 누락을 숨기는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        truth = pd.DataFrame(
            {
                "STN": ["100", "101"],
                "BIAS_none": [0.60, 0.20],
                "BIAS_fixed": [0.62, 0.20],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"] * 2,
                "MECHANISM": ["small_change"] * 2,
                "mean_absdz_geom_km": [0.01, 0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(
                truth,
                {"100": _station_fixture()},
                out,
                allowed_fullnet_root=out.parent,
                enforce_canonical=False,
            )
            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }
            nonprimary = next(
                value
                for value in tables["spatial_structure_sensitivity_summary.csv"][
                    "SETTING_ID"
                ].unique()
                if value != "n6_span100_wsd50_wd2"
            )

            missing_spatial = {name: frame.copy() for name, frame in tables.items()}
            spatial = missing_spatial["spatial_structure_sensitivity_summary.csv"]
            missing_spatial["spatial_structure_sensitivity_summary.csv"] = spatial.loc[
                ~(spatial["STN"].astype(str).eq("100") & spatial["SETTING_ID"].eq(nonprimary))
            ].reset_index(drop=True)
            with self.assertRaisesRegex(
                ValueError, "stationwise spatial sensitivity Cartesian grid"
            ):
                validate_tables(missing_spatial)

            missing_coverage = {name: frame.copy() for name, frame in tables.items()}
            coverage = missing_coverage["matched_state_coverage_sensitivity.csv"]
            missing_coverage["matched_state_coverage_sensitivity.csv"] = coverage.loc[
                ~(coverage["STN"].astype(str).eq("100") & coverage["SETTING_ID"].eq(nonprimary))
            ].reset_index(drop=True)
            with self.assertRaisesRegex(
                ValueError, "stationwise coverage sensitivity Cartesian grid"
            ):
                validate_tables(missing_coverage)

    def test_review_validator_recomputes_physical_denominators_and_direction_rates(self):
        """H2 분모·방향비율을 CSV 자기주장만 믿고 승인하는 회귀를 잡는다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(
                truth,
                {"100": _station_fixture()},
                out,
                allowed_fullnet_root=out.parent,
                enforce_canonical=False,
            )
            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }
            physical = tables["physical_hypothesis_comparison.csv"]
            h2_all = physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL")
            physical.loc[h2_all, "TARGET_CANONICAL_N"] += 1
            physical.loc[h2_all, "TARGET_EXCLUDED_N"] += 1
            with self.assertRaisesRegex(ValueError, "physical canonical denominators"):
                validate_tables(tables)

            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }
            physical = tables["physical_hypothesis_comparison.csv"]
            h2_all = physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL")
            physical.loc[h2_all, "TARGET_DIRECTION_MATCH_N"] = 1
            physical.loc[h2_all, "TARGET_DIRECTION_TESTABLE_N"] = 2
            physical.loc[h2_all, "TARGET_DIRECTION_MATCH_RATE"] = 0.5
            with self.assertRaisesRegex(ValueError, "physical direction"):
                validate_tables(tables)

    def test_review_publish_path_checks_rss_after_large_table_aggregation(self):
        """station loop 뒤 대형 concat·추론·그림 단계의 RSS 초과를 게시하는 회귀를 잡는다."""
        publish = self.function("_publish_precomputed_analysis")
        analyze = self.function("analyze_station_injected")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        station_output = analyze(**_station_fixture(), strict_reproduction=True)
        with _workspace_temp_dir() as temp_dir:
            final = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            identity_paths = {}
            for role in ["db", "station_results", "metadata", "terrain"]:
                path = Path(temp_dir) / f"{role}.dat"
                if role == "db":
                    _create_delete_mode_sqlite(path)
                else:
                    path.write_bytes(role.encode("utf-8"))
                identity_paths[role] = path
            aws_root = Path(temp_dir) / "aws"
            aws_root.mkdir()
            aws_source = aws_root / "Station_100" / "AWS_202401.csv"
            aws_source.parent.mkdir()
            aws_source.write_text("TM,TA\n2024-01-01,1.0\n", encoding="utf-8")
            aws_inventory = [
                {
                    "relative_path": "Station_100/AWS_202401.csv",
                    "bytes": aws_source.stat().st_size,
                    "sha256": hashlib.sha256(aws_source.read_bytes()).hexdigest(),
                    "station_id": "100",
                    "month": "202401",
                }
            ]
            source_metadata = runner._capture_production_source_metadata(
                db_path=identity_paths["db"],
                station_results_path=identity_paths["station_results"],
                metadata_path=identity_paths["metadata"],
                terrain_path=identity_paths["terrain"],
                aws_base_folder=aws_root,
                aws_file_inventory=aws_inventory,
            )
            source_metadata = _attach_fixture_snapshot_metadata(
                source_metadata, identity_paths["db"]
            )
            rss_values = iter([90, 101])
            retained_station_outputs = {"100": station_output}
            with self.assertRaises(MemoryError):
                publish(
                    truth=truth,
                    station_outputs=retained_station_outputs,
                    final=final,
                    canonical=runner._expected_canonical_invariants(),
                    source_metrics={
                        "db_full_hash_verification_count": 4,
                        "db_full_hash_verification_elapsed_seconds": 0.0,
                        "aws_file_read_count": 1,
                        "aws_cache_hit_count": 0,
                        "aws_cache_miss_count": 0,
                        "aws_cache_peak_entries": 0,
                        "aws_cache_eviction_count": 0,
                        "aws_cache_max_entries": 1,
                        "rss_limit_bytes": 100,
                        "process_peak_rss_bytes": 90,
                        "stations_timed": 1,
                        "first_10_station_elapsed_seconds": 1.0,
                        "estimated_total_runtime_seconds_from_first_10": 1.0,
                        "observed_station_elapsed_seconds": 1.0,
                    },
                    source_metadata=source_metadata,
                    rss_reader=lambda: next(rss_values),
                )
            self.assertEqual(
                retained_station_outputs,
                {},
                "post-aggregation staging must not retain per-station result frames",
            )
            self.assertFalse(final.exists())

    def test_review_process_rss_uses_os_peak_not_only_current_sample(self):
        """연산 중 peak가 내려간 뒤 current RSS만 읽어 1.5GiB 초과를 놓치는 회귀를 잡는다."""
        windows_peak = self.function("_windows_peak_working_set_bytes")
        proc_peak = self.function("_parse_proc_peak_rss_bytes")
        counters = type(
            "Counters",
            (),
            {"WorkingSetSize": 100, "PeakWorkingSetSize": 250},
        )()
        self.assertEqual(windows_peak(counters), 250)
        status = "VmRSS:\t100 kB\nVmHWM:\t250 kB\n"
        self.assertEqual(proc_peak(status), 250 * 1024)
        self.assertEqual(proc_peak("VmRSS:\t100 kB\n"), 100 * 1024)

    def test_i3_manifest_and_metadata_schema_reject_rehashed_malformed_json(self):
        """hash를 함께 고친 malformed metadata와 unknown manifest version을 거부한다."""
        run = self.function("run_injected_analysis")
        validate = self.function("validate_published_artifacts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.6],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent, enforce_canonical=False)
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(out, allowed_fullnet_root=out.parent)

        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent, enforce_canonical=False)
            metadata_path = out / "analysis_metadata.json"
            metadata_path.write_text("not-json", encoding="utf-8")
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in manifest["artifacts"] if item["path"] == "analysis_metadata.json"
            )
            entry["bytes"] = metadata_path.stat().st_size
            entry["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(out, allowed_fullnet_root=out.parent)

        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent, enforce_canonical=False)
            metadata_path = out / "analysis_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["analysis_config"]["primary_setting_id"] = "n5_span50_wsd30_wnone"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in manifest["artifacts"] if item["path"] == "analysis_metadata.json"
            )
            entry["bytes"] = metadata_path.stat().st_size
            entry["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate(out, allowed_fullnet_root=out.parent)

    def test_i3_aws_inventory_byte_mutation_is_rejected(self):
        """manifest가 그대로여도 실제 원 AWS 파일이 바뀌면 provenance 검증을 실패시킨다."""
        build_metadata = self.function("_build_analysis_metadata")
        validate_metadata = self.function("_validate_analysis_metadata")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir) / "aws"
            source = root / "Station_100" / "AWS_202401.csv"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"TM,TA\n2024-01-01,1\n")
            inventory = [
                {
                    "relative_path": "Station_100/AWS_202401.csv",
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "station_id": "100",
                    "month": "202401",
                }
            ]
            identity_paths = {}
            for role in ["db", "station_results", "metadata", "terrain"]:
                path = Path(temp_dir) / f"{role}.dat"
                if role == "db":
                    _create_delete_mode_sqlite(path)
                else:
                    path.write_bytes(role.encode("utf-8"))
                identity_paths[role] = path
            metadata = build_metadata(
                canonical=runner._expected_canonical_invariants(),
                metrics={
                    "db_full_hash_verification_count": 4,
                    "db_full_hash_verification_elapsed_seconds": 0.0,
                    "aws_file_read_count": 1,
                    "aws_cache_hit_count": 0,
                    "aws_cache_miss_count": 0,
                    "aws_cache_peak_entries": 0,
                    "aws_cache_eviction_count": 0,
                    "aws_cache_max_entries": 1,
                    "process_peak_rss_bytes": 1,
                    "rss_limit_bytes": 2,
                    "stations_timed": 0,
                    "first_10_station_elapsed_seconds": 0.0,
                    "estimated_total_runtime_seconds_from_first_10": 0.0,
                    "observed_station_elapsed_seconds": 0.0,
                },
                analysis_station_ids=["100"],
                input_sources=_attach_fixture_snapshot_metadata(
                    runner._capture_production_source_metadata(
                        db_path=identity_paths["db"],
                        station_results_path=identity_paths["station_results"],
                        metadata_path=identity_paths["metadata"],
                        terrain_path=identity_paths["terrain"],
                        aws_base_folder=root,
                        aws_file_inventory=inventory,
                    ),
                    identity_paths["db"],
                ),
            )
            validate_metadata(metadata)
            source.write_bytes(source.read_bytes() + b"#")
            with self.assertRaises(ValueError):
                validate_metadata(metadata)

    def test_i3_metadata_requires_exact_core_keys_domains_and_metric_types(self):
        """재해시된 JSON도 필수 의미·형식·허용 키가 틀리면 게시할 수 없다."""
        build_metadata = self.function("_build_analysis_metadata")
        validate_metadata = self.function("_validate_analysis_metadata")
        metadata = build_metadata(
            canonical=None,
            metrics={},
            analysis_station_ids=["100"],
            input_sources={"source_mode": "injected", "aws_file_inventory": []},
        )
        validate_metadata(metadata)

        missing = json.loads(json.dumps(metadata))
        del missing["tpi_status"]
        with self.assertRaises(ValueError):
            validate_metadata(missing)

        mistyped = json.loads(json.dumps(metadata))
        mistyped["processed_paired_hours"] = "1"
        with self.assertRaises(ValueError):
            validate_metadata(mistyped)

        unknown = json.loads(json.dumps(metadata))
        unknown["unreviewed_setting"] = True
        with self.assertRaises(ValueError):
            validate_metadata(unknown)

        wrong_domain = json.loads(json.dumps(metadata))
        wrong_domain["solar_state_thresholds_deg"]["bright"] = ">=0"
        with self.assertRaises(ValueError):
            validate_metadata(wrong_domain)

    def test_i4_acceptance_monitor_rss_runtime_and_lru_shared_source_contract(self):
        """DataFrame 크기만 peak로 쓰거나 무제한 cache를 production acceptance로 부르는 회귀를 잡는다."""
        monitor_type = getattr(runner, "ProductionAcceptanceMonitor", None)
        cache_type = getattr(runner, "BoundedStationSourceCache", None)
        self.assertIsNotNone(monitor_type)
        self.assertIsNotNone(cache_type)

        monitor = monitor_type(total_stations=20, rss_limit_bytes=100)
        for _ in range(10):
            monitor.record_station(elapsed_seconds=2.0, rss_bytes=90)
        summary = monitor.summary()
        self.assertEqual(summary["process_peak_rss_bytes"], 90)
        self.assertEqual(summary["first_10_station_elapsed_seconds"], 20.0)
        self.assertEqual(summary["estimated_total_runtime_seconds_from_first_10"], 40.0)
        with self.assertRaises(MemoryError):
            monitor.record_station(elapsed_seconds=1.0, rss_bytes=101)

        loads = []

        def loader(station_id):
            loads.append(station_id)
            return pd.DataFrame({"TA": [float(station_id)]})

        cache = cache_type(max_entries=2, loader=loader)
        first = cache.get("100")
        second = cache.get("100")
        self.assertIs(first, second)
        cache.get("101")
        cache.get("102")
        self.assertEqual(loads, ["100", "101", "102"])
        metrics = cache.metrics()
        self.assertEqual(metrics["aws_cache_hit_count"], 1)
        self.assertEqual(metrics["aws_cache_miss_count"], 3)
        self.assertEqual(metrics["aws_cache_peak_entries"], 2)
        self.assertEqual(metrics["aws_cache_eviction_count"], 1)

        cached_frame = cache.get("102")
        cached_frame_ref = weakref.ref(cached_frame)
        del cached_frame, first, second
        metrics_before_clear = cache.metrics()
        cache.clear()
        gc.collect()
        self.assertIsNone(cached_frame_ref())
        self.assertEqual(cache.metrics(), metrics_before_clear)

        analyze = self.function("analyze_station_injected")
        station_result = analyze(**_station_fixture(), strict_reproduction=True)
        counters = station_result["metrics"]
        self.assertEqual(counters["unique_geometry_count"], 1)
        self.assertEqual(counters["reproduced_hour_count"], 2)
        self.assertEqual(counters["ols_fit_count"], 2)
        self.assertEqual(counters["aux_fit_count_none"], 2)
        self.assertEqual(counters["aux_fit_count_d1"], 2)
        self.assertEqual(counters["aux_fit_count_d2"], 2)
        # 이 fixture는 exact 이웃 7개 중 고도 결측 1개를 회귀에서만 제외한다.
        self.assertEqual(counters["loo_fit_count_none"], 12)
        self.assertEqual(counters["loo_fit_count_d1"], 12)
        self.assertEqual(counters["loo_fit_count_d2"], 12)
        self.assertEqual(counters["setting_classification_count"], 2 * 81)
        self.assertEqual(counters["coverage_gate_evaluation_count"], 81 * 9)

    def test_memory_contract_canonical_csv_avoids_unicode_whole_file_copy(self):
        """Canonical bytes remain exact without a full-file Unicode duplicate."""

        canonical = self.function("_canonical_csv_bytes")
        small = pd.DataFrame(
            {
                "STN": ["100", "101"],
                "VALUE": [1.25, np.nan],
                "FLAG": [True, False],
                "LABEL": ["가", "comma,value"],
            }
        )
        expected = (
            "\ufeffSTN,VALUE,FLAG,LABEL\n"
            "100,1.25,True,가\n"
            '101,,False,"comma,value"\n'
        ).encode("utf-8")
        self.assertEqual(
            canonical(small, ["STN", "VALUE", "FLAG", "LABEL"]), expected
        )

        n_rows = 50_000
        large = pd.DataFrame(
            {
                "STN": ["100"] * n_rows,
                "VALUE": np.arange(n_rows, dtype=float) / 10.0,
                "FLAG": np.arange(n_rows) % 2 == 0,
            }
        )
        tracemalloc.start()
        try:
            payload = canonical(large, ["STN", "VALUE", "FLAG"])
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(
            peak_bytes,
            len(payload) * 3,
            "canonical serialization allocated a whole-file Unicode duplicate",
        )

    def test_m2_descriptive_state_api_has_no_unused_bootstrap_arguments(self):
        """descriptive-only state API가 bootstrap을 수행하는 것처럼 보이는 회귀를 잡는다."""
        signature = inspect.signature(self.function("build_state_condition_bias"))
        self.assertNotIn("n_boot", signature.parameters)
        self.assertNotIn("seed", signature.parameters)

    def test_i4_two_targets_share_physical_aws_reads_through_bounded_cache(self):
        """두 target의 공통 이웃 원자료가 target별로 물리 재독되는 회귀를 잡는다."""
        production = self.function("run_production_analysis")
        reconstruct = self.function("reconstruct_signature_geometry")
        fixture = _station_fixture()
        times = fixture["pairs"]["TM"]
        ta_sources = fixture["ta_data"].copy()
        ta_sources["100"] = [11.25, 13.75]
        neighbors_b = ["100", "102", "103", "104", "105", "106", "107"]
        pairs_b = fixture["pairs"].copy()
        pairs_b["STN"] = "101"
        pairs_b["NEIGHBOR_SET_ID"] = "sig-B"
        predictions_b = []
        for row in pairs_b.itertuples(index=False):
            geometry = reconstruct(
                target_id="101",
                neighbor_ids=neighbors_b,
                timestamp=row.TM,
                coord_epoch=row.COORD_EPOCH,
                expected_neighbor_count=7,
                meta=fixture["meta"],
                terrain=fixture["terrain"],
            )
            values = ta_sources.loc[row.TM, list(geometry["neighbor_ids"])].to_numpy(float)
            predictions_b.append(float(values @ geometry["idw_weights"]))
        pairs_b["OBSERVED"] = np.asarray(predictions_b) - pairs_b["error_none"].to_numpy(float)
        neighbor_map = {
            **fixture["neighbor_map"],
            "sig-B": neighbors_b,
        }
        truth = pd.DataFrame(
            {
                "STN": ["100", "101"],
                "BIAS_none": [0.60, 0.70],
                "BIAS_fixed": [0.62, 0.75],
                "SIGNIFICANCE_CLASS": [
                    "practically_equivalent",
                    "meaningful_worsening",
                ],
                "MECHANISM": ["small_change", "same_direction_worsened"],
                "mean_absdz_geom_km": [0.01, 0.02],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            truth_path = root / "truth.csv"
            terrain_path = root / "terrain.csv"
            meta_path = root / "meta.csv"
            db_path = root / "readonly.db"
            truth.to_csv(truth_path, index=False, encoding="utf-8-sig")
            fixture["terrain"].to_csv(terrain_path, index=False, encoding="utf-8-sig")
            meta_path.write_text("patched loader", encoding="utf-8")
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE raw (STN TEXT, TM TEXT, interpolable INTEGER, METHOD TEXT)"
            )
            for station_id in ["100", "101"]:
                for timestamp in times:
                    for method in ["none", "fixed_lapse"]:
                        connection.execute(
                            "INSERT INTO raw VALUES (?, ?, 1, ?)",
                            (station_id, str(timestamp), method),
                        )
            connection.commit()
            aws_root = root / "aws"
            for station_id in [str(value) for value in range(100, 108)]:
                folder = aws_root / f"Station_{station_id}"
                folder.mkdir(parents=True)
                for row_index, (timestamp, month) in enumerate(
                    zip(times, ["202406", "202412"])
                ):
                    weather = fixture["weather"].iloc[row_index]
                    pd.DataFrame(
                        {
                            "TM": [timestamp],
                            "TA": [ta_sources.loc[timestamp, station_id]],
                            "WS10": [weather["WS10"]],
                            "WS1": [weather["WS1"]],
                            "RN_60m": [weather["RN_60m"]],
                            "HM": [weather["HM"]],
                            "TD": [weather["TD"]],
                        }
                    ).to_csv(folder / f"AWS_{month}.csv", index=False)
            out_dir = root / "fullnet" / "bias_physical_followup"
            out_dir.parent.mkdir(parents=True)
            receipt_path = root / "bias_physical_followup.acceptance.json"

            def load_pairs(_connection, station_id):
                return fixture["pairs"].copy() if str(station_id) == "100" else pairs_b.copy()

            with (
                patch('analysis.bias_data.open_readonly_db', return_value=connection),
                patch('analysis.bias_data.load_neighbor_map', return_value=neighbor_map),
                patch('analysis.bias_data.load_station_pairs', side_effect=load_pairs),
                patch('weather.find_stations.load_station_meta', return_value=fixture["meta"]),
                patch.object(
                    runner,
                    "validate_canonical_station_results",
                    return_value=runner._expected_canonical_invariants(),
                ),
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out_dir),
            ):
                production(
                    out_dir=out_dir,
                    db_path=db_path,
                    station_results_path=truth_path,
                    metadata_path=meta_path,
                    terrain_path=terrain_path,
                    aws_base_folder=aws_root,
                    acceptance_receipt_path=receipt_path,
                )

            metadata = json.loads(
                (out_dir / "analysis_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["db_full_hash_verification_count"], 4)
            self.assertNotIn("db_hash_verification_count", metadata)
            self.assertNotIn("db_hash_verification_elapsed_seconds", metadata)
            self.assertGreaterEqual(
                metadata["input_sources"]["db_snapshot_creation_elapsed_seconds"],
                0.0,
            )
            self.assertEqual(metadata["aws_file_read_count"], 16)
            self.assertEqual(metadata["aws_cache_miss_count"], 8)
            self.assertGreaterEqual(metadata["aws_cache_hit_count"], 1)
            self.assertLessEqual(
                metadata["aws_cache_peak_entries"], metadata["aws_cache_max_entries"]
            )
            self.assertEqual(metadata["geometry_build_count"], 2)
            self.assertEqual(metadata["processed_neighbor_values"], 28)


class IntegrationReviewFixRound2Tests(RunnerFunctionTestCase):
    @staticmethod
    def _canonical_invariants():
        return {
            "TOTAL_STATIONS": 529,
            "CLASS_COUNTS": {
                "meaningful_improvement": 221,
                "meaningful_worsening": 109,
                "practically_equivalent": 115,
                "uncertain": 84,
            },
            "HIGH_BEFORE_N": 33,
            "HIGH_AFTER_N": 33,
            "HIGH_COMMON_N": 30,
            "HIGH_AFTER_POSITIVE_N": 19,
            "HIGH_AFTER_NEGATIVE_N": 14,
            "EQUIVALENT_AFTER_LEVEL_COUNTS": {
                "low": 20,
                "medium": 62,
                "high": 33,
            },
            "EQUIVALENT_TRANSITION_COUNTS": {
                "low->low": 12,
                "low->medium": 5,
                "low->high": 0,
                "medium->low": 8,
                "medium->medium": 54,
                "medium->high": 3,
                "high->low": 0,
                "high->medium": 3,
                "high->high": 30,
            },
            "WORSENING_MECHANISM_COUNTS": {
                "same_direction_worsened": 91,
                "overshoot_worsened": 18,
            },
            "WORSENING_DZ_BIN_COUNTS": {
                "≤50m": 72,
                "50–100m": 27,
                "100–200m": 7,
                "200–300m": 2,
                ">300m": 1,
            },
            "PRIMARY_HYPOTHESIS_POPULATION": "all_529_stations",
        }

    @staticmethod
    def _production_metrics():
        return {
            "db_full_hash_verification_count": 4,
            "db_full_hash_verification_elapsed_seconds": 0.0,
            "aws_cache_hit_count": 0,
            "aws_cache_miss_count": 0,
            "aws_cache_peak_entries": 0,
            "aws_cache_eviction_count": 0,
            "aws_cache_max_entries": 1,
            "process_peak_rss_bytes": 1,
            "rss_limit_bytes": 2,
            "stations_timed": 0,
            "first_10_station_elapsed_seconds": 0.0,
            "estimated_total_runtime_seconds_from_first_10": 0.0,
            "observed_station_elapsed_seconds": 0.0,
        }

    def test_r1_hourly_absolute_coast_exposure_is_not_abs_signed_or_cell_median(self):
        """시간 가중을 버린 cell 중앙값과 abs(median)을 승인 H2 노출량으로 쓰지 않는다."""
        summarize = self.function("_summarize_hourly_physical_exposure")
        unbalanced = pd.DataFrame(
            {
                "REPRODUCED": True,
                "DELTA_COAST_KM": [1.0] * 1000 + [100.0],
                "SIGNED_DELTA_Z_M": [-50.0] * 1001,
            }
        )
        summary = summarize(unbalanced)
        self.assertEqual(summary["MEDIAN_ABS_DELTA_COAST_KM"], 1.0)
        self.assertEqual(summary["MEAN_SIGNED_DELTA_Z_M"], -50.0)

        sign_change = pd.DataFrame(
            {
                "REPRODUCED": True,
                "DELTA_COAST_KM": [-100.0, -1.0, 2.0],
                "SIGNED_DELTA_Z_M": [-10.0, -20.0, -30.0],
            }
        )
        changed = summarize(sign_change)
        self.assertEqual(changed["MEDIAN_ABS_DELTA_COAST_KM"], 2.0)
        self.assertNotEqual(
            changed["MEDIAN_ABS_DELTA_COAST_KM"],
            abs(float(sign_change["DELTA_COAST_KM"].median())),
        )

        analyzed = self.function("analyze_station_injected")(
            **_station_fixture(), strict_reproduction=True, return_hourly=True
        )
        hourly = analyzed["hourly"].loc[analyzed["hourly"]["REPRODUCED"]]
        station = analyzed["station_summary"].iloc[0]
        self.assertAlmostEqual(
            station["MEDIAN_ABS_DELTA_COAST_KM"],
            float(hourly["DELTA_COAST_KM"].abs().median()),
        )
        self.assertAlmostEqual(
            station["MEAN_SIGNED_DELTA_Z_M"],
            float(hourly["SIGNED_DELTA_Z_M"].mean()),
        )

    def test_r1_h2_uses_hourly_station_exposure_and_preserves_effect_direction(self):
        """1000시간×1 km + 1시간×100 km 반례에서 H2 RBC는 -1이어야 한다."""
        build = self.function("build_physical_hypothesis_comparison")
        equivalent = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "BIAS_none": [0.7, -0.7, 0.2, -0.2],
                "BIAS_fixed": [0.8, -0.8, 0.2, -0.2],
                "ABS_BIAS_NONE": [0.7, 0.7, 0.2, 0.2],
                "ABS_BIAS_FIXED": [0.8, 0.8, 0.2, 0.2],
            }
        )
        structure = pd.DataFrame(
            {
                "STN": ["100", "101", "102", "103"],
                "MEDIAN_ABS_DELTA_COAST_KM": [1.0, 1.0, 2.0, 2.0],
                "DARK_INV_FRACTION": [0.8, 0.7, 0.2, 0.1],
                "DARK_PRIMARY_ELIGIBLE_DAYS": [30] * 4,
                "DARK_PRIMARY_ELIGIBLE_MONTHS": [6] * 4,
                "TARGET_LATITUDE": [37.0, 37.1, 36.9, 37.2],
                "TARGET_LONGITUDE": [127.0, 127.1, 126.9, 127.2],
            }
        )
        rows = []
        for station_id in ["100", "101"]:
            rows.extend(
                [
                    {"STN": station_id, "MONTH": 6, "SOLAR_STATE": "bright", "N_HOURS": 1000, "N_DAYS": 30, "BIAS_fixed": 0.8, "DELTA_COAST_KM": 1.0},
                    {"STN": station_id, "MONTH": 12, "SOLAR_STATE": "dark", "N_HOURS": 1, "N_DAYS": 1, "BIAS_fixed": 0.8, "DELTA_COAST_KM": 100.0},
                ]
            )
        for station_id in ["102", "103"]:
            rows.extend(
                [
                    {"STN": station_id, "MONTH": 6, "SOLAR_STATE": "bright", "N_HOURS": 1000, "N_DAYS": 30, "BIAS_fixed": 0.2, "DELTA_COAST_KM": 2.0},
                    {"STN": station_id, "MONTH": 12, "SOLAR_STATE": "dark", "N_HOURS": 1, "N_DAYS": 1, "BIAS_fixed": 0.2, "DELTA_COAST_KM": 2.0},
                ]
            )
        physical = build(equivalent, structure, pd.DataFrame(rows), n_replicates=20)
        h2 = physical.loc[physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL")].iloc[0]
        self.assertEqual(h2["TARGET_MEDIAN"], 1.0)
        self.assertEqual(h2["REFERENCE_MEDIAN"], 2.0)
        self.assertEqual(h2["EFFECT_ESTIMATE"], -1.0)

    def test_r2_production_metadata_requires_exact_canonical_hashes_and_live_sources(self):
        """production에서 fixture canonical, 가짜 hash, 변경된 비-AWS 원본을 허용하지 않는다."""
        capture = self.function("_capture_production_source_metadata")
        build = self.function("_build_analysis_metadata")
        validate = self.function("_validate_analysis_metadata")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for role in ["db", "station_results", "metadata", "terrain"]:
                path = root / f"{role}.dat"
                if role == "db":
                    _create_delete_mode_sqlite(path)
                else:
                    path.write_bytes(f"{role}-identity".encode("utf-8"))
                paths[role] = path
            aws_root = root / "aws"
            aws_root.mkdir()
            aws_station = aws_root / "Station_100"
            aws_station.mkdir()
            aws_source = aws_station / "AWS_202401.csv"
            aws_source.write_text("TM,TA\n2024-01-01 00:00:00,1.0\n", encoding="utf-8")
            aws_inventory = [
                {
                    "relative_path": "Station_100/AWS_202401.csv",
                    "bytes": aws_source.stat().st_size,
                    "sha256": hashlib.sha256(aws_source.read_bytes()).hexdigest(),
                    "station_id": "100",
                    "month": "202401",
                }
            ]
            sources = capture(
                db_path=paths["db"],
                station_results_path=paths["station_results"],
                metadata_path=paths["metadata"],
                terrain_path=paths["terrain"],
                aws_base_folder=aws_root,
                aws_file_inventory=aws_inventory,
            )
            sources = _attach_fixture_snapshot_metadata(sources, paths["db"])
            production_metrics = self._production_metrics()
            production_metrics["aws_file_read_count"] = 1
            metadata = build(
                canonical=self._canonical_invariants(),
                metrics=production_metrics,
                input_sources=sources,
                analysis_station_ids=["100", "885"],
            )
            validate(metadata)
            for role in ["db", "station_results", "metadata", "terrain"]:
                self.assertEqual(sources[f"{role}_path"], str(paths[role].resolve()))
                self.assertRegex(sources[f"{role}_sha256"], r"^[0-9a-f]{64}$")

            null_canonical = json.loads(json.dumps(metadata))
            null_canonical["canonical_invariants"] = None
            with self.assertRaisesRegex(ValueError, "canonical"):
                validate(null_canonical)
            fixture_canonical = json.loads(json.dumps(metadata))
            fixture_canonical["canonical_invariants"] = {"fixture": True}
            with self.assertRaisesRegex(ValueError, "canonical"):
                validate(fixture_canonical)
            for bad_hash in ["short", "g" * 64]:
                invalid_hash = json.loads(json.dumps(metadata))
                invalid_hash["input_sources"]["db_sha256"] = bad_hash
                with self.assertRaisesRegex(ValueError, "SHA-256|hash"):
                    validate(invalid_hash)

            empty_inventory = json.loads(json.dumps(metadata))
            empty_inventory["input_sources"]["aws_file_inventory"] = []
            empty_inventory["input_sources"]["aws_inventory_aggregate_sha256"] = (
                hashlib.sha256(b"[]").hexdigest()
            )
            empty_inventory["primary_eligibility_source_binding_sha256"] = (
                runner._primary_ledger_source_binding_sha256(
                    ledger_sha256=empty_inventory[
                        "primary_eligibility_ledger_sha256"
                    ],
                    population_sha256=empty_inventory[
                        "analysis_station_ids_sha256"
                    ],
                    config_sha256=runner._canonical_json_hash(
                        runner._analysis_config_payload()
                    ),
                    sources=empty_inventory["input_sources"],
                )
            )
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate(empty_inventory)

            mislabeled_inventory = json.loads(json.dumps(metadata))
            mislabeled_inventory["input_sources"]["aws_file_inventory"][0][
                "station_id"
            ] = "101"
            mislabeled_inventory["input_sources"]["aws_inventory_aggregate_sha256"] = (
                runner._canonical_json_hash(
                    mislabeled_inventory["input_sources"]["aws_file_inventory"]
                )
            )
            mislabeled_inventory["primary_eligibility_source_binding_sha256"] = (
                runner._primary_ledger_source_binding_sha256(
                    ledger_sha256=mislabeled_inventory[
                        "primary_eligibility_ledger_sha256"
                    ],
                    population_sha256=mislabeled_inventory[
                        "analysis_station_ids_sha256"
                    ],
                    config_sha256=runner._canonical_json_hash(
                        runner._analysis_config_payload()
                    ),
                    sources=mislabeled_inventory["input_sources"],
                )
            )
            with self.assertRaisesRegex(ValueError, "inventory.*path|path.*inventory"):
                validate(mislabeled_inventory)

            paths["terrain"].write_bytes(b"changed-after-analysis")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate(metadata)

    def test_r2_production_population_identity_rejects_consistent_885_deletion(self):
        """정본 source의 STN digest와 게시 skeleton을 묶어 885 전표 삭제를 거부한다."""
        capture = self.function("_capture_production_source_metadata")
        build = self.function("_build_analysis_metadata")
        validate_population = self.function("_validate_analysis_population_identity")
        truth = pd.DataFrame(
            {
                "STN": ["100", "885"],
                "BIAS_none": [0.1, 0.272],
                "BIAS_fixed": [0.2, 2.481],
                "SIGNIFICANCE_CLASS": [
                    "practically_equivalent",
                    "meaningful_worsening",
                ],
                "MECHANISM": ["small_change", "same_direction_worsened"],
                "mean_absdz_geom_km": [0.01, 0.34],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            source_paths = {}
            for role in ["db", "metadata", "terrain"]:
                path = root / f"{role}.dat"
                if role == "db":
                    _create_delete_mode_sqlite(path)
                else:
                    path.write_text(role, encoding="utf-8")
                source_paths[role] = path
            station_results = root / "station_results.csv"
            truth.to_csv(station_results, index=False, encoding="utf-8-sig")
            aws_root = root / "aws"
            aws_root.mkdir()
            sources = capture(
                db_path=source_paths["db"],
                station_results_path=station_results,
                metadata_path=source_paths["metadata"],
                terrain_path=source_paths["terrain"],
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            sources = _attach_fixture_snapshot_metadata(sources, source_paths["db"])
            metrics = self._production_metrics()
            metrics["aws_file_read_count"] = 0
            metrics["processed_paired_hours"] = 0
            metadata = build(
                canonical=self._canonical_invariants(),
                metrics=metrics,
                input_sources=sources,
                analysis_station_ids=truth["STN"],
            )
            station = pd.DataFrame({"STN": ["100", "885"]})
            with self.assertRaisesRegex(ValueError, "canonical|529|count"):
                validate_population(metadata, station)
            with patch.object(
                runner,
                "validate_canonical_station_results",
                return_value=self._canonical_invariants(),
            ):
                validate_population(metadata, station)
                with self.assertRaisesRegex(ValueError, "population|station.*identity"):
                    validate_population(
                        metadata, station.loc[station["STN"].ne("885")]
                    )

    def test_r2_published_production_artifact_cannot_downgrade_metadata_to_injected(self):
        """canonical production 경로의 rehashed artifact가 injected schema로 provenance를 지우지 못한다."""
        run = self.function("run_injected_analysis")
        validate = self.function("validate_published_artifacts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent)
            manifest_before = (out / "manifest.json").read_bytes()
            untracked = out / "untracked-directory"
            untracked.mkdir()
            with self.assertRaisesRegex(ValueError, "extra|artifact set"):
                validate(out, allowed_fullnet_root=out.parent)
            untracked.rmdir()
            with patch.object(runner, "CANONICAL_OUTPUT_DIR", out):
                with self.assertRaisesRegex(ValueError, "source_mode|source mode"):
                    validate(out)
                with self.assertRaisesRegex(ValueError, "canonical|source_mode|source mode"):
                    validate(out, allowed_fullnet_root=out.parent)
                with self.assertRaisesRegex(ValueError, "canonical"):
                    run(
                        truth,
                        {"100": _station_fixture()},
                        out,
                        allowed_fullnet_root=out.parent,
                    )
                nested = out / "bias_physical_followup"
                with self.assertRaisesRegex(ValueError, "canonical"):
                    run(
                        truth,
                        {"100": _station_fixture()},
                        nested,
                        allowed_fullnet_root=out,
                    )
                self.assertFalse(nested.exists())
            self.assertEqual((out / "manifest.json").read_bytes(), manifest_before)

    @staticmethod
    def _station_885_gate_inputs():
        months = [1, 2, 3, 4, 5, 6]
        case_rows = []
        for state in ["inversion_like", "normal_negative"]:
            for month in months:
                case_rows.append(
                    {
                        "STN": "885",
                        "MONTH": month,
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": state,
                        "N_HOURS": 120,
                        "N_DAYS": 5,
                        "N_MONTHS": 1,
                        "BIAS_none": 0.1,
                        "BIAS_fixed": 0.6 if state == "inversion_like" else 0.2,
                        "DELTA_ABS_BIAS": 0.5 if state == "inversion_like" else 0.1,
                        "MAE_none": 0.2,
                        "MAE_fixed": 0.7,
                        "DELTA_MAE": 0.5,
                        "DELTA_COAST_KM": 0.0,
                        "H2_EXPECTED_BIAS_SIGN": "not_prespecified",
                        "CASE_SCOPE": "single_station_descriptive_case_not_general_proof",
                        "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                        "LIMITATION": "single_station_descriptive_case_without_full_family_q_gate",
                    }
                )
        station_885 = pd.DataFrame(case_rows, columns=runner.STATION_885_COLUMNS)
        station_structure = pd.DataFrame(
            {
                "STN": ["885"],
                "TOTAL_PAIRED_HOURS": [1440],
                "REPRODUCED_HOURS": [1440],
                "REPRODUCTION_RATE": [1.0],
                "MAX_NONE_ABS_DIFF_C": [0.0],
                "MEAN_SIGNED_DELTA_Z_M": [-80.0],
            }
        )
        state_condition = pd.DataFrame(
            {
                "STN": ["885"],
                "CONFIG": ["primary"],
                "STRUCTURE_STATE": ["inversion_like"],
                "CONDITION_VARIABLE": ["ALL"],
                "CONDITION_LEVEL": ["all"],
                "BIAS_none": [0.1],
                "BIAS_fixed": [0.6],
                "DELTA_ABS_BIAS": [0.5],
            }
        )
        coverage = pd.DataFrame(
            {
                "STN": ["885"],
                "SETTING_ID": ["n6_span100_wsd50_wd2"],
                "COVERAGE_ID": ["d30_m6"],
                "N_INVERSION_DAYS": [30],
                "N_NORMAL_DAYS": [30],
                "N_INVERSION_MONTHS": [6],
                "N_NORMAL_MONTHS": [6],
                "COVERAGE_ELIGIBLE": [True],
            }
        )
        matched = pd.DataFrame(
            {
                "STN": ["885"],
                "CONFIG": ["primary"],
                "SETTING_ID": ["n6_span100_wsd50_wd2"],
                "N_STRATA": [6],
                "DELTA_ABS_BIAS_INVERSION": [0.5],
                "DELTA_ABS_BIAS_NORMAL": [0.1],
                "COMPARISON_ELIGIBLE": [True],
                "CONTRAST_INV_MINUS_NORMAL": [0.4],
                "BLOCK7_CI_LOW": [0.1],
                "Q_VALUE": [0.01],
            }
        )
        return station_885, matched, station_structure, state_condition, coverage

    def test_r3_station_885_h1_gate_is_independent_complete_and_directional(self):
        """885는 H2 coast cell이 아니라 exact-none·양 상태 coverage·저지대 보정방향으로 판정한다."""
        apply_evidence = self.function("_apply_station_885_evidence")
        case, matched, structure, state, coverage = self._station_885_gate_inputs()
        apply_evidence(
            case,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(case["EVIDENCE_GRADE"].eq("문헌과 일치하나 직접 미측정").all())
        self.assertFalse(case["EVIDENCE_GRADE"].eq("자료로 지지").any())

        negative_diff_case, matched, structure, state, coverage = (
            self._station_885_gate_inputs()
        )
        structure.loc[0, "MAX_NONE_ABS_DIFF_C"] = -1.0
        apply_evidence(
            negative_diff_case,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertNotEqual(
            negative_diff_case["EVIDENCE_GRADE"].iloc[0],
            case["EVIDENCE_GRADE"].iloc[0],
        )

        one_cell, matched, structure, state, coverage = self._station_885_gate_inputs()
        one_cell = one_cell.iloc[[0]].copy()
        apply_evidence(
            one_cell,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(one_cell["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all())

        reversed_case, matched, structure, state, coverage = self._station_885_gate_inputs()
        structure.loc[0, "MEAN_SIGNED_DELTA_Z_M"] = 80.0
        apply_evidence(
            reversed_case,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(reversed_case["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all())

        crossing_case, matched, structure, state, coverage = self._station_885_gate_inputs()
        inversion = crossing_case["STRUCTURE_STATE"].eq("inversion_like")
        crossing_case.loc[
            inversion, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
        ] = [-0.1, 0.6, 0.5]
        apply_evidence(
            crossing_case,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            crossing_case["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all()
        )

        simpson_case, matched, structure, state, coverage = self._station_885_gate_inputs()
        inversion = simpson_case["STRUCTURE_STATE"].eq("inversion_like")
        simpson_case.loc[inversion, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]] = [
            0.1,
            -0.55,
            0.45,
        ]
        noncommon = simpson_case.loc[inversion].iloc[[0]].copy()
        noncommon.loc[:, "MONTH"] = 7
        noncommon.loc[:, "SOLAR_STATE"] = "bright"
        noncommon.loc[:, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]] = [
            0.1,
            10.0,
            9.9,
        ]
        simpson_case = pd.concat([simpson_case, noncommon], ignore_index=True)
        state.loc[0, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]] = [0.1, 2.05, 1.95]
        apply_evidence(
            simpson_case,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            simpson_case["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all()
        )

        common_valid, matched, structure, pooled_state, coverage = (
            self._station_885_gate_inputs()
        )
        pooled_state.loc[
            0, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
        ] = [-0.1, -0.6, 0.5]
        apply_evidence(
            common_valid,
            matched,
            station_structure=structure,
            state_condition=pooled_state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            common_valid["EVIDENCE_GRADE"].eq("문헌과 일치하나 직접 미측정").all()
        )

        strata_mismatch, matched, structure, state, coverage = self._station_885_gate_inputs()
        matched.loc[0, "N_STRATA"] = 5
        apply_evidence(
            strata_mismatch,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            strata_mismatch["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all()
        )

        normal_mutation, matched, structure, state, coverage = self._station_885_gate_inputs()
        normal = normal_mutation["STRUCTURE_STATE"].eq("normal_negative")
        normal_mutation.loc[normal, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]] = [
            0.1,
            10.0,
            9.9,
        ]
        apply_evidence(
            normal_mutation,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            normal_mutation["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all()
        )

        dark_reversed, matched, structure, state, coverage = self._station_885_gate_inputs()
        dark_inversion = dark_reversed["STRUCTURE_STATE"].eq("inversion_like")
        dark_reversed.loc[
            dark_inversion, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
        ] = [0.1, -2.0, 1.9]
        bright = dark_reversed.copy()
        bright.loc[:, "SOLAR_STATE"] = "bright"
        bright_inversion = bright["STRUCTURE_STATE"].eq("inversion_like")
        bright.loc[
            bright_inversion, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
        ] = [10.0, 20.0, 10.0]
        dark_reversed = pd.concat([dark_reversed, bright], ignore_index=True)
        matched.loc[
            0,
            [
                "N_STRATA",
                "DELTA_ABS_BIAS_INVERSION",
                "DELTA_ABS_BIAS_NORMAL",
                "CONTRAST_INV_MINUS_NORMAL",
            ],
        ] = [12, 3.95, 0.1, 3.85]
        apply_evidence(
            dark_reversed,
            matched,
            station_structure=structure,
            state_condition=state,
            coverage_sensitivity=coverage,
            analysis_station_ids={"885"},
        )
        self.assertTrue(
            dark_reversed["EVIDENCE_GRADE"].eq("현재 자료에서 지지되지 않음").all()
        )

    def test_r3_validator_rejects_direct_descriptive_support_and_out_of_population_885(self):
        """rehash 여부와 무관하게 기술행 직접지지와 모집단 밖 885 삽입을 semantic 검증한다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent)
            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }
            state_mutation = {name: frame.copy() for name, frame in tables.items()}
            state_mutation["state_condition_bias.csv"].loc[0, "EVIDENCE_GRADE"] = "자료로 지지"
            with self.assertRaisesRegex(ValueError, "descriptive state evidence"):
                validate_tables(state_mutation)

            fabricated_coverage = {
                name: frame.copy() for name, frame in tables.items()
            }
            state_frame = fabricated_coverage["state_condition_bias.csv"]
            insufficient = state_frame["N_DAYS"].lt(30).idxmax()
            state_frame.loc[insufficient, "COVERAGE_ELIGIBLE"] = True
            state_frame.loc[
                insufficient, "EVIDENCE_GRADE"
            ] = "문헌과 일치하나 직접 미측정"
            with self.assertRaisesRegex(ValueError, "descriptive state coverage"):
                validate_tables(fabricated_coverage)

            inserted = {name: frame.copy() for name, frame in tables.items()}
            case, _, _, _, _ = self._station_885_gate_inputs()
            inserted["station_885_diagnostic.csv"] = case.iloc[[0]].copy()
            with self.assertRaisesRegex(ValueError, "885.*population|population.*885"):
                validate_tables(inserted)

    def test_r3_final_885_semantic_validator_rejects_one_cell_and_direction_mutations(self):
        """최종 apply 뒤 case coverage·signed ΔZ·모집단을 함께 바꿔도 재계산이 잡아야 한다."""
        apply_evidence = self.function("_apply_station_885_evidence")
        validate_evidence = self.function("_validate_station_885_evidence_contract")
        case, matched, structure, state, coverage = self._station_885_gate_inputs()
        kwargs = {
            "station_structure": structure,
            "state_condition": state,
            "coverage_sensitivity": coverage,
            "analysis_station_ids": ["885"],
        }
        apply_evidence(case, matched, **kwargs)
        validate_evidence(case, matched, **kwargs)

        one_cell = case.iloc[[0]].copy()
        with self.assertRaisesRegex(ValueError, "independent H1 gate"):
            validate_evidence(one_cell, matched, **kwargs)

        reversed_structure = structure.copy()
        reversed_structure.loc[0, "MEAN_SIGNED_DELTA_Z_M"] = 80.0
        with self.assertRaisesRegex(ValueError, "independent H1 gate"):
            validate_evidence(
                case,
                matched,
                station_structure=reversed_structure,
                state_condition=state,
                coverage_sensitivity=coverage,
                analysis_station_ids=["885"],
            )

        reversed_case = case.copy()
        reversed_inversion = reversed_case["STRUCTURE_STATE"].eq("inversion_like")
        reversed_case.loc[
            reversed_inversion, ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
        ] = [0.1, -0.6, 0.5]
        with self.assertRaisesRegex(ValueError, "independent H1 gate"):
            validate_evidence(
                reversed_case,
                matched,
                station_structure=structure,
                state_condition=state,
                coverage_sensitivity=coverage,
                analysis_station_ids=["885"],
            )

        with self.assertRaisesRegex(ValueError, "population"):
            validate_evidence(
                case,
                matched,
                station_structure=structure,
                state_condition=state,
                coverage_sensitivity=coverage,
                analysis_station_ids=["100"],
            )

        with self.assertRaisesRegex(ValueError, "case.*missing|missing.*case"):
            validate_evidence(
                case.iloc[0:0].copy(),
                matched,
                station_structure=structure,
                state_condition=state,
                coverage_sensitivity=coverage,
                analysis_station_ids=["885"],
            )

        invalid_month = case.copy()
        invalid_month.loc[invalid_month.index[0], "MONTH"] = 13
        with self.assertRaisesRegex(ValueError, "MONTH|month|domain"):
            validate_evidence(invalid_month, matched, **kwargs)

        invalid_scope = case.copy()
        invalid_scope.loc[invalid_scope.index[0], "CASE_SCOPE"] = "general_proof"
        with self.assertRaisesRegex(ValueError, "CASE_SCOPE|scope|domain"):
            validate_evidence(invalid_scope, matched, **kwargs)

    def test_round2_semantic_domains_reject_negative_exact_diff_bad_ci_and_fake_bool(self):
        """재해시 가능한 수치·boolean domain의 불가능 값을 semantic 단계에서 거부한다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        build_matched = self.function("build_matched_state_contrast")
        no_reproduced = pd.DataFrame(
            {
                "STN": ["100"],
                "TM": [pd.Timestamp("2024-01-01")],
                "SOLAR_STATE": ["dark"],
                "REPRODUCED": [False],
                "STATE_PRIMARY": ["ineligible"],
                "error_none": [0.0],
                "error_fixed": [0.0],
            }
        )
        empty_result = build_matched(no_reproduced, n_boot=2000, seed=20260827)
        self.assertEqual(empty_result.loc[0, "DROP_REASON"], "no_reproduced_hours")
        self.assertEqual(empty_result.loc[0, "BLOCK7_VALID_FRACTION"], 0.0)
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent)

            def load_tables():
                return {
                    name: pd.read_csv(out / name, encoding="utf-8-sig")
                    for name in EXPECTED_CSV_SCHEMAS
                }

            negative_exact = load_tables()
            negative_exact["station_structure_coverage.csv"].loc[
                0, "MAX_NONE_ABS_DIFF_C"
            ] = -1.0
            with self.assertRaisesRegex(ValueError, "MAX_NONE|exact-none"):
                validate_tables(negative_exact)

            negative_exposure = load_tables()
            negative_exposure["station_structure_coverage.csv"].loc[
                0, "MEDIAN_ABS_DELTA_COAST_KM"
            ] = -1.0
            with self.assertRaisesRegex(ValueError, "coast|exposure|non-negative"):
                validate_tables(negative_exposure)

            bad_ci = load_tables()
            bad_ci["matched_state_contrast.csv"].loc[
                0, ["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]
            ] = [0.1, -0.1]
            with self.assertRaisesRegex(ValueError, "CI|interval"):
                validate_tables(bad_ci)

            one_replicate = load_tables()
            one_replicate["matched_state_contrast.csv"].loc[
                0,
                [
                    "BLOCK7_REQUESTED_REPLICATES",
                    "BLOCK7_VALID_REPLICATES",
                    "BLOCK7_VALID_FRACTION",
                ],
            ] = [1, 1, 1.0]
            with self.assertRaisesRegex(ValueError, "2000|replicate|bootstrap"):
                validate_tables(one_replicate)

            fake_bool = load_tables()
            fake_bool["high_residual_group_diagnostic.csv"]["HIGH_AFTER"] = (
                fake_bool["high_residual_group_diagnostic.csv"]["HIGH_AFTER"].astype(
                    object
                )
            )
            fake_bool["high_residual_group_diagnostic.csv"].loc[
                0, "HIGH_AFTER"
            ] = "not_boolean"
            with self.assertRaisesRegex(ValueError, "boolean"):
                validate_tables(fake_bool)

    def test_round2_matched_missing_reasons_must_agree_with_station_payload(self):
        """자료가 있는 station을 source-missing/no-reproduced로 위장해 BH family에서 뺄 수 없다."""
        run = self.function("run_injected_analysis")
        validate_tables = self.function("_validate_cross_table_contracts")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            run(truth, {"100": _station_fixture()}, out, allowed_fullnet_root=out.parent)
            tables = {
                name: pd.read_csv(out / name, encoding="utf-8-sig")
                for name in EXPECTED_CSV_SCHEMAS
            }
            station = tables["station_structure_coverage.csv"].iloc[0]
            self.assertGreater(int(station["TOTAL_PAIRED_HOURS"]), 0)
            self.assertGreater(int(station["REPRODUCED_HOURS"]), 0)

            source_missing = {name: frame.copy() for name, frame in tables.items()}
            source_row = source_missing["matched_state_contrast.csv"].index[0]
            source_missing["matched_state_contrast.csv"].loc[
                source_row,
                [
                    "COMPARISON_ELIGIBLE",
                    "DROP_REASON",
                    "ANALYSIS_SUPPORT_CHECKED",
                    "BLOCK7_REQUESTED_REPLICATES",
                    "BLOCK7_VALID_REPLICATES",
                    "BLOCK7_VALID_FRACTION",
                    "BLOCK7_SEED",
                    "BLOCK7_CI_LOW",
                    "BLOCK7_CI_HIGH",
                    "P_VALUE",
                    "Q_VALUE",
                ],
            ] = [
                False,
                "source_payload_missing",
                False,
                0,
                0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
            with self.assertRaisesRegex(ValueError, "source_payload_missing|payload"):
                validate_tables(source_missing)

            no_reproduced = {name: frame.copy() for name, frame in tables.items()}
            no_reproduced_row = no_reproduced["matched_state_contrast.csv"].index[0]
            no_reproduced["matched_state_contrast.csv"].loc[
                no_reproduced_row,
                [
                    "COMPARISON_ELIGIBLE",
                    "DROP_REASON",
                    "ANALYSIS_SUPPORT_CHECKED",
                    "BLOCK7_VALID_REPLICATES",
                    "BLOCK7_VALID_FRACTION",
                    "BLOCK7_CI_LOW",
                    "BLOCK7_CI_HIGH",
                    "P_VALUE",
                    "Q_VALUE",
                ],
            ] = [
                False,
                "no_reproduced_hours",
                False,
                0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
            with self.assertRaisesRegex(ValueError, "no_reproduced_hours|reproduced"):
                validate_tables(no_reproduced)

            early_reason = {name: frame.copy() for name, frame in tables.items()}
            early_row = early_reason["matched_state_contrast.csv"].index[0]
            early_reason["matched_state_contrast.csv"].loc[
                early_row,
                [
                    "COMPARISON_ELIGIBLE",
                    "DROP_REASON",
                    "ANALYSIS_SUPPORT_CHECKED",
                    "BLOCK7_VALID_REPLICATES",
                    "BLOCK7_VALID_FRACTION",
                    "BLOCK7_CI_LOW",
                    "BLOCK7_CI_HIGH",
                    "P_VALUE",
                    "Q_VALUE",
                ],
            ] = [
                False,
                "missing_required_state",
                False,
                0,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
            with self.assertRaisesRegex(ValueError, "primary coverage|DROP_REASON"):
                validate_tables(early_reason)

    def test_r3_canonical_population_cannot_publish_zero_data_and_empty_885_case(self):
        """canonical 529에서는 885를 zero/ineligible로 바꿔 case 전체를 지울 수 없다."""
        validate_presence = self.function("_validate_canonical_station_885_presence")
        station_ids = ["885", *[str(value) for value in range(1000, 1528)]]
        classes = (
            ["meaningful_improvement"] * 221
            + ["meaningful_worsening"] * 109
            + ["practically_equivalent"] * 115
            + ["uncertain"] * 84
        )
        truth = pd.DataFrame({"STN": station_ids, "SIGNIFICANCE_CLASS": classes})
        structure = pd.DataFrame(
            {"STN": ["885"], "TOTAL_PAIRED_HOURS": [0], "REPRODUCED_HOURS": [0]}
        )
        empty_case = pd.DataFrame(columns=runner.STATION_885_COLUMNS)
        with self.assertRaisesRegex(ValueError, "885.*case|case.*885|zero"):
            validate_presence(truth, structure, empty_case)

        structure.loc[0, ["TOTAL_PAIRED_HOURS", "REPRODUCED_HOURS"]] = [1, 1]
        case = pd.DataFrame(
            [{**{column: np.nan for column in runner.STATION_885_COLUMNS}, "STN": "885"}],
            columns=runner.STATION_885_COLUMNS,
        )
        validate_presence(truth, structure, case)


class IntegrationReviewFixRound3Tests(RunnerFunctionTestCase):
    @staticmethod
    def _make_production_sources(root: Path):
        db_path = root / "source.db"
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE probe (value INTEGER)")
        connection.execute("INSERT INTO probe VALUES (1)")
        connection.commit()
        connection.close()
        station_results = root / "station_results.csv"
        pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.1],
                "BIAS_fixed": [0.2],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        ).to_csv(station_results, index=False, encoding="utf-8-sig")
        metadata = root / "metadata.csv"
        metadata.write_text("metadata", encoding="utf-8")
        terrain = root / "terrain.csv"
        terrain.write_text("terrain", encoding="utf-8")
        aws_root = root / "aws"
        aws_root.mkdir()
        return db_path, station_results, metadata, terrain, aws_root

    def test_m1_sha256_domain_is_exact_lowercase_hex(self):
        """부호·underscore·대문자가 digest 문자 domain에 들어오지 못한다."""
        is_hash = self.function("_is_sha256_hex")
        self.assertTrue(is_hash("a" * 64))
        for invalid in [
            "+" + "a" * 63,
            "a_" + "a" * 62,
            "A" * 64,
            "g" * 64,
            "a" * 63,
        ]:
            self.assertFalse(is_hash(invalid), msg=invalid)

    def test_c1_production_ledger_requires_an_external_trusted_binding(self):
        """bundle 내부를 함께 재해시해도 외부에 고정한 binding을 바꾸지 못한다."""
        validate_published = self.function("validate_published_artifacts")
        self.assertIn(
            "trusted_primary_eligibility_binding_sha256",
            inspect.signature(validate_published).parameters,
        )
        validate_binding = self.function(
            "_validate_trusted_primary_eligibility_binding"
        )
        capture = self.function("_capture_production_source_metadata")
        build = self.function("_build_analysis_metadata")
        expected = IntegrationReviewFixRound2Tests._canonical_invariants()
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            sources = _attach_fixture_snapshot_metadata(sources, db_path)
            metadata = build(
                canonical=expected,
                metrics=IntegrationReviewFixRound2Tests._production_metrics(),
                input_sources=sources,
                analysis_station_ids=["100"],
            )
            trusted = metadata["primary_eligibility_source_binding_sha256"]
            validate_binding(metadata, trusted)
            with self.assertRaisesRegex(ValueError, "trusted|external|anchor"):
                validate_binding(metadata, None)

            forged = json.loads(json.dumps(metadata))
            forged["primary_eligibility_ledger_sha256"] = "b" * 64
            forged["primary_eligibility_source_binding_sha256"] = (
                runner._primary_ledger_source_binding_sha256(
                    ledger_sha256=forged["primary_eligibility_ledger_sha256"],
                    population_sha256=forged["analysis_station_ids_sha256"],
                    config_sha256=runner._canonical_json_hash(
                        runner._analysis_config_payload()
                    ),
                    sources=forged["input_sources"],
                )
            )
            with self.assertRaisesRegex(ValueError, "trusted|external|anchor"):
                validate_binding(forged, trusted)

    def test_i2_canonical_nested_counts_reject_bool_and_float_aliases(self):
        """JSON true/false와 정수형 float가 공식 count를 대신할 수 없다."""
        capture = self.function("_capture_production_source_metadata")
        build = self.function("_build_analysis_metadata")
        validate = self.function("_validate_analysis_metadata")
        validate_population = self.function("_validate_analysis_population_identity")
        expected = IntegrationReviewFixRound2Tests._canonical_invariants()
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            sources = _attach_fixture_snapshot_metadata(sources, db_path)
            metadata = build(
                canonical=expected,
                metrics=IntegrationReviewFixRound2Tests._production_metrics(),
                input_sources=sources,
                analysis_station_ids=["100"],
            )
            validate(metadata)
            mutations = []
            as_true = json.loads(json.dumps(metadata))
            as_true["canonical_invariants"]["WORSENING_DZ_BIN_COUNTS"][">300m"] = True
            mutations.append(as_true)
            as_false = json.loads(json.dumps(metadata))
            as_false["canonical_invariants"]["EQUIVALENT_TRANSITION_COUNTS"][
                "low->high"
            ] = False
            mutations.append(as_false)
            as_float = json.loads(json.dumps(metadata))
            as_float["canonical_invariants"]["TOTAL_STATIONS"] = 529.0
            mutations.append(as_float)
            for mutation in mutations:
                if "canonical_invariants_sha256" in mutation:
                    mutation["canonical_invariants_sha256"] = runner._canonical_json_hash(
                        mutation["canonical_invariants"]
                    )
                with self.assertRaisesRegex(ValueError, "canonical|type|integer"):
                    validate(mutation)

            live_alias = json.loads(json.dumps(expected))
            live_alias["WORSENING_DZ_BIN_COUNTS"][">300m"] = True
            station = pd.DataFrame({"STN": ["100"]})
            with patch.object(
                runner, "validate_canonical_station_results", return_value=live_alias
            ):
                with self.assertRaisesRegex(ValueError, "canonical|type|integer"):
                    validate_population(metadata, station)

            def integer_paths(value, prefix=()):
                if isinstance(value, dict):
                    for key, nested in value.items():
                        yield from integer_paths(nested, (*prefix, key))
                elif type(value) is int:
                    yield prefix, value

            for path, original in integer_paths(expected):
                alias = json.loads(json.dumps(expected))
                cursor = alias
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = (
                    bool(original) if original in {0, 1} else float(original)
                )
                mutated_metadata = json.loads(json.dumps(metadata))
                mutated_metadata["canonical_invariants"] = alias
                mutated_metadata["canonical_invariants_sha256"] = (
                    runner._canonical_json_hash(alias)
                )
                with self.assertRaisesRegex(ValueError, "canonical|type|integer"):
                    validate(mutated_metadata)
                with patch.object(
                    runner, "validate_canonical_station_results", return_value=alias
                ):
                    with self.assertRaisesRegex(ValueError, "canonical|type|integer"):
                        validate_population(metadata, station)

    def test_i1_capture_records_delete_mode_and_rejects_wal_source(self):
        """production DB는 sidecar 없는 rollback-journal 단일 main file이어야 한다."""
        capture = self.function("_capture_production_source_metadata")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            self.assertIn("db_journal_mode", sources)
            self.assertEqual(sources["db_journal_mode"], "delete")
            self.assertFalse(sources["db_wal_present"])
            self.assertFalse(sources["db_shm_present"])
            self.assertFalse(sources["db_journal_present"])

            wal_path = root / "wal.db"
            connection = sqlite3.connect(wal_path)
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            self.assertEqual(str(mode).lower(), "wal")
            connection.execute("CREATE TABLE probe (value INTEGER)")
            connection.execute("INSERT INTO probe VALUES (1)")
            connection.commit()
            with self.assertRaisesRegex(ValueError, "WAL|journal|sidecar"):
                capture(
                    db_path=wal_path,
                    station_results_path=station_results,
                    metadata_path=metadata_path,
                    terrain_path=terrain,
                    aws_base_folder=aws_root,
                    aws_file_inventory=[],
                )
            connection.close()

    def test_i1_live_identity_rejects_new_sidecar_with_unchanged_main(self):
        """main hash가 같아도 새 WAL/SHM/journal sidecar가 생기면 snapshot은 실패한다."""
        capture = self.function("_capture_production_source_metadata")
        validate_live = self.function("_validate_live_production_source_identity")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            Path(str(db_path) + "-wal").write_bytes(b"committed-logical-change")
            with self.assertRaisesRegex(ValueError, "WAL|sidecar|journal"):
                validate_live(sources)

    def test_i1_live_identity_rejects_aws_inventory_change_before_backup_cleanup(self):
        """atomic callback이 AWS 개별 파일 변경을 backup 삭제 전에 차단한다."""
        capture = self.function("_capture_production_source_metadata")
        validate_live = self.function("_validate_live_production_source_identity")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            aws_station = aws_root / "Station_100"
            aws_station.mkdir()
            aws_file = aws_station / "AWS_202401.csv"
            aws_file.write_text(
                "TM,TA\n2024-01-01 00:00:00,1.0\n", encoding="utf-8"
            )
            inventory = [
                {
                    "relative_path": "Station_100/AWS_202401.csv",
                    "bytes": aws_file.stat().st_size,
                    "sha256": hashlib.sha256(aws_file.read_bytes()).hexdigest(),
                    "station_id": "100",
                    "month": "202401",
                }
            ]
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=inventory,
            )
            validate_live(sources)
            aws_file.write_text(
                "TM,TA\n2024-01-01 00:00:00,9.0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "AWS|inventory|changed"):
                validate_live(sources)

    def test_i1_snapshot_query_guard_requires_same_file_and_one_read_transaction(self):
        """snapshot query guard가 autocommit과 다른 DB 연결을 거부한다."""
        capture = self.function("_capture_production_source_metadata")
        create_snapshot = self.function("_create_private_sqlite_snapshot")
        open_snapshot = self.function("_open_immutable_sqlite_snapshot")
        guard = self.function("_validate_private_snapshot_after_query")
        cleanup = self.function("_cleanup_private_sqlite_snapshot")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            try:
                snapshot = create_snapshot(
                    db_path,
                    temp_parent=root,
                    source_identity=sources,
                    metrics={"db_full_hash_verification_count": 1},
                )
                connection = open_snapshot(snapshot)
                try:
                    with self.assertRaisesRegex(ValueError, "transaction"):
                        guard(snapshot, connection)
                    connection.execute("BEGIN")
                    guard(snapshot, connection)

                    source_connection = sqlite3.connect(db_path)
                    source_connection.execute("PRAGMA query_only=ON")
                    source_connection.execute("BEGIN")
                    try:
                        with self.assertRaisesRegex(ValueError, "different DB"):
                            guard(snapshot, source_connection)
                    finally:
                        source_connection.rollback()
                        source_connection.close()
                finally:
                    if connection.in_transaction:
                        connection.rollback()
                    connection.close()
            finally:
                if "snapshot" in locals():
                    cleanup(snapshot)

    def test_i1_per_query_full_hash_guard_is_removed(self):
        """폐기된 query별 전체 SHA helper와 O(station) metric은 남지 않는다."""

        self.assertFalse(hasattr(runner, "_validate_production_db_after_query"))
        self.assertNotIn(
            "db_hash_verification_count",
            runner._METADATA_PRODUCTION_INTEGER_METRICS,
        )
        self.assertNotIn(
            "db_hash_verification_elapsed_seconds",
            runner._METADATA_PRODUCTION_FLOAT_METRICS,
        )

    def test_i1_snapshot_query_guards_do_not_increment_full_hash_metric(self):
        """query 경계는 경량 검사이며 capture/create/end에서만 전체 SHA를 읽는다."""
        capture = self.function("_capture_production_source_metadata")
        create_snapshot = self.function("_create_private_sqlite_snapshot")
        open_snapshot = self.function("_open_immutable_sqlite_snapshot")
        guard = self.function("_validate_private_snapshot_after_query")
        full_guard = self.function("_validate_private_snapshot_full_hash")
        cleanup = self.function("_cleanup_private_sqlite_snapshot")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            metrics = {
                "db_full_hash_verification_count": 0,
                "db_full_hash_verification_elapsed_seconds": 0.0,
            }
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
                metrics=metrics,
            )
            try:
                snapshot = create_snapshot(
                    db_path,
                    temp_parent=root,
                    source_identity=sources,
                    metrics=metrics,
                )
                self.assertEqual(metrics["db_full_hash_verification_count"], 2)
                connection = open_snapshot(snapshot)
                try:
                    connection.execute("BEGIN")
                    guard(snapshot, connection)
                    guard(snapshot, connection)
                    self.assertEqual(metrics["db_full_hash_verification_count"], 2)
                finally:
                    connection.rollback()
                    connection.close()
                full_guard(snapshot, metrics=metrics)
                self.assertEqual(metrics["db_full_hash_verification_count"], 3)
            finally:
                if "snapshot" in locals():
                    cleanup(snapshot)

    def test_i1_private_snapshot_keeps_start_value_after_source_changes(self):
        """source 변경 뒤에도 모든 분석 query는 시작 시점 논리 snapshot 값만 읽는다."""
        capture = self.function("_capture_production_source_metadata")
        create_snapshot = self.function("_create_private_sqlite_snapshot")
        open_snapshot = self.function("_open_immutable_sqlite_snapshot")
        guard = self.function("_validate_private_snapshot_after_query")
        validate_original = self.function("_validate_original_db_full_hash")
        cleanup = self.function("_cleanup_private_sqlite_snapshot")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            metrics = {"db_full_hash_verification_count": 1}
            snapshot = create_snapshot(
                db_path,
                temp_parent=root,
                source_identity=sources,
                metrics=metrics,
            )
            try:
                writer = sqlite3.connect(db_path)
                writer.execute("UPDATE probe SET value=2")
                writer.commit()
                writer.close()

                connection = open_snapshot(snapshot)
                try:
                    connection.execute("BEGIN")
                    observed = connection.execute("SELECT value FROM probe").fetchone()[0]
                    self.assertEqual(observed, 1)
                    guard(snapshot, connection)
                finally:
                    if connection.in_transaction:
                        connection.rollback()
                    connection.close()
                self.assertEqual(metrics["db_full_hash_verification_count"], 2)
                with self.assertRaisesRegex(ValueError, "source DB SHA-256|changed"):
                    validate_original(sources, metrics=metrics)
                self.assertEqual(metrics["db_full_hash_verification_count"], 3)
            finally:
                cleanup(snapshot)
            self.assertFalse(Path(snapshot["db_snapshot_private_dir"]).exists())

    def test_i1_snapshot_full_hash_rejects_same_size_mtime_restored_mutation(self):
        """snapshot size·mtime를 복원한 변조도 publish 전 full hash에서 거부한다."""
        capture = self.function("_capture_production_source_metadata")
        create_snapshot = self.function("_create_private_sqlite_snapshot")
        light_guard = self.function("_validate_private_snapshot_light_identity")
        full_guard = self.function("_validate_private_snapshot_full_hash")
        cleanup = self.function("_cleanup_private_sqlite_snapshot")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            sources = capture(
                db_path=db_path,
                station_results_path=station_results,
                metadata_path=metadata_path,
                terrain_path=terrain,
                aws_base_folder=aws_root,
                aws_file_inventory=[],
            )
            metrics = {"db_full_hash_verification_count": 1}
            snapshot = create_snapshot(
                db_path,
                temp_parent=root,
                source_identity=sources,
                metrics=metrics,
            )
            try:
                path = Path(snapshot["db_snapshot_path"])
                stat = path.stat()
                payload = bytearray(path.read_bytes())
                payload[-1] ^= 1
                path.write_bytes(payload)
                os.utime(path, ns=(stat.st_atime_ns, snapshot["db_snapshot_mtime_ns"]))
                self.assertEqual(path.stat().st_size, snapshot["db_snapshot_bytes"])
                light_guard(snapshot)
                with self.assertRaisesRegex(ValueError, "snapshot.*SHA-256|hash"):
                    full_guard(snapshot, metrics=metrics)
                self.assertEqual(metrics["db_full_hash_verification_count"], 3)
            finally:
                cleanup(snapshot)

    def test_i1_primary_binding_includes_private_snapshot_identity(self):
        """ledger trusted binding은 원본 hash뿐 아니라 derivation snapshot hash에도 묶인다."""
        bind = self.function("_primary_ledger_source_binding_sha256")
        sources = {
            "source_mode": "production",
            "db_sha256": "a" * 64,
            "db_snapshot_sha256": "b" * 64,
            "db_snapshot_bytes": 10,
            "db_snapshot_creation_elapsed_seconds": 1.0,
            "db_snapshot_hash_elapsed_seconds": 2.0,
            "station_results_sha256": "c" * 64,
            "metadata_sha256": "d" * 64,
            "terrain_sha256": "e" * 64,
            "aws_inventory_aggregate_sha256": "f" * 64,
        }
        first = bind(
            ledger_sha256="1" * 64,
            population_sha256="2" * 64,
            config_sha256="3" * 64,
            sources=sources,
        )
        changed = dict(sources)
        changed["db_snapshot_sha256"] = "0" * 64
        second = bind(
            ledger_sha256="1" * 64,
            population_sha256="2" * 64,
            config_sha256="3" * 64,
            sources=changed,
        )
        self.assertNotEqual(first, second)

    def test_i1_snapshot_post_creation_hash_failure_cleans_private_directory(self):
        """backup 성공 뒤 hash/quick-check 단계가 실패해도 private snapshot을 남기지 않는다."""

        create_snapshot = self.function("_create_private_sqlite_snapshot")
        original_sha256 = runner._sha256
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.db"
            connection = sqlite3.connect(source)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE probe(value TEXT)")
            connection.execute("INSERT INTO probe VALUES ('start')")
            connection.commit()
            connection.close()
            identity = {
                "db_path": str(source.resolve(strict=True)),
                "db_bytes": int(source.stat().st_size),
                "db_sha256": original_sha256(source),
            }
            metrics = {
                "db_full_hash_verification_count": 1,
                "db_full_hash_verification_elapsed_seconds": 0.0,
            }

            def fail_snapshot_hash(path):
                candidate = Path(path)
                if candidate.name.startswith("snapshot-"):
                    raise OSError("injected snapshot hash failure")
                return original_sha256(candidate)

            with (
                patch.object(runner, "_sha256", side_effect=fail_snapshot_hash),
                self.assertRaisesRegex(OSError, "snapshot hash failure"),
            ):
                create_snapshot(
                    source,
                    temp_parent=root,
                    source_identity=identity,
                    metrics=metrics,
                )
            self.assertEqual(list(root.glob(".bias-db-snapshot-*")), [])

    def test_i1_wal_source_is_rejected_before_production_loader(self):
        """WAL-mode DB는 production DB loader가 한 번도 호출되기 전 중단된다."""
        production = self.function("run_production_analysis")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            connection = sqlite3.connect(db_path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("INSERT INTO probe VALUES (1)")
            connection.commit()
            out = root / "fullnet" / "bias_physical_followup"
            out.parent.mkdir()
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out),
                patch('analysis.bias_data.open_readonly_db') as loader,
                patch.object(runner, "_create_private_sqlite_snapshot") as creator,
                self.assertRaisesRegex(ValueError, "WAL|journal|sidecar"),
            ):
                production(
                    out_dir=out,
                    db_path=db_path,
                    station_results_path=station_results,
                    metadata_path=metadata_path,
                    terrain_path=terrain,
                    aws_base_folder=aws_root,
                    acceptance_receipt_path=root / "acceptance.json",
                )
            self.assertEqual(loader.call_count, 0)
            self.assertEqual(creator.call_count, 0)
            connection.close()

    def test_c1_production_requires_external_acceptance_receipt_before_loader(self):
        """receipt sink가 없으면 production을 성공 경로로 시작하지 않는다."""

        production = self.function("run_production_analysis")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            db_path, station_results, metadata_path, terrain, aws_root = (
                self._make_production_sources(root)
            )
            out = root / "fullnet" / "bias_physical_followup"
            out.parent.mkdir()
            with (
                patch.object(runner, "CANONICAL_OUTPUT_DIR", out),
                patch('analysis.bias_data.open_readonly_db') as loader,
                self.assertRaisesRegex(ValueError, "acceptance receipt"),
            ):
                production(
                    out_dir=out,
                    db_path=db_path,
                    station_results_path=station_results,
                    metadata_path=metadata_path,
                    terrain_path=terrain,
                    aws_base_folder=aws_root,
                )
            loader.assert_not_called()

    def test_i1_pre_promote_source_failure_preserves_previous_final(self):
        """atomic promote 직전 source 변경은 기존 canonical을 backup하기 전 중단된다."""
        publish = self.function("_atomic_publish")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_text("old", encoding="utf-8")
            (staging / "marker.txt").write_text("new", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                publish(
                    staging,
                    final,
                    failure_injection=None,
                    pre_promote_validator=lambda: (_ for _ in ()).throw(
                        RuntimeError("source changed before promote")
                    ),
                )
            self.assertEqual(
                (final / "marker.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertTrue(staging.is_dir())

    def test_i1_source_change_after_promote_restores_previous_final(self):
        """promote 직후 source 변경도 backup 삭제 전 검증하여 원상 복구한다."""
        publish = self.function("_atomic_publish")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_text("old", encoding="utf-8")
            (staging / "marker.txt").write_text("new", encoding="utf-8")
            validation_calls = 0

            def validate_source() -> None:
                nonlocal validation_calls
                validation_calls += 1
                if validation_calls == 2:
                    raise RuntimeError("source changed after promote")

            with self.assertRaisesRegex(RuntimeError, "source changed after promote"):
                publish(
                    staging,
                    final,
                    failure_injection=None,
                    pre_promote_validator=validate_source,
                )
            self.assertEqual(validation_calls, 2)
            self.assertEqual(
                (final / "marker.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertFalse(staging.exists())

    def test_c1_postpublish_anchor_failure_restores_final_and_receipt(self):
        """anchor 검증 실패는 backup 삭제·새 receipt 확정 전에 전부 원복한다."""

        publish = self.function("_atomic_publish")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            receipt = root / "acceptance.json"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_bytes(b"old-final")
            (staging / "marker.txt").write_bytes(b"new-final")
            receipt.write_bytes(b"old-receipt")
            with self.assertRaisesRegex(RuntimeError, "anchor mismatch"):
                publish(
                    staging,
                    final,
                    failure_injection=None,
                    post_promote_validator=lambda: (_ for _ in ()).throw(
                        RuntimeError("anchor mismatch")
                    ),
                    acceptance_receipt_path=receipt,
                    acceptance_receipt_payload_factory=lambda: {"new": True},
                )
            self.assertEqual((final / "marker.txt").read_bytes(), b"old-final")
            self.assertEqual(receipt.read_bytes(), b"old-receipt")

    def test_c1_receipt_failure_after_write_restores_final_and_prior_receipt(self):
        """receipt 기록 뒤 중단돼도 새 final과 receipt를 함께 원자적으로 되돌린다."""

        publish = self.function("_atomic_publish")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            receipt = root / "acceptance.json"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_bytes(b"old-final")
            (staging / "marker.txt").write_bytes(b"new-final")
            receipt.write_bytes(b"old-receipt")
            with self.assertRaisesRegex(RuntimeError, "acceptance receipt"):
                publish(
                    staging,
                    final,
                    failure_injection="after_receipt_before_cleanup",
                    post_promote_validator=lambda: None,
                    acceptance_receipt_path=receipt,
                    acceptance_receipt_payload_factory=lambda: {"new": True},
                )
            self.assertEqual((final / "marker.txt").read_bytes(), b"old-final")
            self.assertEqual(receipt.read_bytes(), b"old-receipt")

    def test_c1_committed_final_survives_receipt_backup_cleanup_failure(self):
        """commit 뒤 orphan backup 정리 실패가 검증된 final/receipt를 지우면 안 된다."""

        publish = self.function("_atomic_publish")
        original_unlink = Path.unlink
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            receipt = root / "acceptance.json"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_bytes(b"old-final")
            (staging / "marker.txt").write_bytes(b"new-final")
            receipt.write_bytes(b"old-receipt")

            def fail_only_receipt_backup_cleanup(path, *args, **kwargs):
                if path.name.startswith(f".{receipt.name}.backup-"):
                    raise PermissionError("injected orphan-receipt cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", new=fail_only_receipt_backup_cleanup):
                publish(
                    staging,
                    final,
                    failure_injection=None,
                    post_promote_validator=lambda: None,
                    acceptance_receipt_path=receipt,
                    acceptance_receipt_payload_factory=lambda: {"new": True},
                )

            self.assertEqual((final / "marker.txt").read_bytes(), b"new-final")
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), {"new": True})

    def test_c1_receipt_cannot_seal_manifest_changed_after_postvalidation(self):
        """receipt는 post-validator가 본 바로 그 manifest hash만 봉인해야 한다."""

        publish = self.function("_atomic_publish")
        build_receipt = self.function("_build_acceptance_receipt_payload")
        with _workspace_temp_dir() as temp_dir:
            root = Path(temp_dir)
            final = root / "bias_physical_followup"
            staging = root / ".bias_physical_followup.staging-probe"
            receipt = root / "acceptance.json"
            final.mkdir()
            staging.mkdir()
            (final / "marker.txt").write_bytes(b"old-final")
            receipt.write_bytes(b"old-receipt")
            (staging / "analysis_summary.md").write_text("validated", encoding="utf-8")
            (staging / "manifest.json").write_text(
                json.dumps({"version": 1, "summary": "validated"}),
                encoding="utf-8",
            )

            def post_validate():
                return runner._sha256(final / "manifest.json")

            def mutate_then_build_receipt():
                (final / "analysis_summary.md").write_text("mutated", encoding="utf-8")
                (final / "manifest.json").write_text(
                    json.dumps({"version": 1, "summary": "mutated"}),
                    encoding="utf-8",
                )
                return build_receipt(
                    final=final,
                    trusted_primary_anchor_sha256="a" * 64,
                    db_snapshot_sha256="b" * 64,
                )

            with self.assertRaisesRegex(ValueError, "manifest.*post|post.*manifest"):
                publish(
                    staging,
                    final,
                    failure_injection=None,
                    post_promote_validator=post_validate,
                    acceptance_receipt_path=receipt,
                    acceptance_receipt_payload_factory=mutate_then_build_receipt,
                )
            self.assertEqual((final / "marker.txt").read_bytes(), b"old-final")
            self.assertEqual(receipt.read_bytes(), b"old-receipt")

    def test_c1_postvalidator_rejects_manifest_changed_during_validator_return(self):
        """validator가 반환되는 사이 바뀐 manifest를 검증 완료 hash로 채택하면 안 된다."""

        stable_validate = self.function(
            "_validate_published_artifacts_stable_manifest"
        )
        with _workspace_temp_dir() as temp_dir:
            final = Path(temp_dir) / "bias_physical_followup"
            final.mkdir()
            manifest = final / "manifest.json"
            manifest.write_text(json.dumps({"version": 1}), encoding="utf-8")

            def mutate_after_semantic_validation(*args, **kwargs):
                manifest.write_text(json.dumps({"version": 2}), encoding="utf-8")
                return {"validated": True}

            with (
                patch.object(
                    runner,
                    "validate_published_artifacts",
                    side_effect=mutate_after_semantic_validation,
                ),
                self.assertRaisesRegex(ValueError, "manifest.*validation|validation.*manifest"),
            ):
                stable_validate(final)

    @staticmethod
    def _primary_hourly_fixture() -> pd.DataFrame:
        rows = []
        for month in range(1, 7):
            for day in range(1, 8):
                base = pd.Timestamp(2024, month, day)
                rows.extend(
                    [
                        {
                            "STN": "100",
                            "TM": base,
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "inversion_like",
                            "error_none": 0.1,
                            "error_fixed": 0.6,
                        },
                        {
                            "STN": "100",
                            "TM": base + pd.Timedelta(hours=1),
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "normal_negative",
                            "error_none": 0.1,
                            "error_fixed": 0.2,
                        },
                    ]
                )
        return pd.DataFrame(rows)

    def test_c1_primary_ledger_preserves_raw_sets_and_sufficient_statistics(self):
        """ledger는 common strata·상태별 날짜/월·signed-error 충분통계를 원표에서 보존한다."""
        build_matched = self.function("build_matched_state_contrast")
        build_ledger = self.function("build_primary_eligibility_ledger")
        hourly = self._primary_hourly_fixture()
        matched = build_matched(hourly, n_boot=2000, seed=20260827)
        primary_coverage = pd.DataFrame(
            [
                {
                    "STN": "100",
                    "SETTING_ID": runner.PRIMARY_SETTING_ID,
                    "COVERAGE_ID": runner.PRIMARY_COVERAGE_ID,
                    "N_COMMON_STRATA": 6,
                    "N_INVERSION_HOURS": 42,
                    "N_NORMAL_HOURS": 42,
                    "N_INVERSION_DAYS": 42,
                    "N_NORMAL_DAYS": 42,
                    "N_INVERSION_MONTHS": 6,
                    "N_NORMAL_MONTHS": 6,
                    "DELTA_ABS_BIAS_INVERSION": 0.5,
                    "DELTA_ABS_BIAS_NORMAL": 0.1,
                    "CONTRAST_INV_MINUS_NORMAL": 0.4,
                    "BASE_MATCHED_DATA_AVAILABLE": True,
                    "COVERAGE_ELIGIBLE": True,
                    "DROP_REASON": "eligible",
                }
            ]
        )
        ledger = build_ledger(
            hourly,
            matched,
            primary_coverage,
            source_population_sha256="a" * 64,
            analysis_config_sha256="b" * 64,
        )
        self.assertEqual(len(ledger), 1)
        row = ledger.iloc[0]
        self.assertEqual(int(row["N_COMMON_STRATA"]), 6)
        self.assertEqual(int(row["N_INVERSION_DAYS"]), 42)
        self.assertEqual(int(row["N_NORMAL_DAYS"]), 42)
        self.assertEqual(json.loads(row["COMMON_STRATA_KEYS_JSON"])[0], ["2024-01", "dark"])
        stats = json.loads(row["STRATUM_SUFFICIENT_STATS_JSON"])
        self.assertEqual(len(stats), 12)
        first = stats[0]
        self.assertEqual(first["n_hours"], 7)
        self.assertAlmostEqual(first["sum_error_none"], 0.7)
        self.assertAlmostEqual(float(row["CONTRAST_INV_MINUS_NORMAL"]), 0.4)

    def test_c1_primary_ledger_keeps_raw_dates_for_early_coverage_failures(self):
        """30일 또는 6개월 미달은 support 미검사와 raw 날짜를 구분한다."""
        build_matched = self.function("build_matched_state_contrast")
        build_ledger = self.function("build_primary_eligibility_ledger")
        cases = [
            (
                "insufficient_unique_days",
                [5, 5, 5, 5, 5, 4],
                29,
                6,
            ),
            (
                "insufficient_months",
                [6, 6, 6, 6, 6],
                30,
                5,
            ),
        ]
        for expected_reason, days_per_month, n_days, n_months in cases:
            with self.subTest(reason=expected_reason):
                rows = []
                for month, days in enumerate(days_per_month, start=1):
                    for day in range(1, days + 1):
                        base = pd.Timestamp(2024, month, day)
                        rows.extend(
                            [
                                {
                                    "STN": "100",
                                    "TM": base,
                                    "SOLAR_STATE": "dark",
                                    "REPRODUCED": True,
                                    "STATE_PRIMARY": "inversion_like",
                                    "error_none": 0.1,
                                    "error_fixed": 0.6,
                                },
                                {
                                    "STN": "100",
                                    "TM": base + pd.Timedelta(hours=1),
                                    "SOLAR_STATE": "dark",
                                    "REPRODUCED": True,
                                    "STATE_PRIMARY": "normal_negative",
                                    "error_none": 0.1,
                                    "error_fixed": 0.2,
                                },
                            ]
                        )
                hourly = pd.DataFrame(rows)
                matched = build_matched(hourly, n_boot=2000, seed=20260827)
                self.assertEqual(matched.iloc[0]["DROP_REASON"], expected_reason)
                coverage_rows = []
                for minimum_days in runner.SENSITIVITY_MIN_DAYS:
                    for minimum_months in runner.SENSITIVITY_MIN_MONTHS:
                        eligible = (
                            n_days >= minimum_days and n_months >= minimum_months
                        )
                        drop_reason = (
                            "eligible"
                            if eligible
                            else (
                                "insufficient_unique_days"
                                if n_days < minimum_days
                                else "insufficient_months"
                            )
                        )
                        coverage_rows.append(
                            {
                            "STN": "100",
                            "SETTING_ID": runner.PRIMARY_SETTING_ID,
                            "COVERAGE_ID": f"d{minimum_days}_m{minimum_months}",
                            "MIN_DAYS_PER_STATE": minimum_days,
                            "MIN_MONTHS_PER_STATE": minimum_months,
                            "N_COMMON_STRATA": n_months,
                            "N_INVERSION_HOURS": n_days,
                            "N_NORMAL_HOURS": n_days,
                            "N_INVERSION_DAYS": n_days,
                            "N_NORMAL_DAYS": n_days,
                            "N_INVERSION_MONTHS": n_months,
                            "N_NORMAL_MONTHS": n_months,
                            "DELTA_ABS_BIAS_INVERSION": 0.5,
                            "DELTA_ABS_BIAS_NORMAL": 0.1,
                            "CONTRAST_INV_MINUS_NORMAL": 0.4,
                            "BASE_MATCHED_DATA_AVAILABLE": True,
                            "COVERAGE_ELIGIBLE": eligible,
                            "DROP_REASON": drop_reason,
                            }
                        )
                coverage = pd.DataFrame(coverage_rows)
                primary_coverage = coverage.loc[
                    coverage["COVERAGE_ID"].eq(runner.PRIMARY_COVERAGE_ID)
                ]
                self.assertEqual(
                    primary_coverage.iloc[0]["DROP_REASON"], expected_reason
                )
                ledger = build_ledger(
                    hourly,
                    matched,
                    primary_coverage,
                    source_population_sha256=runner._station_population_identity(
                        ["100"]
                    )[1],
                    analysis_config_sha256=runner._canonical_json_hash(
                        runner._analysis_config_payload()
                    ),
                )
                row = ledger.iloc[0]
                self.assertEqual(len(json.loads(row["ANALYSIS_DATES_JSON"])), n_days)
                self.assertFalse(bool(row["ANALYSIS_SUPPORT_CHECKED"]))
                self.assertEqual(int(row["ANALYSIS_DATE_COUNT"]), 0)
                self.assertEqual(json.loads(row["SUPPORTED_ANALYSIS_DATES_JSON"]), [])
                self.assertEqual(json.loads(row["UNCOVERED_ANALYSIS_DATES_JSON"]), [])
                runner._validate_primary_eligibility_ledger_contract(
                    ledger,
                    matched,
                    coverage,
                    station_ids=["100"],
                )

    def test_c1_one_required_state_is_missing_not_merely_unmatched(self):
        """필수 두 상태 중 하나만 있으면 matched·coverage·ledger 사유가 같다."""
        hourly = pd.DataFrame(
            [
                {
                    "STN": "100",
                    "TM": pd.Timestamp(2024, 1, day),
                    "SOLAR_STATE": "dark",
                    "REPRODUCED": True,
                    "STATE_PRIMARY": "inversion_like",
                    "error_none": 0.1,
                    "error_fixed": 0.6,
                }
                for day in range(1, 8)
            ]
        )
        descriptive = hourly.rename(
            columns={"STATE_PRIMARY": "STRUCTURE_STATE"}
        )
        point = runner._matched_descriptive_point(descriptive)
        self.assertEqual(point["reason"], "missing_required_state")
        matched = runner.build_matched_state_contrast(hourly)
        self.assertEqual(matched.iloc[0]["DROP_REASON"], "missing_required_state")
        coverage = pd.DataFrame(
            [
                {
                    "STN": "100",
                    "SETTING_ID": runner.PRIMARY_SETTING_ID,
                    "COVERAGE_ID": f"d{minimum_days}_m{minimum_months}",
                    "MIN_DAYS_PER_STATE": minimum_days,
                    "MIN_MONTHS_PER_STATE": minimum_months,
                    "N_COMMON_STRATA": 0,
                    "N_INVERSION_HOURS": 0,
                    "N_NORMAL_HOURS": 0,
                    "N_INVERSION_DAYS": 0,
                    "N_NORMAL_DAYS": 0,
                    "N_INVERSION_MONTHS": 0,
                    "N_NORMAL_MONTHS": 0,
                    "DELTA_ABS_BIAS_INVERSION": np.nan,
                    "DELTA_ABS_BIAS_NORMAL": np.nan,
                    "CONTRAST_INV_MINUS_NORMAL": np.nan,
                    "BASE_MATCHED_DATA_AVAILABLE": False,
                    "COVERAGE_ELIGIBLE": False,
                    "DROP_REASON": "missing_required_state",
                }
                for minimum_days in runner.SENSITIVITY_MIN_DAYS
                for minimum_months in runner.SENSITIVITY_MIN_MONTHS
            ]
        )
        ledger = runner.build_primary_eligibility_ledger(
            hourly,
            matched,
            coverage.loc[coverage["COVERAGE_ID"].eq(runner.PRIMARY_COVERAGE_ID)],
            source_population_sha256=runner._station_population_identity(["100"])[1],
            analysis_config_sha256=runner._canonical_json_hash(
                runner._analysis_config_payload()
            ),
        )
        self.assertEqual(
            ledger.iloc[0]["COVERAGE_DROP_REASON"], "missing_required_state"
        )
        runner._validate_primary_eligibility_ledger_contract(
            ledger,
            matched,
            coverage,
            station_ids=["100"],
        )

    def test_c1_artifact_contract_includes_primary_ledger(self):
        """primary ledger가 schema·manifest·row count가 고정된 독립 CSV로 게시된다."""
        self.assertIn("primary_eligibility_ledger.csv", runner.CSV_SCHEMAS)
        run = self.function("run_injected_analysis")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            result = run(
                truth,
                {"100": _station_fixture()},
                out,
                allowed_fullnet_root=out.parent,
            )
            ledger = pd.read_csv(
                out / "primary_eligibility_ledger.csv", encoding="utf-8-sig"
            )
            self.assertEqual(len(ledger), 1)
            self.assertEqual(result["row_counts"]["primary_eligibility_ledger.csv"], 1)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn(
                "primary_eligibility_ledger.csv",
                {entry["path"] for entry in manifest["artifacts"]},
            )

    def test_c1_staged_primary_csv_must_equal_prewrite_memory_bytes(self):
        """CSV write 경계가 primary ledger를 바꾸면 disk에서 anchor를 다시 만들지 않는다."""

        run = self.function("run_injected_analysis")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        original_write = Path.write_bytes

        def corrupt_primary(path: Path, payload: bytes) -> int:
            if path.name == "primary_eligibility_ledger.csv":
                payload = payload.replace(b"100", b"101", 1)
            return original_write(path, payload)

        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            with (
                patch.object(Path, "write_bytes", new=corrupt_primary),
                self.assertRaisesRegex(ValueError, "pre-write memory bytes"),
            ):
                run(
                    truth,
                    {"100": _station_fixture()},
                    out,
                    allowed_fullnet_root=out.parent,
                )
            self.assertFalse(out.exists())

    def test_c1_primary_anchor_does_not_trust_dataframe_to_csv_boundary(self):
        """primary 기준 bytes는 monkeypatched pandas to_csv 결과에서 만들지 않는다."""

        run = self.function("run_injected_analysis")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            with patch.object(
                pd.DataFrame,
                "to_csv",
                side_effect=RuntimeError("poisoned to_csv boundary"),
            ):
                run(
                    truth,
                    {"100": _station_fixture()},
                    out,
                    allowed_fullnet_root=out.parent,
                )
            self.assertTrue((out / "primary_eligibility_ledger.csv").is_file())

    def test_c1_bootstrap_runs_once_for_generation_and_once_per_validator(self):
        """station당 bootstrap은 생성 1회·게시 전/후 검증 각 1회로 제한한다."""
        run = self.function("run_injected_analysis")
        truth = pd.DataFrame(
            {
                "STN": ["100"],
                "BIAS_none": [0.60],
                "BIAS_fixed": [0.62],
                "SIGNIFICANCE_CLASS": ["practically_equivalent"],
                "MECHANISM": ["small_change"],
                "mean_absdz_geom_km": [0.01],
            }
        )
        with _workspace_temp_dir() as temp_dir:
            out = Path(temp_dir) / "fullnet" / "bias_physical_followup"
            out.parent.mkdir(parents=True)
            with patch.object(
                runner,
                "bootstrap_shared_state_delta_abs_bias_contrast",
                wraps=runner.bootstrap_shared_state_delta_abs_bias_contrast,
            ) as bootstrap:
                run(
                    truth,
                    {"100": _station_fixture()},
                    out,
                    allowed_fullnet_root=out.parent,
                )
            self.assertEqual(bootstrap.call_count, 3)

    @staticmethod
    def _eligible_primary_ledger_fixture():
        rows = []
        for month in range(1, 7):
            for day in range(1, 11):
                base = pd.Timestamp(2024, month, day)
                rows.extend(
                    [
                        {
                            "STN": "100",
                            "TM": base,
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "inversion_like",
                            "error_none": 0.1,
                            "error_fixed": 0.4 + 0.02 * day,
                        },
                        {
                            "STN": "100",
                            "TM": base + pd.Timedelta(hours=1),
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "normal_negative",
                            "error_none": 0.1,
                            "error_fixed": 0.15 + 0.005 * day,
                        },
                    ]
                )
        hourly = pd.DataFrame(rows)
        matched = runner.build_matched_state_contrast(hourly)
        point = matched.iloc[0]
        coverage_rows = []
        for minimum_days in runner.SENSITIVITY_MIN_DAYS:
            for minimum_months in runner.SENSITIVITY_MIN_MONTHS:
                eligible = 60 >= minimum_days and 6 >= minimum_months
                coverage_rows.append(
                    {
                        "STN": "100",
                        "SETTING_ID": runner.PRIMARY_SETTING_ID,
                        "COVERAGE_ID": f"d{minimum_days}_m{minimum_months}",
                        "MIN_DAYS_PER_STATE": minimum_days,
                        "MIN_MONTHS_PER_STATE": minimum_months,
                        "N_COMMON_STRATA": 6,
                        "N_INVERSION_HOURS": 60,
                        "N_NORMAL_HOURS": 60,
                        "N_INVERSION_DAYS": 60,
                        "N_NORMAL_DAYS": 60,
                        "N_INVERSION_MONTHS": 6,
                        "N_NORMAL_MONTHS": 6,
                        "DELTA_ABS_BIAS_INVERSION": point[
                            "DELTA_ABS_BIAS_INVERSION"
                        ],
                        "DELTA_ABS_BIAS_NORMAL": point["DELTA_ABS_BIAS_NORMAL"],
                        "CONTRAST_INV_MINUS_NORMAL": point[
                            "CONTRAST_INV_MINUS_NORMAL"
                        ],
                        "BASE_MATCHED_DATA_AVAILABLE": True,
                        "COVERAGE_ELIGIBLE": eligible,
                        "DROP_REASON": "eligible"
                        if eligible
                        else (
                            "insufficient_unique_days"
                            if 60 < minimum_days
                            else "insufficient_months"
                        ),
                    }
                )
        coverage = pd.DataFrame(coverage_rows)
        ledger = runner.build_primary_eligibility_ledger(
            hourly,
            matched,
            coverage.loc[
                coverage["COVERAGE_ID"].eq(runner.PRIMARY_COVERAGE_ID)
            ],
            source_population_sha256=runner._station_population_identity(["100"])[1],
            analysis_config_sha256=runner._canonical_json_hash(
                runner._analysis_config_payload()
            ),
        )
        runner._apply_primary_ledger_bh_and_sync(ledger, matched)
        return hourly, matched, coverage, ledger

    def test_c1_ledger_rejects_coordinated_29_day_family_shrink(self):
        """coverage 9행과 matched를 함께 줄여도 raw ledger의 60일 family를 숨길 수 없다."""
        _, matched, coverage, ledger = self._eligible_primary_ledger_fixture()
        validate = self.function("_validate_primary_eligibility_ledger_contract")
        validate(ledger, matched, coverage, station_ids=["100"])
        forged_coverage = coverage.copy()
        forged_coverage[["N_INVERSION_DAYS", "N_NORMAL_DAYS"]] = 29
        forged_coverage["COVERAGE_ELIGIBLE"] = (
            forged_coverage["MIN_DAYS_PER_STATE"] <= 20
        )
        forged_coverage["DROP_REASON"] = np.where(
            forged_coverage["COVERAGE_ELIGIBLE"],
            "eligible",
            "insufficient_unique_days",
        )
        forged_matched = matched.copy()
        forged_matched.loc[0, "N_DAYS"] = 29
        forged_matched.loc[0, "COMPARISON_ELIGIBLE"] = False
        forged_matched.loc[0, "DROP_REASON"] = "insufficient_unique_days"
        forged_matched.loc[0, ["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH", "P_VALUE", "Q_VALUE"]] = np.nan
        forged_matched.loc[0, "EVIDENCE_GRADE"] = "현재 자료에서 지지되지 않음"
        with self.assertRaisesRegex(ValueError, "ledger|coverage|raw"):
            validate(ledger, forged_matched, forged_coverage, station_ids=["100"])

    def test_c1_ledger_rejects_coordinated_bootstrap_family_expansion(self):
        """ledger/matched의 support·CI·p를 함께 꾸며도 날짜 통계 fixed-seed 재계산이 거부한다."""
        _, matched, coverage, ledger = self._eligible_primary_ledger_fixture()
        validate = self.function("_validate_primary_eligibility_ledger_contract")
        forged_ledger = ledger.copy()
        forged_matched = matched.copy()
        for frame, reason_column in [
            (forged_ledger, "COMPARISON_DROP_REASON"),
            (forged_matched, "DROP_REASON"),
        ]:
            frame.loc[0, "BLOCK7_VALID_REPLICATES"] = 1899
            frame.loc[0, "BLOCK7_VALID_FRACTION"] = 1899 / 2000
            frame.loc[0, "COMPARISON_ELIGIBLE"] = False
            frame.loc[0, reason_column] = "insufficient_valid_replicates"
            frame.loc[0, ["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH", "P_VALUE", "Q_VALUE"]] = np.nan
            frame.loc[0, "EVIDENCE_GRADE"] = "현재 자료에서 지지되지 않음"
        with self.assertRaisesRegex(ValueError, "bootstrap|ledger"):
            validate(
                forged_ledger,
                forged_matched,
                coverage,
                station_ids=["100"],
            )

    def test_c1_ledger_rejects_fractional_counts_and_stringified_sums(self):
        """top-level/nested count와 signed sum의 JSON 타입·값 domain을 exact 고정한다."""

        _, matched, coverage, ledger = self._eligible_primary_ledger_fixture()
        validate = self.function("_validate_primary_eligibility_ledger_contract")
        mutations = []

        fractional = ledger.copy()
        fractional["N_INVERSION_DAYS"] = fractional[
            "N_INVERSION_DAYS"
        ].astype(float)
        fractional.loc[0, "N_INVERSION_DAYS"] = 60.75
        mutations.append(fractional)

        nested_count = ledger.copy()
        nested = json.loads(nested_count.loc[0, "DATE_SUFFICIENT_STATS_JSON"])
        nested[0]["n_hours"] = float(nested[0]["n_hours"])
        nested_count.loc[0, "DATE_SUFFICIENT_STATS_JSON"] = json.dumps(nested)
        mutations.append(nested_count)

        string_sum = ledger.copy()
        nested = json.loads(string_sum.loc[0, "DATE_SUFFICIENT_STATS_JSON"])
        nested[0]["sum_error_none"] = str(nested[0]["sum_error_none"])
        string_sum.loc[0, "DATE_SUFFICIENT_STATS_JSON"] = json.dumps(nested)
        mutations.append(string_sum)

        support_fractional = ledger.copy()
        nested = json.loads(
            support_fractional.loc[0, "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON"]
        )
        nested[0]["analysis_unique_dates"] = 0.75
        support_fractional.loc[0, "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON"] = (
            json.dumps(nested)
        )
        mutations.append(support_fractional)

        for mutation in mutations:
            with self.subTest(payload=mutation.to_dict("records")[0]):
                with self.assertRaisesRegex(ValueError, "integer|count|sum|domain"):
                    validate(mutation, matched, coverage, station_ids=["100"])

    def test_c1_ledger_rejects_noncanonical_iso_date_in_early_failure(self):
        """support 전 탈락 행도 YYYY-MM-DD가 아닌 날짜 alias를 허용하지 않는다."""

        rows = []
        for month, days in enumerate([5, 5, 5, 5, 5, 4], start=1):
            for day in range(1, days + 1):
                base = pd.Timestamp(2024, month, day)
                rows.extend(
                    [
                        {
                            "STN": "100",
                            "TM": base,
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "inversion_like",
                            "error_none": 0.1,
                            "error_fixed": 0.6,
                        },
                        {
                            "STN": "100",
                            "TM": base + pd.Timedelta(hours=1),
                            "SOLAR_STATE": "dark",
                            "REPRODUCED": True,
                            "STATE_PRIMARY": "normal_negative",
                            "error_none": 0.1,
                            "error_fixed": 0.2,
                        },
                    ]
                )
        hourly = pd.DataFrame(rows)
        matched = runner.build_matched_state_contrast(
            hourly, n_boot=2000, seed=20260827
        )
        coverage_rows = []
        for minimum_days in runner.SENSITIVITY_MIN_DAYS:
            for minimum_months in runner.SENSITIVITY_MIN_MONTHS:
                eligible = 29 >= minimum_days and 6 >= minimum_months
                coverage_rows.append(
                    {
                        "STN": "100",
                        "SETTING_ID": runner.PRIMARY_SETTING_ID,
                        "COVERAGE_ID": f"d{minimum_days}_m{minimum_months}",
                        "MIN_DAYS_PER_STATE": minimum_days,
                        "MIN_MONTHS_PER_STATE": minimum_months,
                        "N_COMMON_STRATA": 6,
                        "N_INVERSION_HOURS": 29,
                        "N_NORMAL_HOURS": 29,
                        "N_INVERSION_DAYS": 29,
                        "N_NORMAL_DAYS": 29,
                        "N_INVERSION_MONTHS": 6,
                        "N_NORMAL_MONTHS": 6,
                        "DELTA_ABS_BIAS_INVERSION": 0.5,
                        "DELTA_ABS_BIAS_NORMAL": 0.1,
                        "CONTRAST_INV_MINUS_NORMAL": 0.4,
                        "BASE_MATCHED_DATA_AVAILABLE": True,
                        "COVERAGE_ELIGIBLE": eligible,
                        "DROP_REASON": "eligible" if eligible else "insufficient_unique_days",
                    }
                )
        coverage = pd.DataFrame(coverage_rows)
        primary_coverage = coverage.loc[
            coverage["COVERAGE_ID"].eq(runner.PRIMARY_COVERAGE_ID)
        ]
        ledger = runner.build_primary_eligibility_ledger(
            hourly,
            matched,
            primary_coverage,
            source_population_sha256=runner._station_population_identity(["100"])[1],
            analysis_config_sha256=runner._canonical_json_hash(
                runner._analysis_config_payload()
            ),
        )
        self.assertEqual(ledger.iloc[0]["COVERAGE_DROP_REASON"], "insufficient_unique_days")

        forged = ledger.copy()
        canonical = "2024-01-01"
        alias = "2024-1-1"
        for column in (
            "INVERSION_UNIQUE_DATES_JSON",
            "NORMAL_UNIQUE_DATES_JSON",
            "ANALYSIS_DATES_JSON",
        ):
            values = json.loads(forged.loc[0, column])
            forged.loc[0, column] = json.dumps(
                sorted(alias if value == canonical else value for value in values)
            )
        daily = json.loads(forged.loc[0, "DATE_SUFFICIENT_STATS_JSON"])
        for item in daily:
            if item["date"] == canonical:
                item["date"] = alias
        forged.loc[0, "DATE_SUFFICIENT_STATS_JSON"] = json.dumps(daily)
        strata = json.loads(forged.loc[0, "STRATUM_SUFFICIENT_STATS_JSON"])
        for item in strata:
            item["unique_dates"] = sorted(
                alias if value == canonical else value for value in item["unique_dates"]
            )
        forged.loc[0, "STRATUM_SUFFICIENT_STATS_JSON"] = json.dumps(strata)

        with self.assertRaisesRegex(ValueError, "date|ISO|canonical"):
            runner._validate_primary_eligibility_ledger_contract(
                forged,
                matched,
                coverage,
                station_ids=["100"],
            )

    def test_c1_zero_date_support_flag_cannot_be_forged(self):
        """0-date skeleton의 support_checked를 두 파생표와 함께 바꿔도 ledger가 거부한다."""
        digest = runner._station_population_identity(["100"])[1]
        config_digest = runner._canonical_json_hash(runner._analysis_config_payload())
        ledger = runner._empty_primary_eligibility_ledger(
            "100",
            "source_payload_missing",
            source_population_sha256=digest,
            analysis_config_sha256=config_digest,
        )
        matched = runner._empty_matched_contrast("100", "source_payload_missing")
        coverage = runner._empty_matched_coverage_sensitivity(
            "100", "source_payload_missing"
        )
        forged_ledger = ledger.copy()
        forged_matched = matched.copy()
        forged_ledger.loc[0, "ANALYSIS_SUPPORT_CHECKED"] = True
        forged_matched.loc[0, "ANALYSIS_SUPPORT_CHECKED"] = True
        with self.assertRaisesRegex(ValueError, "matched|bootstrap|ledger"):
            runner._validate_primary_eligibility_ledger_contract(
                forged_ledger,
                forged_matched,
                coverage,
                station_ids=["100"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
