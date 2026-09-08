"""고도보정 BIAS 물리요인 후속분석 원자료 runner.

이 모듈은 작은 주입형 자료와 실제 production 자료를 같은 계산 계약으로 처리한다.
핵심 원칙은 (1) SQLite 정본을 읽기 전용으로 사용하고, (2) exact 이웃과 기존
``none`` IDW를 먼저 재현하며, (3) 529개 전체 모집단의 시간별 공간 기온구조를
계산한 뒤에만 사전 정의한 부분집단 표지를 결합하는 것이다.

여기서 ``inversion_like``는 주변 지상 관측소 사이의 *공간적* 양의 기온-고도
기울기이다. 실제 연직 프로파일 역전을 직접 측정한 값이 아니다.
"""

from __future__ import annotations

from project_paths import AWS_DIR

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
import csv
import gc
from itertools import product
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

from analysis.bias_physical_followup import (
    benjamini_hochberg,
    build_equivalent_diagnostics,
    build_worsening_diagnostics,
    classify_spatial_temperature_structure,
    exact_idw_none_weighted_mean,
    ols_temperature_elevation_slope,
    rank_biserial_effect,
    weighted_temperature_elevation_slope,
)
from analysis.bias_physical_block_bootstrap import (
    bootstrap_shared_state_delta_abs_bias_contrast,
)
from analysis.bias_geometry import compute_geometry_metrics
from weather.find_stations import filter_valid_station_meta, get_distance_km


STRUCTURE_STATES = (
    "inversion_like",
    "normal_negative",
    "uncertain",
    "ineligible",
)
STRUCTURE_CONFIGS = {
    "primary": "STATE_PRIMARY",
    "span200": "STATE_SPAN200",
    "neff4": "STATE_NEFF4",
    "ols_only": "STATE_OLS_ONLY",
}
SENSITIVITY_MIN_N = (5, 6, 7)
SENSITIVITY_MIN_SPAN_M = (50, 100, 200)
SENSITIVITY_MIN_WEIGHTED_SD_M = (30, 50, 75)
SENSITIVITY_AUX_WEIGHTS = ("none", "d1", "d2")
SENSITIVITY_MIN_DAYS = (20, 30, 60)
SENSITIVITY_MIN_MONTHS = (3, 6, 9)
PRIMARY_SETTING_ID = "n6_span100_wsd50_wd2"
PRIMARY_COVERAGE_ID = "d30_m6"
ARTIFACT_SCHEMA_VERSION = 6
PROCESS_RSS_LIMIT_BYTES = 1_610_612_736
_METADATA_CORE_METRICS = (
    "db_query_count",
    "aws_file_read_count",
    "geometry_build_count",
    "unique_geometry_count",
    "processed_paired_hours",
    "reproduced_hour_count",
    "processed_neighbor_values",
    "ols_fit_count",
    "aux_fit_count_none",
    "aux_fit_count_d1",
    "aux_fit_count_d2",
    "loo_fit_count_none",
    "loo_fit_count_d1",
    "loo_fit_count_d2",
    "setting_classification_count",
    "coverage_gate_evaluation_count",
    "peak_chunk_memory_bytes",
)
_METADATA_PRODUCTION_INTEGER_METRICS = (
    "db_full_hash_verification_count",
    "aws_cache_hit_count",
    "aws_cache_miss_count",
    "aws_cache_peak_entries",
    "aws_cache_eviction_count",
    "aws_cache_max_entries",
    "process_peak_rss_bytes",
    "rss_limit_bytes",
    "stations_timed",
)
_METADATA_PRODUCTION_FLOAT_METRICS = (
    "db_full_hash_verification_elapsed_seconds",
    "first_10_station_elapsed_seconds",
    "estimated_total_runtime_seconds_from_first_10",
    "observed_station_elapsed_seconds",
)
CANONICAL_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "WeatherData_AWS"
    / "interpolation_batch"
    / "fullnet"
    / "bias_physical_followup"
)
SIGNIFICANCE_CLASSES = {
    "meaningful_improvement",
    "meaningful_worsening",
    "practically_equivalent",
    "uncertain",
}

_T_975 = np.array(
    [
        np.nan,
        12.706204736432095,
        4.302652729696142,
        3.182446305284263,
        2.7764451051977987,
        2.570581835636314,
        2.4469118487916806,
        2.3646242510102993,
        2.3060041350333704,
    ],
    dtype=float,
)

STATION_STRUCTURE_COLUMNS = [
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
    "DARK_PRIMARY_ELIGIBLE_HOURS",
    "DARK_PRIMARY_ELIGIBLE_DAYS",
    "DARK_PRIMARY_ELIGIBLE_MONTHS",
    "DARK_INV_FRACTION",
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
    "MEDIAN_ABS_DELTA_COAST_KM",
    "MEAN_SIGNED_DELTA_Z_M",
    "MEDIAN_WEIGHTED_COASTAL_FRACTION",
    "MEDIAN_WEIGHTED_ISLAND_FRACTION",
    "MEDIAN_RELIEF_MISMATCH_M",
    "MEDIAN_TERRAIN_MISMATCH_FRACTION",
    "TARGET_LATITUDE",
    "TARGET_LONGITUDE",
    "TPI_STATUS",
    "ANALYSIS_POPULATION",
]

STATE_CONDITION_COLUMNS = [
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
]

COAST_TIME_COLUMNS = [
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
    "H2_OBSERVED_BIAS_SIGN",
    "H2_DIRECTION_MATCH",
    "H2_PLACEMENT_SCOPE",
    "H2_EVIDENCE_SCOPE",
    "EVIDENCE_GRADE",
    "LIMITATION",
    "ANALYSIS_POPULATION",
    "SUBSET_MEANINGFUL_WORSENING",
    "SUBSET_HIGH_AFTER",
]

EQUIVALENT_COLUMNS = [
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
]

WORSENING_COLUMNS = [
    "STN",
    "BIAS_none",
    "BIAS_fixed",
    "MECHANISM",
    "CORRECTION_SHIFT",
    "DELTA_ABS_BIAS",
    "ABS_DZ_M",
    "DZ_BIN",
]

HIGH_RESIDUAL_COLUMNS = [
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
]

STATION_885_COLUMNS = [
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
]

MATCHED_STATE_CONTRAST_COLUMNS = [
    "STN",
    "CONFIG",
    "SETTING_ID",
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
    "P_VALUE",
    "Q_VALUE",
    "EVIDENCE_GRADE",
    "LIMITATION",
    "ANALYSIS_POPULATION",
    "SUBSET_MEANINGFUL_WORSENING",
    "SUBSET_HIGH_AFTER",
]

SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS = [
    "STN",
    "SETTING_ID",
    "MIN_N",
    "MIN_SPAN_M",
    "MIN_WEIGHTED_SD_M",
    "AUX_WEIGHT",
    "STRUCTURE_STATE",
    "TOTAL_PAIRED_HOURS",
    "REPRODUCED_HOURS",
    "N_HOURS",
    "STATE_FRACTION_OF_REPRODUCED",
    "ELIGIBLE_HOURS",
    "ELIGIBLE_FRACTION_OF_REPRODUCED",
    "INVERSION_FRACTION_OF_ELIGIBLE",
    "N_EXACT_REPRODUCTION_FAILURE",
    "N_NONFINITE_ELEVATION",
    "N_FAIL_MIN_N",
    "N_FAIL_SPAN",
    "N_FAIL_WEIGHTED_SD",
    "N_FAIL_N_EFF",
    "N_OLS_CI_CROSSES_ZERO",
    "N_OLS_AUX_SIGN_DISAGREE",
    "N_LOO_SIGN_UNSTABLE",
    "MEDIAN_OLS_SLOPE_C_PER_KM",
    "MEDIAN_AUX_SLOPE_C_PER_KM",
    "MEDIAN_LOO_SIGN_STABILITY",
    "ANALYSIS_POPULATION",
]

MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS = [
    "STN",
    "SETTING_ID",
    "MIN_N",
    "MIN_SPAN_M",
    "MIN_WEIGHTED_SD_M",
    "AUX_WEIGHT",
    "COVERAGE_ID",
    "MIN_DAYS_PER_STATE",
    "MIN_MONTHS_PER_STATE",
    "N_COMMON_STRATA",
    "N_INVERSION_HOURS",
    "N_NORMAL_HOURS",
    "N_INVERSION_DAYS",
    "N_NORMAL_DAYS",
    "N_INVERSION_MONTHS",
    "N_NORMAL_MONTHS",
    "DELTA_ABS_BIAS_INVERSION",
    "DELTA_ABS_BIAS_NORMAL",
    "CONTRAST_INV_MINUS_NORMAL",
    "BASE_MATCHED_DATA_AVAILABLE",
    "COVERAGE_ELIGIBLE",
    "DROP_REASON",
    "SUBSET_MEANINGFUL_WORSENING",
    "SUBSET_HIGH_AFTER",
    "ANALYSIS_POPULATION",
]

PRIMARY_ELIGIBILITY_LEDGER_COLUMNS = [
    "STN",
    "SETTING_ID",
    "COVERAGE_ID",
    "SOURCE_POPULATION_SHA256",
    "ANALYSIS_CONFIG_SHA256",
    "COMMON_STRATA_KEYS_JSON",
    "INVERSION_UNIQUE_DATES_JSON",
    "NORMAL_UNIQUE_DATES_JSON",
    "INVERSION_MONTHS_JSON",
    "NORMAL_MONTHS_JSON",
    "ANALYSIS_DATES_JSON",
    "SUPPORTED_ANALYSIS_DATES_JSON",
    "UNCOVERED_ANALYSIS_DATES_JSON",
    "STRATUM_SUFFICIENT_STATS_JSON",
    "DATE_SUFFICIENT_STATS_JSON",
    "N_COMMON_STRATA",
    "N_INVERSION_HOURS",
    "N_NORMAL_HOURS",
    "N_INVERSION_DAYS",
    "N_NORMAL_DAYS",
    "N_INVERSION_MONTHS",
    "N_NORMAL_MONTHS",
    "DELTA_ABS_BIAS_INVERSION",
    "DELTA_ABS_BIAS_NORMAL",
    "CONTRAST_INV_MINUS_NORMAL",
    "BASE_MATCHED_DATA_AVAILABLE",
    "COVERAGE_ELIGIBLE",
    "COVERAGE_DROP_REASON",
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
    "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON",
    "BLOCK7_CI_LOW",
    "BLOCK7_CI_HIGH",
    "P_VALUE",
    "Q_VALUE",
    "COMPARISON_ELIGIBLE",
    "COMPARISON_DROP_REASON",
    "EVIDENCE_GRADE",
    "ANALYSIS_POPULATION",
]

PHYSICAL_HYPOTHESIS_COLUMNS = """HYPOTHESIS_ID,CONTRAST_ID,ROW_KIND,ANALYSIS_ROLE,SIGN_STRATUM,TARGET_GROUP,REFERENCE_GROUP,ROW_UNIT,TARGET_CANONICAL_N,REFERENCE_CANONICAL_N,TARGET_ELIGIBLE_N,REFERENCE_ELIGIBLE_N,TARGET_EXCLUDED_N,REFERENCE_EXCLUDED_N,VARIABLE,VARIABLE_UNIT,TARGET_MEDIAN,REFERENCE_MEDIAN,MEDIAN_DIFFERENCE,EFFECT_SIZE_NAME,EFFECT_ESTIMATE,EXPECTED_EFFECT_DIRECTION,OBSERVED_EFFECT_DIRECTION,EFFECT_DIRECTION_MATCH,EXPECTED_BIAS_DIRECTION_RULE,TARGET_DIRECTION_MATCH_N,TARGET_DIRECTION_TESTABLE_N,TARGET_DIRECTION_MATCH_RATE,REFERENCE_DIRECTION_MATCH_N,REFERENCE_DIRECTION_TESTABLE_N,REFERENCE_DIRECTION_MATCH_RATE,BIAS_DIRECTION_MATCH,COVERAGE_GATE,COVERAGE_ELIGIBLE,INFERENCE_MODE,SPATIAL_BLOCK_METHOD,PRIMARY_BLOCK_KM,N_BLOCKS_30KM,TARGET_BLOCKS_30KM,REFERENCE_BLOCKS_30KM,VALID_REPLICATES_30KM,CI_LOW_30KM,CI_HIGH_30KM,P_VALUE_30KM,N_BLOCKS_50KM,TARGET_BLOCKS_50KM,REFERENCE_BLOCKS_50KM,VALID_REPLICATES_50KM,CI_LOW_50KM,CI_HIGH_50KM,P_VALUE,N_BLOCKS_100KM,TARGET_BLOCKS_100KM,REFERENCE_BLOCKS_100KM,VALID_REPLICATES_100KM,CI_LOW_100KM,CI_HIGH_100KM,P_VALUE_100KM,FAMILY_ID,FAMILY_SIZE,Q_VALUE,Q_METHOD,SPATIAL_SENSITIVITY_DIRECTION_STABLE,EVIDENCE_GRADE,LIMITATION,ANALYSIS_POPULATION""".split(",")

CSV_SCHEMAS = {
    "station_structure_coverage.csv": STATION_STRUCTURE_COLUMNS,
    "state_condition_bias.csv": STATE_CONDITION_COLUMNS,
    "coast_time_signed_bias.csv": COAST_TIME_COLUMNS,
    "equivalent_diagnostics.csv": EQUIVALENT_COLUMNS,
    "worsening_diagnostics.csv": WORSENING_COLUMNS,
    "high_residual_group_diagnostic.csv": HIGH_RESIDUAL_COLUMNS,
    "station_885_diagnostic.csv": STATION_885_COLUMNS,
    "matched_state_contrast.csv": MATCHED_STATE_CONTRAST_COLUMNS,
    "spatial_structure_sensitivity_summary.csv": SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS,
    "matched_state_coverage_sensitivity.csv": MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS,
    "primary_eligibility_ledger.csv": PRIMARY_ELIGIBILITY_LEDGER_COLUMNS,
    "physical_hypothesis_comparison.csv": PHYSICAL_HYPOTHESIS_COLUMNS,
}

PRIMARY_ANCHORED_CSV_NAMES = (
    "primary_eligibility_ledger.csv",
    "matched_state_contrast.csv",
    "matched_state_coverage_sensitivity.csv",
)

PNG_NAMES = (
    "equivalent_before_after.png",
    "worsening_by_dz.png",
    "high_residual_factors.png",
    "structure_condition_bias.png",
    "station_885_case.png",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_station_id(value: Any, *, field: str = "station ID") -> str:
    """공식 KMA 지점 ID를 공백 없는 숫자 문자열로 정규화한다."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError(f"{field} must be finite")
    text = str(value).strip()
    if not text or not text.isdigit():
        raise ValueError(f"{field} must be a non-empty numeric identifier")
    return text


def validate_exact_neighbor_ids(
    target_id: Any,
    neighbor_ids: Sequence[Any],
    expected_count: int | None = None,
) -> tuple[str, ...]:
    """exact signature의 이웃 ID 불변식을 검증한다."""

    target = _normalize_station_id(target_id, field="target_id")
    if isinstance(neighbor_ids, (str, bytes)):
        raise ValueError("neighbor_ids must be a sequence of identifiers")
    ids = tuple(
        _normalize_station_id(value, field="neighbor_id") for value in neighbor_ids
    )
    if not ids or len(ids) > 10:
        raise ValueError("exact neighbor set must contain between 1 and 10 IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("exact neighbor IDs must be unique")
    if target in ids:
        raise ValueError("target must not appear in exact neighbor IDs")
    if expected_count is not None:
        if isinstance(expected_count, bool):
            raise ValueError("expected_count must be an integer")
        try:
            count = int(expected_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("expected_count must be an integer") from exc
        if count != expected_count or count != len(ids):
            raise ValueError("NEIGHBOR_COUNT does not match exact neighbor IDs")
    return ids


def _stable_distance_weights(distances_km: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances_km, dtype=float)
    if distances.ndim != 1 or not len(distances):
        raise ValueError("distances must be a non-empty vector")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("distances must be finite and non-negative")
    zero = np.flatnonzero(distances == 0)
    if len(zero):
        result = np.zeros(len(distances), dtype=float)
        result[int(zero[0])] = 1.0
        return result
    scaled = (float(distances.min()) / distances) ** 2
    return scaled / scaled.sum()


def _terrain_lookup(terrain: pd.DataFrame | None) -> dict[str, pd.Series]:
    if terrain is None or len(terrain) == 0:
        return {}
    _require_columns(terrain, ["STN"], "terrain")
    normalized = terrain.copy()
    normalized["STN"] = normalized["STN"].map(
        lambda value: _normalize_station_id(value, field="terrain STN")
    )
    if normalized["STN"].duplicated().any():
        duplicated = normalized.loc[normalized["STN"].duplicated(), "STN"].tolist()
        raise ValueError(f"duplicate terrain rows: {duplicated}")
    return {str(row.STN): normalized.iloc[index] for index, row in enumerate(normalized.itertuples(index=False))}


def _metadata_validity_boundary_lookup(meta: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, ...]]:
    """Stage1 날짜 key에서 좌표가 바뀌는 관측소별 달력일을 반환한다."""

    _require_columns(meta, ["지점", "시작일", "종료일"], "meta")
    work = meta.loc[:, ["지점", "시작일", "종료일"]].copy()
    work["지점"] = work["지점"].map(
        lambda value: _normalize_station_id(value, field="metadata station ID")
    )
    work["시작일"] = pd.to_datetime(work["시작일"], errors="raise")
    work["종료일"] = pd.to_datetime(work["종료일"], errors="coerce")
    result: dict[str, tuple[pd.Timestamp, ...]] = {}
    for station_id, rows in work.groupby("지점", sort=False):
        values = pd.concat([rows["시작일"], rows["종료일"]], ignore_index=True).dropna()
        effective_days = set()
        for value in values.unique():
            timestamp = pd.Timestamp(value)
            day = timestamp.normalize()
            effective_days.add(day if timestamp == day else day + pd.DateOffset(days=1))
        result[str(station_id)] = tuple(sorted(effective_days))
    return result


def _iter_signature_metadata_segments(
    group: pd.DataFrame,
    *,
    station_ids: Sequence[Any],
    boundary_lookup: Mapping[str, Sequence[pd.Timestamp]],
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    """signature 구성원의 metadata 경계가 없는 구간별 paired row를 반환한다."""

    if group.empty:
        return
    _require_columns(group, ["TM"], "signature group")
    timestamps = pd.to_datetime(group["TM"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("signature group TM must not contain NaT")
    members = tuple(
        _normalize_station_id(value, field="geometry member station ID")
        for value in station_ids
    )
    calendar_days = timestamps.dt.normalize()
    lower = pd.Timestamp(calendar_days.min())
    upper = pd.Timestamp(calendar_days.max())
    boundaries = sorted(
        {
            pd.Timestamp(boundary)
            for station_id in members
            for boundary in boundary_lookup.get(station_id, ())
            if lower < pd.Timestamp(boundary) <= upper
        }
    )
    if not boundaries:
        yield lower, group
        return

    boundary_values = np.asarray(boundaries, dtype="datetime64[ns]")
    segment_ids = np.searchsorted(
        boundary_values,
        calendar_days.to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    segmented = group.copy()
    segmented["__GEOMETRY_SEGMENT"] = segment_ids
    for _, segment in segmented.groupby("__GEOMETRY_SEGMENT", sort=False):
        segment = segment.drop(columns="__GEOMETRY_SEGMENT")
        yield pd.Timestamp(segment["TM"].min()).normalize(), segment


def reconstruct_signature_geometry(
    *,
    target_id: Any,
    neighbor_ids: Sequence[Any],
    timestamp: Any,
    coord_epoch: Any,
    expected_neighbor_count: int,
    meta: pd.DataFrame,
    terrain: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """한 시각 exact signature의 실제 거리순 기하와 환경 차이를 재구성한다."""

    required = [
        "지점",
        "시작일",
        "종료일",
        "위도",
        "경도",
        "노장해발고도(m)",
    ]
    _require_columns(meta, required, "meta")
    target = _normalize_station_id(target_id, field="target_id")
    exact_ids = validate_exact_neighbor_ids(
        target, neighbor_ids, expected_count=expected_neighbor_count
    )

    work = meta.copy()
    work["__metadata_order"] = np.arange(len(work), dtype=int)
    work["지점"] = work["지점"].map(
        lambda value: _normalize_station_id(value, field="metadata station ID")
    )
    work["시작일"] = pd.to_datetime(work["시작일"], errors="raise")
    work["종료일"] = pd.to_datetime(work["종료일"], errors="coerce")
    for column in ["위도", "경도", "노장해발고도(m)"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = filter_valid_station_meta(work, pd.Timestamp(timestamp))
    target_rows = valid.loc[valid["지점"].eq(target)]
    if len(target_rows) != 1:
        raise ValueError("target metadata is not uniquely valid at the paired hour")
    target_row = target_rows.iloc[0]
    if not np.isfinite([target_row["위도"], target_row["경도"]]).all():
        raise ValueError("target coordinates must be finite")

    epoch = pd.Timestamp(coord_epoch).normalize()
    if pd.Timestamp(target_row["시작일"]).normalize() != epoch:
        raise ValueError("COORD_EPOCH does not match target metadata epoch")

    records: list[tuple[float, int, str, pd.Series]] = []
    for sid in exact_ids:
        rows = valid.loc[valid["지점"].eq(sid)]
        if len(rows) != 1:
            raise ValueError(f"neighbor {sid} metadata is not uniquely valid")
        row = rows.iloc[0]
        if not np.isfinite([row["위도"], row["경도"]]).all():
            raise ValueError(f"neighbor {sid} coordinates must be finite")
        distance = float(
            get_distance_km(
                float(target_row["위도"]),
                float(target_row["경도"]),
                float(row["위도"]),
                float(row["경도"]),
            )
        )
        records.append((distance, int(row["__metadata_order"]), sid, row))
    records.sort(key=lambda item: (item[0], item[1]))

    ordered_ids = tuple(item[2] for item in records)
    distances = np.asarray([item[0] for item in records], dtype=float)
    elevations = np.asarray(
        [pd.to_numeric(item[3]["노장해발고도(m)"], errors="coerce") for item in records],
        dtype=float,
    )
    regression_mask = np.isfinite(elevations)
    weights = _stable_distance_weights(distances)

    # 기존 독립 geometry 모듈과 exact 이웃수·최단거리를 교차검산한다.
    legacy_geometry = compute_geometry_metrics(target, exact_ids, valid)
    if legacy_geometry.neighbor_count_geom != len(ordered_ids) or not np.isclose(
        legacy_geometry.nearest_distance_km,
        float(distances.min()),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("reconstructed exact geometry differs from bias_geometry")
    # 정본 geometry의 signed ΔZ 정의를 그대로 재사용한다. target 또는 이웃
    # 고도가 결측인 해당 이웃의 ΔZ는 0이며, 거리 가중은 그대로 보존된다.
    signed_delta_z_m = float(legacy_geometry.dz_geom_km * 1000.0)

    terrain_by_id = _terrain_lookup(terrain)
    target_terrain = terrain_by_id.get(target)
    neighbor_terrain = [terrain_by_id.get(sid) for sid in ordered_ids]

    def weighted_neighbor_metric(column: str) -> float:
        values = np.asarray(
            [
                np.nan
                if row is None
                else pd.to_numeric(row.get(column, np.nan), errors="coerce")
                for row in neighbor_terrain
            ],
            dtype=float,
        )
        mask = np.isfinite(values)
        if not mask.any():
            return float("nan")
        local = weights[mask]
        if local.sum() <= 0:
            local = np.ones(mask.sum(), dtype=float)
        local = local / local.sum()
        return float(np.dot(local, values[mask]))

    target_coast = (
        np.nan
        if target_terrain is None
        else pd.to_numeric(target_terrain.get("coast_dist_km", np.nan), errors="coerce")
    )
    neighbor_coast = weighted_neighbor_metric("coast_dist_km")
    delta_coast = (
        float(target_coast - neighbor_coast)
        if np.isfinite(target_coast) and np.isfinite(neighbor_coast)
        else float("nan")
    )
    coastal_fraction = weighted_neighbor_metric("is_coastal")
    island_fraction = weighted_neighbor_metric("is_island")
    target_relief = (
        np.nan
        if target_terrain is None
        else pd.to_numeric(target_terrain.get("relief_1km_m", np.nan), errors="coerce")
    )
    neighbor_relief = weighted_neighbor_metric("relief_1km_m")
    relief_mismatch = (
        float(target_relief - neighbor_relief)
        if np.isfinite(target_relief) and np.isfinite(neighbor_relief)
        else float("nan")
    )

    target_group = None if target_terrain is None else target_terrain.get("terrain_group")
    mismatch_values = np.asarray(
        [
            np.nan
            if row is None or pd.isna(target_group) or pd.isna(row.get("terrain_group"))
            else float(str(row.get("terrain_group")) != str(target_group))
            for row in neighbor_terrain
        ],
        dtype=float,
    )
    mismatch_mask = np.isfinite(mismatch_values)
    if mismatch_mask.any():
        local_weights = weights[mismatch_mask]
        if local_weights.sum() <= 0:
            local_weights = np.ones(mismatch_mask.sum(), dtype=float)
        terrain_mismatch = float(
            np.dot(local_weights / local_weights.sum(), mismatch_values[mismatch_mask])
        )
    else:
        terrain_mismatch = float("nan")

    return {
        "target_id": target,
        "target_latitude": float(target_row["위도"]),
        "target_longitude": float(target_row["경도"]),
        "neighbor_ids": ordered_ids,
        "distances_km": distances,
        "elevations_m": elevations,
        "idw_weights": weights,
        "zero_neighbor_distance": bool(np.any(distances == 0)),
        "regression_neighbor_ids": tuple(
            sid for sid, keep in zip(ordered_ids, regression_mask) if keep
        ),
        "regression_distances_km": distances[regression_mask],
        "regression_elevations_m": elevations[regression_mask],
        "regression_neighbor_count": int(regression_mask.sum()),
        "delta_coast_km": delta_coast,
        "signed_delta_z_m": signed_delta_z_m,
        "weighted_coastal_fraction": coastal_fraction,
        "weighted_island_fraction": island_fraction,
        "relief_mismatch_m": relief_mismatch,
        "terrain_mismatch_fraction": terrain_mismatch,
        "tpi_status": "NOT_MEASURED",
    }


def _neighbor_fingerprint(ids: Sequence[str]) -> str:
    canonical = "".join(f"{len(value)}:{value};" for value in sorted(ids)).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def vectorized_structure_diagnostics(
    *,
    elevations_m: Sequence[float],
    temperature_matrix_c: Any,
    distances_km: Sequence[float],
    neighbor_ids: Sequence[Any],
    target_id: Any,
) -> pd.DataFrame:
    """한 exact 기하의 여러 시각 OLS/WLS·4분류를 벡터화해 계산한다.

    첫 행은 Task2 scalar 함수와 교차검산하여 batch 최적화가 계산 의미를
    바꾸지 않았는지 즉시 확인한다.
    """

    elevation = np.asarray(elevations_m, dtype=float)
    distance = np.asarray(distances_km, dtype=float)
    matrix = np.asarray(temperature_matrix_c, dtype=float)
    if elevation.ndim != 1 or distance.ndim != 1 or matrix.ndim != 2:
        raise ValueError("elevation/distance must be vectors and temperature a matrix")
    n = len(elevation)
    if n < 3 or n > 10 or len(distance) != n or matrix.shape[1] != n:
        raise ValueError("vectorized slope inputs must share 3–10 exact neighbors")
    if not np.isfinite(elevation).all() or not np.isfinite(distance).all():
        raise ValueError("elevation and distance must be finite")
    if not np.isfinite(matrix).all():
        raise ValueError("temperature matrix must be finite")
    if np.any(distance <= 0):
        raise ValueError("structure regression requires positive distances")
    ids = validate_exact_neighbor_ids(target_id, neighbor_ids, expected_count=n)
    if not len(matrix):
        return pd.DataFrame()

    # 승인된 Task2 순수함수를 계산 오라클로 사용한다.
    oracle_ols = ols_temperature_elevation_slope(
        elevation, matrix[0], neighbor_ids=ids, target_id=str(target_id).strip()
    )
    oracle_wls = weighted_temperature_elevation_slope(
        elevation,
        matrix[0],
        distances_km=distance,
        neighbor_ids=ids,
        target_id=str(target_id).strip(),
    )
    classify_spatial_temperature_structure(oracle_ols, oracle_wls)

    x = elevation / 1000.0
    xc = x - x.mean()
    sxx = float(np.dot(xc, xc))
    if sxx <= 0:
        raise ValueError("elevation spread must be positive")
    y_mean = matrix.mean(axis=1)
    slopes = (matrix - y_mean[:, None]) @ xc / sxx
    intercepts = y_mean - slopes * x.mean()
    fitted = intercepts[:, None] + slopes[:, None] * x
    residual = matrix - fitted
    sse = np.einsum("ij,ij->i", residual, residual)
    df = n - 2
    standard_error = np.sqrt(np.maximum(sse / df, 0.0) / sxx)
    tcrit = _T_975[df]
    ci_low = slopes - tcrit * standard_error
    ci_high = slopes + tcrit * standard_error
    centered_y = matrix - y_mean[:, None]
    sst = np.einsum("ij,ij->i", centered_y, centered_y)
    r2 = np.where(sst > 0, 1.0 - sse / sst, np.nan)
    leverage = 1.0 / n + xc**2 / sxx
    if np.any(leverage >= 1):
        raise ValueError("HC3 leverage is not identifiable")
    adjusted = residual / (1.0 - leverage)[None, :]
    hc3_se = np.sqrt(
        np.maximum(
            np.sum((xc[None, :] ** 2) * (adjusted**2), axis=1) / (sxx**2),
            0.0,
        )
    )

    def auxiliary_diagnostics(power: int) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        raw = np.ones(n, dtype=float) if power == 0 else (float(distance.min()) / distance) ** power
        normalized = raw / raw.sum()
        x_mean = float(np.dot(normalized, x))
        y_mean_weighted = matrix @ normalized
        x_centered = x - x_mean
        denominator = float(np.dot(normalized, x_centered**2))
        if denominator <= 0:
            raise ValueError("weighted elevation spread must be positive")
        full_slopes = (
            ((matrix - y_mean_weighted[:, None]) * x_centered[None, :])
            @ normalized
            / denominator
        )
        effective_n = float(raw.sum() ** 2 / np.dot(raw, raw))
        elevation_sd = float(
            np.sqrt(
                np.dot(
                    normalized,
                    (elevation - np.dot(normalized, elevation)) ** 2,
                )
            )
        )
        leave_one = np.full((len(matrix), n), np.nan, dtype=float)
        for omitted in range(n):
            keep = np.arange(n) != omitted
            local_raw = raw[keep]
            local_weight = local_raw / local_raw.sum()
            local_x = x[keep]
            local_x_mean = float(np.dot(local_weight, local_x))
            local_xc = local_x - local_x_mean
            local_denominator = float(np.dot(local_weight, local_xc**2))
            if local_denominator <= 0:
                continue
            local_y = matrix[:, keep]
            local_y_mean = local_y @ local_weight
            leave_one[:, omitted] = (
                ((local_y - local_y_mean[:, None]) * local_xc[None, :])
                @ local_weight
                / local_denominator
            )
        full_sign = np.sign(full_slopes)
        valid = np.isfinite(leave_one) & (leave_one != 0)
        agreement = valid & (np.sign(leave_one) == full_sign[:, None]) & (full_sign[:, None] != 0)
        stable = valid.all(axis=1) & agreement.all(axis=1)
        stability = agreement.sum(axis=1) / float(n)
        return full_slopes, effective_n, elevation_sd, stable, stability

    auxiliary = {
        "none": auxiliary_diagnostics(0),
        "d1": auxiliary_diagnostics(1),
        "d2": auxiliary_diagnostics(2),
    }
    weighted_slopes, n_eff, weighted_sd, _, _ = auxiliary["d2"]
    span = float(elevation.max() - elevation.min())
    fingerprint = _neighbor_fingerprint(ids)

    if not np.isclose(slopes[0], oracle_ols["slope_c_per_km"], atol=1e-11, rtol=0):
        raise RuntimeError("vectorized OLS does not match Task2 scalar contract")
    if not np.isclose(
        weighted_slopes[0], oracle_wls["slope_c_per_km"], atol=1e-11, rtol=0
    ):
        raise RuntimeError("vectorized WLS does not match Task2 scalar contract")

    base_eligible = (
        (n >= 6) and (span >= 100.0) and (weighted_sd >= 50.0) and (n_eff >= 3.0)
    )
    span200_eligible = (
        (n >= 6) and (span >= 200.0) and (weighted_sd >= 50.0) and (n_eff >= 3.0)
    )
    neff4_eligible = (
        (n >= 6) and (span >= 100.0) and (weighted_sd >= 50.0) and (n_eff >= 4.0)
    )
    ols_only_eligible = (n >= 6) and (span >= 100.0)

    def state_array(
        eligible: bool,
        *,
        require_weighted: bool = True,
        require_loo_stable: bool = False,
    ) -> np.ndarray:
        state = np.full(len(matrix), "uncertain", dtype=object)
        if not eligible:
            state[:] = "ineligible"
            return state
        positive = (slopes > 0) & (ci_low > 0)
        negative = (slopes < 0) & (ci_high < 0)
        if require_weighted:
            positive &= weighted_slopes > 0
            negative &= weighted_slopes < 0
        if require_loo_stable:
            positive &= auxiliary["d2"][3]
            negative &= auxiliary["d2"][3]
        state[positive] = "inversion_like"
        state[negative] = "normal_negative"
        return state

    reasons: list[str] = []
    if n < 6:
        reasons.append("n_below_threshold")
    if span < 100:
        reasons.append("elevation_span_below_threshold")
    if weighted_sd < 50:
        reasons.append("weighted_elevation_sd_below_threshold")
    if n_eff < 3:
        reasons.append("n_eff_below_threshold")
    primary_reason = "eligible" if not reasons else ";".join(reasons)

    return pd.DataFrame(
        {
            "REGRESSION_N": np.full(len(matrix), n, dtype=int),
            "NEIGHBOR_FINGERPRINT": np.full(len(matrix), fingerprint, dtype=object),
            "ELEVATION_SPAN_M": np.full(len(matrix), span),
            "WEIGHTED_ELEVATION_SD_M": np.full(len(matrix), weighted_sd),
            "N_EFF": np.full(len(matrix), n_eff),
            "OLS_SLOPE_C_PER_KM": slopes,
            "OLS_CI95_LOW_C_PER_KM": ci_low,
            "OLS_CI95_HIGH_C_PER_KM": ci_high,
            "OLS_R2": r2,
            "OLS_HC3_SE_C_PER_KM": hc3_se,
            "OLS_HC3_CI95_LOW_C_PER_KM": slopes - tcrit * hc3_se,
            "OLS_HC3_CI95_HIGH_C_PER_KM": slopes + tcrit * hc3_se,
            "WLS_SLOPE_C_PER_KM": weighted_slopes,
            "AUX_SLOPE_NONE_C_PER_KM": auxiliary["none"][0],
            "AUX_SLOPE_D1_C_PER_KM": auxiliary["d1"][0],
            "AUX_SLOPE_D2_C_PER_KM": auxiliary["d2"][0],
            "WEIGHTED_ELEVATION_SD_NONE_M": np.full(len(matrix), auxiliary["none"][2]),
            "WEIGHTED_ELEVATION_SD_D1_M": np.full(len(matrix), auxiliary["d1"][2]),
            "WEIGHTED_ELEVATION_SD_D2_M": np.full(len(matrix), auxiliary["d2"][2]),
            "N_EFF_NONE": np.full(len(matrix), auxiliary["none"][1]),
            "N_EFF_D1": np.full(len(matrix), auxiliary["d1"][1]),
            "N_EFF_D2": np.full(len(matrix), auxiliary["d2"][1]),
            "LOO_STABLE_NONE": auxiliary["none"][3],
            "LOO_STABLE_D1": auxiliary["d1"][3],
            "LOO_STABLE_D2": auxiliary["d2"][3],
            "LOO_SIGN_STABILITY_NONE": auxiliary["none"][4],
            "LOO_SIGN_STABILITY_D1": auxiliary["d1"][4],
            "LOO_SIGN_STABILITY_D2": auxiliary["d2"][4],
            # 주 판정은 81-setting 표의 사전 고정 primary와 동일하게
            # 거리제곱 기울기의 leave-one-neighbor 부호 안정성까지 요구한다.
            "STATE_PRIMARY": state_array(
                base_eligible, require_loo_stable=True
            ),
            "STATE_PRIMARY_REASON": np.full(len(matrix), primary_reason, dtype=object),
            "STATE_SPAN200": state_array(span200_eligible),
            "STATE_NEFF4": state_array(neff4_eligible),
            "STATE_OLS_ONLY": state_array(ols_only_eligible, require_weighted=False),
        }
    )


def _sensitivity_settings() -> list[dict[str, Any]]:
    """사전 고정된 81개 공간구조 민감도 setting을 결정론적 순서로 반환한다."""

    rows: list[dict[str, Any]] = []
    for minimum_n, minimum_span, minimum_sd, auxiliary_weight in product(
        SENSITIVITY_MIN_N,
        SENSITIVITY_MIN_SPAN_M,
        SENSITIVITY_MIN_WEIGHTED_SD_M,
        SENSITIVITY_AUX_WEIGHTS,
    ):
        rows.append(
            {
                "SETTING_ID": (
                    f"n{minimum_n}_span{minimum_span}_wsd{minimum_sd}_w{auxiliary_weight}"
                ),
                "MIN_N": minimum_n,
                "MIN_SPAN_M": minimum_span,
                "MIN_WEIGHTED_SD_M": minimum_sd,
                "AUX_WEIGHT": auxiliary_weight,
            }
        )
    return rows


def _sensitivity_state_for_setting(
    frame: pd.DataFrame, setting: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """한 setting의 4상태와 배제/불확실 사유 mask를 시간축으로 계산한다."""

    suffix = str(setting["AUX_WEIGHT"]).upper()
    required = [
        "REPRODUCED",
        "REGRESSION_N",
        "ELEVATION_SPAN_M",
        f"WEIGHTED_ELEVATION_SD_{suffix}_M",
        f"N_EFF_{suffix}",
        "OLS_SLOPE_C_PER_KM",
        "OLS_CI95_LOW_C_PER_KM",
        "OLS_CI95_HIGH_C_PER_KM",
        f"AUX_SLOPE_{suffix}_C_PER_KM",
        f"LOO_STABLE_{suffix}",
    ]
    _require_columns(frame, required, "structure sensitivity hourly")
    reproduced = frame["REPRODUCED"].astype(bool).to_numpy()
    n_values = pd.to_numeric(frame["REGRESSION_N"], errors="coerce").to_numpy(float)
    span = pd.to_numeric(frame["ELEVATION_SPAN_M"], errors="coerce").to_numpy(float)
    weighted_sd = pd.to_numeric(
        frame[f"WEIGHTED_ELEVATION_SD_{suffix}_M"], errors="coerce"
    ).to_numpy(float)
    n_eff = pd.to_numeric(frame[f"N_EFF_{suffix}"], errors="coerce").to_numpy(float)
    ols = pd.to_numeric(frame["OLS_SLOPE_C_PER_KM"], errors="coerce").to_numpy(float)
    ci_low = pd.to_numeric(frame["OLS_CI95_LOW_C_PER_KM"], errors="coerce").to_numpy(float)
    ci_high = pd.to_numeric(frame["OLS_CI95_HIGH_C_PER_KM"], errors="coerce").to_numpy(float)
    auxiliary = pd.to_numeric(
        frame[f"AUX_SLOPE_{suffix}_C_PER_KM"], errors="coerce"
    ).to_numpy(float)
    loo_stable = frame[f"LOO_STABLE_{suffix}"].fillna(False).astype(bool).to_numpy()

    finite_geometry = np.isfinite(n_values) & np.isfinite(span) & np.isfinite(weighted_sd) & np.isfinite(n_eff)
    finite_slopes = np.isfinite(ols) & np.isfinite(ci_low) & np.isfinite(ci_high) & np.isfinite(auxiliary)
    fail_n = reproduced & (~np.isfinite(n_values) | (n_values < float(setting["MIN_N"])))
    fail_span = reproduced & (~np.isfinite(span) | (span < float(setting["MIN_SPAN_M"])))
    fail_sd = reproduced & (
        ~np.isfinite(weighted_sd) | (weighted_sd < float(setting["MIN_WEIGHTED_SD_M"]))
    )
    fail_neff = reproduced & (~np.isfinite(n_eff) | (n_eff < 3.0))
    eligible = reproduced & finite_geometry & finite_slopes & ~fail_n & ~fail_span & ~fail_sd & ~fail_neff
    positive_ci = (ols > 0) & (ci_low > 0)
    negative_ci = (ols < 0) & (ci_high < 0)
    sign_agree = ((ols > 0) & (auxiliary > 0)) | ((ols < 0) & (auxiliary < 0))
    states = np.full(len(frame), "ineligible", dtype=object)
    states[eligible] = "uncertain"
    states[eligible & positive_ci & (auxiliary > 0) & loo_stable] = "inversion_like"
    states[eligible & negative_ci & (auxiliary < 0) & loo_stable] = "normal_negative"
    masks = {
        "eligible": eligible,
        "fail_exact": ~reproduced,
        "fail_nonfinite": reproduced & (~finite_geometry | ~finite_slopes),
        "fail_n": fail_n,
        "fail_span": fail_span,
        "fail_sd": fail_sd,
        "fail_neff": fail_neff,
        "ci_crosses": eligible & ~(positive_ci | negative_ci),
        "sign_disagree": eligible & ~sign_agree,
        "loo_unstable": eligible & ~loo_stable,
    }
    return states, masks


def build_structure_sensitivity_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """한 관측소 시간진단을 81 setting×4상태 long-form 요약으로 변환한다."""

    _require_columns(diagnostics, ["STN", "REPRODUCED"], "structure diagnostics")
    stations = diagnostics["STN"].astype(str).str.strip().unique()
    if len(stations) != 1:
        raise ValueError("structure sensitivity accepts exactly one station")
    station_id = _normalize_station_id(stations[0], field="sensitivity STN")
    rows: list[dict[str, Any]] = []
    total = int(len(diagnostics))
    reproduced_mask = diagnostics["REPRODUCED"].astype(bool).to_numpy()
    reproduced = int(reproduced_mask.sum())
    for setting in _sensitivity_settings():
        states, masks = _sensitivity_state_for_setting(diagnostics, setting)
        eligible_hours = int(masks["eligible"].sum())
        inversion_hours = int(np.count_nonzero(states == "inversion_like"))
        suffix = str(setting["AUX_WEIGHT"]).upper()
        ols = pd.to_numeric(diagnostics["OLS_SLOPE_C_PER_KM"], errors="coerce")
        aux = pd.to_numeric(
            diagnostics[f"AUX_SLOPE_{suffix}_C_PER_KM"], errors="coerce"
        )
        stability_column = f"LOO_SIGN_STABILITY_{suffix}"
        stability = (
            pd.to_numeric(diagnostics[stability_column], errors="coerce")
            if stability_column in diagnostics
            else diagnostics[f"LOO_STABLE_{suffix}"].astype(float)
        )
        for state in STRUCTURE_STATES:
            # Exact-none 재현에 실패한 시간은 민감도 상태의 모집단이 아니다.
            # 별도 실패 수에는 남기되 네 상태의 분모/합계에서는 제외한다.
            state_mask = (states == state) & reproduced_mask
            state_count = int(np.count_nonzero(state_mask))
            rows.append(
                {
                    "STN": station_id,
                    **setting,
                    "STRUCTURE_STATE": state,
                    "TOTAL_PAIRED_HOURS": total,
                    "REPRODUCED_HOURS": reproduced,
                    "N_HOURS": state_count,
                    "STATE_FRACTION_OF_REPRODUCED": state_count / reproduced if reproduced else np.nan,
                    "ELIGIBLE_HOURS": eligible_hours,
                    "ELIGIBLE_FRACTION_OF_REPRODUCED": eligible_hours / reproduced if reproduced else np.nan,
                    "INVERSION_FRACTION_OF_ELIGIBLE": inversion_hours / eligible_hours if eligible_hours else np.nan,
                    "N_EXACT_REPRODUCTION_FAILURE": int(masks["fail_exact"].sum()),
                    "N_NONFINITE_ELEVATION": int(masks["fail_nonfinite"].sum()),
                    "N_FAIL_MIN_N": int(masks["fail_n"].sum()),
                    "N_FAIL_SPAN": int(masks["fail_span"].sum()),
                    "N_FAIL_WEIGHTED_SD": int(masks["fail_sd"].sum()),
                    "N_FAIL_N_EFF": int(masks["fail_neff"].sum()),
                    "N_OLS_CI_CROSSES_ZERO": int(masks["ci_crosses"].sum()),
                    "N_OLS_AUX_SIGN_DISAGREE": int(masks["sign_disagree"].sum()),
                    "N_LOO_SIGN_UNSTABLE": int(masks["loo_unstable"].sum()),
                    "MEDIAN_OLS_SLOPE_C_PER_KM": float(ols[masks["eligible"]].median()) if eligible_hours else np.nan,
                    "MEDIAN_AUX_SLOPE_C_PER_KM": float(aux[masks["eligible"]].median()) if eligible_hours else np.nan,
                    "MEDIAN_LOO_SIGN_STABILITY": float(stability[masks["eligible"]].median()) if eligible_hours else np.nan,
                    "ANALYSIS_POPULATION": "all_stations_before_subset_join",
                }
            )
    return pd.DataFrame(rows, columns=SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS)


def _matched_descriptive_point(frame: pd.DataFrame) -> dict[str, Any]:
    """공통 월×태양 층을 동일 가중해 두 상태의 signed-BIAS 효과를 계산한다."""

    work = frame.loc[
        frame["STRUCTURE_STATE"].isin(["inversion_like", "normal_negative"])
    ].copy()
    required_states = {"inversion_like", "normal_negative"}
    if not required_states.issubset(set(work["STRUCTURE_STATE"])):
        return {"available": False, "reason": "missing_required_state"}
    work["TM"] = pd.to_datetime(work["TM"], errors="raise")
    work["MONTH_KEY"] = work["TM"].dt.to_period("M").astype(str)
    strata = ["MONTH_KEY", "SOLAR_STATE"]
    common = None
    for state in ["inversion_like", "normal_negative"]:
        keys = set(map(tuple, work.loc[work["STRUCTURE_STATE"].eq(state), strata].drop_duplicates().to_numpy()))
        common = keys if common is None else common & keys
    if not common:
        return {"available": False, "reason": "no_common_month_solar_strata"}
    common_frame = pd.DataFrame(sorted(common), columns=strata)
    matched = work.merge(common_frame, on=strata, how="inner", validate="many_to_one")
    effects: dict[str, float] = {}
    for state in ["inversion_like", "normal_negative"]:
        state_frame = matched.loc[matched["STRUCTURE_STATE"].eq(state)]
        per_stratum = state_frame.groupby(strata, sort=True)[["error_none", "error_fixed"]].mean()
        bias_none = float(per_stratum["error_none"].mean())
        bias_fixed = float(per_stratum["error_fixed"].mean())
        effects[state] = abs(bias_fixed) - abs(bias_none)
    inversion = matched.loc[matched["STRUCTURE_STATE"].eq("inversion_like")]
    normal = matched.loc[matched["STRUCTURE_STATE"].eq("normal_negative")]
    return {
        "available": True,
        "reason": "eligible",
        "n_strata": len(common),
        "n_inversion_hours": len(inversion),
        "n_normal_hours": len(normal),
        "n_inversion_days": inversion["TM"].dt.normalize().nunique(),
        "n_normal_days": normal["TM"].dt.normalize().nunique(),
        "n_inversion_months": inversion["TM"].dt.to_period("M").nunique(),
        "n_normal_months": normal["TM"].dt.to_period("M").nunique(),
        "inversion_effect": effects["inversion_like"],
        "normal_effect": effects["normal_negative"],
        "contrast": effects["inversion_like"] - effects["normal_negative"],
    }


def build_matched_state_outputs(
    hourly: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 20260827,
) -> dict[str, pd.DataFrame]:
    """81 공간 setting의 9 coverage 기술민감도와 주 30일·6개월 추론행을 만든다."""

    _require_columns(hourly, ["STN", "TM", "SOLAR_STATE", "REPRODUCED", "error_none", "error_fixed"], "matched sensitivity hourly")
    station_ids = hourly["STN"].astype(str).str.strip().unique()
    if len(station_ids) != 1:
        raise ValueError("matched sensitivity accepts exactly one station")
    station_id = _normalize_station_id(station_ids[0], field="matched sensitivity STN")
    rows: list[dict[str, Any]] = []
    primary_hourly: pd.DataFrame | None = None
    point_cache: dict[bytes, dict[str, Any]] = {}
    for setting in _sensitivity_settings():
        states, _ = _sensitivity_state_for_setting(hourly, setting)
        state_key = "\x1f".join(map(str, states)).encode("utf-8")
        point = point_cache.get(state_key)
        if point is None:
            local_point = hourly[
                ["TM", "SOLAR_STATE", "REPRODUCED", "error_none", "error_fixed"]
            ].copy()
            local_point["STRUCTURE_STATE"] = states
            point = _matched_descriptive_point(
                local_point.loc[local_point["REPRODUCED"].astype(bool)]
            )
            point_cache[state_key] = point
        if setting["SETTING_ID"] == PRIMARY_SETTING_ID:
            primary_hourly = hourly.copy()
            primary_hourly["STATE_PRIMARY"] = states
        for minimum_days, minimum_months in product(
            SENSITIVITY_MIN_DAYS, SENSITIVITY_MIN_MONTHS
        ):
            available = bool(point.get("available", False))
            eligible = bool(
                available
                and point["n_inversion_days"] >= minimum_days
                and point["n_normal_days"] >= minimum_days
                and point["n_inversion_months"] >= minimum_months
                and point["n_normal_months"] >= minimum_months
            )
            if not available:
                drop_reason = str(point.get("reason", "missing_required_state"))
            elif (
                point["n_inversion_days"] < minimum_days
                or point["n_normal_days"] < minimum_days
            ):
                drop_reason = "insufficient_unique_days"
            elif (
                point["n_inversion_months"] < minimum_months
                or point["n_normal_months"] < minimum_months
            ):
                drop_reason = "insufficient_months"
            else:
                drop_reason = "eligible"
            rows.append(
                {
                    "STN": station_id,
                    **setting,
                    "COVERAGE_ID": f"d{minimum_days}_m{minimum_months}",
                    "MIN_DAYS_PER_STATE": minimum_days,
                    "MIN_MONTHS_PER_STATE": minimum_months,
                    "N_COMMON_STRATA": int(point.get("n_strata", 0)),
                    "N_INVERSION_HOURS": int(point.get("n_inversion_hours", 0)),
                    "N_NORMAL_HOURS": int(point.get("n_normal_hours", 0)),
                    "N_INVERSION_DAYS": int(point.get("n_inversion_days", 0)),
                    "N_NORMAL_DAYS": int(point.get("n_normal_days", 0)),
                    "N_INVERSION_MONTHS": int(point.get("n_inversion_months", 0)),
                    "N_NORMAL_MONTHS": int(point.get("n_normal_months", 0)),
                    "DELTA_ABS_BIAS_INVERSION": point.get("inversion_effect", np.nan),
                    "DELTA_ABS_BIAS_NORMAL": point.get("normal_effect", np.nan),
                    "CONTRAST_INV_MINUS_NORMAL": point.get("contrast", np.nan),
                    "BASE_MATCHED_DATA_AVAILABLE": available,
                    "COVERAGE_ELIGIBLE": eligible,
                    "DROP_REASON": drop_reason,
                    "SUBSET_MEANINGFUL_WORSENING": False,
                    "SUBSET_HIGH_AFTER": False,
                    "ANALYSIS_POPULATION": "all_stations_before_subset_join",
                }
            )
    sensitivity = pd.DataFrame(rows, columns=MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS)
    if primary_hourly is None:
        raise RuntimeError("primary sensitivity setting was not generated")
    primary = build_matched_state_contrast(primary_hourly, n_boot=n_boot, seed=seed)
    return {"primary": primary, "sensitivity": sensitivity}


def solar_elevation_noaa(
    timestamps: Iterable[Any],
    *,
    latitude_deg: float,
    longitude_deg: float,
) -> np.ndarray:
    """NOAA 근사식으로 굴절 전 태양고도(도)를 계산한다.

    timezone 없는 ``TM``은 KST(UTC+9)로 해석한다. timezone이 있는 입력은 동일
    순간의 KST로 변환한다.
    """

    latitude = float(latitude_deg)
    longitude = float(longitude_deg)
    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("latitude and longitude must be finite")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude are out of range")
    index = pd.DatetimeIndex(timestamps)
    if index.isna().any():
        raise ValueError("timestamps must not contain NaT")
    if index.tz is None:
        local = index.tz_localize("Asia/Seoul")
    else:
        local = index.tz_convert("Asia/Seoul")

    day = local.dayofyear.to_numpy(dtype=float)
    hour = (
        local.hour.to_numpy(dtype=float)
        + local.minute.to_numpy(dtype=float) / 60.0
        + local.second.to_numpy(dtype=float) / 3600.0
        + local.microsecond.to_numpy(dtype=float) / 3.6e9
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    equation_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    # KST 표준 자오선 135°E: timezone offset 9시간.
    time_offset = equation_time + 4.0 * longitude - 60.0 * 9.0
    true_solar_minutes = hour * 60.0 + time_offset
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = math.radians(latitude)
    cosine_zenith = (
        math.sin(latitude_rad) * np.sin(declination)
        + math.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    zenith = np.arccos(np.clip(cosine_zenith, -1.0, 1.0))
    return 90.0 - np.rad2deg(zenith)


def classify_solar_state(solar_elevation_deg: Any) -> np.ndarray:
    """태양고도 기술 상태: bright >0°, transition -6~0°, dark <-6°."""

    values = np.asarray(solar_elevation_deg, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("solar elevation must be finite")
    return np.where(values > 0.0, "bright", np.where(values < -6.0, "dark", "transition"))


def _normalize_pair_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, ["STN", "TM"], name)
    work = frame.copy()
    work["STN"] = work["STN"].map(
        lambda value: _normalize_station_id(value, field=f"{name} STN")
    )
    work["TM"] = pd.to_datetime(work["TM"], errors="raise")
    if work["TM"].isna().any():
        raise ValueError(f"{name} TM must not contain NaT")
    if work.duplicated(["STN", "TM"]).any():
        raise ValueError(f"{name} must be unique by (STN, TM)")
    return work


def merge_weather_conditions(
    pairs: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    min_coverage: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """target의 같은 ``(STN,TM)`` 기상조건을 결합하고 paired coverage를 계산한다."""

    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must lie in [0, 1]")
    paired = _normalize_pair_keys(pairs, "pairs")
    measured = _normalize_pair_keys(weather, "weather")
    _require_columns(measured, ["WS10", "WS1", "RN_60m", "HM", "TD"], "weather")
    for column in ["WS10", "WS1", "RN_60m", "HM", "TD"]:
        measured[column] = pd.to_numeric(measured[column], errors="coerce")
    for column in ["WS10", "WS1", "RN_60m", "HM"]:
        if (measured[column].dropna() < 0).any():
            raise ValueError(f"{column} must be non-negative when observed")

    joined = paired.merge(
        measured[["STN", "TM", "WS10", "WS1", "RN_60m", "HM", "TD"]],
        on=["STN", "TM"],
        how="left",
        validate="one_to_one",
    )
    joined["WS_EFFECTIVE"] = joined["WS10"].where(joined["WS10"].notna(), joined["WS1"])
    joined["WS_SOURCE"] = np.select(
        [joined["WS10"].notna(), joined["WS1"].notna()],
        ["WS10", "WS1"],
        default="missing",
    )
    joined["WIND_QUARTILE"] = pd.Series(pd.NA, index=joined.index, dtype="object")
    for _, indices in joined.groupby("STN", sort=False).groups.items():
        index = pd.Index(indices)
        values = joined.loc[index, "WS_EFFECTIVE"]
        valid = values.notna()
        if valid.any():
            ranks = values.loc[valid].rank(method="average", pct=True)
            labels = np.ceil(ranks.to_numpy(float) * 4.0).astype(int)
            labels = np.clip(labels, 1, 4)
            joined.loc[ranks.index, "WIND_QUARTILE"] = [f"Q{value}" for value in labels]
    joined["PRECIPITATION_STATE"] = np.where(
        joined["RN_60m"].isna(),
        pd.NA,
        np.where(joined["RN_60m"].eq(0), "non_precipitation", "precipitation"),
    )

    coverage_rows = []
    for station_id, group in joined.groupby("STN", sort=False):
        denominator = len(group)
        coverage = {
            "STN": station_id,
            "PAIRED_HOURS": int(denominator),
            "WIND_VALID_HOURS": int(group["WS_EFFECTIVE"].notna().sum()),
            "RN_VALID_HOURS": int(group["RN_60m"].notna().sum()),
            "HM_VALID_HOURS": int(group["HM"].notna().sum()),
            "TD_VALID_HOURS": int(group["TD"].notna().sum()),
        }
        for short, source in [
            ("WIND", "WS_EFFECTIVE"),
            ("RN", "RN_60m"),
            ("HM", "HM"),
            ("TD", "TD"),
        ]:
            coverage[f"{short}_COVERAGE"] = (
                float(group[source].notna().sum() / denominator) if denominator else 0.0
            )
        coverage["WEATHER_PRIMARY_ELIGIBLE"] = bool(
            denominator
            and coverage["WIND_COVERAGE"] >= min_coverage
            and coverage["RN_COVERAGE"] >= min_coverage
            and coverage["HM_COVERAGE"] >= min_coverage
            and coverage["TD_COVERAGE"] >= min_coverage
        )
        coverage["WEATHER_DROP_REASON"] = (
            "eligible"
            if coverage["WEATHER_PRIMARY_ELIGIBLE"]
            else "paired_weather_coverage_below_80pct"
        )
        coverage_rows.append(coverage)
    return {
        "hourly": joined,
        "coverage": pd.DataFrame(coverage_rows),
    }


def _bias_metrics(group: pd.DataFrame) -> dict[str, float]:
    """같은 mask에서 signed BIAS를 먼저 평균하고 MAE를 별도로 계산한다."""

    if group.empty:
        return {
            "BIAS_none": np.nan,
            "BIAS_fixed": np.nan,
            "DELTA_ABS_BIAS": np.nan,
            "MAE_none": np.nan,
            "MAE_fixed": np.nan,
            "DELTA_MAE": np.nan,
        }
    none = pd.to_numeric(group["error_none"], errors="coerce").to_numpy(float)
    fixed = pd.to_numeric(group["error_fixed"], errors="coerce").to_numpy(float)
    mask = np.isfinite(none) & np.isfinite(fixed)
    if not mask.any():
        return {
            "BIAS_none": np.nan,
            "BIAS_fixed": np.nan,
            "DELTA_ABS_BIAS": np.nan,
            "MAE_none": np.nan,
            "MAE_fixed": np.nan,
            "DELTA_MAE": np.nan,
        }
    none = none[mask]
    fixed = fixed[mask]
    bias_none = float(none.mean())
    bias_fixed = float(fixed.mean())
    mae_none = float(np.abs(none).mean())
    mae_fixed = float(np.abs(fixed).mean())
    return {
        "BIAS_none": bias_none,
        "BIAS_fixed": bias_fixed,
        "DELTA_ABS_BIAS": abs(bias_fixed) - abs(bias_none),
        "MAE_none": mae_none,
        "MAE_fixed": mae_fixed,
        "DELTA_MAE": mae_fixed - mae_none,
    }


def _condition_domains(hourly: pd.DataFrame) -> list[tuple[str, list[Any]]]:
    # 고정 domain으로 0개 cell도 보존한다.
    return [
        ("ALL", ["all"]),
        ("SOLAR_STATE", ["bright", "transition", "dark"]),
        ("WIND_QUARTILE", ["Q1", "Q2", "Q3", "Q4"]),
        ("PRECIPITATION_STATE", ["non_precipitation", "precipitation"]),
    ]


def build_state_condition_bias(
    hourly: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    min_days: int = 30,
    min_months: int = 6,
) -> pd.DataFrame:
    """관측소×민감도×공간구조×조건의 완전한 BIAS skeleton을 만든다."""

    required = [
        "STN",
        "TM",
        "REPRODUCED",
        *STRUCTURE_CONFIGS.values(),
        "SOLAR_STATE",
        "WIND_QUARTILE",
        "PRECIPITATION_STATE",
        "error_none",
        "error_fixed",
    ]
    _require_columns(hourly, required, "hourly")
    if hourly.empty:
        return pd.DataFrame(columns=STATE_CONDITION_COLUMNS)
    work = hourly.copy()
    work["STN"] = work["STN"].map(
        lambda value: _normalize_station_id(value, field="hourly STN")
    )
    work["TM"] = pd.to_datetime(work["TM"], errors="raise")
    work["error_none"] = pd.to_numeric(work["error_none"], errors="coerce")
    work["error_fixed"] = pd.to_numeric(work["error_fixed"], errors="coerce")
    if work[["error_none", "error_fixed"]].isna().any().any():
        raise ValueError("paired error columns must be finite")
    if not np.isfinite(work[["error_none", "error_fixed"]].to_numpy(float)).all():
        raise ValueError("paired error columns must be finite")

    coverage_lookup: dict[str, pd.Series] = {}
    if isinstance(coverage, pd.DataFrame) and len(coverage):
        cov = coverage.copy()
        if "STN" not in cov.columns:
            stations = work["STN"].drop_duplicates().tolist()
            if len(stations) != 1 or len(cov) != 1:
                raise ValueError("coverage without STN is only valid for one station")
            cov["STN"] = stations[0]
        cov["STN"] = cov["STN"].map(
            lambda value: _normalize_station_id(value, field="coverage STN")
        )
        if cov["STN"].duplicated().any():
            raise ValueError("coverage must contain one row per station")
        coverage_lookup = {str(row.STN): row for row in cov.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    for station_id, station in work.groupby("STN", sort=True):
        reproduced = station.loc[station["REPRODUCED"].astype(bool)].copy()
        cov_record = coverage_lookup.get(station_id)
        if cov_record is None:
            wind_coverage = rn_coverage = 0.0
        else:
            wind_coverage = float(getattr(cov_record, "WIND_COVERAGE", 0.0))
            rn_coverage = float(getattr(cov_record, "RN_COVERAGE", 0.0))

        for config, state_column in STRUCTURE_CONFIGS.items():
            for state in STRUCTURE_STATES:
                state_data = reproduced.loc[reproduced[state_column].eq(state)].copy()
                for condition_variable, levels in _condition_domains(reproduced):
                    for level in levels:
                        if condition_variable == "ALL":
                            group = state_data
                        else:
                            group = state_data.loc[state_data[condition_variable].eq(level)]
                        metrics = _bias_metrics(group)
                        n_hours = int(len(group))
                        n_days = int(group["TM"].dt.normalize().nunique()) if n_hours else 0
                        n_months = (
                            int(group["TM"].dt.to_period("M").nunique()) if n_hours else 0
                        )
                        n_strata = (
                            int(
                                group.assign(__MONTH=group["TM"].dt.to_period("M"))[
                                    ["__MONTH", "SOLAR_STATE"]
                                ]
                                .drop_duplicates()
                                .shape[0]
                            )
                            if n_hours
                            else 0
                        )
                        coverage_eligible = bool(
                            n_days >= min_days and n_months >= min_months
                        )
                        if condition_variable == "WIND_QUARTILE":
                            coverage_eligible = coverage_eligible and wind_coverage >= 0.8
                        if condition_variable == "PRECIPITATION_STATE":
                            coverage_eligible = coverage_eligible and rn_coverage >= 0.8
                        drop_reason = "eligible" if coverage_eligible else "insufficient_coverage"
                        # 상태 단독 CI는 주 질문(역전 유사-정상 상태의 효과 차이)을
                        # 검정하지 않으므로 추론값을 두지 않는다. 비교 추론은 별도
                        # matched_state_contrast 표의 공통 날짜 block으로만 수행한다.
                        ci_low, ci_high, replicates = np.nan, np.nan, 0
                        rows.append(
                            {
                                "STN": station_id,
                                "CONFIG": config,
                                "STRUCTURE_STATE": state,
                                "CONDITION_VARIABLE": condition_variable,
                                "CONDITION_LEVEL": level,
                                "N_HOURS": n_hours,
                                "N_DAYS": n_days,
                                "N_MONTHS": n_months,
                                "N_STRATA": n_strata,
                                **metrics,
                                "BLOCK7_CI_LOW": ci_low,
                                "BLOCK7_CI_HIGH": ci_high,
                                "BLOCK7_REPLICATES": replicates,
                                "COVERAGE_ELIGIBLE": coverage_eligible,
                                "DROP_REASON": drop_reason,
                                "EVIDENCE_GRADE": (
                                    "문헌과 일치하나 직접 미측정"
                                    if coverage_eligible
                                    else "현재 자료에서 지지되지 않음"
                                ),
                                "LIMITATION": (
                                    "regional_inversion_like_not_vertical_profile;"
                                    "same_predictor_shared_by_none_and_fixed;"
                                    "state_row_descriptive_only_inference_in_matched_contrast"
                                ),
                                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                                "SUBSET_MEANINGFUL_WORSENING": False,
                                "SUBSET_HIGH_AFTER": False,
                            }
                        )
    return pd.DataFrame(rows, columns=STATE_CONDITION_COLUMNS)


def build_matched_state_contrast(
    hourly: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 20260827,
) -> pd.DataFrame:
    """같은 월×태양상태와 공통 날짜 block에서 두 공간상태 효과를 비교한다."""

    required = [
        "STN",
        "TM",
        "SOLAR_STATE",
        "REPRODUCED",
        "STATE_PRIMARY",
        "error_none",
        "error_fixed",
    ]
    _require_columns(hourly, required, "hourly")
    if hourly.empty:
        return pd.DataFrame(columns=MATCHED_STATE_CONTRAST_COLUMNS)
    station_ids = hourly["STN"].astype(str).str.strip().unique()
    if len(station_ids) != 1:
        raise ValueError("matched contrast accepts exactly one station")
    station_id = _normalize_station_id(station_ids[0], field="matched contrast STN")
    reproduced = hourly.loc[hourly["REPRODUCED"].astype(bool)].copy()
    rows = []
    for config_index, (config, state_column) in enumerate((("primary", "STATE_PRIMARY"),)):
        frame = reproduced[
            ["STN", "TM", "SOLAR_STATE", state_column, "error_none", "error_fixed"]
        ].rename(columns={state_column: "STRUCTURE_STATE"})
        if frame.empty:
            row = {
                "STN": station_id,
                "CONFIG": config,
                "SETTING_ID": PRIMARY_SETTING_ID,
                "N_STRATA": 0,
                "N_DAYS": 0,
                "N_MONTHS": 0,
                "DELTA_ABS_BIAS_INVERSION": np.nan,
                "DELTA_ABS_BIAS_NORMAL": np.nan,
                "CONTRAST_INV_MINUS_NORMAL": np.nan,
                "BLOCK7_CI_LOW": np.nan,
                "BLOCK7_CI_HIGH": np.nan,
                "BLOCK7_REQUESTED_REPLICATES": n_boot,
                "BLOCK7_VALID_REPLICATES": 0,
                "BLOCK7_VALID_FRACTION": 0.0,
                "BLOCK7_SEED": seed + int(station_id) * 17 + config_index * 1009,
                "BLOCK7_BLOCK_DAYS": 7,
                "STRATA_WEIGHTING": "equal_common_month_solar_strata",
                "ESTIMAND": "standardized_signed_bias_then_delta_abs_bias_contrast",
                "SAMPLING_SCHEME": "monthly_calendar_contiguous_fixed_length_blocks",
                "ANALYSIS_SUPPORT_CHECKED": False,
                "ANALYSIS_DATE_COUNT": 0,
                "SUPPORTED_ANALYSIS_DATE_COUNT": 0,
                "UNCOVERED_ANALYSIS_DATE_COUNT": 0,
                "ANALYSIS_DATE_SUPPORT_FRACTION": np.nan,
                "UNCOVERED_ANALYSIS_DATES_JSON": "[]",
                "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON": "[]",
                "COMPARISON_ELIGIBLE": False,
                "DROP_REASON": "no_reproduced_hours",
                "P_VALUE": np.nan,
                "Q_VALUE": np.nan,
                "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                "LIMITATION": "no_reproduced_hours",
                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                "SUBSET_MEANINGFUL_WORSENING": False,
                "SUBSET_HIGH_AFTER": False,
            }
        else:
            result = bootstrap_shared_state_delta_abs_bias_contrast(
                frame,
                n_replicates=n_boot,
                block_length_days=7,
                seed=seed + int(station_id) * 17 + config_index * 1009,
                min_unique_days_per_state=30,
                min_months_per_state=6,
            )
            if result.valid_replicates:
                values = result.replicate_contrasts
                # 관측 추정치 중심의 bootstrap 편차를 0 귀무와 비교하는 basic
                # 근사검정이다. 주 근거는 percentile CI이며 p/q는 보조로만 쓴다.
                deviations = values - float(result.point_contrast)
                p_value = float(
                    (
                        1
                        + np.count_nonzero(
                            np.abs(deviations) >= abs(float(result.point_contrast))
                        )
                    )
                    / (len(values) + 1)
                )
            else:
                p_value = np.nan
            row = {
                "STN": station_id,
                "CONFIG": config,
                "SETTING_ID": PRIMARY_SETTING_ID,
                "N_STRATA": result.common_strata_count,
                "N_DAYS": min(result.inversion_unique_days, result.normal_unique_days),
                "N_MONTHS": min(result.inversion_months, result.normal_months),
                "DELTA_ABS_BIAS_INVERSION": result.inversion_delta_abs_bias,
                "DELTA_ABS_BIAS_NORMAL": result.normal_delta_abs_bias,
                "CONTRAST_INV_MINUS_NORMAL": result.point_contrast,
                "BLOCK7_CI_LOW": result.ci_low,
                "BLOCK7_CI_HIGH": result.ci_high,
                "BLOCK7_REQUESTED_REPLICATES": result.requested_replicates,
                "BLOCK7_VALID_REPLICATES": result.valid_replicates,
                "BLOCK7_VALID_FRACTION": result.valid_replicate_fraction,
                "BLOCK7_SEED": result.seed,
                "BLOCK7_BLOCK_DAYS": result.block_length_days,
                "STRATA_WEIGHTING": result.strata_weighting,
                "ESTIMAND": result.estimand,
                "SAMPLING_SCHEME": result.sampling_scheme,
                "ANALYSIS_SUPPORT_CHECKED": result.analysis_support_checked,
                "ANALYSIS_DATE_COUNT": result.analysis_date_count,
                "SUPPORTED_ANALYSIS_DATE_COUNT": result.supported_analysis_date_count,
                "UNCOVERED_ANALYSIS_DATE_COUNT": result.uncovered_analysis_date_count,
                "ANALYSIS_DATE_SUPPORT_FRACTION": result.analysis_date_support_fraction,
                "UNCOVERED_ANALYSIS_DATES_JSON": json.dumps(
                    result.uncovered_analysis_dates,
                    ensure_ascii=False,
                ),
                "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON": json.dumps(
                    result.analysis_support_diagnostics,
                    ensure_ascii=False,
                ),
                "COMPARISON_ELIGIBLE": result.coverage_eligible,
                "DROP_REASON": result.reason,
                "P_VALUE": p_value if result.coverage_eligible else np.nan,
                "Q_VALUE": np.nan,
                "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                "LIMITATION": (
                    "regional_inversion_like_not_vertical_profile;"
                    "shared_predictor_and_month_solar_matched_dates;"
                    "basic_bootstrap_p_is_exploratory_CI_is_primary"
                ),
                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                "SUBSET_MEANINGFUL_WORSENING": False,
                "SUBSET_HIGH_AFTER": False,
            }
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCHED_STATE_CONTRAST_COLUMNS)


def _json_date_list(values: Iterable[Any]) -> list[str]:
    timestamps = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    return sorted({value.strftime("%Y-%m-%d") for value in timestamps})


def _same_number(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=atol))


def build_primary_eligibility_ledger(
    hourly: pd.DataFrame,
    matched_contrast: pd.DataFrame,
    primary_coverage: pd.DataFrame,
    *,
    source_population_sha256: str,
    analysis_config_sha256: str,
) -> pd.DataFrame:
    """primary eligibility를 시간 원표에서 독립적으로 고정한다.

    coverage/matched 두 파생표가 서로 같다는 이유만으로 family 멤버십을
    승인하지 않고, common month×solar×state 원표의 날짜 집합과 signed
    error 충분통계를 보존한다.
    """

    _require_columns(
        hourly,
        [
            "STN",
            "TM",
            "SOLAR_STATE",
            "REPRODUCED",
            "STATE_PRIMARY",
            "error_none",
            "error_fixed",
        ],
        "primary eligibility hourly",
    )
    _require_columns(
        matched_contrast,
        MATCHED_STATE_CONTRAST_COLUMNS,
        "primary matched contrast",
    )
    required_coverage = [
        "STN",
        "SETTING_ID",
        "COVERAGE_ID",
        "N_COMMON_STRATA",
        "N_INVERSION_HOURS",
        "N_NORMAL_HOURS",
        "N_INVERSION_DAYS",
        "N_NORMAL_DAYS",
        "N_INVERSION_MONTHS",
        "N_NORMAL_MONTHS",
        "DELTA_ABS_BIAS_INVERSION",
        "DELTA_ABS_BIAS_NORMAL",
        "CONTRAST_INV_MINUS_NORMAL",
        "BASE_MATCHED_DATA_AVAILABLE",
        "COVERAGE_ELIGIBLE",
        "DROP_REASON",
    ]
    _require_columns(primary_coverage, required_coverage, "primary coverage")
    if not _is_sha256_hex(source_population_sha256) or not _is_sha256_hex(
        analysis_config_sha256
    ):
        raise ValueError("primary ledger source/config identities must be SHA-256")
    station_ids = hourly["STN"].map(
        lambda value: _normalize_station_id(value, field="primary ledger STN")
    ).unique()
    if len(station_ids) != 1 or len(matched_contrast) != 1 or len(primary_coverage) != 1:
        raise ValueError("primary ledger requires exactly one station/matched/coverage row")
    station_id = station_ids[0]
    matched = matched_contrast.iloc[0]
    coverage = primary_coverage.iloc[0]
    if any(
        _normalize_station_id(value, field="primary ledger joined STN") != station_id
        for value in [matched["STN"], coverage["STN"]]
    ):
        raise ValueError("primary ledger station joins disagree")
    if (
        str(matched["SETTING_ID"]) != PRIMARY_SETTING_ID
        or str(coverage["SETTING_ID"]) != PRIMARY_SETTING_ID
        or str(coverage["COVERAGE_ID"]) != PRIMARY_COVERAGE_ID
    ):
        raise ValueError("primary ledger setting/coverage IDs disagree with contract")

    reproduced_all = hourly.loc[hourly["REPRODUCED"].astype(bool)].copy()
    reproduced_all["TM"] = pd.to_datetime(reproduced_all["TM"], errors="raise")
    reproduced_all["error_none"] = pd.to_numeric(
        reproduced_all["error_none"], errors="raise"
    )
    reproduced_all["error_fixed"] = pd.to_numeric(
        reproduced_all["error_fixed"], errors="raise"
    )
    reproduced_all["MONTH_KEY"] = reproduced_all["TM"].dt.to_period("M").astype(str)
    reproduced_all["DATE_KEY"] = reproduced_all["TM"].dt.strftime("%Y-%m-%d")
    work = reproduced_all.loc[
        reproduced_all["STATE_PRIMARY"].isin(
            ["inversion_like", "normal_negative"]
        )
    ].copy()
    strata_columns = ["MONTH_KEY", "SOLAR_STATE"]
    keys_by_state = {
        state: set(
            map(
                tuple,
                work.loc[work["STATE_PRIMARY"].eq(state), strata_columns]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        for state in ("inversion_like", "normal_negative")
    }
    common = sorted(keys_by_state["inversion_like"] & keys_by_state["normal_negative"])
    if common:
        common_frame = pd.DataFrame(common, columns=strata_columns)
        analysis = work.merge(common_frame, on=strata_columns, how="inner", validate="many_to_one")
    else:
        analysis = work.iloc[0:0].copy()

    state_rows: dict[str, pd.DataFrame] = {
        state: analysis.loc[analysis["STATE_PRIMARY"].eq(state)].copy()
        for state in ("inversion_like", "normal_negative")
    }
    effects: dict[str, float] = {}
    for state, state_frame in state_rows.items():
        if state_frame.empty:
            effects[state] = np.nan
            continue
        means = state_frame.groupby(strata_columns, sort=True)[
            ["error_none", "error_fixed"]
        ].mean()
        effects[state] = abs(float(means["error_fixed"].mean())) - abs(
            float(means["error_none"].mean())
        )
    contrast = effects["inversion_like"] - effects["normal_negative"]
    raw = {
        "N_COMMON_STRATA": len(common),
        "N_INVERSION_HOURS": len(state_rows["inversion_like"]),
        "N_NORMAL_HOURS": len(state_rows["normal_negative"]),
        "N_INVERSION_DAYS": state_rows["inversion_like"]["DATE_KEY"].nunique(),
        "N_NORMAL_DAYS": state_rows["normal_negative"]["DATE_KEY"].nunique(),
        "N_INVERSION_MONTHS": state_rows["inversion_like"]["MONTH_KEY"].nunique(),
        "N_NORMAL_MONTHS": state_rows["normal_negative"]["MONTH_KEY"].nunique(),
        "DELTA_ABS_BIAS_INVERSION": effects["inversion_like"],
        "DELTA_ABS_BIAS_NORMAL": effects["normal_negative"],
        "CONTRAST_INV_MINUS_NORMAL": contrast,
    }
    available = bool(common and all(len(frame) for frame in state_rows.values()))
    coverage_eligible = bool(
        available
        and raw["N_INVERSION_DAYS"] >= 30
        and raw["N_NORMAL_DAYS"] >= 30
        and raw["N_INVERSION_MONTHS"] >= 6
        and raw["N_NORMAL_MONTHS"] >= 6
    )
    if not available:
        required_states = {"inversion_like", "normal_negative"}
        coverage_reason = (
            "missing_required_state"
            if not required_states.issubset(set(work["STATE_PRIMARY"]))
            else "no_common_month_solar_strata"
        )
    elif min(raw["N_INVERSION_DAYS"], raw["N_NORMAL_DAYS"]) < 30:
        coverage_reason = "insufficient_unique_days"
    elif min(raw["N_INVERSION_MONTHS"], raw["N_NORMAL_MONTHS"]) < 6:
        coverage_reason = "insufficient_months"
    else:
        coverage_reason = "eligible"

    for column, value in raw.items():
        published = coverage[column]
        if column.startswith("N_"):
            if isinstance(published, (bool, np.bool_)) or int(published) != int(value):
                raise ValueError(f"primary coverage differs from raw ledger: {column}")
        elif not _same_number(published, value):
            raise ValueError(f"primary coverage differs from raw ledger: {column}")
    if bool(coverage["BASE_MATCHED_DATA_AVAILABLE"]) != available:
        raise ValueError("primary coverage availability differs from raw ledger")
    if bool(coverage["COVERAGE_ELIGIBLE"]) != coverage_eligible:
        raise ValueError("primary coverage eligibility differs from raw ledger")
    if str(coverage["DROP_REASON"]) != coverage_reason:
        raise ValueError("primary coverage reason differs from raw ledger")

    matched_point_map = {
        "N_STRATA": raw["N_COMMON_STRATA"],
        "N_DAYS": min(raw["N_INVERSION_DAYS"], raw["N_NORMAL_DAYS"]),
        "N_MONTHS": min(raw["N_INVERSION_MONTHS"], raw["N_NORMAL_MONTHS"]),
        "DELTA_ABS_BIAS_INVERSION": raw["DELTA_ABS_BIAS_INVERSION"],
        "DELTA_ABS_BIAS_NORMAL": raw["DELTA_ABS_BIAS_NORMAL"],
        "CONTRAST_INV_MINUS_NORMAL": raw["CONTRAST_INV_MINUS_NORMAL"],
    }
    for column, value in matched_point_map.items():
        if column.startswith("N_"):
            if int(matched[column]) != int(value):
                raise ValueError(f"matched contrast differs from raw ledger: {column}")
        elif not _same_number(matched[column], value):
            raise ValueError(f"matched contrast differs from raw ledger: {column}")

    independent_bootstrap: dict[str, Any] = {
        column: matched[column]
        for column in [
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
            "BLOCK7_CI_LOW",
            "BLOCK7_CI_HIGH",
            "P_VALUE",
            "COMPARISON_ELIGIBLE",
            "DROP_REASON",
        ]
    }
    # inference fields are generated once in ``build_matched_state_contrast``.
    # The raw date/stratum sufficient statistics below remain independent, and
    # ``_validate_primary_eligibility_ledger_contract`` reruns the fixed-seed
    # bootstrap from those statistics both before and after publication.  A
    # second bootstrap here would only duplicate the same station calculation.

    stats: list[dict[str, Any]] = []
    for (month, solar, state), group in analysis.groupby(
        ["MONTH_KEY", "SOLAR_STATE", "STATE_PRIMARY"], sort=True
    ):
        stats.append(
            {
                "month": str(month),
                "solar_state": str(solar),
                "structure_state": str(state),
                "n_hours": int(len(group)),
                "sum_error_none": float(group["error_none"].sum()),
                "sum_error_fixed": float(group["error_fixed"].sum()),
                "unique_dates": sorted(group["DATE_KEY"].unique().tolist()),
            }
        )
    date_stats: list[dict[str, Any]] = []
    for (month, solar, state, date), group in reproduced_all.groupby(
        ["MONTH_KEY", "SOLAR_STATE", "STATE_PRIMARY", "DATE_KEY"], sort=True
    ):
        date_stats.append(
            {
                "month": str(month),
                "solar_state": str(solar),
                "structure_state": str(state),
                "date": str(date),
                "n_hours": int(len(group)),
                "sum_error_none": float(group["error_none"].sum()),
                "sum_error_fixed": float(group["error_fixed"].sum()),
            }
        )
    inversion_dates = sorted(state_rows["inversion_like"]["DATE_KEY"].unique().tolist())
    normal_dates = sorted(state_rows["normal_negative"]["DATE_KEY"].unique().tolist())
    analysis_dates = sorted(analysis["DATE_KEY"].unique().tolist())
    try:
        uncovered_dates = sorted(set(json.loads(str(
            independent_bootstrap["UNCOVERED_ANALYSIS_DATES_JSON"]
        ))))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("matched uncovered analysis dates JSON is invalid") from exc
    support_checked = bool(independent_bootstrap["ANALYSIS_SUPPORT_CHECKED"])
    if support_checked:
        supported_dates = sorted(set(analysis_dates) - set(uncovered_dates))
        if int(independent_bootstrap["ANALYSIS_DATE_COUNT"]) != len(analysis_dates):
            raise ValueError("matched analysis date count differs from raw ledger")
        if int(independent_bootstrap["SUPPORTED_ANALYSIS_DATE_COUNT"]) != len(supported_dates):
            raise ValueError("matched supported date count differs from raw ledger")
        if int(independent_bootstrap["UNCOVERED_ANALYSIS_DATE_COUNT"]) != len(uncovered_dates):
            raise ValueError("matched uncovered date count differs from raw ledger")
    else:
        # 30일·6개월 coverage를 먼저 통과하지 못한 경우 bootstrap은
        # calendar support 검사를 시작하지 않는다. raw 날짜집합은 ledger에
        # 그대로 보존하되, support 결과 필드는 helper 계약대로 0/빈 값이다.
        supported_dates = []
        if (
            uncovered_dates
            or int(independent_bootstrap["ANALYSIS_DATE_COUNT"]) != 0
            or int(independent_bootstrap["SUPPORTED_ANALYSIS_DATE_COUNT"]) != 0
            or int(independent_bootstrap["UNCOVERED_ANALYSIS_DATE_COUNT"]) != 0
        ):
            raise ValueError("unchecked bootstrap support fields must remain empty")

    row = {
        "STN": station_id,
        "SETTING_ID": PRIMARY_SETTING_ID,
        "COVERAGE_ID": PRIMARY_COVERAGE_ID,
        "SOURCE_POPULATION_SHA256": source_population_sha256,
        "ANALYSIS_CONFIG_SHA256": analysis_config_sha256,
        "COMMON_STRATA_KEYS_JSON": json.dumps([list(key) for key in common], ensure_ascii=False),
        "INVERSION_UNIQUE_DATES_JSON": json.dumps(inversion_dates),
        "NORMAL_UNIQUE_DATES_JSON": json.dumps(normal_dates),
        "INVERSION_MONTHS_JSON": json.dumps(sorted(state_rows["inversion_like"]["MONTH_KEY"].unique().tolist())),
        "NORMAL_MONTHS_JSON": json.dumps(sorted(state_rows["normal_negative"]["MONTH_KEY"].unique().tolist())),
        "ANALYSIS_DATES_JSON": json.dumps(analysis_dates),
        "SUPPORTED_ANALYSIS_DATES_JSON": json.dumps(supported_dates),
        "UNCOVERED_ANALYSIS_DATES_JSON": json.dumps(uncovered_dates),
        "STRATUM_SUFFICIENT_STATS_JSON": json.dumps(stats, ensure_ascii=False, sort_keys=True),
        "DATE_SUFFICIENT_STATS_JSON": json.dumps(date_stats, ensure_ascii=False, sort_keys=True),
        **raw,
        "BASE_MATCHED_DATA_AVAILABLE": available,
        "COVERAGE_ELIGIBLE": coverage_eligible,
        "COVERAGE_DROP_REASON": coverage_reason,
        **{key: value for key, value in independent_bootstrap.items() if key not in {"DROP_REASON", "UNCOVERED_ANALYSIS_DATES_JSON"}},
        "Q_VALUE": matched["Q_VALUE"],
        "COMPARISON_DROP_REASON": independent_bootstrap["DROP_REASON"],
        "EVIDENCE_GRADE": matched["EVIDENCE_GRADE"],
        "ANALYSIS_POPULATION": "all_stations_before_subset_join",
    }
    return pd.DataFrame([row], columns=PRIMARY_ELIGIBILITY_LEDGER_COLUMNS)


def _apply_primary_ledger_bh_and_sync(
    ledger: pd.DataFrame,
    matched_contrast: pd.DataFrame,
) -> None:
    """primary ledger에서 full family BH를 계산한 뒤 matched 표에 전파한다."""

    _require_columns(ledger, PRIMARY_ELIGIBILITY_LEDGER_COLUMNS, "primary ledger")
    _require_columns(matched_contrast, MATCHED_STATE_CONTRAST_COLUMNS, "matched contrast")
    if ledger["STN"].duplicated().any() or matched_contrast["STN"].duplicated().any():
        raise ValueError("primary ledger and matched contrast must be one row per station")
    eligible_values = ledger["COMPARISON_ELIGIBLE"]
    if eligible_values.isna().any() or eligible_values.map(
        lambda value: not isinstance(value, (bool, np.bool_))
    ).any():
        raise ValueError("primary ledger eligibility must be exact boolean")
    eligible = eligible_values.astype(bool)
    p_values = pd.to_numeric(ledger["P_VALUE"], errors="coerce")
    if p_values.loc[eligible].isna().any() or (
        (p_values.loc[eligible] < 0) | (p_values.loc[eligible] > 1)
    ).any():
        raise ValueError("eligible primary ledger rows require p-values in [0,1]")
    ledger.loc[~eligible, ["P_VALUE", "Q_VALUE"]] = np.nan
    ledger["Q_VALUE"] = np.nan
    if eligible.any():
        ledger.loc[eligible, "Q_VALUE"] = benjamini_hochberg(
            p_values.loc[eligible].to_numpy(float)
        )
    ci_low = pd.to_numeric(ledger["BLOCK7_CI_LOW"], errors="coerce")
    ci_high = pd.to_numeric(ledger["BLOCK7_CI_HIGH"], errors="coerce")
    supported = eligible & ((ci_low > 0) | (ci_high < 0)) & (
        pd.to_numeric(ledger["Q_VALUE"], errors="coerce") <= 0.05
    )
    ledger["EVIDENCE_GRADE"] = np.where(
        supported, "자료로 지지", "현재 자료에서 지지되지 않음"
    )
    lookup = ledger.set_index("STN")
    if set(matched_contrast["STN"].astype(str)) != set(lookup.index.astype(str)):
        raise ValueError("primary ledger and matched station skeleton differ")
    for index, row in matched_contrast.iterrows():
        source = lookup.loc[str(row["STN"])]
        matched_contrast.at[index, "P_VALUE"] = source["P_VALUE"]
        matched_contrast.at[index, "Q_VALUE"] = source["Q_VALUE"]
        matched_contrast.at[index, "COMPARISON_ELIGIBLE"] = source[
            "COMPARISON_ELIGIBLE"
        ]
        matched_contrast.at[index, "DROP_REASON"] = source[
            "COMPARISON_DROP_REASON"
        ]
        matched_contrast.at[index, "EVIDENCE_GRADE"] = source["EVIDENCE_GRADE"]


def _apply_bh_within_config(frame: pd.DataFrame) -> None:
    """전체 station의 eligible 비교를 CONFIG별 family로 한 번만 BH 보정한다."""

    _require_columns(
        frame,
        [
            "CONFIG",
            "COMPARISON_ELIGIBLE",
            "P_VALUE",
            "Q_VALUE",
            "BLOCK7_CI_LOW",
            "BLOCK7_CI_HIGH",
            "EVIDENCE_GRADE",
        ],
        "matched inference",
    )
    eligible_values = frame["COMPARISON_ELIGIBLE"]
    if eligible_values.isna().any() or eligible_values.map(
        lambda value: not isinstance(value, (bool, np.bool_))
    ).any():
        raise ValueError("matched COMPARISON_ELIGIBLE must contain exact boolean values")
    eligible = eligible_values.astype(bool)
    p_values = pd.to_numeric(frame["P_VALUE"], errors="coerce")
    if (
        p_values.loc[eligible].isna().any()
        or (p_values.loc[eligible] < 0).any()
        or (p_values.loc[eligible] > 1).any()
    ):
        raise ValueError("eligible full-family rows must contain P_VALUE in [0, 1]")
    frame.loc[~eligible, ["P_VALUE", "Q_VALUE"]] = np.nan
    frame["Q_VALUE"] = np.nan
    for _, indices in frame.groupby("CONFIG", sort=False).groups.items():
        index = pd.Index(indices)
        eligible_index = index[eligible.loc[index].to_numpy()]
        if len(eligible_index):
            frame.loc[eligible_index, "Q_VALUE"] = benjamini_hochberg(
                frame.loc[eligible_index, "P_VALUE"].to_numpy(float)
            )
    ci_excludes_zero = (
        (pd.to_numeric(frame["BLOCK7_CI_LOW"], errors="coerce") > 0)
        | (pd.to_numeric(frame["BLOCK7_CI_HIGH"], errors="coerce") < 0)
    )
    supported = eligible & ci_excludes_zero & (pd.to_numeric(frame["Q_VALUE"], errors="coerce") <= 0.05)
    frame["EVIDENCE_GRADE"] = np.where(
        supported, "자료로 지지", "현재 자료에서 지지되지 않음"
    )


def _validate_matched_inference_contract(frame: pd.DataFrame) -> None:
    """게시된 P_VALUE에서 full-family BH와 evidence gate를 독립 재계산한다."""

    required = [
        "CONFIG",
        "COMPARISON_ELIGIBLE",
        "P_VALUE",
        "Q_VALUE",
        "BLOCK7_CI_LOW",
        "BLOCK7_CI_HIGH",
        "EVIDENCE_GRADE",
    ]
    _require_columns(frame, required, "matched inference")
    eligible_values = frame["COMPARISON_ELIGIBLE"]
    if eligible_values.isna().any() or eligible_values.map(
        lambda value: not isinstance(value, (bool, np.bool_))
    ).any():
        raise ValueError("matched COMPARISON_ELIGIBLE must contain exact boolean values")
    eligible = eligible_values.astype(bool)
    if "DROP_REASON" in frame and not np.array_equal(
        eligible.to_numpy(bool),
        frame["DROP_REASON"].astype(str).eq("eligible").to_numpy(bool),
    ):
        raise ValueError(
            "matched COMPARISON_ELIGIBLE must agree exactly with DROP_REASON=eligible"
        )
    diagnostic_columns = {
        "ANALYSIS_SUPPORT_CHECKED",
        "UNCOVERED_ANALYSIS_DATE_COUNT",
        "ANALYSIS_DATE_SUPPORT_FRACTION",
        "BLOCK7_VALID_FRACTION",
        "BLOCK7_VALID_REPLICATES",
    }
    if diagnostic_columns.issubset(frame.columns):
        support_checked = frame["ANALYSIS_SUPPORT_CHECKED"]
        if support_checked.isna().any() or support_checked.map(
            lambda value: not isinstance(value, (bool, np.bool_))
        ).any():
            raise ValueError("matched ANALYSIS_SUPPORT_CHECKED must be exact boolean")
        diagnostic_eligible = (
            support_checked.astype(bool)
            & pd.to_numeric(
                frame["UNCOVERED_ANALYSIS_DATE_COUNT"], errors="coerce"
            ).eq(0)
            & np.isclose(
                pd.to_numeric(
                    frame["ANALYSIS_DATE_SUPPORT_FRACTION"], errors="coerce"
                ),
                1.0,
                rtol=0,
                atol=1e-12,
            )
            & (
                pd.to_numeric(frame["BLOCK7_VALID_FRACTION"], errors="coerce")
                >= 0.95
            )
            & (
                pd.to_numeric(frame["BLOCK7_VALID_REPLICATES"], errors="coerce")
                >= 1900
            )
        )
        if not np.array_equal(
            eligible.to_numpy(bool), np.asarray(diagnostic_eligible, dtype=bool)
        ):
            raise ValueError(
                "matched eligibility differs from published support/replicate diagnostics"
            )
    ineligible = ~eligible
    if frame.loc[ineligible, ["P_VALUE", "Q_VALUE"]].notna().any().any():
        raise ValueError("ineligible matched rows must have P_VALUE/Q_VALUE NA")
    eligible_pq = frame.loc[eligible, ["P_VALUE", "Q_VALUE"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        eligible_pq.isna().any().any()
        or (eligible_pq < 0).any().any()
        or (eligible_pq > 1).any().any()
    ):
        raise ValueError(
            "eligible full-family rows must contain finite P_VALUE/Q_VALUE in [0, 1]"
        )
    expected = frame.copy()
    expected["Q_VALUE"] = np.nan
    _apply_bh_within_config(expected)
    actual_q = pd.to_numeric(frame["Q_VALUE"], errors="coerce").to_numpy(float)
    expected_q = pd.to_numeric(expected["Q_VALUE"], errors="coerce").to_numpy(float)
    if not np.allclose(actual_q, expected_q, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError("published matched Q_VALUE does not equal eligible full-family BH")
    if not np.array_equal(
        frame["EVIDENCE_GRADE"].astype(str).to_numpy(),
        expected["EVIDENCE_GRADE"].astype(str).to_numpy(),
    ):
        raise ValueError("matched evidence grade violates eligible+CI+q gate")


def _h2_expected_sign(delta_coast_km: float, month: int, solar_state: str) -> str:
    if not np.isfinite(delta_coast_km):
        return "not_prespecified"
    summer = int(month) in {6, 7, 8}
    winter = int(month) in {12, 1, 2}
    if delta_coast_km < 0 and summer and solar_state == "bright":
        return "positive"
    if delta_coast_km < 0 and winter and solar_state == "dark":
        return "negative"
    if delta_coast_km > 0 and summer and solar_state == "bright":
        return "negative"
    if delta_coast_km > 0 and winter and solar_state == "dark":
        return "positive"
    return "not_prespecified"


def _build_coast_time_bias(hourly: pd.DataFrame) -> pd.DataFrame:
    reproduced = hourly.loc[hourly["REPRODUCED"].astype(bool)].copy()
    rows = []
    for (station_id, month, solar), group in reproduced.groupby(
        ["STN", "MONTH", "SOLAR_STATE"], sort=True, dropna=False
    ):
        metrics = _bias_metrics(group)
        delta_coast = float(group["DELTA_COAST_KM"].median())
        expected = _h2_expected_sign(delta_coast, int(month), str(solar))
        observed = (
            "positive"
            if metrics["BIAS_fixed"] > 0
            else "negative"
            if metrics["BIAS_fixed"] < 0
            else "zero"
        )
        direction_match = bool(expected in {"positive", "negative"} and observed == expected)
        rows.append(
            {
                "STN": station_id,
                "MONTH": int(month),
                "SOLAR_STATE": solar,
                "N_HOURS": int(len(group)),
                "N_DAYS": int(group["TM"].dt.normalize().nunique()),
                "N_MONTHS": int(group["TM"].dt.to_period("M").nunique()),
                **metrics,
                "DELTA_COAST_KM": delta_coast,
                "H2_EXPECTED_BIAS_SIGN": expected,
                "H2_OBSERVED_BIAS_SIGN": observed,
                "H2_DIRECTION_MATCH": direction_match,
                "H2_PLACEMENT_SCOPE": "prespecified_delta_coast_signed_direction",
                "H2_EVIDENCE_SCOPE": "hourly_extrapolation_from_Kim_Tmax_Tmin",
                "EVIDENCE_GRADE": (
                    "문헌과 일치하나 직접 미측정"
                    if direction_match
                    else "현재 자료에서 지지되지 않음"
                ),
                "LIMITATION": "Kim_Tmax_Tmin_relation_extrapolated_to_hourly_bias",
                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                "SUBSET_MEANINGFUL_WORSENING": False,
                "SUBSET_HIGH_AFTER": False,
            }
        )
    return pd.DataFrame(rows, columns=COAST_TIME_COLUMNS)


def _summarize_hourly_physical_exposure(hourly: pd.DataFrame) -> dict[str, float]:
    """재현된 시간 원표에서 station 물리 노출량을 집계한다.

    H2 노출량은 cell을 먼저 요약하지 않고 각 시간의 ``|Δcoast|`` 중앙값을
    사용한다. H1 사례의 signed ΔZ는 기존 `bias_geometry`와 같은 target-이웃
    고도차(결측 이웃은 0) 거리 가중값을 시간에 대해 산술평균한다.
    """

    _require_columns(
        hourly,
        ["REPRODUCED", "DELTA_COAST_KM", "SIGNED_DELTA_Z_M"],
        "hourly physical exposure",
    )
    reproduced = hourly.loc[hourly["REPRODUCED"].astype(bool)]

    def finite_values(column: str) -> np.ndarray:
        values = pd.to_numeric(reproduced[column], errors="coerce").to_numpy(float)
        return values[np.isfinite(values)]

    coast = finite_values("DELTA_COAST_KM")
    signed_delta_z = finite_values("SIGNED_DELTA_Z_M")
    return {
        "MEDIAN_ABS_DELTA_COAST_KM": (
            float(np.median(np.abs(coast))) if len(coast) else float("nan")
        ),
        "MEAN_SIGNED_DELTA_Z_M": (
            float(np.mean(signed_delta_z)) if len(signed_delta_z) else float("nan")
        ),
    }


def _project_epsg5179(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GRS80 Transverse Mercator(EPSG:5179)를 외부 패키지 없이 계산한다."""

    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    a = 6_378_137.0
    inverse_f = 298.257222101
    flattening = 1.0 / inverse_f
    e2 = flattening * (2.0 - flattening)
    ep2 = e2 / (1.0 - e2)
    lon0 = math.radians(127.5)
    lat0 = math.radians(38.0)
    k0 = 0.9996

    def meridional(phi: np.ndarray | float) -> np.ndarray:
        return a * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * phi)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * phi)
            - (35 * e2**3 / 3072) * np.sin(6 * phi)
        )

    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)
    tan_lat = np.tan(latitude)
    n_radius = a / np.sqrt(1 - e2 * sin_lat**2)
    t = tan_lat**2
    c = ep2 * cos_lat**2
    A = cos_lat * (longitude - lon0)
    M = meridional(latitude)
    M0 = float(meridional(lat0))
    east = 1_000_000.0 + k0 * n_radius * (
        A
        + (1 - t + c) * A**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * A**5 / 120
    )
    north = 2_000_000.0 + k0 * (
        M
        - M0
        + n_radius
        * tan_lat
        * (
            A**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * A**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * A**6 / 720
        )
    )
    return east, north


def _weighted_rank_biserial(
    target: np.ndarray,
    reference: np.ndarray,
    target_weight: np.ndarray,
    reference_weight: np.ndarray,
) -> float:
    pair_weight = target_weight[:, None] * reference_weight[None, :]
    denominator = float(pair_weight.sum())
    if denominator <= 0:
        return np.nan
    signs = np.sign(target[:, None] - reference[None, :])
    return float(np.sum(signs * pair_weight) / denominator)


def _spatial_block_rbc(
    values: np.ndarray,
    groups: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    block_km: int,
    n_replicates: int,
    seed: int,
) -> dict[str, Any]:
    finite = np.isfinite(values) & np.isfinite(latitude) & np.isfinite(longitude)
    values = values[finite]
    groups = groups[finite]
    latitude = latitude[finite]
    longitude = longitude[finite]
    target = values[groups == "high_after"]
    reference = values[groups == "control"]
    point = rank_biserial_effect(target, reference) if len(target) and len(reference) else np.nan
    east, north = _project_epsg5179(latitude, longitude)
    blocks = np.array(
        [f"{int(math.floor(x / (block_km * 1000)))}:{int(math.floor(y / (block_km * 1000)))}" for x, y in zip(east, north)],
        dtype=object,
    )
    unique_blocks = np.array(sorted(set(blocks)), dtype=object)
    target_blocks = len(set(blocks[groups == "high_after"]))
    reference_blocks = len(set(blocks[groups == "control"]))
    result = {
        "n_blocks": len(unique_blocks),
        "target_blocks": target_blocks,
        "reference_blocks": reference_blocks,
        "valid_replicates": 0,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "eligible": False,
    }
    if len(unique_blocks) < 8 or target_blocks < 4 or reference_blocks < 4 or not np.isfinite(point):
        return result
    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(n_replicates):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        counts = Counter(sampled.tolist())
        station_weight = np.array([counts.get(value, 0) for value in blocks], dtype=float)
        target_mask = groups == "high_after"
        reference_mask = groups == "control"
        estimate = _weighted_rank_biserial(
            values[target_mask],
            values[reference_mask],
            station_weight[target_mask],
            station_weight[reference_mask],
        )
        if np.isfinite(estimate):
            replicates.append(estimate)
    result["valid_replicates"] = len(replicates)
    if len(replicates) < math.ceil(0.95 * n_replicates):
        return result
    array = np.asarray(replicates, dtype=float)
    deviations = array - point
    result.update(
        {
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "p_value": float((1 + np.count_nonzero(np.abs(deviations) >= abs(point))) / (len(array) + 1)),
            "eligible": True,
        }
    )
    return result


def _direction_station_summary(coast: pd.DataFrame) -> pd.DataFrame:
    """사전 지정 6개 month×solar cell을 모두 가진 station만 방향 gate에 포함한다."""

    if coast.empty:
        return pd.DataFrame(columns=["STN", "MATCH_N", "TESTABLE_N", "DIRECTION_GATE"])
    work = coast.copy()
    _require_columns(
        work,
        ["MONTH", "SOLAR_STATE", "DELTA_COAST_KM", "BIAS_fixed"],
        "coast direction summary",
    )
    work["H2_EXPECTED_BIAS_SIGN"] = [
        _h2_expected_sign(float(delta), int(month), str(solar))
        for delta, month, solar in zip(
            pd.to_numeric(work["DELTA_COAST_KM"], errors="coerce"),
            pd.to_numeric(work["MONTH"], errors="raise"),
            work["SOLAR_STATE"].astype(str),
        )
    ]
    observed = np.where(
        pd.to_numeric(work["BIAS_fixed"], errors="coerce") > 0,
        "positive",
        np.where(
            pd.to_numeric(work["BIAS_fixed"], errors="coerce") < 0,
            "negative",
            "zero",
        ),
    )
    work["H2_DIRECTION_MATCH"] = (
        work["H2_EXPECTED_BIAS_SIGN"].isin(["positive", "negative"])
        & (observed == work["H2_EXPECTED_BIAS_SIGN"].astype(str))
    )
    expected_cells = {(month, "bright") for month in (6, 7, 8)} | {
        (month, "dark") for month in (12, 1, 2)
    }
    rows = []
    for station_id, group in work.groupby("STN", sort=True):
        testable = group.loc[
            group.apply(lambda row: (int(row["MONTH"]), str(row["SOLAR_STATE"])) in expected_cells, axis=1)
            & (pd.to_numeric(group["N_DAYS"], errors="coerce") >= 20)
            & pd.to_numeric(group["DELTA_COAST_KM"], errors="coerce").notna()
            & pd.to_numeric(group["BIAS_fixed"], errors="coerce").notna()
            & pd.to_numeric(group["DELTA_COAST_KM"], errors="coerce").ne(0)
        ]
        keys = set(zip(testable["MONTH"].astype(int), testable["SOLAR_STATE"].astype(str)))
        gate = keys == expected_cells
        rows.append(
            {
                "STN": str(station_id),
                "MATCH_N": int(testable["H2_DIRECTION_MATCH"].astype(bool).sum()) if gate else 0,
                "TESTABLE_N": 6 if gate else 0,
                "DIRECTION_GATE": gate,
            }
        )
    return pd.DataFrame(rows)


def build_physical_hypothesis_comparison(
    equivalent: pd.DataFrame,
    station_structure: pd.DataFrame,
    coast_time: pd.DataFrame,
    *,
    state_condition: pd.DataFrame | None = None,
    n_replicates: int = 2000,
    seed: int = 20260827,
) -> pd.DataFrame:
    """equivalent 모집단의 high-after/control에 대한 H1/H2 6개 고정 contrast를 만든다."""

    _require_columns(equivalent, ["STN", "BIAS_fixed", "ABS_BIAS_FIXED"], "equivalent")
    base = equivalent[["STN", "BIAS_fixed", "ABS_BIAS_FIXED"]].copy()
    base["STN"] = base["STN"].astype(str).str.strip()
    base["GROUP"] = np.where(base["ABS_BIAS_FIXED"] > 0.5, "high_after", "control")
    base["SIGN"] = np.where(base["BIAS_fixed"] > 0, "positive", np.where(base["BIAS_fixed"] < 0, "negative", "zero"))

    structure = station_structure.copy()
    structure["STN"] = structure["STN"].astype(str).str.strip()
    _require_columns(
        structure,
        ["MEDIAN_ABS_DELTA_COAST_KM"],
        "station physical exposure",
    )
    if "DARK_INV_FRACTION" not in structure:
        structure["DARK_INV_FRACTION"] = np.nan
    if {
        "DARK_PRIMARY_ELIGIBLE_DAYS",
        "DARK_PRIMARY_ELIGIBLE_MONTHS",
    }.issubset(structure):
        h1_station_gate = (
            pd.to_numeric(structure["DARK_PRIMARY_ELIGIBLE_DAYS"], errors="coerce") >= 30
        ) & (
            pd.to_numeric(structure["DARK_PRIMARY_ELIGIBLE_MONTHS"], errors="coerce") >= 6
        )
        structure.loc[~h1_station_gate, "DARK_INV_FRACTION"] = np.nan
    elif state_condition is not None and len(state_condition):
        dark = state_condition.loc[
            state_condition["CONFIG"].eq("primary")
            & state_condition["CONDITION_VARIABLE"].eq("SOLAR_STATE")
            & state_condition["CONDITION_LEVEL"].eq("dark")
            & state_condition["STRUCTURE_STATE"].isin(["inversion_like", "normal_negative", "uncertain"])
        ].copy()
        if len(dark):
            denominator = dark.groupby("STN")["N_HOURS"].sum()
            numerator = dark.loc[dark["STRUCTURE_STATE"].eq("inversion_like")].groupby("STN")["N_HOURS"].sum()
            fraction = (numerator / denominator).rename("DARK_INV_FRACTION")
            structure = structure.drop(columns=["DARK_INV_FRACTION"]).merge(fraction, on="STN", how="left")
    structure["ABS_DCOAST_KM"] = pd.to_numeric(
        structure["MEDIAN_ABS_DELTA_COAST_KM"], errors="coerce"
    )
    coast = coast_time.copy()
    if len(coast):
        coast["STN"] = coast["STN"].astype(str).str.strip()
    direction = _direction_station_summary(coast)
    joined = base.merge(structure, on="STN", how="left", suffixes=("", "_STRUCTURE"))
    joined = joined.merge(direction, on="STN", how="left")
    for column in ["MATCH_N", "TESTABLE_N"]:
        joined[column] = pd.to_numeric(joined.get(column, 0), errors="coerce").fillna(0).astype(int)
    joined["DIRECTION_GATE"] = joined.get("DIRECTION_GATE", False).fillna(False).astype(bool)

    definitions = [
        ("H1_LOCAL_TOPOGRAPHY", "H1", "DARK_INV_FRACTION", "fraction"),
        ("H2_COAST_MISMATCH", "H2", "ABS_DCOAST_KM", "km"),
    ]
    rows: list[dict[str, Any]] = []
    for hypothesis_id, prefix, variable, unit in definitions:
        for sign in ("all", "positive", "negative"):
            subset = joined if sign == "all" else joined.loc[joined["SIGN"].eq(sign)]
            target_all = subset.loc[subset["GROUP"].eq("high_after")]
            reference_all = subset.loc[subset["GROUP"].eq("control")]
            target = target_all.loc[pd.to_numeric(target_all[variable], errors="coerce").notna()]
            reference = reference_all.loc[pd.to_numeric(reference_all[variable], errors="coerce").notna()]
            target_values = pd.to_numeric(target[variable], errors="coerce").to_numpy(float)
            reference_values = pd.to_numeric(reference[variable], errors="coerce").to_numpy(float)
            target_median = float(np.median(target_values)) if len(target_values) else np.nan
            reference_median = float(np.median(reference_values)) if len(reference_values) else np.nan
            median_difference = target_median - reference_median if np.isfinite(target_median) and np.isfinite(reference_median) else np.nan
            effect = rank_biserial_effect(target_values, reference_values) if len(target_values) and len(reference_values) else np.nan
            coverage = bool(
                len(target_values) >= math.ceil(0.8 * len(target_all))
                and len(reference_values) >= math.ceil(0.8 * len(reference_all))
                and len(target_all) > 0
                and len(reference_all) > 0
            )
            record = {column: np.nan for column in PHYSICAL_HYPOTHESIS_COLUMNS}
            record.update(
                {
                    "HYPOTHESIS_ID": hypothesis_id,
                    "CONTRAST_ID": f"{prefix}_{'ALL' if sign == 'all' else 'POS' if sign == 'positive' else 'NEG'}_HIGH_V_CONTROL",
                    "ROW_KIND": "group_comparison",
                    "ANALYSIS_ROLE": "primary" if sign == "all" else "sign_stratified_descriptive",
                    "SIGN_STRATUM": sign,
                    "TARGET_GROUP": "high_after",
                    "REFERENCE_GROUP": "control_low_or_medium",
                    "ROW_UNIT": "station_group_contrast",
                    "TARGET_CANONICAL_N": len(target_all),
                    "REFERENCE_CANONICAL_N": len(reference_all),
                    "TARGET_ELIGIBLE_N": len(target),
                    "REFERENCE_ELIGIBLE_N": len(reference),
                    "TARGET_EXCLUDED_N": len(target_all) - len(target),
                    "REFERENCE_EXCLUDED_N": len(reference_all) - len(reference),
                    "VARIABLE": variable,
                    "VARIABLE_UNIT": unit,
                    "TARGET_MEDIAN": target_median,
                    "REFERENCE_MEDIAN": reference_median,
                    "MEDIAN_DIFFERENCE": median_difference,
                    "EFFECT_SIZE_NAME": "rank_biserial",
                    "EFFECT_ESTIMATE": effect,
                    "EXPECTED_EFFECT_DIRECTION": "greater",
                    "OBSERVED_EFFECT_DIRECTION": "greater" if effect > 0 else "lower" if effect < 0 else "tie" if effect == 0 else "not_estimable",
                    "EFFECT_DIRECTION_MATCH": bool(effect > 0) if np.isfinite(effect) else np.nan,
                    "COVERAGE_GATE": "target_and_reference_station_coverage_at_least_80pct",
                    "COVERAGE_ELIGIBLE": coverage,
                    "INFERENCE_MODE": "descriptive_only",
                    "SPATIAL_BLOCK_METHOD": "none",
                    "FAMILY_ID": "NONE_DESCRIPTIVE",
                    "FAMILY_SIZE": 0,
                    "Q_METHOD": "none",
                    "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                    "ANALYSIS_POPULATION": "practically_equivalent_115_after_bias_level",
                }
            )
            if hypothesis_id == "H1_LOCAL_TOPOGRAPHY":
                record["EXPECTED_BIAS_DIRECTION_RULE"] = "NOT_TESTABLE_TPI_NOT_MEASURED"
                record["LIMITATION"] = "TPI_and_valley_floor_position_NOT_MEASURED;regional_pattern_descriptive_only"
                if coverage and median_difference > 0 and effect > 0:
                    record["EVIDENCE_GRADE"] = "문헌과 일치하나 직접 미측정"
            else:
                target_gate = target_all["DIRECTION_GATE"].astype(bool)
                reference_gate = reference_all["DIRECTION_GATE"].astype(bool)
                target_match = int(target_all.loc[target_gate, "MATCH_N"].sum())
                target_testable = int(target_all.loc[target_gate, "TESTABLE_N"].sum())
                reference_match = int(reference_all.loc[reference_gate, "MATCH_N"].sum())
                reference_testable = int(reference_all.loc[reference_gate, "TESTABLE_N"].sum())
                target_rate = target_match / target_testable if target_testable else np.nan
                reference_rate = reference_match / reference_testable if reference_testable else np.nan
                direction_coverage = bool(
                    target_gate.sum() >= math.ceil(0.8 * len(target_all))
                    and reference_gate.sum() >= math.ceil(0.8 * len(reference_all))
                    and len(target_all) > 0
                    and len(reference_all) > 0
                )
                direction_match = bool(
                    direction_coverage
                    and np.isfinite(target_rate)
                    and np.isfinite(reference_rate)
                    and target_rate > 0.5
                    and target_rate > reference_rate
                )
                record.update(
                    {
                        "EXPECTED_BIAS_DIRECTION_RULE": "summer_bright=-sign_delta_coast;winter_dark=sign_delta_coast",
                        "TARGET_DIRECTION_MATCH_N": target_match,
                        "TARGET_DIRECTION_TESTABLE_N": target_testable,
                        "TARGET_DIRECTION_MATCH_RATE": target_rate,
                        "REFERENCE_DIRECTION_MATCH_N": reference_match,
                        "REFERENCE_DIRECTION_TESTABLE_N": reference_testable,
                        "REFERENCE_DIRECTION_MATCH_RATE": reference_rate,
                        "BIAS_DIRECTION_MATCH": direction_match,
                        "LIMITATION": "Kim_Tmax_Tmin_to_hourly_extrapolation;delta_coast_proxy_not_causal_SST_or_seabreeze",
                    }
                )
                if sign == "all":
                    record.update(
                        {
                            "INFERENCE_MODE": "spatial_block_bootstrap",
                            "SPATIAL_BLOCK_METHOD": "EPSG5179_fixed_square_cluster_bootstrap",
                            "PRIMARY_BLOCK_KM": 50,
                            "FAMILY_ID": "PHYSICAL_H2_PRIMARY_50KM",
                            "FAMILY_SIZE": 1,
                            "Q_METHOD": "BH_eligible_primary_rows_only",
                        }
                    )
                    latitude_column = (
                        next(
                            (
                                name
                                for name in ["TARGET_LATITUDE", "위도"]
                                if name in joined
                            ),
                            None,
                        )
                        if coverage
                        else None
                    )
                    longitude_column = (
                        next(
                            (
                                name
                                for name in ["TARGET_LONGITUDE", "경도"]
                                if name in joined
                            ),
                            None,
                        )
                        if coverage
                        else None
                    )
                    spatial = {}
                    if latitude_column and longitude_column:
                        spatial_source = joined.loc[joined["GROUP"].isin(["high_after", "control"])]
                        spatial_values = pd.to_numeric(spatial_source[variable], errors="coerce").to_numpy(float)
                        spatial_groups = spatial_source["GROUP"].to_numpy(str)
                        lat = pd.to_numeric(spatial_source[latitude_column], errors="coerce").to_numpy(float)
                        lon = pd.to_numeric(spatial_source[longitude_column], errors="coerce").to_numpy(float)
                        for index, block in enumerate((30, 50, 100)):
                            spatial[block] = _spatial_block_rbc(
                                spatial_values,
                                spatial_groups,
                                lat,
                                lon,
                                block_km=block,
                                n_replicates=n_replicates,
                                seed=seed + index * 1009,
                            )
                            current = spatial[block]
                            record.update(
                                {
                                    f"N_BLOCKS_{block}KM": current["n_blocks"],
                                    f"TARGET_BLOCKS_{block}KM": current["target_blocks"],
                                    f"REFERENCE_BLOCKS_{block}KM": current["reference_blocks"],
                                    f"VALID_REPLICATES_{block}KM": current["valid_replicates"],
                                    f"CI_LOW_{block}KM": current["ci_low"],
                                    f"CI_HIGH_{block}KM": current["ci_high"],
                                }
                            )
                            record["P_VALUE" if block == 50 else f"P_VALUE_{block}KM"] = current["p_value"]
                        # 50 km가 사전 지정 주 분석이다. 30/100 km의 자료 부족은
                        # 주 p/q를 지우지 않고 근거 등급의 민감도 gate에서만 반영한다.
                        if spatial[50]["eligible"]:
                            record["Q_VALUE"] = record["P_VALUE"]
                        if all(spatial[block]["eligible"] for block in (30, 50, 100)):
                            record["SPATIAL_SENSITIVITY_DIRECTION_STABLE"] = all(
                                spatial[block]["ci_low"] > 0 for block in (30, 50, 100)
                            )
                    fully_supported = bool(
                        coverage
                        and median_difference > 0
                        and effect > 0
                        and direction_match
                        and pd.notna(record["Q_VALUE"])
                        and record["Q_VALUE"] <= 0.05
                        and all(pd.notna(record[f"CI_LOW_{block}KM"]) and record[f"CI_LOW_{block}KM"] > 0 for block in (30, 50, 100))
                    )
                    if fully_supported:
                        record["EVIDENCE_GRADE"] = "자료로 지지"
                    elif not direction_match or not coverage or not (median_difference > 0 and effect > 0):
                        record["EVIDENCE_GRADE"] = "현재 자료에서 지지되지 않음"
                    else:
                        record["EVIDENCE_GRADE"] = "문헌과 일치하나 직접 미측정"
                elif coverage and median_difference > 0 and effect > 0 and direction_match:
                    record["EVIDENCE_GRADE"] = "문헌과 일치하나 직접 미측정"
            rows.append(record)
    return pd.DataFrame(rows, columns=PHYSICAL_HYPOTHESIS_COLUMNS)


def _empty_structure_batch(length: int, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "REGRESSION_N": np.zeros(length, dtype=int),
            "NEIGHBOR_FINGERPRINT": [None] * length,
            "ELEVATION_SPAN_M": np.full(length, np.nan),
            "WEIGHTED_ELEVATION_SD_M": np.full(length, np.nan),
            "N_EFF": np.full(length, np.nan),
            "OLS_SLOPE_C_PER_KM": np.full(length, np.nan),
            "OLS_CI95_LOW_C_PER_KM": np.full(length, np.nan),
            "OLS_CI95_HIGH_C_PER_KM": np.full(length, np.nan),
            "OLS_R2": np.full(length, np.nan),
            "OLS_HC3_SE_C_PER_KM": np.full(length, np.nan),
            "OLS_HC3_CI95_LOW_C_PER_KM": np.full(length, np.nan),
            "OLS_HC3_CI95_HIGH_C_PER_KM": np.full(length, np.nan),
            "WLS_SLOPE_C_PER_KM": np.full(length, np.nan),
            "AUX_SLOPE_NONE_C_PER_KM": np.full(length, np.nan),
            "AUX_SLOPE_D1_C_PER_KM": np.full(length, np.nan),
            "AUX_SLOPE_D2_C_PER_KM": np.full(length, np.nan),
            "WEIGHTED_ELEVATION_SD_NONE_M": np.full(length, np.nan),
            "WEIGHTED_ELEVATION_SD_D1_M": np.full(length, np.nan),
            "WEIGHTED_ELEVATION_SD_D2_M": np.full(length, np.nan),
            "N_EFF_NONE": np.full(length, np.nan),
            "N_EFF_D1": np.full(length, np.nan),
            "N_EFF_D2": np.full(length, np.nan),
            "LOO_STABLE_NONE": np.zeros(length, dtype=bool),
            "LOO_STABLE_D1": np.zeros(length, dtype=bool),
            "LOO_STABLE_D2": np.zeros(length, dtype=bool),
            "LOO_SIGN_STABILITY_NONE": np.full(length, np.nan),
            "LOO_SIGN_STABILITY_D1": np.full(length, np.nan),
            "LOO_SIGN_STABILITY_D2": np.full(length, np.nan),
            "STATE_PRIMARY": ["ineligible"] * length,
            "STATE_PRIMARY_REASON": [reason] * length,
            "STATE_SPAN200": ["ineligible"] * length,
            "STATE_NEFF4": ["ineligible"] * length,
            "STATE_OLS_ONLY": ["ineligible"] * length,
        }
    )


def _ensure_ta_frame(ta_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ta_data, pd.DataFrame):
        raise TypeError("ta_data must be a DataFrame")
    out = ta_data.copy()
    out.index = pd.to_datetime(out.index, errors="raise")
    if out.index.has_duplicates:
        raise ValueError("ta_data index must be unique")
    normalized_columns = [
        _normalize_station_id(column, field="ta_data station column")
        for column in out.columns
    ]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("ta_data station columns must be unique after normalization")
    out.columns = normalized_columns
    return out.sort_index()


def analyze_station_injected(
    *,
    pairs: pd.DataFrame,
    neighbor_map: Mapping[str, Sequence[Any]],
    meta: pd.DataFrame,
    ta_data: pd.DataFrame,
    weather: pd.DataFrame,
    terrain: pd.DataFrame,
    strict_reproduction: bool = True,
    return_hourly: bool = False,
    reproduction_tolerance: float = 1e-10,
    state_boot_replicates: int = 2000,
    state_boot_seed: int = 20260827,
    source_population_sha256: str | None = None,
    analysis_config_sha256: str | None = None,
) -> dict[str, Any]:
    """주입된 한 관측소 자료를 production과 같은 계약으로 분석한다."""

    started = time.perf_counter()
    required = [
        "TM",
        "STN",
        "OBSERVED",
        "error_none",
        "error_fixed",
        "NEIGHBOR_SET_ID",
        "COORD_EPOCH",
        "NEIGHBOR_COUNT",
    ]
    _require_columns(pairs, required, "pairs")
    paired = _normalize_pair_keys(pairs, "pairs")
    station_ids = paired["STN"].drop_duplicates().tolist()
    if len(station_ids) != 1:
        raise ValueError("analyze_station_injected accepts exactly one target STN")
    station_id = station_ids[0]
    if source_population_sha256 is None:
        _, source_population_sha256 = _station_population_identity([station_id])
    if analysis_config_sha256 is None:
        analysis_config_sha256 = _canonical_json_hash(_analysis_config_payload())
    for column in ["OBSERVED", "error_none", "error_fixed", "NEIGHBOR_COUNT"]:
        paired[column] = pd.to_numeric(paired[column], errors="raise")
    if not np.isfinite(paired[["OBSERVED", "error_none", "error_fixed"]].to_numpy(float)).all():
        raise ValueError("paired observed/error values must be finite")
    paired["NEIGHBOR_SET_ID"] = paired["NEIGHBOR_SET_ID"].astype(str)
    paired["COORD_EPOCH"] = pd.to_datetime(paired["COORD_EPOCH"], errors="raise").dt.normalize()
    temperature = _ensure_ta_frame(ta_data)
    geometry_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], dict[str, Any]] = {}
    metadata_boundaries = _metadata_validity_boundary_lookup(meta)
    pieces: list[pd.DataFrame] = []
    processed_neighbor_values = 0

    for (signature, epoch), signature_group in paired.groupby(
        ["NEIGHBOR_SET_ID", "COORD_EPOCH"], sort=False
    ):
        if signature not in neighbor_map:
            raise ValueError(f"unknown exact neighbor signature: {signature}")
        counts = signature_group["NEIGHBOR_COUNT"].drop_duplicates().tolist()
        if len(counts) != 1:
            raise ValueError("NEIGHBOR_COUNT must be constant within a signature epoch")
        exact_ids = validate_exact_neighbor_ids(
            station_id,
            neighbor_map[signature],
            expected_count=int(counts[0]),
        )
        for geometry_timestamp, group in _iter_signature_metadata_segments(
            signature_group,
            station_ids=(station_id, *exact_ids),
            boundary_lookup=metadata_boundaries,
        ):
            key = (str(signature), pd.Timestamp(epoch), geometry_timestamp)
            geometry = reconstruct_signature_geometry(
                target_id=station_id,
                neighbor_ids=exact_ids,
                timestamp=geometry_timestamp,
                coord_epoch=epoch,
                expected_neighbor_count=int(counts[0]),
                meta=meta,
                terrain=terrain,
            )
            geometry_cache[key] = geometry
            missing_columns = set(geometry["neighbor_ids"]) - set(temperature.columns)
            if missing_columns:
                raise ValueError(f"TA columns missing exact neighbors: {sorted(missing_columns)}")
            missing_times = set(group["TM"]) - set(temperature.index)
            if missing_times:
                raise ValueError("TA data missing paired timestamps")
            all_matrix = temperature.loc[
                group["TM"], list(geometry["neighbor_ids"])
            ].to_numpy(float)
            processed_neighbor_values += int(all_matrix.size)
            if not np.isfinite(all_matrix).all():
                raise ValueError("neighbor TA is missing or non-finite at a paired hour")

            # 동일 geometry의 시각은 배열 연산으로 처리한다. 첫 시각은 Task2 scalar와 검산한다.
            if geometry["zero_neighbor_distance"]:
                predictions = all_matrix[
                    :, int(np.flatnonzero(geometry["distances_km"] == 0)[0])
                ]
            else:
                predictions = all_matrix @ geometry["idw_weights"]
            scalar_first = exact_idw_none_weighted_mean(
                all_matrix[0],
                geometry["distances_km"],
                neighbor_ids=geometry["neighbor_ids"],
                target_id=station_id,
            )
            if not np.isclose(predictions[0], scalar_first, rtol=0, atol=1e-12):
                raise RuntimeError("vectorized none IDW differs from Task2 scalar contract")
            expected_prediction = group["OBSERVED"].to_numpy(float) + group[
                "error_none"
            ].to_numpy(float)
            differences = np.abs(predictions - expected_prediction)
            reproduced = differences <= reproduction_tolerance
            if strict_reproduction and not reproduced.all():
                raise ValueError(
                    f"none reproduction mismatch for {station_id}: max={differences.max():.12g}"
                )

            if geometry["zero_neighbor_distance"]:
                diagnostics = _empty_structure_batch(len(group), "zero_neighbor_distance")
                diagnostics["REGRESSION_N"] = geometry["regression_neighbor_count"]
            elif geometry["regression_neighbor_count"] < 3:
                diagnostics = _empty_structure_batch(
                    len(group), "fewer_than_three_finite_elevation_neighbors"
                )
                diagnostics["REGRESSION_N"] = geometry["regression_neighbor_count"]
            else:
                regression_ids = list(geometry["regression_neighbor_ids"])
                regression_matrix = temperature.loc[
                    group["TM"], regression_ids
                ].to_numpy(float)
                diagnostics = vectorized_structure_diagnostics(
                    elevations_m=geometry["regression_elevations_m"],
                    temperature_matrix_c=regression_matrix,
                    distances_km=geometry["regression_distances_km"],
                    neighbor_ids=regression_ids,
                    target_id=station_id,
                )
            diagnostics.index = group.index

            piece = group.copy()
            piece["PRED_NONE_RECONSTRUCTED"] = predictions
            piece["NONE_ABS_DIFF_C"] = differences
            piece["REPRODUCED"] = reproduced
            piece["EXCLUSION_REASON"] = np.where(
                reproduced, "eligible", "none_reproduction_mismatch"
            )
            piece = pd.concat([piece, diagnostics], axis=1)
            piece["DELTA_COAST_KM"] = geometry["delta_coast_km"]
            piece["SIGNED_DELTA_Z_M"] = geometry["signed_delta_z_m"]
            piece["WEIGHTED_COASTAL_FRACTION"] = geometry["weighted_coastal_fraction"]
            piece["WEIGHTED_ISLAND_FRACTION"] = geometry["weighted_island_fraction"]
            piece["RELIEF_MISMATCH_M"] = geometry["relief_mismatch_m"]
            piece["TERRAIN_MISMATCH_FRACTION"] = geometry["terrain_mismatch_fraction"]
            piece["TPI_STATUS"] = geometry["tpi_status"]
            piece["TARGET_LATITUDE"] = geometry["target_latitude"]
            piece["TARGET_LONGITUDE"] = geometry["target_longitude"]
            pieces.append(piece)

    hourly = pd.concat(pieces, ignore_index=True).sort_values("TM").reset_index(drop=True)
    # 이설 epoch별 좌표가 달라질 수 있으므로 첫 행 좌표를 연중 고정하지 않는다.
    solar = np.empty(len(hourly), dtype=float)
    for (_, latitude, longitude), indices in hourly.groupby(
        ["COORD_EPOCH", "TARGET_LATITUDE", "TARGET_LONGITUDE"], sort=False
    ).groups.items():
        index = pd.Index(indices)
        solar[index.to_numpy(int)] = solar_elevation_noaa(
            hourly.loc[index, "TM"],
            latitude_deg=float(latitude),
            longitude_deg=float(longitude),
        )
    hourly["SOLAR_ELEVATION_DEG"] = solar
    hourly["SOLAR_STATE"] = classify_solar_state(solar)
    hourly["MONTH"] = hourly["TM"].dt.month

    weather_result = merge_weather_conditions(
        hourly.drop(columns=[column for column in ["WS10", "WS1", "RN_60m", "HM", "TD"] if column in hourly]),
        weather,
        min_coverage=0.8,
    )
    hourly = weather_result["hourly"]
    weather_coverage = weather_result["coverage"]
    state_bias = build_state_condition_bias(hourly, weather_coverage)
    sensitivity_summary = build_structure_sensitivity_summary(hourly)
    matched_outputs = build_matched_state_outputs(
        hourly,
        n_boot=state_boot_replicates,
        seed=state_boot_seed,
    )
    matched_contrast = matched_outputs["primary"]
    coverage_sensitivity = matched_outputs["sensitivity"]
    primary_coverage = coverage_sensitivity.loc[
        coverage_sensitivity["SETTING_ID"].eq(PRIMARY_SETTING_ID)
        & coverage_sensitivity["COVERAGE_ID"].eq(PRIMARY_COVERAGE_ID)
    ]
    primary_ledger = build_primary_eligibility_ledger(
        hourly,
        matched_contrast,
        primary_coverage,
        source_population_sha256=source_population_sha256,
        analysis_config_sha256=analysis_config_sha256,
    )
    coast_time = _build_coast_time_bias(hourly)

    def count_state(column: str, state: str | None = None) -> int:
        series = hourly[column]
        return int((series.ne("ineligible") if state is None else series.eq(state)).sum())

    exclusion_counts = Counter(
        hourly.loc[hourly["EXCLUSION_REASON"].ne("eligible"), "EXCLUSION_REASON"]
    )
    ineligible_reasons = Counter(
        hourly.loc[hourly["STATE_PRIMARY"].eq("ineligible"), "STATE_PRIMARY_REASON"]
    )
    coverage_row = weather_coverage.iloc[0]
    physical_exposure = _summarize_hourly_physical_exposure(hourly)
    summary = pd.DataFrame(
        [
            {
                "STN": station_id,
                "TOTAL_PAIRED_HOURS": int(len(hourly)),
                "REPRODUCED_HOURS": int(hourly["REPRODUCED"].sum()),
                "REPRODUCTION_RATE": float(hourly["REPRODUCED"].mean()),
                "MAX_NONE_ABS_DIFF_C": float(hourly["NONE_ABS_DIFF_C"].max()),
                "PRIMARY_ELIGIBLE_HOURS": count_state("STATE_PRIMARY"),
                "PRIMARY_INVERSION_LIKE_HOURS": count_state("STATE_PRIMARY", "inversion_like"),
                "PRIMARY_NORMAL_NEGATIVE_HOURS": count_state("STATE_PRIMARY", "normal_negative"),
                "PRIMARY_UNCERTAIN_HOURS": count_state("STATE_PRIMARY", "uncertain"),
                "PRIMARY_INELIGIBLE_HOURS": count_state("STATE_PRIMARY", "ineligible"),
                "DARK_PRIMARY_ELIGIBLE_HOURS": int(
                    (
                        hourly["SOLAR_STATE"].eq("dark")
                        & hourly["STATE_PRIMARY"].ne("ineligible")
                    ).sum()
                ),
                "DARK_PRIMARY_ELIGIBLE_DAYS": int(
                    hourly.loc[
                        hourly["SOLAR_STATE"].eq("dark")
                        & hourly["STATE_PRIMARY"].ne("ineligible"),
                        "TM",
                    ].dt.normalize().nunique()
                ),
                "DARK_PRIMARY_ELIGIBLE_MONTHS": int(
                    hourly.loc[
                        hourly["SOLAR_STATE"].eq("dark")
                        & hourly["STATE_PRIMARY"].ne("ineligible"),
                        "TM",
                    ].dt.to_period("M").nunique()
                ),
                "DARK_INV_FRACTION": (
                    float(
                        (
                            hourly["SOLAR_STATE"].eq("dark")
                            & hourly["STATE_PRIMARY"].eq("inversion_like")
                        ).sum()
                        / (
                            hourly["SOLAR_STATE"].eq("dark")
                            & hourly["STATE_PRIMARY"].ne("ineligible")
                        ).sum()
                    )
                    if (
                        hourly["SOLAR_STATE"].eq("dark")
                        & hourly["STATE_PRIMARY"].ne("ineligible")
                    ).any()
                    else np.nan
                ),
                "SPAN200_ELIGIBLE_HOURS": count_state("STATE_SPAN200"),
                "SPAN200_INVERSION_LIKE_HOURS": count_state("STATE_SPAN200", "inversion_like"),
                "NEFF4_ELIGIBLE_HOURS": count_state("STATE_NEFF4"),
                "NEFF4_INVERSION_LIKE_HOURS": count_state("STATE_NEFF4", "inversion_like"),
                "OLS_ONLY_ELIGIBLE_HOURS": count_state("STATE_OLS_ONLY"),
                "OLS_ONLY_INVERSION_LIKE_HOURS": count_state("STATE_OLS_ONLY", "inversion_like"),
                "WIND_COVERAGE": float(coverage_row["WIND_COVERAGE"]),
                "RN_COVERAGE": float(coverage_row["RN_COVERAGE"]),
                "HM_COVERAGE": float(coverage_row["HM_COVERAGE"]),
                "TD_COVERAGE": float(coverage_row["TD_COVERAGE"]),
                "WEATHER_PRIMARY_ELIGIBLE": bool(coverage_row["WEATHER_PRIMARY_ELIGIBLE"]),
                "WEATHER_DROP_REASON": coverage_row["WEATHER_DROP_REASON"],
                "EXCLUSION_COUNTS_JSON": json.dumps(exclusion_counts, ensure_ascii=False, sort_keys=True),
                "PRIMARY_INELIGIBLE_REASONS_JSON": json.dumps(ineligible_reasons, ensure_ascii=False, sort_keys=True),
                "MEDIAN_DELTA_COAST_KM": float(hourly["DELTA_COAST_KM"].median()),
                **physical_exposure,
                "MEDIAN_WEIGHTED_COASTAL_FRACTION": float(hourly["WEIGHTED_COASTAL_FRACTION"].median()),
                "MEDIAN_WEIGHTED_ISLAND_FRACTION": float(hourly["WEIGHTED_ISLAND_FRACTION"].median()),
                "MEDIAN_RELIEF_MISMATCH_M": float(hourly["RELIEF_MISMATCH_M"].median()),
                "MEDIAN_TERRAIN_MISMATCH_FRACTION": float(hourly["TERRAIN_MISMATCH_FRACTION"].median()),
                "TARGET_LATITUDE": float(hourly["TARGET_LATITUDE"].iloc[-1]),
                "TARGET_LONGITUDE": float(hourly["TARGET_LONGITUDE"].iloc[-1]),
                "TPI_STATUS": "NOT_MEASURED",
                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
            }
        ],
        columns=STATION_STRUCTURE_COLUMNS,
    )
    result: dict[str, Any] = {
        "station_summary": summary,
        "state_condition_bias": state_bias,
        "matched_state_contrast": matched_contrast,
        "spatial_structure_sensitivity_summary": sensitivity_summary,
        "matched_state_coverage_sensitivity": coverage_sensitivity,
        "primary_eligibility_ledger": primary_ledger,
        "coast_time_signed_bias": coast_time,
        "metrics": {
            "geometry_build_count": len(geometry_cache),
            "unique_geometry_count": len(geometry_cache),
            "processed_paired_hours": int(len(hourly)),
            "reproduced_hour_count": int(hourly["REPRODUCED"].sum()),
            "processed_neighbor_values": processed_neighbor_values,
            "ols_fit_count": int(hourly["OLS_SLOPE_C_PER_KM"].notna().sum()),
            "aux_fit_count_none": int(hourly["AUX_SLOPE_NONE_C_PER_KM"].notna().sum()),
            "aux_fit_count_d1": int(hourly["AUX_SLOPE_D1_C_PER_KM"].notna().sum()),
            "aux_fit_count_d2": int(hourly["AUX_SLOPE_D2_C_PER_KM"].notna().sum()),
            "loo_fit_count_none": int(
                hourly.loc[hourly["AUX_SLOPE_NONE_C_PER_KM"].notna(), "REGRESSION_N"].sum()
            ),
            "loo_fit_count_d1": int(
                hourly.loc[hourly["AUX_SLOPE_D1_C_PER_KM"].notna(), "REGRESSION_N"].sum()
            ),
            "loo_fit_count_d2": int(
                hourly.loc[hourly["AUX_SLOPE_D2_C_PER_KM"].notna(), "REGRESSION_N"].sum()
            ),
            "setting_classification_count": int(len(hourly) * len(_sensitivity_settings())),
            "coverage_gate_evaluation_count": int(
                len(_sensitivity_settings())
                * len(SENSITIVITY_MIN_DAYS)
                * len(SENSITIVITY_MIN_MONTHS)
            ),
            "peak_chunk_memory_bytes": int(hourly.memory_usage(deep=True).sum()),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }
    if return_hourly:
        result["hourly"] = hourly
    return result


def _prepare_station_results(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "SIGNIFICANCE_CLASS",
        "mean_absdz_geom_km",
    ]
    _require_columns(frame, required, "station_results")
    work = frame.copy()
    work["STN"] = work["STN"].map(
        lambda value: _normalize_station_id(value, field="station_results STN")
    )
    if work["STN"].duplicated().any():
        raise ValueError("station_results STN must be unique")
    for column in ["BIAS_none", "BIAS_fixed", "mean_absdz_geom_km"]:
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(float)).all():
            raise ValueError(f"station_results {column} must be finite")
    if (work["mean_absdz_geom_km"] < 0).any():
        raise ValueError("mean_absdz_geom_km must be non-negative")
    invalid = set(work["SIGNIFICANCE_CLASS"].astype(str)) - SIGNIFICANCE_CLASSES
    if invalid:
        raise ValueError(f"unknown significance classes: {sorted(invalid)}")
    if "MECHANISM" not in work.columns:
        work["MECHANISM"] = ""
    return work


def _expected_canonical_invariants() -> dict[str, Any]:
    """정본 station-results에서 반드시 재현되어야 하는 exact 불변식."""

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
        "EQUIVALENT_AFTER_LEVEL_COUNTS": {"low": 20, "medium": 62, "high": 33},
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


def validate_canonical_station_results(frame: pd.DataFrame) -> dict[str, Any]:
    """공식 529개 정본의 분모·부분집단 불변식을 검증한다."""

    work = _prepare_station_results(frame)
    counts = work["SIGNIFICANCE_CLASS"].value_counts().to_dict()
    expected = {
        "meaningful_improvement": 221,
        "meaningful_worsening": 109,
        "practically_equivalent": 115,
        "uncertain": 84,
    }
    equivalent_result = build_equivalent_diagnostics(work)
    equivalent = equivalent_result["stations"]
    transition = equivalent_result["transition"]
    worsening = build_worsening_diagnostics(work)["stations"]
    high_before = equivalent["ABS_BIAS_NONE"] > 0.5
    high_after = equivalent["ABS_BIAS_FIXED"] > 0.5
    positive_after = equivalent["BIAS_fixed"] > 0
    result = {
        "TOTAL_STATIONS": int(len(work)),
        "CLASS_COUNTS": {key: int(counts.get(key, 0)) for key in expected},
        "HIGH_BEFORE_N": int(high_before.sum()),
        "HIGH_AFTER_N": int(high_after.sum()),
        "HIGH_COMMON_N": int((high_before & high_after).sum()),
        "HIGH_AFTER_POSITIVE_N": int((high_after & positive_after).sum()),
        "HIGH_AFTER_NEGATIVE_N": int((high_after & ~positive_after).sum()),
        "EQUIVALENT_AFTER_LEVEL_COUNTS": {
            level: int(equivalent["BIAS_FIXED_LEVEL"].eq(level).sum())
            for level in ["low", "medium", "high"]
        },
        "EQUIVALENT_TRANSITION_COUNTS": {
            f"{row.BIAS_NONE_LEVEL}->{row.BIAS_FIXED_LEVEL}": int(row.N)
            for row in transition.itertuples(index=False)
        },
        "WORSENING_MECHANISM_COUNTS": {
            mechanism: int(worsening["MECHANISM"].eq(mechanism).sum())
            for mechanism in ["same_direction_worsened", "overshoot_worsened"]
        },
        "WORSENING_DZ_BIN_COUNTS": {
            label: int(worsening["DZ_BIN"].eq(label).sum())
            for label in ["≤50m", "50–100m", "100–200m", "200–300m", ">300m"]
        },
        "PRIMARY_HYPOTHESIS_POPULATION": "all_529_stations",
    }
    expected_result = _expected_canonical_invariants()
    if result != expected_result:
        raise ValueError(
            "canonical station-result invariants failed: "
            f"observed={result}, expected={expected_result}"
        )
    return result


def _empty_station_summary(station_id: str, reason: str) -> pd.DataFrame:
    row = {column: np.nan for column in STATION_STRUCTURE_COLUMNS}
    row.update(
        {
            "STN": station_id,
            "TOTAL_PAIRED_HOURS": 0,
            "REPRODUCED_HOURS": 0,
            "REPRODUCTION_RATE": np.nan,
            "PRIMARY_ELIGIBLE_HOURS": 0,
            "PRIMARY_INVERSION_LIKE_HOURS": 0,
            "PRIMARY_NORMAL_NEGATIVE_HOURS": 0,
            "PRIMARY_UNCERTAIN_HOURS": 0,
            "PRIMARY_INELIGIBLE_HOURS": 0,
            "DARK_PRIMARY_ELIGIBLE_HOURS": 0,
            "DARK_PRIMARY_ELIGIBLE_DAYS": 0,
            "DARK_PRIMARY_ELIGIBLE_MONTHS": 0,
            "SPAN200_ELIGIBLE_HOURS": 0,
            "SPAN200_INVERSION_LIKE_HOURS": 0,
            "NEFF4_ELIGIBLE_HOURS": 0,
            "NEFF4_INVERSION_LIKE_HOURS": 0,
            "OLS_ONLY_ELIGIBLE_HOURS": 0,
            "OLS_ONLY_INVERSION_LIKE_HOURS": 0,
            "WEATHER_PRIMARY_ELIGIBLE": False,
            "WEATHER_DROP_REASON": reason,
            "EXCLUSION_COUNTS_JSON": json.dumps({reason: 1}, ensure_ascii=False),
            "PRIMARY_INELIGIBLE_REASONS_JSON": "{}",
            "TPI_STATUS": "NOT_MEASURED",
            "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
        }
    )
    return pd.DataFrame([row], columns=STATION_STRUCTURE_COLUMNS)


def _empty_state_skeleton(station_id: str, reason: str) -> pd.DataFrame:
    rows = []
    for config in STRUCTURE_CONFIGS:
        for state in STRUCTURE_STATES:
            for variable, levels in _condition_domains(pd.DataFrame()):
                for level in levels:
                    row = {column: np.nan for column in STATE_CONDITION_COLUMNS}
                    row.update(
                        {
                            "STN": station_id,
                            "CONFIG": config,
                            "STRUCTURE_STATE": state,
                            "CONDITION_VARIABLE": variable,
                            "CONDITION_LEVEL": level,
                            "N_HOURS": 0,
                            "N_DAYS": 0,
                            "N_MONTHS": 0,
                            "N_STRATA": 0,
                            "BLOCK7_REPLICATES": 0,
                            "COVERAGE_ELIGIBLE": False,
                            "DROP_REASON": reason,
                            "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                            "LIMITATION": "no_hourly_source_data",
                            "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                            "SUBSET_MEANINGFUL_WORSENING": False,
                            "SUBSET_HIGH_AFTER": False,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=STATE_CONDITION_COLUMNS)


def _empty_matched_contrast(station_id: str, reason: str) -> pd.DataFrame:
    rows = []
    for config in ("primary",):
        row = {column: np.nan for column in MATCHED_STATE_CONTRAST_COLUMNS}
        row.update(
            {
                "STN": station_id,
                "CONFIG": config,
                "SETTING_ID": PRIMARY_SETTING_ID,
                "N_STRATA": 0,
                "N_DAYS": 0,
                "N_MONTHS": 0,
                "BLOCK7_VALID_REPLICATES": 0,
                "BLOCK7_REQUESTED_REPLICATES": 0,
                "BLOCK7_VALID_FRACTION": np.nan,
                "BLOCK7_BLOCK_DAYS": 7,
                "STRATA_WEIGHTING": "equal_common_month_solar_strata",
                "ESTIMAND": "standardized_signed_bias_then_delta_abs_bias_contrast",
                "SAMPLING_SCHEME": "monthly_calendar_contiguous_fixed_length_blocks",
                "ANALYSIS_SUPPORT_CHECKED": False,
                "ANALYSIS_DATE_COUNT": 0,
                "SUPPORTED_ANALYSIS_DATE_COUNT": 0,
                "UNCOVERED_ANALYSIS_DATE_COUNT": 0,
                "ANALYSIS_DATE_SUPPORT_FRACTION": np.nan,
                "UNCOVERED_ANALYSIS_DATES_JSON": "[]",
                "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON": "[]",
                "COMPARISON_ELIGIBLE": False,
                "DROP_REASON": reason,
                "P_VALUE": np.nan,
                "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                "LIMITATION": "no_hourly_source_data",
                "ANALYSIS_POPULATION": "all_eligible_stations_before_subset_join",
                "SUBSET_MEANINGFUL_WORSENING": False,
                "SUBSET_HIGH_AFTER": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCHED_STATE_CONTRAST_COLUMNS)


def _empty_primary_eligibility_ledger(
    station_id: str,
    reason: str,
    *,
    source_population_sha256: str,
    analysis_config_sha256: str,
) -> pd.DataFrame:
    row = {column: np.nan for column in PRIMARY_ELIGIBILITY_LEDGER_COLUMNS}
    row.update(
        {
            "STN": _normalize_station_id(station_id, field="primary ledger STN"),
            "SETTING_ID": PRIMARY_SETTING_ID,
            "COVERAGE_ID": PRIMARY_COVERAGE_ID,
            "SOURCE_POPULATION_SHA256": source_population_sha256,
            "ANALYSIS_CONFIG_SHA256": analysis_config_sha256,
            "COMMON_STRATA_KEYS_JSON": "[]",
            "INVERSION_UNIQUE_DATES_JSON": "[]",
            "NORMAL_UNIQUE_DATES_JSON": "[]",
            "INVERSION_MONTHS_JSON": "[]",
            "NORMAL_MONTHS_JSON": "[]",
            "ANALYSIS_DATES_JSON": "[]",
            "SUPPORTED_ANALYSIS_DATES_JSON": "[]",
            "UNCOVERED_ANALYSIS_DATES_JSON": "[]",
            "STRATUM_SUFFICIENT_STATS_JSON": "[]",
            "DATE_SUFFICIENT_STATS_JSON": "[]",
            "N_COMMON_STRATA": 0,
            "N_INVERSION_HOURS": 0,
            "N_NORMAL_HOURS": 0,
            "N_INVERSION_DAYS": 0,
            "N_NORMAL_DAYS": 0,
            "N_INVERSION_MONTHS": 0,
            "N_NORMAL_MONTHS": 0,
            "BASE_MATCHED_DATA_AVAILABLE": False,
            "COVERAGE_ELIGIBLE": False,
            "COVERAGE_DROP_REASON": reason,
            "BLOCK7_REQUESTED_REPLICATES": 0,
            "BLOCK7_VALID_REPLICATES": 0,
            "BLOCK7_VALID_FRACTION": np.nan,
            "BLOCK7_BLOCK_DAYS": 7,
            "STRATA_WEIGHTING": "equal_common_month_solar_strata",
            "ESTIMAND": "standardized_signed_bias_then_delta_abs_bias_contrast",
            "SAMPLING_SCHEME": "monthly_calendar_contiguous_fixed_length_blocks",
            "ANALYSIS_SUPPORT_CHECKED": False,
            "ANALYSIS_DATE_COUNT": 0,
            "SUPPORTED_ANALYSIS_DATE_COUNT": 0,
            "UNCOVERED_ANALYSIS_DATE_COUNT": 0,
            "ANALYSIS_DATE_SUPPORT_FRACTION": np.nan,
            "ANALYSIS_SUPPORT_DIAGNOSTICS_JSON": "[]",
            "BLOCK7_CI_LOW": np.nan,
            "BLOCK7_CI_HIGH": np.nan,
            "P_VALUE": np.nan,
            "Q_VALUE": np.nan,
            "COMPARISON_ELIGIBLE": False,
            "COMPARISON_DROP_REASON": reason,
            "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
            "ANALYSIS_POPULATION": "all_stations_before_subset_join",
        }
    )
    return pd.DataFrame([row], columns=PRIMARY_ELIGIBILITY_LEDGER_COLUMNS)


def _empty_spatial_sensitivity(station_id: str, reason: str) -> pd.DataFrame:
    rows = []
    for setting in _sensitivity_settings():
        for state in STRUCTURE_STATES:
            row = {column: np.nan for column in SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS}
            row.update(
                {
                    "STN": station_id,
                    **setting,
                    "STRUCTURE_STATE": state,
                    "TOTAL_PAIRED_HOURS": 0,
                    "REPRODUCED_HOURS": 0,
                    "N_HOURS": 0,
                    "ELIGIBLE_HOURS": 0,
                    "N_EXACT_REPRODUCTION_FAILURE": 0,
                    "N_NONFINITE_ELEVATION": 0,
                    "N_FAIL_MIN_N": 0,
                    "N_FAIL_SPAN": 0,
                    "N_FAIL_WEIGHTED_SD": 0,
                    "N_FAIL_N_EFF": 0,
                    "N_OLS_CI_CROSSES_ZERO": 0,
                    "N_OLS_AUX_SIGN_DISAGREE": 0,
                    "N_LOO_SIGN_UNSTABLE": 0,
                    "ANALYSIS_POPULATION": "all_stations_before_subset_join",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS)


def _empty_matched_coverage_sensitivity(station_id: str, reason: str) -> pd.DataFrame:
    rows = []
    for setting in _sensitivity_settings():
        for minimum_days, minimum_months in product(
            SENSITIVITY_MIN_DAYS, SENSITIVITY_MIN_MONTHS
        ):
            row = {column: np.nan for column in MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS}
            row.update(
                {
                    "STN": station_id,
                    **setting,
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
                    "BASE_MATCHED_DATA_AVAILABLE": False,
                    "COVERAGE_ELIGIBLE": False,
                    "DROP_REASON": reason,
                    "SUBSET_MEANINGFUL_WORSENING": False,
                    "SUBSET_HIGH_AFTER": False,
                    "ANALYSIS_POPULATION": "all_stations_before_subset_join",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS)


def _high_residual_table(
    equivalent: pd.DataFrame,
    station_structure: pd.DataFrame,
) -> pd.DataFrame:
    if equivalent.empty:
        return pd.DataFrame(columns=HIGH_RESIDUAL_COLUMNS)
    summary = station_structure.set_index("STN", drop=False)
    rows = []
    for record in equivalent.itertuples(index=False):
        station_id = str(record.STN)
        high_before = bool(record.ABS_BIAS_NONE > 0.5)
        high_after = bool(record.ABS_BIAS_FIXED > 0.5)
        if station_id in summary.index:
            station = summary.loc[station_id]
            primary_eligible = bool(float(station["PRIMARY_ELIGIBLE_HOURS"] or 0) > 0)
            delta_coast = station["MEDIAN_DELTA_COAST_KM"]
            relief = station["MEDIAN_RELIEF_MISMATCH_M"]
            tpi = station["TPI_STATUS"]
        else:
            primary_eligible = False
            delta_coast = relief = np.nan
            tpi = "NOT_MEASURED"
        rows.append(
            {
                "STN": station_id,
                "BIAS_none": float(record.BIAS_none),
                "BIAS_fixed": float(record.BIAS_fixed),
                "BIAS_FIXED_SIGN": "positive" if record.BIAS_fixed > 0 else "negative",
                "BIAS_NONE_LEVEL": record.BIAS_NONE_LEVEL,
                "BIAS_FIXED_LEVEL": record.BIAS_FIXED_LEVEL,
                "HIGH_BEFORE": high_before,
                "HIGH_AFTER": high_after,
                "HIGH_COMMON": high_before and high_after,
                "IS_PRIMARY_HIGH_RESIDUAL": high_after,
                "PRIMARY_STRUCTURE_ELIGIBLE": primary_eligible,
                "MEDIAN_DELTA_COAST_KM": delta_coast,
                "MEDIAN_RELIEF_MISMATCH_M": relief,
                "TPI_STATUS": tpi,
                "ANALYSIS_POPULATION": "practically_equivalent_descriptive_subset",
            }
        )
    return pd.DataFrame(rows, columns=HIGH_RESIDUAL_COLUMNS)


def _station_885_table(hourly: pd.DataFrame | None) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=STATION_885_COLUMNS)
    work = hourly.loc[hourly["STN"].eq("885") & hourly["REPRODUCED"].astype(bool)]
    rows = []
    for (month, solar, state), group in work.groupby(
        ["MONTH", "SOLAR_STATE", "STATE_PRIMARY"], sort=True
    ):
        metrics = _bias_metrics(group)
        delta_coast = float(group["DELTA_COAST_KM"].median())
        rows.append(
            {
                "STN": "885",
                "MONTH": int(month),
                "SOLAR_STATE": solar,
                "STRUCTURE_STATE": state,
                "N_HOURS": int(len(group)),
                "N_DAYS": int(group["TM"].dt.normalize().nunique()),
                "N_MONTHS": int(group["TM"].dt.to_period("M").nunique()),
                **metrics,
                "DELTA_COAST_KM": delta_coast,
                "H2_EXPECTED_BIAS_SIGN": _h2_expected_sign(delta_coast, int(month), str(solar)),
                "CASE_SCOPE": "single_station_descriptive_case_not_general_proof",
                "EVIDENCE_GRADE": "현재 자료에서 지지되지 않음",
                "LIMITATION": "single_station_descriptive_case_without_full_family_q_gate",
            }
        )
    return pd.DataFrame(rows, columns=STATION_885_COLUMNS)


def _station_885_common_strata_direction_gate(
    station_885: pd.DataFrame,
) -> tuple[bool, str, int, float, float, float]:
    """전체 공통층 point와 사전 고정 dark H1 signed 방향을 함께 재계산한다."""

    empty_result = (False, "station_885_common_strata_missing", 0, np.nan, np.nan, np.nan)

    required = [
        "MONTH",
        "SOLAR_STATE",
        "STRUCTURE_STATE",
        "N_HOURS",
        "N_DAYS",
        "BIAS_none",
        "BIAS_fixed",
    ]
    if station_885.empty or any(column not in station_885 for column in required):
        return empty_result
    work = station_885.loc[
        station_885["STRUCTURE_STATE"].isin(
            ["inversion_like", "normal_negative"]
        ),
        required,
    ].copy()
    key_columns = ["MONTH", "SOLAR_STATE", "STRUCTURE_STATE"]
    if work.duplicated(key_columns).any():
        return False, "station_885_common_strata_duplicate", 0, np.nan, np.nan, np.nan
    for column in ["N_HOURS", "N_DAYS", "BIAS_none", "BIAS_fixed"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if (
        work[["N_HOURS", "N_DAYS", "BIAS_none", "BIAS_fixed"]]
        .isna()
        .any()
        .any()
        or (work["N_HOURS"] <= 0).any()
        or (work["N_DAYS"] <= 0).any()
    ):
        return False, "station_885_common_strata_nonfinite", 0, np.nan, np.nan, np.nan
    key_sets: dict[str, set[tuple[int, str]]] = {}
    for state_name in ("inversion_like", "normal_negative"):
        rows = work.loc[work["STRUCTURE_STATE"].eq(state_name)]
        key_sets[state_name] = {
            (int(row.MONTH), str(row.SOLAR_STATE))
            for row in rows.itertuples(index=False)
        }
    common_keys = key_sets["inversion_like"] & key_sets["normal_negative"]
    if len({month for month, _ in common_keys}) < 6:
        return (
            False,
            "station_885_common_strata_months_insufficient",
            len(common_keys),
            np.nan,
            np.nan,
            np.nan,
        )
    key_values = pd.Series(
        list(zip(work["MONTH"].astype(int), work["SOLAR_STATE"].astype(str))),
        index=work.index,
    )

    common_rows: dict[str, pd.DataFrame] = {}
    effects: dict[str, float] = {}
    for state_name in ("inversion_like", "normal_negative"):
        rows = work.loc[
            work["STRUCTURE_STATE"].eq(state_name) & key_values.isin(common_keys)
        ]
        if len(rows) != len(common_keys):
            return (
                False,
                "station_885_common_strata_incomplete",
                len(common_keys),
                np.nan,
                np.nan,
                np.nan,
            )
        common_rows[state_name] = rows
        signed_none = float(rows["BIAS_none"].mean())
        signed_fixed = float(rows["BIAS_fixed"].mean())
        effects[state_name] = abs(signed_fixed) - abs(signed_none)
    delta_inversion = effects["inversion_like"]
    delta_normal = effects["normal_negative"]
    contrast = delta_inversion - delta_normal

    dark_keys = {key for key in common_keys if key[1] == "dark"}
    dark_rows: dict[str, pd.DataFrame] = {}
    for state_name in ("inversion_like", "normal_negative"):
        rows = common_rows[state_name].loc[
            pd.Series(
                list(
                    zip(
                        common_rows[state_name]["MONTH"].astype(int),
                        common_rows[state_name]["SOLAR_STATE"].astype(str),
                    )
                ),
                index=common_rows[state_name].index,
            ).isin(dark_keys)
        ]
        dark_rows[state_name] = rows
        if (
            rows["MONTH"].nunique() < 6
            or float(rows["N_DAYS"].sum()) < 30
        ):
            return (
                False,
                "station_885_dark_common_coverage_insufficient",
                len(common_keys),
                delta_inversion,
                delta_normal,
                contrast,
            )
    inversion_dark = dark_rows["inversion_like"]
    bias_none = float(inversion_dark["BIAS_none"].mean())
    bias_fixed = float(inversion_dark["BIAS_fixed"].mean())
    dark_delta_abs_bias = abs(bias_fixed) - abs(bias_none)
    if not (
        np.isfinite(bias_none)
        and np.isfinite(bias_fixed)
        and bias_none > 0
        and bias_fixed > bias_none
        and bias_fixed > 0
        and dark_delta_abs_bias > 0
    ):
        return (
            False,
            "station_885_dark_common_signed_direction_failed",
            len(common_keys),
            delta_inversion,
            delta_normal,
            contrast,
        )
    return (
        True,
        "eligible",
        len(common_keys),
        delta_inversion,
        delta_normal,
        contrast,
    )


def _evaluate_station_885_h1_gate(
    station_885: pd.DataFrame,
    matched_contrast: pd.DataFrame,
    *,
    station_structure: pd.DataFrame,
    state_condition: pd.DataFrame,
    coverage_sensitivity: pd.DataFrame,
    analysis_station_ids: Iterable[Any],
) -> tuple[bool, str]:
    """885 저지대 H1 사례의 독립적인 사전 gate를 원표 요약에서 재계산한다."""

    reasons: list[str] = []
    common_strata_count = 0
    common_delta_inversion = np.nan
    common_delta_normal = np.nan
    common_contrast = np.nan
    population = [str(value).strip() for value in analysis_station_ids]
    if population.count("885") != 1:
        reasons.append("station_885_not_unique_in_analysis_population")
    if station_885.empty:
        reasons.append("station_885_case_rows_missing")
    elif set(station_885["STN"].astype(str).str.strip()) != {"885"}:
        reasons.append("station_885_case_contains_other_station")
    else:
        # case 표 자체도 두 primary 상태가 6개월·30일 이상 나타나야 한다.
        for state_name in ("inversion_like", "normal_negative"):
            rows = station_885.loc[station_885["STRUCTURE_STATE"].eq(state_name)]
            months = pd.to_numeric(rows.get("MONTH"), errors="coerce")
            days = pd.to_numeric(rows.get("N_DAYS"), errors="coerce")
            if (
                len(rows) == 0
                or months.dropna().nunique() < 6
                or days.isna().any()
                or float(days.sum()) < 30
            ):
                reasons.append(f"station_885_case_{state_name}_coverage_insufficient")
        (
            common_direction,
            common_reason,
            common_strata_count,
            common_delta_inversion,
            common_delta_normal,
            common_contrast,
        ) = (
            _station_885_common_strata_direction_gate(station_885)
        )
        if not common_direction:
            reasons.append(common_reason)

    structure = station_structure.loc[
        station_structure["STN"].astype(str).str.strip().eq("885")
    ]
    if len(structure) != 1:
        reasons.append("station_885_structure_row_not_unique")
    else:
        row = structure.iloc[0]
        total = pd.to_numeric(row.get("TOTAL_PAIRED_HOURS"), errors="coerce")
        reproduced = pd.to_numeric(row.get("REPRODUCED_HOURS"), errors="coerce")
        rate = pd.to_numeric(row.get("REPRODUCTION_RATE"), errors="coerce")
        max_diff = pd.to_numeric(row.get("MAX_NONE_ABS_DIFF_C"), errors="coerce")
        mean_signed_delta_z = pd.to_numeric(
            row.get("MEAN_SIGNED_DELTA_Z_M"), errors="coerce"
        )
        if not (
            np.isfinite(total)
            and total > 0
            and reproduced == total
            and np.isfinite(rate)
            and np.isclose(rate, 1.0, rtol=0, atol=1e-12)
            and np.isfinite(max_diff)
            and max_diff >= 0
            and max_diff <= 1e-10
        ):
            reasons.append("station_885_exact_none_reproduction_failed")
        if not (np.isfinite(mean_signed_delta_z) and mean_signed_delta_z < 0):
            reasons.append("station_885_target_not_lower_than_weighted_neighbors")

    state = state_condition.loc[
        state_condition["STN"].astype(str).str.strip().eq("885")
        & state_condition["CONFIG"].eq("primary")
        & state_condition["STRUCTURE_STATE"].eq("inversion_like")
        & state_condition["CONDITION_VARIABLE"].eq("ALL")
        & state_condition["CONDITION_LEVEL"].eq("all")
    ]
    if len(state) != 1:
        reasons.append("station_885_primary_inversion_summary_not_unique")

    coverage = coverage_sensitivity.loc[
        coverage_sensitivity["STN"].astype(str).str.strip().eq("885")
        & coverage_sensitivity["SETTING_ID"].eq(PRIMARY_SETTING_ID)
        & coverage_sensitivity["COVERAGE_ID"].eq(PRIMARY_COVERAGE_ID)
    ]
    if len(coverage) != 1:
        reasons.append("station_885_primary_coverage_row_not_unique")
    else:
        row = coverage.iloc[0]
        coverage_gate = bool(row.get("COVERAGE_ELIGIBLE", False)) and all(
            pd.to_numeric(row.get(column), errors="coerce") >= threshold
            for column, threshold in [
                ("N_INVERSION_DAYS", 30),
                ("N_NORMAL_DAYS", 30),
                ("N_INVERSION_MONTHS", 6),
                ("N_NORMAL_MONTHS", 6),
            ]
        )
        if not coverage_gate:
            reasons.append("station_885_two_state_coverage_failed")

    matched = matched_contrast.loc[
        matched_contrast["STN"].astype(str).str.strip().eq("885")
        & matched_contrast["SETTING_ID"].eq(PRIMARY_SETTING_ID)
    ]
    if len(matched) != 1:
        reasons.append("station_885_matched_contrast_not_unique")
    else:
        row = matched.iloc[0]
        contrast = pd.to_numeric(row.get("CONTRAST_INV_MINUS_NORMAL"), errors="coerce")
        delta_inversion = pd.to_numeric(
            row.get("DELTA_ABS_BIAS_INVERSION"), errors="coerce"
        )
        delta_normal = pd.to_numeric(
            row.get("DELTA_ABS_BIAS_NORMAL"), errors="coerce"
        )
        ci_low = pd.to_numeric(row.get("BLOCK7_CI_LOW"), errors="coerce")
        q_value = pd.to_numeric(row.get("Q_VALUE"), errors="coerce")
        n_strata = pd.to_numeric(row.get("N_STRATA"), errors="coerce")
        if not (
            bool(row.get("COMPARISON_ELIGIBLE", False))
            and np.isfinite(n_strata)
            and int(n_strata) == n_strata
            and int(n_strata) == common_strata_count
            and np.isfinite(delta_inversion)
            and np.isclose(
                delta_inversion, common_delta_inversion, rtol=0, atol=1e-12
            )
            and np.isfinite(delta_normal)
            and np.isclose(delta_normal, common_delta_normal, rtol=0, atol=1e-12)
            and np.isfinite(contrast)
            and np.isclose(contrast, common_contrast, rtol=0, atol=1e-12)
            and contrast > 0
            and np.isfinite(ci_low)
            and ci_low > 0
            and np.isfinite(q_value)
            and q_value <= 0.05
        ):
            reasons.append("station_885_full_family_inference_gate_failed")
    return not reasons, "eligible" if not reasons else ";".join(reasons)


def _apply_station_885_evidence(
    station_885: pd.DataFrame,
    matched_contrast: pd.DataFrame,
    *,
    station_structure: pd.DataFrame,
    state_condition: pd.DataFrame,
    coverage_sensitivity: pd.DataFrame,
    analysis_station_ids: Iterable[Any],
) -> None:
    """885 H1 사례를 독립 gate 통과 시에만 간접·기술 근거로 표시한다."""

    if station_885.empty:
        return
    supported, reason = _evaluate_station_885_h1_gate(
        station_885,
        matched_contrast,
        station_structure=station_structure,
        state_condition=state_condition,
        coverage_sensitivity=coverage_sensitivity,
        analysis_station_ids=analysis_station_ids,
    )
    station_885["EVIDENCE_GRADE"] = (
        "문헌과 일치하나 직접 미측정"
        if supported
        else "현재 자료에서 지지되지 않음"
    )
    station_885["LIMITATION"] = (
        "single_station_descriptive_case;H1_low_target_signed_direction;"
        "full_family_q_gate;TPI_vertical_profile_not_measured;gate=" + reason
    )


def _validate_station_885_evidence_contract(
    station_885: pd.DataFrame,
    matched_contrast: pd.DataFrame,
    *,
    station_structure: pd.DataFrame,
    state_condition: pd.DataFrame,
    coverage_sensitivity: pd.DataFrame,
    analysis_station_ids: Iterable[Any],
) -> None:
    """게시된 885 등급을 source 요약에서 독립 재계산해 의미 변조를 막는다."""

    population = [str(value).strip() for value in analysis_station_ids]
    if station_885.empty:
        if population.count("885") > 1:
            raise ValueError("station 885 is duplicated in the analysis population")
        if population.count("885") == 1:
            structure = station_structure.loc[
                station_structure["STN"].astype(str).str.strip().eq("885")
            ]
            if len(structure) != 1:
                raise ValueError("station 885 structure row is missing or duplicated")
            total = pd.to_numeric(
                structure.iloc[0].get("TOTAL_PAIRED_HOURS"), errors="coerce"
            )
            reproduced = pd.to_numeric(
                structure.iloc[0].get("REPRODUCED_HOURS"), errors="coerce"
            )
            if (
                np.isfinite(total)
                and total > 0
                or np.isfinite(reproduced)
                and reproduced > 0
            ):
                raise ValueError(
                    "station 885 case rows are missing despite available analysis data"
                )
        return
    if population.count("885") != 1:
        raise ValueError("station 885 diagnostic is outside or duplicated in the population")
    months = pd.to_numeric(station_885["MONTH"], errors="coerce")
    if (
        months.isna().any()
        or (months != np.floor(months)).any()
        or (~months.between(1, 12)).any()
    ):
        raise ValueError("station 885 MONTH must be an integer in the 1..12 domain")
    expected_scope = "single_station_descriptive_case_not_general_proof"
    if not station_885["CASE_SCOPE"].astype(str).eq(expected_scope).all():
        raise ValueError("station 885 CASE_SCOPE violates the descriptive-case domain")
    expected = station_885.copy()
    _apply_station_885_evidence(
        expected,
        matched_contrast,
        station_structure=station_structure,
        state_condition=state_condition,
        coverage_sensitivity=coverage_sensitivity,
        analysis_station_ids=population,
    )
    if not np.array_equal(
        station_885["EVIDENCE_GRADE"].astype(str).to_numpy(),
        expected["EVIDENCE_GRADE"].astype(str).to_numpy(),
    ) or not np.array_equal(
        station_885["LIMITATION"].astype(str).to_numpy(),
        expected["LIMITATION"].astype(str).to_numpy(),
    ):
        raise ValueError("station 885 evidence differs from the independent H1 gate")
    if station_885["EVIDENCE_GRADE"].eq("자료로 지지").any():
        raise ValueError("station 885 evidence may not claim direct support")


def _validate_canonical_station_885_presence(
    truth: pd.DataFrame | None,
    station_structure: pd.DataFrame,
    station_885: pd.DataFrame,
) -> None:
    """정본 529 분석에서 885 원자료·case 전체 삭제를 금지한다."""

    if truth is None or len(truth) != 529 or "SIGNIFICANCE_CLASS" not in truth:
        return
    expected_counts = {
        "meaningful_improvement": 221,
        "meaningful_worsening": 109,
        "practically_equivalent": 115,
        "uncertain": 84,
    }
    if truth["SIGNIFICANCE_CLASS"].value_counts().to_dict() != expected_counts:
        return
    truth_ids = [
        _normalize_station_id(value, field="canonical truth STN")
        for value in truth["STN"]
    ]
    if truth_ids.count("885") != 1:
        raise ValueError("canonical population must contain station 885 exactly once")
    structure = station_structure.loc[
        station_structure["STN"].astype(str).str.strip().eq("885")
    ]
    if len(structure) != 1:
        raise ValueError("canonical station 885 structure row is missing or duplicated")
    total = pd.to_numeric(
        structure.iloc[0].get("TOTAL_PAIRED_HOURS"), errors="coerce"
    )
    reproduced = pd.to_numeric(
        structure.iloc[0].get("REPRODUCED_HOURS"), errors="coerce"
    )
    if not (
        np.isfinite(total)
        and total > 0
        and np.isfinite(reproduced)
        and reproduced > 0
        and not station_885.empty
        and set(station_885["STN"].astype(str).str.strip()) == {"885"}
    ):
        raise ValueError("canonical station 885 cannot have zero data or an empty case")


def _has_reparse_component(path: Path) -> bool:
    current = path
    while True:
        if current.exists():
            is_junction = bool(getattr(current, "is_junction", lambda: False)())
            if current.is_symlink() or is_junction:
                return True
        if current == current.parent:
            break
        current = current.parent
    return False


def guard_production_output_directory(out_dir: os.PathLike[str] | str) -> Path:
    """production output을 프로젝트의 단 하나인 canonical leaf로 제한한다."""

    raw = Path(out_dir).expanduser()
    if ".." in raw.parts:
        raise ValueError("production out_dir must not contain parent traversal")
    lexical = Path(os.path.abspath(str(raw)))
    canonical = Path(CANONICAL_OUTPUT_DIR)
    if not canonical.parent.is_dir():
        raise ValueError("canonical production parent does not exist")
    if str(lexical) != str(canonical):
        raise ValueError("production out_dir must equal the canonical project leaf exactly")
    if _has_reparse_component(lexical.parent) or (lexical.exists() and _has_reparse_component(lexical)):
        raise ValueError("production output path must not traverse a symlink or junction")
    resolved = lexical.resolve(strict=False)
    if resolved != canonical.resolve(strict=False):
        raise ValueError("resolved production output differs from canonical leaf")
    return resolved


def guard_injected_output_directory(
    out_dir: os.PathLike[str] | str,
    *,
    allowed_fullnet_root: os.PathLike[str] | str,
) -> Path:
    """주입형 fixture output을 명시한 fullnet root의 정확한 child로만 허용한다."""

    root_raw = Path(allowed_fullnet_root).expanduser()
    output_raw = Path(out_dir).expanduser()
    if ".." in root_raw.parts or ".." in output_raw.parts:
        raise ValueError("injected output paths must not contain parent traversal")
    root = Path(os.path.abspath(str(root_raw)))
    output = Path(os.path.abspath(str(output_raw)))
    canonical = Path(os.path.abspath(str(Path(CANONICAL_OUTPUT_DIR).expanduser())))
    resolved_root = root.resolve(strict=False)
    resolved_output = output.resolve(strict=False)
    resolved_canonical = canonical.resolve(strict=False)

    def overlaps(left: Path, right: Path) -> bool:
        return (
            left == right
            or left.is_relative_to(right)
            or right.is_relative_to(left)
        )

    if overlaps(root, canonical) or overlaps(output, canonical) or overlaps(
        resolved_root, resolved_canonical
    ) or overlaps(resolved_output, resolved_canonical):
        raise ValueError("injected output must not use the canonical production path")
    if not root.is_dir():
        raise ValueError("allowed injected fullnet root must already exist")
    if _has_reparse_component(root):
        raise ValueError("allowed injected root must not be a symlink or junction")
    expected = root / "bias_physical_followup"
    if str(output) != str(expected):
        raise ValueError("injected output must be the exact bias_physical_followup child")
    if output.exists() and _has_reparse_component(output):
        raise ValueError("injected output leaf must not be a symlink or junction")
    return output.resolve(strict=False)


def guard_output_directory(out_dir: os.PathLike[str] | str) -> Path:
    """호환용 production alias. 새 코드는 명시적 production/injected guard를 사용한다."""

    return guard_production_output_directory(out_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sqlite_sidecar_paths(db_path: os.PathLike[str] | str) -> dict[str, Path]:
    """SQLite main file과 같은 이름을 쓰는 저널 sidecar 경로를 반환한다."""

    resolved = Path(db_path).expanduser().resolve(strict=True)
    return {
        "wal": Path(f"{resolved}-wal"),
        "shm": Path(f"{resolved}-shm"),
        "journal": Path(f"{resolved}-journal"),
    }


def _sqlite_rollback_snapshot(
    db_path: os.PathLike[str] | str,
    *,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """sidecar가 없는 DELETE rollback-journal SQLite snapshot인지 검증한다.

    ``immutable=1``을 쓰지 않는다. WAL의 main file만 보고 commit을
    놓치는 경로를 원천 차단하기 위해 PRAGMA 조회 전·후 sidecar를 모두 본다.
    """

    resolved = Path(db_path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("production DB source must be a regular file")
    sidecars = _sqlite_sidecar_paths(resolved)

    def reject_sidecars() -> None:
        present = [name for name, path in sidecars.items() if path.exists()]
        if present:
            raise ValueError(
                "production SQLite WAL/SHM/journal sidecar must be absent: "
                + ", ".join(present)
            )

    reject_sidecars()
    owns_connection = connection is None
    handle = connection
    try:
        if handle is None:
            uri = resolved.as_uri() + "?mode=ro"
            handle = sqlite3.connect(uri, uri=True)
        mode_row = handle.execute("PRAGMA journal_mode").fetchone()
        mode = "" if not mode_row else str(mode_row[0]).strip().lower()
    except sqlite3.Error as exc:
        raise ValueError("production DB must be a readable SQLite database") from exc
    finally:
        if owns_connection and handle is not None:
            handle.close()
    reject_sidecars()
    if mode != "delete":
        raise ValueError(
            f"production SQLite journal_mode must be delete rollback mode, got {mode!r}"
        )
    return {
        "db_journal_mode": mode,
        "db_wal_present": False,
        "db_shm_present": False,
        "db_journal_present": False,
    }


def _file_identity(
    path: os.PathLike[str] | str,
    *,
    role: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"production {role} source must be a regular file")
    hash_started = time.perf_counter()
    source_hash = _sha256(resolved)
    hash_elapsed = max(0.0, time.perf_counter() - hash_started)
    if role == "db" and metrics is not None:
        _record_db_full_hash(metrics, elapsed_seconds=hash_elapsed)
    return {
        f"{role}_path": str(resolved),
        f"{role}_bytes": int(resolved.stat().st_size),
        f"{role}_sha256": source_hash,
    }


def _capture_production_source_metadata(
    *,
    db_path: os.PathLike[str] | str,
    station_results_path: os.PathLike[str] | str,
    metadata_path: os.PathLike[str] | str,
    terrain_path: os.PathLike[str] | str,
    aws_base_folder: os.PathLike[str] | str,
    aws_file_inventory: Sequence[Mapping[str, Any]],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """분석 시작 시점의 production 원본 identity를 exact schema로 고정한다."""

    aws_root = Path(aws_base_folder).expanduser().resolve(strict=True)
    if not aws_root.is_dir():
        raise ValueError("production AWS base folder must be a directory")
    result: dict[str, Any] = {"source_mode": "production"}
    for role, path in [
        ("db", db_path),
        ("station_results", station_results_path),
        ("metadata", metadata_path),
        ("terrain", terrain_path),
    ]:
        result.update(_file_identity(path, role=role, metrics=metrics))
    result.update(_sqlite_rollback_snapshot(db_path))
    result["aws_base_folder"] = str(aws_root)
    result["aws_file_inventory"] = [dict(entry) for entry in aws_file_inventory]
    return result


def _record_db_full_hash(
    metrics: dict[str, Any],
    *,
    elapsed_seconds: float,
) -> None:
    """station 수와 무관한 DB 전체 hash 횟수·시간을 누적한다."""

    count = metrics.get("db_full_hash_verification_count", 0)
    elapsed = metrics.get("db_full_hash_verification_elapsed_seconds", 0.0)
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or int(count) < 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float, np.integer, np.floating))
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0
        or not np.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0
    ):
        raise ValueError("production DB full-hash metrics are invalid")
    metrics["db_full_hash_verification_count"] = int(count) + 1
    metrics["db_full_hash_verification_elapsed_seconds"] = (
        float(elapsed) + float(elapsed_seconds)
    )


def _create_private_sqlite_snapshot(
    source_path: os.PathLike[str] | str,
    *,
    temp_parent: os.PathLike[str] | str,
    source_identity: Mapping[str, Any],
    metrics: dict[str, Any],
    source_connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """rollback source를 무작위 private SQLite 논리 snapshot으로 한 번 복제한다."""

    source = Path(source_path).expanduser().resolve(strict=True)
    parent = Path(temp_parent).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("private snapshot parent must be an existing directory")
    if str(source) != source_identity.get("db_path"):
        raise ValueError("private snapshot source differs from captured DB path")
    if source.stat().st_size != source_identity.get("db_bytes"):
        raise ValueError("private snapshot source size differs from capture")
    if not _is_sha256_hex(source_identity.get("db_sha256")):
        raise ValueError("private snapshot source capture hash is invalid")
    _sqlite_rollback_snapshot(source)

    if os.name == "nt":
        # bundled Python의 tempfile 0700 DACL이 OneDrive에서 생성자 자신도
        # 접근하지 못하게 되는 Windows 런타임 버그를 피한다. UUID 이름과
        # cleanup containment 검사는 동일하게 유지한다.
        private_dir = parent / f".bias-db-snapshot-{uuid.uuid4().hex}"
        private_dir.mkdir()
    else:
        private_dir = Path(
            tempfile.mkdtemp(prefix=".bias-db-snapshot-", dir=str(parent))
        )
        os.chmod(private_dir, 0o700)
    private_dir = private_dir.resolve(strict=True)
    snapshot_path = private_dir / f"snapshot-{uuid.uuid4().hex}.db"
    source_handle = source_connection
    owns_source_connection = source_handle is None
    destination: sqlite3.Connection | None = None
    creation_started = time.perf_counter()
    try:
        if source_handle is None:
            source_uri = source.as_uri() + "?mode=ro"
            source_handle = sqlite3.connect(source_uri, uri=True)
        if source_handle.in_transaction:
            raise ValueError("private snapshot source connection must start outside a transaction")
        source_handle.execute("BEGIN")
        source_handle.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        rows = source_handle.execute("PRAGMA database_list").fetchall()
        mains = [row[2] for row in rows if len(row) >= 3 and row[1] == "main"]
        if len(mains) != 1 or Path(mains[0]).expanduser().resolve(strict=True) != source:
            raise ValueError("private snapshot source connection points to another DB")
        _sqlite_rollback_snapshot(source, connection=source_handle)
        destination = sqlite3.connect(snapshot_path)
        source_handle.backup(destination)
        destination.commit()
    except Exception:
        if destination is not None:
            destination.close()
        if source_handle is not None and source_handle.in_transaction:
            source_handle.rollback()
        if owns_source_connection and source_handle is not None:
            source_handle.close()
        shutil.rmtree(private_dir, ignore_errors=True)
        raise
    else:
        creation_elapsed = max(0.0, time.perf_counter() - creation_started)
        destination.close()
        if source_handle.in_transaction:
            source_handle.rollback()
        if owns_source_connection:
            source_handle.close()

    try:
        _sqlite_rollback_snapshot(snapshot_path)
        check = sqlite3.connect(snapshot_path)
        try:
            result = check.execute("PRAGMA quick_check").fetchone()
        finally:
            check.close()
        if not result or str(result[0]).strip().lower() != "ok":
            raise ValueError("private SQLite snapshot failed quick_check")

        hash_started = time.perf_counter()
        snapshot_hash = _sha256(snapshot_path)
        hash_elapsed = max(0.0, time.perf_counter() - hash_started)
        _record_db_full_hash(metrics, elapsed_seconds=hash_elapsed)
        stat = snapshot_path.stat()
        return {
            "db_snapshot_private_dir": str(private_dir),
            "db_snapshot_path": str(snapshot_path.resolve(strict=True)),
            "db_snapshot_parent": str(parent),
            "db_snapshot_bytes": int(stat.st_size),
            "db_snapshot_mtime_ns": int(stat.st_mtime_ns),
            "db_snapshot_sha256": snapshot_hash,
            "db_snapshot_creation_elapsed_seconds": float(creation_elapsed),
            "db_snapshot_hash_elapsed_seconds": float(hash_elapsed),
        }
    except Exception:
        shutil.rmtree(private_dir, ignore_errors=True)
        raise


def _open_immutable_sqlite_snapshot(
    snapshot_identity: Mapping[str, Any],
) -> sqlite3.Connection:
    """private snapshot을 sidecar를 만들지 않는 immutable read-only 연결로 연다."""

    _validate_private_snapshot_light_identity(snapshot_identity)
    path = Path(str(snapshot_identity["db_snapshot_path"]))
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validate_private_snapshot_light_identity(
    snapshot_identity: Mapping[str, Any],
) -> None:
    """query 경계에서 전체 hash 없이 private snapshot identity를 검사한다."""

    path_value = snapshot_identity.get("db_snapshot_path")
    if not isinstance(path_value, str):
        raise ValueError("private DB snapshot path is missing")
    path = Path(path_value).expanduser().resolve(strict=True)
    if str(path) != path_value or not path.is_file():
        raise ValueError("private DB snapshot resolved path changed")
    stat = path.stat()
    if (
        stat.st_size != snapshot_identity.get("db_snapshot_bytes")
        or stat.st_mtime_ns != snapshot_identity.get("db_snapshot_mtime_ns")
    ):
        raise ValueError("private DB snapshot lightweight identity changed")
    present = [
        name
        for name, sidecar in _sqlite_sidecar_paths(path).items()
        if sidecar.exists()
    ]
    if present:
        raise ValueError(
            "private DB snapshot sidecar must remain absent: " + ", ".join(present)
        )


def _validate_private_snapshot_after_query(
    snapshot_identity: Mapping[str, Any],
    connection: sqlite3.Connection,
) -> None:
    """각 분석 query 뒤 immutable connection과 snapshot 경량 identity를 확인한다."""

    _validate_private_snapshot_light_identity(snapshot_identity)
    if not connection.in_transaction:
        raise ValueError("private DB snapshot queries must share one read transaction")
    rows = connection.execute("PRAGMA database_list").fetchall()
    main_files = [row[2] for row in rows if len(row) >= 3 and row[1] == "main"]
    if len(main_files) != 1:
        raise ValueError("private snapshot connection must expose exactly one main DB")
    connected = Path(main_files[0]).expanduser().resolve(strict=True)
    expected = Path(str(snapshot_identity["db_snapshot_path"])).resolve(strict=True)
    if connected != expected:
        raise ValueError("analysis connection points to a different DB than the snapshot")
    query_only = connection.execute("PRAGMA query_only").fetchone()
    if not query_only or int(query_only[0]) != 1:
        raise ValueError("private snapshot analysis connection must be query-only")


def _validate_private_snapshot_full_hash(
    snapshot_identity: dict[str, Any],
    *,
    metrics: dict[str, Any],
) -> None:
    """게시 직전 private snapshot 전체 hash를 생성 시 identity와 대조한다."""

    _validate_private_snapshot_light_identity(snapshot_identity)
    path = Path(str(snapshot_identity["db_snapshot_path"]))
    started = time.perf_counter()
    observed = _sha256(path)
    elapsed = max(0.0, time.perf_counter() - started)
    _record_db_full_hash(metrics, elapsed_seconds=elapsed)
    snapshot_identity["db_snapshot_hash_elapsed_seconds"] = float(
        snapshot_identity.get("db_snapshot_hash_elapsed_seconds", 0.0)
    ) + elapsed
    if observed != snapshot_identity.get("db_snapshot_sha256"):
        raise ValueError("private DB snapshot SHA-256 changed before publication")


def _cleanup_private_sqlite_snapshot(snapshot_identity: Mapping[str, Any]) -> None:
    """생성한 무작위 private snapshot directory만 안전하게 제거한다."""

    directory_value = snapshot_identity.get("db_snapshot_private_dir")
    path_value = snapshot_identity.get("db_snapshot_path")
    parent_value = snapshot_identity.get("db_snapshot_parent")
    if not all(isinstance(value, str) for value in (directory_value, path_value, parent_value)):
        raise ValueError("private DB snapshot cleanup identity is incomplete")
    directory = Path(str(directory_value)).resolve(strict=False)
    path = Path(str(path_value)).resolve(strict=False)
    parent = Path(str(parent_value)).resolve(strict=True)
    if (
        directory.parent != parent
        or not directory.name.startswith(".bias-db-snapshot-")
        or path.parent != directory
        or not path.name.startswith("snapshot-")
        or path.suffix != ".db"
    ):
        raise ValueError("refusing unsafe private DB snapshot cleanup target")
    if directory.exists():
        shutil.rmtree(directory)


def _validate_live_production_source_identity(
    sources: Mapping[str, Any],
    *,
    verify_db_hash: bool = True,
) -> None:
    """분석 시작 identity와 현재 원본이 같은지 게시 직전 다시 확인한다."""

    for role in ("db", "station_results", "metadata", "terrain"):
        path_value = sources.get(f"{role}_path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"production source {role} path is missing")
        try:
            resolved = Path(path_value).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"production source {role} changed or disappeared") from exc
        if str(resolved) != path_value or not resolved.is_file():
            raise ValueError(f"production source {role} resolved path changed")
        expected_bytes = sources.get(f"{role}_bytes")
        expected_hash = sources.get(f"{role}_sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError(f"production source {role} byte identity is invalid")
        if not _is_sha256_hex(expected_hash):
            raise ValueError(f"production source {role} SHA-256 hash is invalid")
        if resolved.stat().st_size != expected_bytes:
            raise ValueError(f"production source {role} changed during/after analysis")
        if (role != "db" or verify_db_hash) and _sha256(resolved) != expected_hash:
            raise ValueError(f"production source {role} changed during/after analysis")
        if role == "db":
            snapshot = _sqlite_rollback_snapshot(resolved)
            for key, expected in snapshot.items():
                if sources.get(key) is not expected and sources.get(key) != expected:
                    raise ValueError(
                        "production SQLite journal/sidecar state changed during analysis"
                    )
    aws_root_value = sources.get("aws_base_folder")
    if not isinstance(aws_root_value, str) or not aws_root_value:
        raise ValueError("production AWS base folder path is missing")
    try:
        aws_root = Path(aws_root_value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("production AWS base folder changed or disappeared") from exc
    if str(aws_root) != aws_root_value or not aws_root.is_dir():
        raise ValueError("production AWS base folder resolved path changed")
    inventory = sources.get("aws_file_inventory")
    if not isinstance(inventory, list):
        raise ValueError("production AWS inventory is missing")
    seen: set[str] = set()
    for entry in inventory:
        if not isinstance(entry, Mapping) or set(entry) != {
            "relative_path",
            "bytes",
            "sha256",
            "station_id",
            "month",
        }:
            raise ValueError("production AWS inventory entry schema mismatch")
        relative = str(entry["relative_path"])
        relative_path = Path(relative)
        if (
            relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError("production AWS inventory path is duplicate or unsafe")
        seen.add(relative)
        expected_bytes = entry["bytes"]
        expected_hash = entry["sha256"]
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not _is_sha256_hex(expected_hash)
        ):
            raise ValueError("production AWS inventory identity is invalid")
        source = aws_root / relative_path
        if (
            not source.is_file()
            or source.stat().st_size != expected_bytes
            or _sha256(source) != expected_hash
        ):
            raise ValueError(f"inventoried AWS source changed: {relative}")


def _validate_original_db_full_hash(
    sources: Mapping[str, Any],
    *,
    metrics: dict[str, Any],
) -> None:
    """게시 직전 원본 DB가 시작 capture와 동일한지 전체 hash로 한 번 대조한다."""

    path_value = sources.get("db_path")
    expected_hash = sources.get("db_sha256")
    expected_bytes = sources.get("db_bytes")
    if not isinstance(path_value, str) or not _is_sha256_hex(expected_hash):
        raise ValueError("production source DB identity is invalid")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("production source DB size changed during analysis")
    _sqlite_rollback_snapshot(path)
    started = time.perf_counter()
    observed = _sha256(path)
    elapsed = max(0.0, time.perf_counter() - started)
    _record_db_full_hash(metrics, elapsed_seconds=elapsed)
    if observed != expected_hash:
        raise ValueError("production source DB SHA-256 changed during analysis")


def _analysis_config_payload() -> dict[str, Any]:
    return {
        "spatial_settings": _sensitivity_settings(),
        "coverage_settings": [
            {
                "COVERAGE_ID": f"d{days}_m{months}",
                "MIN_DAYS_PER_STATE": days,
                "MIN_MONTHS_PER_STATE": months,
            }
            for days, months in product(SENSITIVITY_MIN_DAYS, SENSITIVITY_MIN_MONTHS)
        ],
        "primary_setting_id": PRIMARY_SETTING_ID,
        "primary_coverage_id": PRIMARY_COVERAGE_ID,
        "minimum_n_eff": 3,
        "block_length_days": 7,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20260827,
    }


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_csv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """DataFrame을 게시 전 메모리에서 고정하는 결정적 UTF-8-SIG CSV bytes."""

    _validate_table(frame, columns, "canonical CSV payload")
    # StringIO -> str -> encode는 큰 CSV 전체의 Unicode 복제본을 한 번 더 만든다.
    # UTF-8-SIG text wrapper로 BytesIO에 직접 기록해 정본 bytes는 그대로 유지한다.
    stream = io.BytesIO()
    text_stream = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    writer = csv.writer(text_stream, lineterminator="\n")
    writer.writerow(columns)
    for values in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        canonical: list[Any] = []
        for value in values:
            if pd.isna(value):
                canonical.append("")
            elif isinstance(value, (bool, np.bool_)):
                canonical.append("True" if bool(value) else "False")
            elif isinstance(value, (int, np.integer)):
                canonical.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError("canonical CSV cannot contain infinite numbers")
                canonical.append(repr(numeric))
            else:
                canonical.append(str(value))
        writer.writerow(canonical)
    text_stream.flush()
    text_stream.detach()
    return stream.getvalue()


def _canonical_csv_payloads(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, bytes]:
    if set(tables) != set(CSV_SCHEMAS):
        raise ValueError("canonical CSV payload table set mismatch")
    return {
        name: _canonical_csv_bytes(tables[name], columns)
        for name, columns in CSV_SCHEMAS.items()
    }


def _primary_inference_table_hashes(
    canonical_csv_payloads: Mapping[str, bytes],
) -> dict[str, str]:
    if not set(PRIMARY_ANCHORED_CSV_NAMES).issubset(canonical_csv_payloads):
        raise ValueError("primary inference canonical payload is incomplete")
    return {
        name: hashlib.sha256(canonical_csv_payloads[name]).hexdigest()
        for name in PRIMARY_ANCHORED_CSV_NAMES
    }


def _validate_canonical_invariants_exact(value: Any) -> None:
    """canonical nested JSON을 값과 *JSON 타입*까지 exact 검증한다.

    Python에서 ``bool``은 ``int``의 하위 타입이며 ``True == 1``이므로,
    단순 dict 동등성으로는 count의 ``1 -> true`` 변조를 막지 못한다.
    """

    expected = _expected_canonical_invariants()

    def compare(observed: Any, contract: Any, path: str) -> None:
        if isinstance(contract, dict):
            if type(observed) is not dict or set(observed) != set(contract):
                raise ValueError(f"canonical invariant schema/type mismatch at {path}")
            for key in contract:
                compare(observed[key], contract[key], f"{path}.{key}")
            return
        if isinstance(contract, list):
            if type(observed) is not list or len(observed) != len(contract):
                raise ValueError(f"canonical invariant schema/type mismatch at {path}")
            for index, (left, right) in enumerate(zip(observed, contract)):
                compare(left, right, f"{path}[{index}]")
            return
        if type(observed) is not type(contract):
            raise ValueError(f"canonical invariant JSON type mismatch at {path}")
        if observed != contract:
            raise ValueError(f"canonical invariant value mismatch at {path}")

    compare(value, expected, "canonical_invariants")


def _station_population_identity(station_ids: Iterable[Any]) -> tuple[int, str]:
    """순서와 무관한 station 모집단 count·SHA-256 identity를 만든다."""

    normalized = [
        _normalize_station_id(value, field="analysis population STN")
        for value in station_ids
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("analysis population STN must be unique")
    ordered = sorted(normalized, key=lambda value: (int(value), value))
    return len(ordered), _canonical_json_hash(ordered)


def _primary_ledger_source_binding_sha256(
    *,
    ledger_sha256: str,
    population_sha256: str,
    config_sha256: str,
    sources: Mapping[str, Any],
    primary_table_sha256s: Mapping[str, Any] | None = None,
) -> str:
    """ledger identity를 게시된 source snapshot·모집단·설정과 결합한다."""

    if not all(
        _is_sha256_hex(value)
        for value in (ledger_sha256, population_sha256, config_sha256)
    ):
        raise ValueError("primary ledger binding inputs must be SHA-256 hashes")
    if primary_table_sha256s is None:
        primary_table_sha256s = {
            name: ledger_sha256 for name in PRIMARY_ANCHORED_CSV_NAMES
        }
    if (
        type(primary_table_sha256s) is not dict
        or set(primary_table_sha256s) != set(PRIMARY_ANCHORED_CSV_NAMES)
        or not all(_is_sha256_hex(value) for value in primary_table_sha256s.values())
        or primary_table_sha256s["primary_eligibility_ledger.csv"]
        != ledger_sha256
    ):
        raise ValueError("primary inference table binding hashes are invalid")
    source_mode = sources.get("source_mode")
    if source_mode == "production":
        source_hashes = {
            key: sources.get(key)
            for key in (
                "db_sha256",
                "db_snapshot_sha256",
                "station_results_sha256",
                "metadata_sha256",
                "terrain_sha256",
                "aws_inventory_aggregate_sha256",
            )
        }
        if not all(_is_sha256_hex(value) for value in source_hashes.values()):
            raise ValueError("production ledger binding source hashes are invalid")
        snapshot_metadata = {
            key: sources.get(key)
            for key in (
                "db_snapshot_bytes",
                "db_snapshot_creation_elapsed_seconds",
                "db_snapshot_hash_elapsed_seconds",
            )
        }
        snapshot_bytes = snapshot_metadata["db_snapshot_bytes"]
        snapshot_times = [
            snapshot_metadata["db_snapshot_creation_elapsed_seconds"],
            snapshot_metadata["db_snapshot_hash_elapsed_seconds"],
        ]
        if (
            isinstance(snapshot_bytes, bool)
            or not isinstance(snapshot_bytes, (int, np.integer))
            or int(snapshot_bytes) <= 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) < 0
                for value in snapshot_times
            )
        ):
            raise ValueError("production ledger snapshot metadata is invalid")
    elif source_mode == "injected":
        source_hashes = {
            "aws_inventory_aggregate_sha256": sources.get(
                "aws_inventory_aggregate_sha256"
            )
        }
        snapshot_metadata = None
    else:
        raise ValueError("primary ledger binding source mode is invalid")
    return _canonical_json_hash(
        {
            "ledger_sha256": ledger_sha256,
            "population_sha256": population_sha256,
            "config_sha256": config_sha256,
            "source_mode": source_mode,
            "source_hashes": source_hashes,
            "db_snapshot_metadata": snapshot_metadata,
            "primary_table_sha256s": dict(primary_table_sha256s),
        }
    )


def _validate_trusted_primary_eligibility_binding(
    metadata: Mapping[str, Any],
    trusted_binding_sha256: str | None,
) -> None:
    """production bundle 밖에 고정한 ledger-source binding과 대조한다.

    ``analysis_metadata.json``과 manifest는 같은 게시 트리 안에 있어
    함께 재해시할 수 있다. 따라서 production의 사후 검증은 runner가
    원표 검증 직후 메모리에 고정하고 반환한 외부 digest가 없으면
    성공을 선언하지 않는다.
    """

    sources = metadata.get("input_sources")
    if not isinstance(sources, Mapping) or sources.get("source_mode") != "production":
        raise ValueError("trusted primary ledger anchor is only valid for production")
    if not _is_sha256_hex(trusted_binding_sha256):
        raise ValueError("external trusted primary ledger anchor is required")
    observed = metadata.get("primary_eligibility_source_binding_sha256")
    if observed != trusted_binding_sha256:
        raise ValueError("primary ledger differs from the external trusted anchor")


def _build_analysis_metadata(
    *,
    canonical: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    analysis_station_ids: Iterable[Any],
    input_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources = dict(input_sources or {})
    sources.setdefault(
        "source_mode", "production" if "aws_base_folder" in sources else "injected"
    )
    inventory = list(sources.get("aws_file_inventory", []))
    sources["aws_file_inventory"] = inventory
    sources["aws_inventory_aggregate_sha256"] = _canonical_json_hash(inventory)
    metric_payload: dict[str, Any] = {
        key: 0 for key in _METADATA_CORE_METRICS
    }
    metric_payload.update(dict(metrics))
    station_count, station_ids_sha256 = _station_population_identity(
        analysis_station_ids
    )
    pending_ledger_hash = _canonical_json_hash(
        {"pending_station_ids_sha256": station_ids_sha256}
    )
    pending_primary_hashes = {
        name: pending_ledger_hash for name in PRIMARY_ANCHORED_CSV_NAMES
    }
    config_hash = _canonical_json_hash(_analysis_config_payload())
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "analysis_population": "all_stations_before_descriptive_subset_join",
        "analysis_station_count": station_count,
        "analysis_station_ids_sha256": station_ids_sha256,
        "primary_eligibility_ledger_rows": station_count,
        "primary_eligibility_ledger_sha256": pending_ledger_hash,
        "primary_inference_table_sha256s": pending_primary_hashes,
        "primary_eligibility_source_binding_sha256": (
            _primary_ledger_source_binding_sha256(
                ledger_sha256=pending_ledger_hash,
                population_sha256=station_ids_sha256,
                config_sha256=config_hash,
                sources=sources,
                primary_table_sha256s=pending_primary_hashes,
            )
        ),
        "canonical_invariants": canonical,
        "canonical_invariants_sha256": (
            _canonical_json_hash(canonical) if canonical is not None else None
        ),
        "time_interpretation": "naive_TM_is_KST",
        "solar_state_thresholds_deg": {
            "bright": ">0",
            "transition": "[-6,0]",
            "dark": "<-6",
        },
        "spatial_state_term": "regional_inversion_like",
        "spatial_state_not_equivalent_to": "measured_vertical_temperature_inversion",
        "tpi_status": "NOT_MEASURED",
        "analysis_config": _analysis_config_payload(),
        "state_contrast_uncertainty": (
            "equal_common_month_solar_strata_calendar_contiguous_7day_blocks_"
            "shared_across_methods_and_states_without_month_wrap"
        ),
        "h2_evidence_scope": "hourly_extrapolation_from_Kim_Tmax_Tmin",
        "input_sources": sources,
        **metric_payload,
    }


def _validate_analysis_metadata(
    metadata: Any,
    *,
    expected_source_mode: str | None = None,
    validate_live_sources: bool = True,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("analysis metadata must be a JSON object")
    fixed_keys = {
        "schema_version",
        "analysis_population",
        "analysis_station_count",
        "analysis_station_ids_sha256",
        "primary_eligibility_ledger_rows",
        "primary_eligibility_ledger_sha256",
        "primary_inference_table_sha256s",
        "primary_eligibility_source_binding_sha256",
        "canonical_invariants",
        "canonical_invariants_sha256",
        "time_interpretation",
        "solar_state_thresholds_deg",
        "spatial_state_term",
        "spatial_state_not_equivalent_to",
        "tpi_status",
        "analysis_config",
        "state_contrast_uncertainty",
        "h2_evidence_scope",
        "input_sources",
    }
    required_keys = fixed_keys | set(_METADATA_CORE_METRICS)
    allowed_keys = required_keys | set(_METADATA_PRODUCTION_INTEGER_METRICS) | set(
        _METADATA_PRODUCTION_FLOAT_METRICS
    )
    if not required_keys.issubset(metadata) or not set(metadata).issubset(allowed_keys):
        raise ValueError("analysis metadata required/exact key contract mismatch")
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("analysis metadata schema_version mismatch")
    if metadata.get("analysis_population") != "all_stations_before_descriptive_subset_join":
        raise ValueError("analysis metadata population mismatch")
    station_count = metadata.get("analysis_station_count")
    if (
        isinstance(station_count, bool)
        or not isinstance(station_count, (int, np.integer))
        or station_count < 0
    ):
        raise ValueError("analysis metadata station count must be a non-negative integer")
    if not _is_sha256_hex(metadata.get("analysis_station_ids_sha256")):
        raise ValueError("analysis metadata station identity must be a SHA-256 hash")
    ledger_rows = metadata.get("primary_eligibility_ledger_rows")
    if (
        isinstance(ledger_rows, bool)
        or not isinstance(ledger_rows, (int, np.integer))
        or int(ledger_rows) != int(station_count)
    ):
        raise ValueError("primary ledger row count must equal the station population")
    if not _is_sha256_hex(metadata.get("primary_eligibility_ledger_sha256")):
        raise ValueError("primary ledger identity must be a SHA-256 hash")
    primary_table_sha256s = metadata.get("primary_inference_table_sha256s")
    if (
        type(primary_table_sha256s) is not dict
        or set(primary_table_sha256s) != set(PRIMARY_ANCHORED_CSV_NAMES)
        or not all(_is_sha256_hex(value) for value in primary_table_sha256s.values())
        or primary_table_sha256s["primary_eligibility_ledger.csv"]
        != metadata["primary_eligibility_ledger_sha256"]
    ):
        raise ValueError("primary inference table identity contract mismatch")
    expected_ledger_binding = _primary_ledger_source_binding_sha256(
        ledger_sha256=metadata["primary_eligibility_ledger_sha256"],
        population_sha256=metadata["analysis_station_ids_sha256"],
        config_sha256=_canonical_json_hash(_analysis_config_payload()),
        sources=metadata.get("input_sources", {}),
        primary_table_sha256s=primary_table_sha256s,
    )
    if metadata.get("primary_eligibility_source_binding_sha256") != expected_ledger_binding:
        raise ValueError("primary ledger source/config binding hash mismatch")
    canonical = metadata.get("canonical_invariants")
    canonical_hash = metadata.get("canonical_invariants_sha256")
    if canonical is not None and not isinstance(canonical, dict):
        raise ValueError("analysis metadata canonical invariants must be an object or null")
    if metadata.get("analysis_config") != _analysis_config_payload():
        raise ValueError("analysis metadata config differs from fixed contract")
    if metadata.get("time_interpretation") != "naive_TM_is_KST":
        raise ValueError("analysis metadata time interpretation mismatch")
    if metadata.get("solar_state_thresholds_deg") != {
        "bright": ">0",
        "transition": "[-6,0]",
        "dark": "<-6",
    }:
        raise ValueError("analysis metadata solar-state domain mismatch")
    if metadata.get("spatial_state_term") != "regional_inversion_like":
        raise ValueError("analysis metadata spatial state term mismatch")
    if metadata.get("spatial_state_not_equivalent_to") != "measured_vertical_temperature_inversion":
        raise ValueError("analysis metadata vertical-inversion limitation mismatch")
    if metadata.get("tpi_status") != "NOT_MEASURED":
        raise ValueError("analysis metadata TPI status mismatch")
    if metadata.get("state_contrast_uncertainty") != (
        "equal_common_month_solar_strata_calendar_contiguous_7day_blocks_"
        "shared_across_methods_and_states_without_month_wrap"
    ):
        raise ValueError("analysis metadata bootstrap estimand mismatch")
    if metadata.get("h2_evidence_scope") != "hourly_extrapolation_from_Kim_Tmax_Tmin":
        raise ValueError("analysis metadata H2 evidence scope mismatch")

    for key in _METADATA_CORE_METRICS:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"analysis metadata metric {key} must be a non-negative integer")
    sources = metadata.get("input_sources")
    if not isinstance(sources, dict):
        raise ValueError("analysis metadata input_sources must be an object")
    source_mode = sources.get("source_mode")
    if source_mode not in {"injected", "production"}:
        raise ValueError("analysis metadata source_mode is invalid")
    if expected_source_mode is not None:
        if expected_source_mode not in {"injected", "production"}:
            raise ValueError("expected analysis metadata source mode is invalid")
        if source_mode != expected_source_mode:
            raise ValueError(
                "analysis metadata source_mode does not match the publication path"
            )
    injected_source_keys = {
        "source_mode",
        "aws_file_inventory",
        "aws_inventory_aggregate_sha256",
    }
    production_source_keys = injected_source_keys | {
        "db_path",
        "db_bytes",
        "db_sha256",
        "db_snapshot_sha256",
        "db_snapshot_bytes",
        "db_snapshot_creation_elapsed_seconds",
        "db_snapshot_hash_elapsed_seconds",
        "db_journal_mode",
        "db_wal_present",
        "db_shm_present",
        "db_journal_present",
        "station_results_path",
        "station_results_bytes",
        "station_results_sha256",
        "metadata_path",
        "metadata_bytes",
        "metadata_sha256",
        "terrain_path",
        "terrain_bytes",
        "terrain_sha256",
        "aws_base_folder",
    }
    expected_source_keys = (
        injected_source_keys if source_mode == "injected" else production_source_keys
    )
    if set(sources) != expected_source_keys:
        raise ValueError("analysis metadata input-source schema mismatch")
    production_metrics = set(_METADATA_PRODUCTION_INTEGER_METRICS) | set(
        _METADATA_PRODUCTION_FLOAT_METRICS
    )
    if source_mode == "production" and not production_metrics.issubset(metadata):
        raise ValueError("production analysis metadata lacks acceptance metrics")
    if source_mode == "injected" and set(metadata).intersection(production_metrics):
        raise ValueError("injected analysis metadata must not claim production acceptance metrics")
    for key in _METADATA_PRODUCTION_INTEGER_METRICS:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"production metadata metric {key} must be a non-negative integer")
    for key in _METADATA_PRODUCTION_FLOAT_METRICS:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"production metadata metric {key} must be finite and non-negative")
    if source_mode == "production":
        snapshot_hash = sources.get("db_snapshot_sha256")
        snapshot_bytes = sources.get("db_snapshot_bytes")
        snapshot_times = [
            sources.get("db_snapshot_creation_elapsed_seconds"),
            sources.get("db_snapshot_hash_elapsed_seconds"),
        ]
        if not _is_sha256_hex(snapshot_hash):
            raise ValueError("production DB snapshot SHA-256 is invalid")
        if (
            isinstance(snapshot_bytes, bool)
            or not isinstance(snapshot_bytes, (int, np.integer))
            or int(snapshot_bytes) <= 0
        ):
            raise ValueError("production DB snapshot byte count is invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
            or float(value) < 0
            for value in snapshot_times
        ):
            raise ValueError("production DB snapshot timing metadata is invalid")
        _validate_canonical_invariants_exact(canonical)
        expected_canonical_hash = _canonical_json_hash(_expected_canonical_invariants())
        if canonical_hash != expected_canonical_hash:
            raise ValueError("production canonical invariant hash differs from exact contract")
        if validate_live_sources:
            _validate_live_production_source_identity(sources)
    elif canonical is not None or canonical_hash is not None:
        raise ValueError("injected metadata must not claim production canonical invariants")
    inventory = sources.get("aws_file_inventory")
    if not isinstance(inventory, list):
        raise ValueError("AWS source inventory must be a list")
    if source_mode == "production":
        read_count = int(metadata["aws_file_read_count"])
        processed_hours = int(metadata["processed_paired_hours"])
        if (read_count > 0 or processed_hours > 0) and not inventory:
            raise ValueError("production AWS source inventory must not be empty")
        if len(inventory) > read_count:
            raise ValueError("AWS source inventory exceeds the physical file-read count")
    aggregate_hash = sources.get("aws_inventory_aggregate_sha256")
    if not _is_sha256_hex(aggregate_hash) or aggregate_hash != _canonical_json_hash(inventory):
        raise ValueError("AWS source inventory aggregate hash mismatch")
    seen: set[str] = set()
    aws_root_value = sources.get("aws_base_folder")
    aws_root = Path(aws_root_value) if isinstance(aws_root_value, str) else None
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "bytes",
            "sha256",
            "station_id",
            "month",
        }:
            raise ValueError("AWS inventory entry schema mismatch")
        relative = str(entry["relative_path"])
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("AWS inventory path is duplicate or unsafe")
        seen.add(relative)
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError("AWS inventory byte count is invalid")
        if not _is_sha256_hex(entry["sha256"]):
            raise ValueError("AWS inventory hash is invalid")
        if not isinstance(entry["station_id"], str):
            raise ValueError("AWS inventory station_id must be a string")
        station_id = _normalize_station_id(
            entry["station_id"], field="AWS inventory station_id"
        )
        month = entry["month"]
        if (
            not isinstance(month, str)
            or len(month) != 6
            or not month.isdigit()
        ):
            raise ValueError("AWS inventory month must be YYYYMM")
        expected_relative = f"Station_{station_id}/AWS_{month}.csv"
        if Path(relative).as_posix() != expected_relative:
            raise ValueError("AWS inventory path disagrees with station_id/month labels")
        if validate_live_sources and aws_root is not None:
            source = aws_root / relative
            if not source.is_file():
                raise ValueError(f"inventoried AWS source is missing: {relative}")
            if source.stat().st_size != entry["bytes"] or _sha256(source) != entry["sha256"]:
                raise ValueError(f"inventoried AWS source changed during/after analysis: {relative}")


def _validate_analysis_population_identity(
    metadata: Mapping[str, Any],
    station_structure: pd.DataFrame,
    *,
    require_live_source: bool = True,
) -> pd.DataFrame | None:
    """게시 skeleton을 metadata에 묶고, 게시 시점에는 live truth도 대조한다."""

    _require_columns(station_structure, ["STN"], "station_structure")
    count, digest = _station_population_identity(station_structure["STN"])
    if (
        count != int(metadata.get("analysis_station_count", -1))
        or digest != metadata.get("analysis_station_ids_sha256")
    ):
        raise ValueError("published station population identity differs from metadata")
    sources = metadata.get("input_sources")
    if not isinstance(sources, Mapping) or sources.get("source_mode") != "production":
        return None
    if not require_live_source:
        # 외부 receipt 재시작 검증은 당시 원본 경로가 이동·보관된 뒤에도
        # final bundle 자체의 population count/hash를 검증할 수 있어야 한다.
        # canonical metadata, ledger population/config digest, primary hashes와
        # 외부 anchor/manifest는 호출 경로의 다른 validator가 계속 검사한다.
        return None
    source_path = sources.get("station_results_path")
    if not isinstance(source_path, str):
        raise ValueError("production station-results source path is missing")
    truth = _prepare_station_results(
        pd.read_csv(source_path, encoding="utf-8-sig")
    )
    canonical = validate_canonical_station_results(truth)
    _validate_canonical_invariants_exact(canonical)
    if (
        _canonical_json_hash(canonical)
        != metadata.get("canonical_invariants_sha256")
        or canonical != metadata.get("canonical_invariants")
    ):
        raise ValueError(
            "production station-results canonical invariants differ from metadata"
        )
    source_count, source_digest = _station_population_identity(truth["STN"])
    if (
        source_count != count
        or source_digest != digest
        or set(truth["STN"]) != set(station_structure["STN"].astype(str))
    ):
        raise ValueError(
            "published station identity differs from immutable station-results source"
        )
    return truth


def _validate_table(frame: pd.DataFrame, expected_columns: Sequence[str], name: str) -> None:
    if frame.columns.tolist() != list(expected_columns):
        raise ValueError(f"{name} schema mismatch")
    numeric = frame.select_dtypes(include=[np.number])
    if len(numeric) and np.isinf(numeric.to_numpy(float)).any():
        raise ValueError(f"{name} contains infinite numeric values")


def _is_canonical_month_label(value: Any) -> bool:
    """월 label은 유효한 달력의 정확한 YYYY-MM 문자열만 허용한다."""

    return bool(
        type(value) is str
        and re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", value)
    )


def _is_canonical_date_label(value: Any) -> bool:
    """날짜 label은 유효한 달력의 정확한 YYYY-MM-DD 문자열만 허용한다."""

    if type(value) is not str or re.fullmatch(
        r"[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])", value
    ) is None:
        return False
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return parsed.strftime("%Y-%m-%d") == value


def _validate_primary_eligibility_ledger_contract(
    ledger: pd.DataFrame,
    matched: pd.DataFrame,
    coverage_sensitivity: pd.DataFrame,
    *,
    station_ids: Iterable[Any],
) -> None:
    """ledger 원시 충분통계에서 primary coverage/matched/BH를 재유도한다."""

    _validate_table(ledger, PRIMARY_ELIGIBILITY_LEDGER_COLUMNS, "primary ledger")
    exact_boolean_columns = (
        "BASE_MATCHED_DATA_AVAILABLE",
        "COVERAGE_ELIGIBLE",
        "ANALYSIS_SUPPORT_CHECKED",
        "COMPARISON_ELIGIBLE",
    )
    for column in exact_boolean_columns:
        if ledger[column].isna().any() or ledger[column].map(
            lambda value: not isinstance(value, (bool, np.bool_))
        ).any():
            raise ValueError(f"primary ledger {column} must contain exact booleans")
    exact_count_columns = (
        "N_COMMON_STRATA",
        "N_INVERSION_HOURS",
        "N_NORMAL_HOURS",
        "N_INVERSION_DAYS",
        "N_NORMAL_DAYS",
        "N_INVERSION_MONTHS",
        "N_NORMAL_MONTHS",
        "BLOCK7_REQUESTED_REPLICATES",
        "BLOCK7_VALID_REPLICATES",
        "BLOCK7_SEED",
        "BLOCK7_BLOCK_DAYS",
        "ANALYSIS_DATE_COUNT",
        "SUPPORTED_ANALYSIS_DATE_COUNT",
        "UNCOVERED_ANALYSIS_DATE_COUNT",
    )
    for column in exact_count_columns:
        values = ledger[column]
        for value in values.dropna():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer, float, np.floating)
            ):
                raise ValueError(f"primary ledger {column} must be an integer count")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise ValueError(f"primary ledger {column} must be an integer count")
    for column in ("BLOCK7_VALID_FRACTION", "ANALYSIS_DATE_SUPPORT_FRACTION"):
        for value in ledger[column].dropna():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer, float, np.floating)
            ):
                raise ValueError(f"primary ledger {column} must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"primary ledger {column} must be within [0,1]")
    fixed_domains = {
        "SETTING_ID": {PRIMARY_SETTING_ID},
        "COVERAGE_ID": {PRIMARY_COVERAGE_ID},
        "STRATA_WEIGHTING": {"equal_common_month_solar_strata"},
        "ESTIMAND": {"standardized_signed_bias_then_delta_abs_bias_contrast"},
        "SAMPLING_SCHEME": {"monthly_calendar_contiguous_fixed_length_blocks"},
        "ANALYSIS_POPULATION": {"all_stations_before_subset_join"},
    }
    for column, allowed in fixed_domains.items():
        if ledger[column].isna().any() or not ledger[column].map(
            lambda value: type(value) is str and value in allowed
        ).all():
            raise ValueError(f"primary ledger {column} domain mismatch")
    normalized_stations = {
        _normalize_station_id(value, field="primary ledger population STN")
        for value in station_ids
    }
    if ledger["STN"].duplicated().any() or set(ledger["STN"].astype(str)) != normalized_stations:
        raise ValueError("primary ledger must contain exactly one row per station")
    _, population_digest = _station_population_identity(normalized_stations)
    config_digest = _canonical_json_hash(_analysis_config_payload())
    if (
        not ledger["SOURCE_POPULATION_SHA256"].eq(population_digest).all()
        or not ledger["ANALYSIS_CONFIG_SHA256"].eq(config_digest).all()
        or not ledger["SETTING_ID"].eq(PRIMARY_SETTING_ID).all()
        or not ledger["COVERAGE_ID"].eq(PRIMARY_COVERAGE_ID).all()
    ):
        raise ValueError("primary ledger source population/config identity mismatch")

    indexed_ledger = ledger.set_index("STN").sort_index()
    indexed_matched = matched.set_index("STN").sort_index()
    if set(indexed_matched.index) != set(indexed_ledger.index):
        raise ValueError("primary ledger and matched station skeleton differ")
    primary_coverage = coverage_sensitivity.loc[
        coverage_sensitivity["SETTING_ID"].eq(PRIMARY_SETTING_ID)
    ].copy()
    if set(primary_coverage["STN"].astype(str)) != set(indexed_ledger.index):
        raise ValueError("primary ledger and primary coverage station skeleton differ")

    raw_columns = [
        "N_COMMON_STRATA",
        "N_INVERSION_HOURS",
        "N_NORMAL_HOURS",
        "N_INVERSION_DAYS",
        "N_NORMAL_DAYS",
        "N_INVERSION_MONTHS",
        "N_NORMAL_MONTHS",
        "DELTA_ABS_BIAS_INVERSION",
        "DELTA_ABS_BIAS_NORMAL",
        "CONTRAST_INV_MINUS_NORMAL",
    ]
    for row in indexed_ledger.itertuples():
        station_id = str(row.Index)
        try:
            common_keys = json.loads(row.COMMON_STRATA_KEYS_JSON)
            inversion_dates = json.loads(row.INVERSION_UNIQUE_DATES_JSON)
            normal_dates = json.loads(row.NORMAL_UNIQUE_DATES_JSON)
            inversion_months = json.loads(row.INVERSION_MONTHS_JSON)
            normal_months = json.loads(row.NORMAL_MONTHS_JSON)
            analysis_dates = json.loads(row.ANALYSIS_DATES_JSON)
            supported_dates = json.loads(row.SUPPORTED_ANALYSIS_DATES_JSON)
            uncovered_dates = json.loads(row.UNCOVERED_ANALYSIS_DATES_JSON)
            stratum_stats = json.loads(row.STRATUM_SUFFICIENT_STATS_JSON)
            date_stats = json.loads(row.DATE_SUFFICIENT_STATS_JSON)
            support_diagnostics = json.loads(row.ANALYSIS_SUPPORT_DIAGNOSTICS_JSON)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("primary ledger contains invalid JSON") from exc
        list_values = [
            common_keys,
            inversion_dates,
            normal_dates,
            inversion_months,
            normal_months,
            analysis_dates,
            supported_dates,
            uncovered_dates,
            stratum_stats,
            date_stats,
            support_diagnostics,
        ]
        if any(not isinstance(value, list) for value in list_values):
            raise ValueError("primary ledger JSON fields must be lists")
        required_support = {
            "month",
            "solar_state",
            "structure_state",
            "analysis_unique_dates",
            "supported_unique_dates",
            "uncovered_unique_dates",
            "support_fraction",
            "uncovered_dates",
        }
        for item in support_diagnostics:
            if not isinstance(item, dict) or set(item) != required_support:
                raise ValueError("primary ledger support diagnostic schema mismatch")
            count_values = [
                item["analysis_unique_dates"],
                item["supported_unique_dates"],
                item["uncovered_unique_dates"],
            ]
            if (
                not _is_canonical_month_label(item["month"])
                or type(item["solar_state"]) is not str
                or type(item["structure_state"]) is not str
                or item["solar_state"] not in {"bright", "transition", "dark"}
                or item["structure_state"]
                not in {"inversion_like", "normal_negative"}
                or any(type(value) is not int or value < 0 for value in count_values)
                or isinstance(item["support_fraction"], bool)
                or type(item["support_fraction"]) not in {int, float}
                or not np.isfinite(float(item["support_fraction"]))
                or not 0 <= float(item["support_fraction"]) <= 1
                or type(item["uncovered_dates"]) is not list
                or any(
                    not _is_canonical_date_label(value)
                    for value in item["uncovered_dates"]
                )
                or item["uncovered_dates"]
                != sorted(set(item["uncovered_dates"]))
                or item["supported_unique_dates"]
                + item["uncovered_unique_dates"]
                != item["analysis_unique_dates"]
                or len(item["uncovered_dates"])
                != item["uncovered_unique_dates"]
            ):
                raise ValueError("primary ledger support diagnostic domain mismatch")
        if any(value != sorted(set(value)) for value in [
            inversion_dates,
            normal_dates,
            inversion_months,
            normal_months,
            analysis_dates,
            supported_dates,
            uncovered_dates,
        ]):
            raise ValueError("primary ledger date/month sets must be sorted and unique")
        if any(
            not _is_canonical_date_label(value)
            for values in (
                inversion_dates,
                normal_dates,
                analysis_dates,
                supported_dates,
                uncovered_dates,
            )
            for value in values
        ) or any(
            not _is_canonical_month_label(value)
            for values in (inversion_months, normal_months)
            for value in values
        ):
            raise ValueError("primary ledger date/month labels must be canonical ISO")
        parsed_common: list[tuple[str, str]] = []
        for item in common_keys:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not _is_canonical_month_label(item[0])
                or item[1] not in {"bright", "transition", "dark"}
            ):
                raise ValueError("primary ledger common stratum key is invalid")
            parsed_common.append((item[0], item[1]))
        if parsed_common != sorted(set(parsed_common)):
            raise ValueError("primary ledger common strata must be sorted and unique")

        required_daily = {
            "month",
            "solar_state",
            "structure_state",
            "date",
            "n_hours",
            "sum_error_none",
            "sum_error_fixed",
        }
        daily_rows: list[dict[str, Any]] = []
        for item in date_stats:
            if not isinstance(item, dict) or set(item) != required_daily:
                raise ValueError("primary ledger date sufficient-stat schema mismatch")
            if (
                not _is_canonical_month_label(item["month"])
                or not _is_canonical_date_label(item["date"])
                or type(item["solar_state"]) is not str
                or type(item["structure_state"]) is not str
            ):
                raise ValueError("primary ledger date-stat labels must be strings")
            if item["structure_state"] not in set(STRUCTURE_STATES):
                raise ValueError("primary ledger date-stat structure state is invalid")
            if item["solar_state"] not in {"bright", "transition", "dark"}:
                raise ValueError("primary ledger date-stat solar state is invalid")
            n_hours = item["n_hours"]
            if isinstance(n_hours, bool) or not isinstance(n_hours, int) or n_hours <= 0:
                raise ValueError("primary ledger date-stat hour count is invalid")
            date = pd.Timestamp(item["date"])
            if date.strftime("%Y-%m") != item["month"]:
                raise ValueError("primary ledger date-stat month/date disagree")
            sums = [item["sum_error_none"], item["sum_error_fixed"]]
            if any(
                isinstance(value, bool)
                or type(value) not in {int, float}
                or not np.isfinite(float(value))
                for value in sums
            ):
                raise ValueError("primary ledger date-stat sums must be finite")
            daily_rows.append(item)
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            computed_common: list[tuple[str, str]] = []
            computed = {column: 0 for column in raw_columns[:7]}
            computed.update({column: np.nan for column in raw_columns[7:]})
            computed_inversion_dates: list[str] = []
            computed_normal_dates: list[str] = []
            computed_inversion_months: list[str] = []
            computed_normal_months: list[str] = []
            computed_analysis_dates: list[str] = []
        else:
            duplicated = daily.duplicated(
                ["month", "solar_state", "structure_state", "date"]
            )
            if duplicated.any():
                raise ValueError("primary ledger date-stat keys must be unique")
            relevant_daily = daily.loc[
                daily["structure_state"].isin(
                    ["inversion_like", "normal_negative"]
                )
            ].copy()
            keys_by_state = {
                state: set(
                    map(
                        tuple,
                        relevant_daily.loc[
                            relevant_daily["structure_state"].eq(state),
                            ["month", "solar_state"],
                        ]
                        .drop_duplicates()
                        .to_numpy(),
                    )
                )
                for state in ("inversion_like", "normal_negative")
            }
            computed_common = sorted(
                keys_by_state["inversion_like"] & keys_by_state["normal_negative"]
            )
            if computed_common:
                common_frame = pd.DataFrame(
                    computed_common, columns=["month", "solar_state"]
                )
                analysis_daily = relevant_daily.merge(
                    common_frame,
                    on=["month", "solar_state"],
                    how="inner",
                    validate="many_to_one",
                )
            else:
                analysis_daily = relevant_daily.iloc[0:0].copy()
            state_daily = {
                state: analysis_daily.loc[
                    analysis_daily["structure_state"].eq(state)
                ]
                for state in ("inversion_like", "normal_negative")
            }
            effects: dict[str, float] = {}
            for state, state_frame in state_daily.items():
                if state_frame.empty:
                    effects[state] = np.nan
                    continue
                grouped = state_frame.groupby(["month", "solar_state"], sort=True)[
                    ["n_hours", "sum_error_none", "sum_error_fixed"]
                ].sum()
                none_bias = float((grouped["sum_error_none"] / grouped["n_hours"]).mean())
                fixed_bias = float((grouped["sum_error_fixed"] / grouped["n_hours"]).mean())
                effects[state] = abs(fixed_bias) - abs(none_bias)
            computed_inversion_dates = sorted(
                state_daily["inversion_like"]["date"].unique().tolist()
            )
            computed_normal_dates = sorted(
                state_daily["normal_negative"]["date"].unique().tolist()
            )
            computed_inversion_months = sorted(
                state_daily["inversion_like"]["month"].unique().tolist()
            )
            computed_normal_months = sorted(
                state_daily["normal_negative"]["month"].unique().tolist()
            )
            computed_analysis_dates = sorted(
                analysis_daily["date"].unique().tolist()
            )
            computed = {
                "N_COMMON_STRATA": len(computed_common),
                "N_INVERSION_HOURS": int(state_daily["inversion_like"]["n_hours"].sum()),
                "N_NORMAL_HOURS": int(state_daily["normal_negative"]["n_hours"].sum()),
                "N_INVERSION_DAYS": len(computed_inversion_dates),
                "N_NORMAL_DAYS": len(computed_normal_dates),
                "N_INVERSION_MONTHS": len(computed_inversion_months),
                "N_NORMAL_MONTHS": len(computed_normal_months),
                "DELTA_ABS_BIAS_INVERSION": effects["inversion_like"],
                "DELTA_ABS_BIAS_NORMAL": effects["normal_negative"],
                "CONTRAST_INV_MINUS_NORMAL": effects["inversion_like"]
                - effects["normal_negative"],
            }
        support_checked = bool(row.ANALYSIS_SUPPORT_CHECKED)
        if (
            parsed_common != computed_common
            or inversion_dates != computed_inversion_dates
            or normal_dates != computed_normal_dates
            or inversion_months != computed_inversion_months
            or normal_months != computed_normal_months
            or analysis_dates != computed_analysis_dates
            or set(supported_dates).intersection(uncovered_dates)
            or (
                support_checked
                and sorted(set(supported_dates) | set(uncovered_dates))
                != analysis_dates
            )
            or (not support_checked and bool(supported_dates or uncovered_dates))
        ):
            raise ValueError("primary ledger raw date/stratum sets disagree")
        for column, value in computed.items():
            published = getattr(row, column)
            if column.startswith("N_"):
                if int(published) != int(value):
                    raise ValueError(f"primary ledger raw count differs: {column}")
            elif not _same_number(published, value):
                raise ValueError(f"primary ledger point estimate differs: {column}")
        required_stratum = {
            "month",
            "solar_state",
            "structure_state",
            "n_hours",
            "sum_error_none",
            "sum_error_fixed",
            "unique_dates",
        }
        expected_strata: dict[tuple[str, str, str], dict[str, Any]] = {}
        if not daily.empty and computed_common:
            common_set = set(computed_common)
            relevant = daily.loc[
                daily["structure_state"].isin(
                    ["inversion_like", "normal_negative"]
                )
                & pd.Series(
                    [
                        (month, solar) in common_set
                        for month, solar in zip(
                            daily["month"], daily["solar_state"]
                        )
                    ],
                    index=daily.index,
                )
            ]
            for (month, solar, state), group in relevant.groupby(
                ["month", "solar_state", "structure_state"], sort=True
            ):
                expected_strata[(str(month), str(solar), str(state))] = {
                    "n_hours": int(group["n_hours"].sum()),
                    "sum_error_none": float(group["sum_error_none"].sum()),
                    "sum_error_fixed": float(group["sum_error_fixed"].sum()),
                    "unique_dates": sorted(group["date"].unique().tolist()),
                }
        observed_keys: set[tuple[str, str, str]] = set()
        for item in stratum_stats:
            if not isinstance(item, dict) or set(item) != required_stratum:
                raise ValueError("primary ledger stratum sufficient-stat schema mismatch")
            if (
                not _is_canonical_month_label(item["month"])
                or type(item["solar_state"]) is not str
                or type(item["structure_state"]) is not str
                or item["solar_state"] not in {"bright", "transition", "dark"}
                or item["structure_state"]
                not in {"inversion_like", "normal_negative"}
                or type(item["n_hours"]) is not int
                or item["n_hours"] <= 0
                or type(item["unique_dates"]) is not list
                or any(
                    not _is_canonical_date_label(value)
                    for value in item["unique_dates"]
                )
                or item["unique_dates"] != sorted(set(item["unique_dates"]))
                or any(
                    isinstance(item[column], bool)
                    or type(item[column]) not in {int, float}
                    or not np.isfinite(float(item[column]))
                    for column in ("sum_error_none", "sum_error_fixed")
                )
            ):
                raise ValueError("primary ledger stratum sufficient-stat domain mismatch")
            key = (
                str(item["month"]),
                str(item["solar_state"]),
                str(item["structure_state"]),
            )
            if key in observed_keys or key not in expected_strata:
                raise ValueError("primary ledger stratum sufficient-stat key mismatch")
            observed_keys.add(key)
            expected_item = expected_strata[key]
            if (
                isinstance(item["n_hours"], bool)
                or int(item["n_hours"]) != expected_item["n_hours"]
                or item["unique_dates"] != expected_item["unique_dates"]
                or not _same_number(
                    item["sum_error_none"], expected_item["sum_error_none"]
                )
                or not _same_number(
                    item["sum_error_fixed"], expected_item["sum_error_fixed"]
                )
            ):
                raise ValueError("primary ledger stratum sufficient stats differ from dates")
        if observed_keys != set(expected_strata):
            raise ValueError("primary ledger stratum sufficient-stat count disagrees")
        available = bool(computed_common)
        expected_coverage = bool(
            available
            and computed["N_INVERSION_DAYS"] >= 30
            and computed["N_NORMAL_DAYS"] >= 30
            and computed["N_INVERSION_MONTHS"] >= 6
            and computed["N_NORMAL_MONTHS"] >= 6
        )
        if row.COVERAGE_DROP_REASON == "source_payload_missing":
            expected_reason = "source_payload_missing"
        elif not available:
            required_states_present = set(daily["structure_state"]) & {
                "inversion_like",
                "normal_negative",
            } if not daily.empty else set()
            expected_reason = (
                "missing_required_state"
                if not {"inversion_like", "normal_negative"}.issubset(
                    required_states_present
                )
                else "no_common_month_solar_strata"
            )
        elif min(computed["N_INVERSION_DAYS"], computed["N_NORMAL_DAYS"]) < 30:
            expected_reason = "insufficient_unique_days"
        elif min(computed["N_INVERSION_MONTHS"], computed["N_NORMAL_MONTHS"]) < 6:
            expected_reason = "insufficient_months"
        else:
            expected_reason = "eligible"
        if (
            bool(row.BASE_MATCHED_DATA_AVAILABLE) != available
            or bool(row.COVERAGE_ELIGIBLE) != expected_coverage
            or str(row.COVERAGE_DROP_REASON) != expected_reason
        ):
            raise ValueError("primary ledger coverage gate differs from raw sufficient stats")
        if support_checked:
            if int(row.ANALYSIS_DATE_COUNT) != len(analysis_dates):
                raise ValueError("primary ledger analysis date count disagrees with raw set")
            if int(row.SUPPORTED_ANALYSIS_DATE_COUNT) != len(supported_dates) or int(
                row.UNCOVERED_ANALYSIS_DATE_COUNT
            ) != len(uncovered_dates):
                raise ValueError("primary ledger calendar-block support counts disagree")
        elif (
            int(row.ANALYSIS_DATE_COUNT) != 0
            or int(row.SUPPORTED_ANALYSIS_DATE_COUNT) != 0
            or int(row.UNCOVERED_ANALYSIS_DATE_COUNT) != 0
            or support_diagnostics
        ):
            raise ValueError("unchecked primary ledger support fields must remain empty")
        if not analysis_dates and bool(row.ANALYSIS_SUPPORT_CHECKED):
            raise ValueError("zero-date primary ledger cannot claim checked block support")

        # 날짜별 n/sum은 7일 block 재표집에 필요한 최소 충분통계다. 같은 날짜의
        # 평균 error를 n회 복원하면 날짜 block weight 아래 signed 합/분모가 정확히
        # 보존되므로 fixed-seed support/CI/p/eligibility를 독립 재계산할 수 있다.
        if daily_rows:
            expanded_rows: list[dict[str, Any]] = []
            date_offsets: Counter[str] = Counter()
            for item in daily_rows:
                n_hours = int(item["n_hours"])
                mean_none = float(item["sum_error_none"]) / n_hours
                mean_fixed = float(item["sum_error_fixed"]) / n_hours
                for _ in range(n_hours):
                    offset = date_offsets[str(item["date"])]
                    date_offsets[str(item["date"])] += 1
                    expanded_rows.append({
                        "STN": station_id,
                        "TM": pd.Timestamp(item["date"]) + pd.Timedelta(
                            microseconds=offset
                        ),
                        "SOLAR_STATE": item["solar_state"],
                        "STRUCTURE_STATE": item["structure_state"],
                        "error_none": mean_none,
                        "error_fixed": mean_fixed,
                    })
            independent_result = bootstrap_shared_state_delta_abs_bias_contrast(
                pd.DataFrame(expanded_rows),
                n_replicates=2000,
                block_length_days=7,
                seed=20260827 + int(station_id) * 17,
                min_unique_days_per_state=30,
                min_months_per_state=6,
            )
            if independent_result.valid_replicates:
                deviations = (
                    independent_result.replicate_contrasts
                    - float(independent_result.point_contrast)
                )
                independent_p = float(
                    (
                        1
                        + np.count_nonzero(
                            np.abs(deviations)
                            >= abs(float(independent_result.point_contrast))
                        )
                    )
                    / (len(deviations) + 1)
                )
            else:
                independent_p = np.nan
            bootstrap_contract = {
                "BLOCK7_REQUESTED_REPLICATES": independent_result.requested_replicates,
                "BLOCK7_VALID_REPLICATES": independent_result.valid_replicates,
                "BLOCK7_VALID_FRACTION": independent_result.valid_replicate_fraction,
                "BLOCK7_SEED": independent_result.seed,
                "BLOCK7_BLOCK_DAYS": independent_result.block_length_days,
                "STRATA_WEIGHTING": independent_result.strata_weighting,
                "ESTIMAND": independent_result.estimand,
                "SAMPLING_SCHEME": independent_result.sampling_scheme,
                "ANALYSIS_SUPPORT_CHECKED": independent_result.analysis_support_checked,
                "ANALYSIS_DATE_COUNT": independent_result.analysis_date_count,
                "SUPPORTED_ANALYSIS_DATE_COUNT": independent_result.supported_analysis_date_count,
                "UNCOVERED_ANALYSIS_DATE_COUNT": independent_result.uncovered_analysis_date_count,
                "ANALYSIS_DATE_SUPPORT_FRACTION": independent_result.analysis_date_support_fraction,
                "BLOCK7_CI_LOW": independent_result.ci_low,
                "BLOCK7_CI_HIGH": independent_result.ci_high,
                "P_VALUE": independent_p
                if independent_result.coverage_eligible
                else np.nan,
                "COMPARISON_ELIGIBLE": independent_result.coverage_eligible,
                "COMPARISON_DROP_REASON": independent_result.reason,
            }
            for column, expected_value in bootstrap_contract.items():
                observed_value = getattr(row, column)
                if isinstance(expected_value, (bool, np.bool_)):
                    equal = isinstance(observed_value, (bool, np.bool_)) and bool(
                        observed_value
                    ) == bool(expected_value)
                elif isinstance(expected_value, str):
                    equal = str(observed_value) == expected_value
                else:
                    equal = _same_number(observed_value, expected_value)
                if not equal:
                    raise ValueError(
                        f"primary ledger bootstrap differs from date sufficient stats: {column}"
                    )
            if sorted(independent_result.uncovered_analysis_dates) != uncovered_dates:
                raise ValueError(
                    "primary ledger uncovered dates differ from fixed-seed bootstrap"
                )
            normalized_support = json.loads(
                json.dumps(independent_result.analysis_support_diagnostics)
            )
            if support_diagnostics != normalized_support:
                raise ValueError(
                    "primary ledger support diagnostics differ from fixed-seed bootstrap"
                )
        elif str(row.COMPARISON_DROP_REASON) not in {
            "source_payload_missing",
            "no_reproduced_hours",
            "missing_required_state",
        }:
            raise ValueError("empty primary ledger cannot claim a data-derived comparison")

        coverage_rows = primary_coverage.loc[primary_coverage["STN"].eq(station_id)]
        if len(coverage_rows) != 9:
            raise ValueError("primary ledger requires all nine coverage thresholds")
        for coverage_row in coverage_rows.itertuples(index=False):
            for column in raw_columns:
                published = getattr(coverage_row, column)
                value = computed[column]
                if column.startswith("N_"):
                    if int(published) != int(value):
                        raise ValueError("primary coverage raw count differs from ledger")
                elif not _same_number(published, value):
                    raise ValueError("primary coverage point differs from ledger")
            threshold_eligible = bool(
                available
                and computed["N_INVERSION_DAYS"] >= int(coverage_row.MIN_DAYS_PER_STATE)
                and computed["N_NORMAL_DAYS"] >= int(coverage_row.MIN_DAYS_PER_STATE)
                and computed["N_INVERSION_MONTHS"] >= int(coverage_row.MIN_MONTHS_PER_STATE)
                and computed["N_NORMAL_MONTHS"] >= int(coverage_row.MIN_MONTHS_PER_STATE)
            )
            if bool(coverage_row.COVERAGE_ELIGIBLE) != threshold_eligible:
                raise ValueError("primary coverage threshold result differs from ledger")
        matched_row = indexed_matched.loc[station_id]
        ledger_to_matched = {
            "N_COMMON_STRATA": "N_STRATA",
            "N_INVERSION_DAYS": None,
            "N_INVERSION_MONTHS": None,
            "DELTA_ABS_BIAS_INVERSION": "DELTA_ABS_BIAS_INVERSION",
            "DELTA_ABS_BIAS_NORMAL": "DELTA_ABS_BIAS_NORMAL",
            "CONTRAST_INV_MINUS_NORMAL": "CONTRAST_INV_MINUS_NORMAL",
            "BLOCK7_CI_LOW": "BLOCK7_CI_LOW",
            "BLOCK7_CI_HIGH": "BLOCK7_CI_HIGH",
            "BLOCK7_REQUESTED_REPLICATES": "BLOCK7_REQUESTED_REPLICATES",
            "BLOCK7_VALID_REPLICATES": "BLOCK7_VALID_REPLICATES",
            "BLOCK7_VALID_FRACTION": "BLOCK7_VALID_FRACTION",
            "BLOCK7_SEED": "BLOCK7_SEED",
            "BLOCK7_BLOCK_DAYS": "BLOCK7_BLOCK_DAYS",
            "ANALYSIS_SUPPORT_CHECKED": "ANALYSIS_SUPPORT_CHECKED",
            "ANALYSIS_DATE_COUNT": "ANALYSIS_DATE_COUNT",
            "SUPPORTED_ANALYSIS_DATE_COUNT": "SUPPORTED_ANALYSIS_DATE_COUNT",
            "UNCOVERED_ANALYSIS_DATE_COUNT": "UNCOVERED_ANALYSIS_DATE_COUNT",
            "ANALYSIS_DATE_SUPPORT_FRACTION": "ANALYSIS_DATE_SUPPORT_FRACTION",
            "P_VALUE": "P_VALUE",
            "Q_VALUE": "Q_VALUE",
            "COMPARISON_ELIGIBLE": "COMPARISON_ELIGIBLE",
            "EVIDENCE_GRADE": "EVIDENCE_GRADE",
        }
        for ledger_column, matched_column in ledger_to_matched.items():
            if matched_column is None:
                continue
            left = getattr(row, ledger_column)
            right = matched_row[matched_column]
            if isinstance(left, (bool, np.bool_)):
                equal = isinstance(right, (bool, np.bool_)) and bool(left) == bool(right)
            elif isinstance(left, str):
                equal = str(right) == left
            else:
                equal = _same_number(left, right)
            if not equal:
                raise ValueError(
                    f"matched primary inference differs from ledger: {matched_column}"
                )
        if int(matched_row["N_DAYS"]) != min(
            computed["N_INVERSION_DAYS"], computed["N_NORMAL_DAYS"]
        ) or int(matched_row["N_MONTHS"]) != min(
            computed["N_INVERSION_MONTHS"], computed["N_NORMAL_MONTHS"]
        ):
            raise ValueError("matched primary coverage count differs from ledger")
        if str(matched_row["DROP_REASON"]) != str(row.COMPARISON_DROP_REASON):
            raise ValueError("matched primary reason differs from ledger")

    eligible = indexed_ledger["COMPARISON_ELIGIBLE"].astype(bool)
    p_values = pd.to_numeric(indexed_ledger["P_VALUE"], errors="coerce")
    expected_q = pd.Series(np.nan, index=indexed_ledger.index, dtype=float)
    if eligible.any():
        if p_values.loc[eligible].isna().any():
            raise ValueError("eligible primary ledger family contains missing p-values")
        expected_q.loc[eligible] = benjamini_hochberg(
            p_values.loc[eligible].to_numpy(float)
        )
    if not np.allclose(
        pd.to_numeric(indexed_ledger["Q_VALUE"], errors="coerce"),
        expected_q,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("primary ledger Q_VALUE differs from the full eligible family")
    ci_low = pd.to_numeric(indexed_ledger["BLOCK7_CI_LOW"], errors="coerce")
    ci_high = pd.to_numeric(indexed_ledger["BLOCK7_CI_HIGH"], errors="coerce")
    expected_grade = np.where(
        eligible & ((ci_low > 0) | (ci_high < 0)) & (expected_q <= 0.05),
        "자료로 지지",
        "현재 자료에서 지지되지 않음",
    )
    if not np.array_equal(indexed_ledger["EVIDENCE_GRADE"].to_numpy(), expected_grade):
        raise ValueError("primary ledger evidence grade differs from its CI/q gate")


def _validate_cross_table_contracts(
    tables: Mapping[str, pd.DataFrame],
    truth: pd.DataFrame | None = None,
) -> None:
    """schema만으로 잡히지 않는 key·분모·category·subset 모순을 차단한다."""

    tables = {name: frame.copy() for name, frame in tables.items()}
    for frame in tables.values():
        if "STN" in frame:
            frame["STN"] = frame["STN"].map(
                lambda value: _normalize_station_id(value, field="published STN")
            )
    if truth is not None:
        truth = truth.copy()
        truth["STN"] = truth["STN"].map(
            lambda value: _normalize_station_id(value, field="truth STN")
        )

    key_columns = {
        "station_structure_coverage.csv": ["STN"],
        "state_condition_bias.csv": [
            "STN",
            "CONFIG",
            "STRUCTURE_STATE",
            "CONDITION_VARIABLE",
            "CONDITION_LEVEL",
        ],
        "coast_time_signed_bias.csv": ["STN", "MONTH", "SOLAR_STATE"],
        "equivalent_diagnostics.csv": ["STN"],
        "worsening_diagnostics.csv": ["STN"],
        "high_residual_group_diagnostic.csv": ["STN"],
        "station_885_diagnostic.csv": [
            "STN",
            "MONTH",
            "SOLAR_STATE",
            "STRUCTURE_STATE",
        ],
        "matched_state_contrast.csv": ["STN", "CONFIG"],
        "spatial_structure_sensitivity_summary.csv": [
            "STN",
            "SETTING_ID",
            "STRUCTURE_STATE",
        ],
        "matched_state_coverage_sensitivity.csv": [
            "STN",
            "SETTING_ID",
            "COVERAGE_ID",
        ],
        "primary_eligibility_ledger.csv": ["STN"],
        "physical_hypothesis_comparison.csv": ["CONTRAST_ID"],
    }
    for name, keys in key_columns.items():
        if tables[name].duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate result keys")

    station = tables["station_structure_coverage.csv"]
    state = tables["state_condition_bias.csv"]
    matched = tables["matched_state_contrast.csv"]
    spatial_sensitivity = tables["spatial_structure_sensitivity_summary.csv"]
    coverage_sensitivity = tables["matched_state_coverage_sensitivity.csv"]
    primary_ledger = tables["primary_eligibility_ledger.csv"]
    physical = tables["physical_hypothesis_comparison.csv"]

    boolean_columns = {
        "station_structure_coverage.csv": ["WEATHER_PRIMARY_ELIGIBLE"],
        "state_condition_bias.csv": [
            "COVERAGE_ELIGIBLE",
            "SUBSET_MEANINGFUL_WORSENING",
            "SUBSET_HIGH_AFTER",
        ],
        "coast_time_signed_bias.csv": [
            "H2_DIRECTION_MATCH",
            "SUBSET_MEANINGFUL_WORSENING",
            "SUBSET_HIGH_AFTER",
        ],
        "high_residual_group_diagnostic.csv": [
            "HIGH_BEFORE",
            "HIGH_AFTER",
            "HIGH_COMMON",
            "IS_PRIMARY_HIGH_RESIDUAL",
            "PRIMARY_STRUCTURE_ELIGIBLE",
        ],
        "matched_state_contrast.csv": [
            "ANALYSIS_SUPPORT_CHECKED",
            "COMPARISON_ELIGIBLE",
            "SUBSET_MEANINGFUL_WORSENING",
            "SUBSET_HIGH_AFTER",
        ],
        "matched_state_coverage_sensitivity.csv": [
            "BASE_MATCHED_DATA_AVAILABLE",
            "COVERAGE_ELIGIBLE",
            "SUBSET_MEANINGFUL_WORSENING",
            "SUBSET_HIGH_AFTER",
        ],
        "primary_eligibility_ledger.csv": [
            "BASE_MATCHED_DATA_AVAILABLE",
            "COVERAGE_ELIGIBLE",
            "ANALYSIS_SUPPORT_CHECKED",
            "COMPARISON_ELIGIBLE",
        ],
        "physical_hypothesis_comparison.csv": [
            "EFFECT_DIRECTION_MATCH",
            "BIAS_DIRECTION_MATCH",
            "COVERAGE_ELIGIBLE",
            "SPATIAL_SENSITIVITY_DIRECTION_STABLE",
        ],
    }
    optional_boolean_columns = {
        (
            "physical_hypothesis_comparison.csv",
            "EFFECT_DIRECTION_MATCH",
        ),
        (
            "physical_hypothesis_comparison.csv",
            "BIAS_DIRECTION_MATCH",
        ),
        (
            "physical_hypothesis_comparison.csv",
            "SPATIAL_SENSITIVITY_DIRECTION_STABLE",
        ),
    }
    for name, columns in boolean_columns.items():
        for column in columns:
            values = tables[name][column]
            if (name, column) not in optional_boolean_columns and values.isna().any():
                raise ValueError(f"{name} {column} must not contain missing booleans")
            invalid = values.dropna().map(
                lambda value: not isinstance(value, (bool, np.bool_))
            )
            if invalid.any():
                raise ValueError(f"{name} {column} must contain exact boolean values")

    max_none_diff = pd.to_numeric(
        station["MAX_NONE_ABS_DIFF_C"], errors="coerce"
    ).dropna()
    if (max_none_diff < 0).any() or (max_none_diff > 1e-10).any():
        raise ValueError("MAX_NONE_ABS_DIFF_C violates the exact-none domain")
    coast_exposure = pd.to_numeric(
        station["MEDIAN_ABS_DELTA_COAST_KM"], errors="coerce"
    ).dropna()
    if (coast_exposure < 0).any():
        raise ValueError("absolute coast exposure must be non-negative")
    expected_stations = set(station["STN"])
    if truth is not None and expected_stations != set(truth["STN"]):
        raise ValueError("station skeleton does not match the analysis population")
    if set(state["STN"]) != expected_stations:
        raise ValueError("state skeleton does not match the analysis population")
    if set(matched["STN"]) != expected_stations:
        raise ValueError("matched contrast skeleton does not match the analysis population")
    if set(spatial_sensitivity["STN"]) != expected_stations:
        raise ValueError("spatial sensitivity skeleton does not match the analysis population")
    if set(coverage_sensitivity["STN"]) != expected_stations:
        raise ValueError("coverage sensitivity skeleton does not match the analysis population")
    if set(primary_ledger["STN"]) != expected_stations:
        raise ValueError("primary ledger skeleton does not match the analysis population")

    allowed_evidence = {
        "자료로 지지",
        "문헌과 일치하나 직접 미측정",
        "현재 자료에서 지지되지 않음",
    }
    evidence_frames = [
        state,
        tables["coast_time_signed_bias.csv"],
        tables["station_885_diagnostic.csv"],
        matched,
        primary_ledger,
        physical,
    ]
    for frame in evidence_frames:
        if len(frame) and not set(frame["EVIDENCE_GRADE"].dropna()).issubset(allowed_evidence):
            raise ValueError("unknown evidence grade")
    if not set(state["CONFIG"]).issubset(STRUCTURE_CONFIGS):
        raise ValueError("unknown state CONFIG")
    if set(matched["CONFIG"]) != {"primary"}:
        raise ValueError("unknown matched-contrast CONFIG")
    if set(matched["SETTING_ID"]) != {PRIMARY_SETTING_ID}:
        raise ValueError("matched contrast must contain only the prespecified primary setting")
    if not set(state["STRUCTURE_STATE"]).issubset(STRUCTURE_STATES):
        raise ValueError("unknown spatial structure state")
    condition_domains = {
        "ALL": {"all"},
        "SOLAR_STATE": {"bright", "transition", "dark"},
        "WIND_QUARTILE": {"Q1", "Q2", "Q3", "Q4"},
        "PRECIPITATION_STATE": {"non_precipitation", "precipitation"},
    }
    for variable, group in state.groupby("CONDITION_VARIABLE", sort=False):
        if variable not in condition_domains or not set(group["CONDITION_LEVEL"]).issubset(
            condition_domains[variable]
        ):
            raise ValueError("unknown condition category")

    count_columns = [
        "N_HOURS",
        "N_DAYS",
        "N_MONTHS",
    ]
    for frame in [state, tables["coast_time_signed_bias.csv"], tables["station_885_diagnostic.csv"]]:
        for column in count_columns:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"invalid count column: {column}")
        zero = frame["N_HOURS"].eq(0)
        metric_columns = [
            "BIAS_none",
            "BIAS_fixed",
            "DELTA_ABS_BIAS",
            "MAE_none",
            "MAE_fixed",
            "DELTA_MAE",
        ]
        if frame.loc[zero, metric_columns].notna().any().any():
            raise ValueError("zero-hour row must not contain finite BIAS/MAE metrics")
        if frame.loc[~zero, metric_columns].isna().any().any():
            raise ValueError("positive-hour row must contain complete BIAS/MAE metrics")
    if state[["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]].notna().any().any() or (
        pd.to_numeric(state["BLOCK7_REPLICATES"], errors="raise") != 0
    ).any():
        raise ValueError("state rows are descriptive and must not contain inferential CI")
    state_station_coverage = station.set_index("STN")[["WIND_COVERAGE", "RN_COVERAGE"]]
    state_station_ids = state["STN"]
    wind_coverage = pd.to_numeric(
        state_station_ids.map(state_station_coverage["WIND_COVERAGE"]), errors="coerce"
    )
    rn_coverage = pd.to_numeric(
        state_station_ids.map(state_station_coverage["RN_COVERAGE"]), errors="coerce"
    )
    expected_state_coverage = (
        (pd.to_numeric(state["N_DAYS"], errors="raise") >= 30)
        & (pd.to_numeric(state["N_MONTHS"], errors="raise") >= 6)
    )
    expected_state_coverage &= ~state["CONDITION_VARIABLE"].eq("WIND_QUARTILE") | (
        wind_coverage >= 0.8
    )
    expected_state_coverage &= ~state["CONDITION_VARIABLE"].eq(
        "PRECIPITATION_STATE"
    ) | (rn_coverage >= 0.8)
    if not np.array_equal(
        state["COVERAGE_ELIGIBLE"].astype(bool).to_numpy(),
        expected_state_coverage.to_numpy(bool),
    ):
        raise ValueError("descriptive state coverage differs from the generator rule")
    expected_state_grade = np.where(
        expected_state_coverage,
        "문헌과 일치하나 직접 미측정",
        "현재 자료에서 지지되지 않음",
    )
    if not np.array_equal(
        state["EVIDENCE_GRADE"].astype(str).to_numpy(), expected_state_grade
    ) or state["EVIDENCE_GRADE"].eq("자료로 지지").any():
        raise ValueError("descriptive state evidence grade violates the generator rule")

    station_885 = tables["station_885_diagnostic.csv"]
    _validate_canonical_station_885_presence(truth, station, station_885)
    _validate_station_885_evidence_contract(
        station_885,
        matched,
        station_structure=station,
        state_condition=state,
        coverage_sensitivity=coverage_sensitivity,
        analysis_station_ids=list(station["STN"]),
    )

    for column in [
        "N_STRATA",
        "N_DAYS",
        "N_MONTHS",
        "BLOCK7_REQUESTED_REPLICATES",
        "BLOCK7_VALID_REPLICATES",
        "BLOCK7_BLOCK_DAYS",
        "ANALYSIS_DATE_COUNT",
        "SUPPORTED_ANALYSIS_DATE_COUNT",
        "UNCOVERED_ANALYSIS_DATE_COUNT",
    ]:
        values = pd.to_numeric(matched[column], errors="raise").to_numpy(float)
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"invalid matched-contrast count: {column}")
    if (
        matched["BLOCK7_VALID_REPLICATES"]
        > matched["BLOCK7_REQUESTED_REPLICATES"]
    ).any():
        raise ValueError("valid bootstrap replicates exceed requested replicates")
    source_missing = matched["DROP_REASON"].astype(str).eq("source_payload_missing")
    matched_station = station.set_index("STN").loc[matched["STN"]]
    station_total_hours = pd.to_numeric(
        matched_station["TOTAL_PAIRED_HOURS"], errors="raise"
    ).reset_index(drop=True)
    station_reproduced_hours = pd.to_numeric(
        matched_station["REPRODUCED_HOURS"], errors="raise"
    ).reset_index(drop=True)
    station_source_missing = (
        matched_station["WEATHER_DROP_REASON"]
        .astype(str)
        .eq("source_payload_missing")
        .reset_index(drop=True)
        & station_total_hours.eq(0)
        & station_reproduced_hours.eq(0)
    )
    if not np.array_equal(
        source_missing.to_numpy(bool), station_source_missing.to_numpy(bool)
    ):
        raise ValueError(
            "matched source_payload_missing must agree with the station payload skeleton"
        )
    no_reproduced = matched["DROP_REASON"].astype(str).eq("no_reproduced_hours")
    expected_no_reproduced = station_reproduced_hours.eq(0) & ~station_source_missing
    if not np.array_equal(
        no_reproduced.to_numpy(bool), expected_no_reproduced.to_numpy(bool)
    ):
        raise ValueError(
            "matched no_reproduced_hours must agree with station reproduced hours"
        )
    requested = pd.to_numeric(
        matched["BLOCK7_REQUESTED_REPLICATES"], errors="raise"
    )
    valid_replicates = pd.to_numeric(
        matched["BLOCK7_VALID_REPLICATES"], errors="raise"
    )
    block_days = pd.to_numeric(matched["BLOCK7_BLOCK_DAYS"], errors="raise")
    seeds = pd.to_numeric(matched["BLOCK7_SEED"], errors="coerce")
    expected_seeds = matched["STN"].map(
        lambda station_id: 20260827 + int(str(station_id)) * 17
    )
    if (
        (requested.loc[~source_missing] != 2000).any()
        or (block_days != 7).any()
        or not np.array_equal(
            seeds.loc[~source_missing].to_numpy(float),
            expected_seeds.loc[~source_missing].to_numpy(float),
        )
        or (requested.loc[source_missing] != 0).any()
        or (valid_replicates.loc[source_missing] != 0).any()
        or seeds.loc[source_missing].notna().any()
    ):
        raise ValueError("matched bootstrap must use the fixed 2000-replicate/seed contract")
    eligible = matched["COMPARISON_ELIGIBLE"].astype(bool)
    complete_ci = matched[["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]].notna().all(axis=1)
    if (
        pd.to_numeric(matched.loc[complete_ci, "BLOCK7_CI_LOW"], errors="raise")
        > pd.to_numeric(matched.loc[complete_ci, "BLOCK7_CI_HIGH"], errors="raise")
    ).any():
        raise ValueError("matched bootstrap CI is not an ordered interval")
    if matched.loc[eligible, ["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]].isna().any().any():
        raise ValueError("eligible matched contrast must contain a complete CI")
    if matched.loc[~eligible, ["BLOCK7_CI_LOW", "BLOCK7_CI_HIGH"]].notna().any().any():
        raise ValueError("ineligible matched contrast must not contain an inferential CI")
    fractions = pd.to_numeric(matched["BLOCK7_VALID_FRACTION"], errors="coerce")
    if (fractions.dropna() < 0).any() or (fractions.dropna() > 1).any():
        raise ValueError("bootstrap valid fraction must lie in [0, 1]")
    expected_fraction = valid_replicates.loc[~source_missing] / requested.loc[
        ~source_missing
    ]
    if (
        fractions.loc[source_missing].notna().any()
        or fractions.loc[~source_missing].isna().any()
        or not np.allclose(
            fractions.loc[~source_missing].to_numpy(float),
            expected_fraction.to_numpy(float),
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError("bootstrap valid fraction differs from valid/requested counts")
    if (fractions.loc[eligible] < 0.95).any():
        raise ValueError("eligible matched contrast violates the 95% valid-replicate gate")
    if (valid_replicates.loc[eligible] < 1900).any():
        raise ValueError("eligible matched contrast requires at least 1900 valid replicates")

    support_checked = matched["ANALYSIS_SUPPORT_CHECKED"].astype(bool)
    analysis_dates = pd.to_numeric(matched["ANALYSIS_DATE_COUNT"], errors="raise")
    supported_dates = pd.to_numeric(
        matched["SUPPORTED_ANALYSIS_DATE_COUNT"], errors="raise"
    )
    uncovered_dates = pd.to_numeric(
        matched["UNCOVERED_ANALYSIS_DATE_COUNT"], errors="raise"
    )
    if not np.array_equal(
        (supported_dates + uncovered_dates).to_numpy(float),
        analysis_dates.to_numpy(float),
    ):
        raise ValueError("analysis-date support counts do not sum to the point population")
    support_fraction = pd.to_numeric(
        matched["ANALYSIS_DATE_SUPPORT_FRACTION"], errors="coerce"
    )
    positive_analysis = analysis_dates > 0
    if support_fraction.loc[positive_analysis].isna().any() or not np.allclose(
        support_fraction.loc[positive_analysis].to_numpy(float),
        (
            supported_dates.loc[positive_analysis]
            / analysis_dates.loc[positive_analysis]
        ).to_numpy(float),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("analysis-date support fraction is inconsistent with counts")
    if support_fraction.loc[~positive_analysis].notna().any():
        raise ValueError("zero-date contrast must leave support fraction unavailable")
    if (
        ~support_checked.loc[eligible]
    ).any() or (uncovered_dates.loc[eligible] != 0).any() or not np.allclose(
        support_fraction.loc[eligible].to_numpy(float),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("eligible matched contrast lacks complete block support")

    allowed_drop_reasons = {
        "eligible",
        "missing_required_state",
        "no_common_month_solar_strata",
        "insufficient_unique_days",
        "insufficient_months",
        "insufficient_calendar_blocks",
        "uncovered_analysis_dates",
        "resampling_degenerate",
        "no_valid_replicates",
        "insufficient_valid_replicates",
        "no_reproduced_hours",
        "source_payload_missing",
    }
    if not set(matched["DROP_REASON"].astype(str)).issubset(allowed_drop_reasons):
        raise ValueError("unknown matched-contrast DROP_REASON")
    uncovered_reason = matched["DROP_REASON"].eq("uncovered_analysis_dates")
    if (
        (~support_checked.loc[uncovered_reason]).any()
        or (uncovered_dates.loc[uncovered_reason] <= 0).any()
        or matched.loc[uncovered_reason, "COMPARISON_ELIGIBLE"].astype(bool).any()
        or (matched.loc[uncovered_reason, "BLOCK7_VALID_REPLICATES"] != 0).any()
    ):
        raise ValueError("uncovered-analysis-date rows violate the ineligible contract")
    if (uncovered_dates.gt(0) != uncovered_reason).any():
        raise ValueError("uncovered analysis dates must use the dedicated drop reason")

    for row in matched.itertuples(index=False):
        try:
            uncovered_list = json.loads(row.UNCOVERED_ANALYSIS_DATES_JSON)
            support_diagnostics = json.loads(row.ANALYSIS_SUPPORT_DIAGNOSTICS_JSON)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid analysis-support JSON") from exc
        if not isinstance(uncovered_list, list) or len(uncovered_list) != int(
            row.UNCOVERED_ANALYSIS_DATE_COUNT
        ):
            raise ValueError("uncovered analysis-date JSON disagrees with its count")
        if not isinstance(support_diagnostics, list):
            raise ValueError("analysis-support diagnostics must be a JSON list")
        if bool(row.ANALYSIS_SUPPORT_CHECKED) and int(row.ANALYSIS_DATE_COUNT) > 0:
            if not support_diagnostics:
                raise ValueError("checked analysis support requires stratum diagnostics")

    q_values = pd.to_numeric(matched["Q_VALUE"], errors="coerce")
    if (q_values.dropna() < 0).any() or (q_values.dropna() > 1).any():
        raise ValueError("Q_VALUE must lie in [0, 1]")
    _validate_matched_inference_contract(matched)

    expected_setting_ids = {row["SETTING_ID"] for row in _sensitivity_settings()}
    setting_registry = {
        str(row["SETTING_ID"]): row for row in _sensitivity_settings()
    }
    if set(spatial_sensitivity["SETTING_ID"]) != expected_setting_ids:
        raise ValueError("spatial sensitivity does not contain the exact 81-setting grid")
    if set(spatial_sensitivity["STRUCTURE_STATE"]) != set(STRUCTURE_STATES):
        raise ValueError("spatial sensitivity state domain mismatch")
    spatial_group_size = spatial_sensitivity.groupby(["STN", "SETTING_ID"], sort=False).size()
    if not (spatial_group_size == 4).all():
        raise ValueError("every station-setting requires exactly four spatial-state rows")
    spatial_station_sizes = spatial_sensitivity.groupby("STN", sort=False).size().reindex(
        sorted(expected_stations)
    )
    spatial_station_setting_counts = spatial_sensitivity.groupby(
        "STN", sort=False
    )["SETTING_ID"].nunique().reindex(sorted(expected_stations))
    if (
        spatial_station_sizes.isna().any()
        or spatial_station_setting_counts.isna().any()
        or not spatial_station_sizes.eq(len(expected_setting_ids) * len(STRUCTURE_STATES)).all()
        or not spatial_station_setting_counts.eq(len(expected_setting_ids)).all()
    ):
        raise ValueError("stationwise spatial sensitivity Cartesian grid is incomplete")
    spatial_count_columns = [
        "TOTAL_PAIRED_HOURS",
        "REPRODUCED_HOURS",
        "N_HOURS",
        "ELIGIBLE_HOURS",
        "N_EXACT_REPRODUCTION_FAILURE",
        "N_NONFINITE_ELEVATION",
        "N_FAIL_MIN_N",
        "N_FAIL_SPAN",
        "N_FAIL_WEIGHTED_SD",
        "N_FAIL_N_EFF",
        "N_OLS_CI_CROSSES_ZERO",
        "N_OLS_AUX_SIGN_DISAGREE",
        "N_LOO_SIGN_UNSTABLE",
    ]
    for column in spatial_count_columns:
        values = pd.to_numeric(spatial_sensitivity[column], errors="raise").to_numpy(float)
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"invalid spatial sensitivity count: {column}")
    repeated_spatial_columns = [
        "TOTAL_PAIRED_HOURS",
        "REPRODUCED_HOURS",
        "ELIGIBLE_HOURS",
        "INVERSION_FRACTION_OF_ELIGIBLE",
        "N_EXACT_REPRODUCTION_FAILURE",
        "N_NONFINITE_ELEVATION",
        "N_FAIL_MIN_N",
        "N_FAIL_SPAN",
        "N_FAIL_WEIGHTED_SD",
        "N_FAIL_N_EFF",
        "N_OLS_CI_CROSSES_ZERO",
        "N_OLS_AUX_SIGN_DISAGREE",
        "N_LOO_SIGN_UNSTABLE",
        "MEDIAN_OLS_SLOPE_C_PER_KM",
        "MEDIAN_AUX_SLOPE_C_PER_KM",
        "MEDIAN_LOO_SIGN_STABILITY",
        "ANALYSIS_POPULATION",
    ]
    for column in ["MIN_N", "MIN_SPAN_M", "MIN_WEIGHTED_SD_M", "AUX_WEIGHT"]:
        expected = spatial_sensitivity["SETTING_ID"].map(
            {setting_id: registry[column] for setting_id, registry in setting_registry.items()}
        )
        if not spatial_sensitivity[column].eq(expected).all():
            raise ValueError("spatial sensitivity setting columns disagree with SETTING_ID")
    spatial_keys = ["STN", "SETTING_ID"]
    repeated_counts = spatial_sensitivity.groupby(spatial_keys, sort=False)[
        repeated_spatial_columns
    ].nunique(dropna=False)
    if (repeated_counts > 1).any().any():
        raise ValueError("spatial sensitivity repeated setting metrics are inconsistent")
    total_hours = pd.to_numeric(spatial_sensitivity["TOTAL_PAIRED_HOURS"], errors="raise").to_numpy(float)
    reproduced_hours = pd.to_numeric(spatial_sensitivity["REPRODUCED_HOURS"], errors="raise").to_numpy(float)
    n_hours = pd.to_numeric(spatial_sensitivity["N_HOURS"], errors="raise").to_numpy(float)
    eligible_hours = pd.to_numeric(spatial_sensitivity["ELIGIBLE_HOURS"], errors="raise").to_numpy(float)
    if np.any(reproduced_hours > total_hours):
        raise ValueError("spatial reproduced hours exceed paired hours")
    grouped_n = spatial_sensitivity.groupby(spatial_keys, sort=False)["N_HOURS"].transform("sum").to_numpy(float)
    if not np.array_equal(grouped_n, reproduced_hours):
        raise ValueError("spatial sensitivity state counts do not sum to reproduced hours")
    exact_failures = pd.to_numeric(
        spatial_sensitivity["N_EXACT_REPRODUCTION_FAILURE"], errors="raise"
    ).to_numpy(float)
    if not np.array_equal(exact_failures, total_hours - reproduced_hours):
        raise ValueError("spatial exact-reproduction failure count is inconsistent")
    eligible_component = np.where(
        spatial_sensitivity["STRUCTURE_STATE"].ne("ineligible"), n_hours, 0.0
    )
    expected_eligible = pd.Series(eligible_component, index=spatial_sensitivity.index).groupby(
        [spatial_sensitivity["STN"], spatial_sensitivity["SETTING_ID"]], sort=False
    ).transform("sum").to_numpy(float)
    if not np.array_equal(eligible_hours, expected_eligible):
        raise ValueError("spatial eligible-hour denominator is inconsistent")
    inversion_component = np.where(
        spatial_sensitivity["STRUCTURE_STATE"].eq("inversion_like"), n_hours, 0.0
    )
    inversion_hours = pd.Series(inversion_component, index=spatial_sensitivity.index).groupby(
        [spatial_sensitivity["STN"], spatial_sensitivity["SETTING_ID"]], sort=False
    ).transform("sum").to_numpy(float)
    expected_state_fraction = np.divide(
        n_hours,
        reproduced_hours,
        out=np.full(len(spatial_sensitivity), np.nan),
        where=reproduced_hours > 0,
    )
    if not np.allclose(
        pd.to_numeric(spatial_sensitivity["STATE_FRACTION_OF_REPRODUCED"], errors="coerce"),
        expected_state_fraction,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("spatial state fraction uses the wrong denominator")
    expected_eligible_fraction = np.divide(
        eligible_hours,
        reproduced_hours,
        out=np.full(len(spatial_sensitivity), np.nan),
        where=reproduced_hours > 0,
    )
    if not np.allclose(
        pd.to_numeric(spatial_sensitivity["ELIGIBLE_FRACTION_OF_REPRODUCED"], errors="coerce"),
        expected_eligible_fraction,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("spatial eligible fraction uses the wrong denominator")
    expected_inversion_fraction = np.divide(
        inversion_hours,
        eligible_hours,
        out=np.full(len(spatial_sensitivity), np.nan),
        where=eligible_hours > 0,
    )
    if not np.allclose(
        pd.to_numeric(spatial_sensitivity["INVERSION_FRACTION_OF_ELIGIBLE"], errors="coerce"),
        expected_inversion_fraction,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("spatial inversion fraction uses the wrong denominator")

    expected_coverage_ids = {
        f"d{days}_m{months}"
        for days, months in product(SENSITIVITY_MIN_DAYS, SENSITIVITY_MIN_MONTHS)
    }
    if set(coverage_sensitivity["SETTING_ID"]) != expected_setting_ids or set(
        coverage_sensitivity["COVERAGE_ID"]
    ) != expected_coverage_ids:
        raise ValueError("matched coverage sensitivity grid is incomplete")
    coverage_group_size = coverage_sensitivity.groupby(["STN", "SETTING_ID"], sort=False).size()
    if not (coverage_group_size == 9).all():
        raise ValueError("every station-setting requires the nine coverage combinations")
    coverage_station_sizes = coverage_sensitivity.groupby("STN", sort=False).size().reindex(
        sorted(expected_stations)
    )
    coverage_station_setting_counts = coverage_sensitivity.groupby(
        "STN", sort=False
    )["SETTING_ID"].nunique().reindex(sorted(expected_stations))
    if (
        coverage_station_sizes.isna().any()
        or coverage_station_setting_counts.isna().any()
        or not coverage_station_sizes.eq(
            len(expected_setting_ids) * len(expected_coverage_ids)
        ).all()
        or not coverage_station_setting_counts.eq(len(expected_setting_ids)).all()
    ):
        raise ValueError("stationwise coverage sensitivity Cartesian grid is incomplete")
    coverage_count_columns = [
        "MIN_DAYS_PER_STATE",
        "MIN_MONTHS_PER_STATE",
        "N_COMMON_STRATA",
        "N_INVERSION_HOURS",
        "N_NORMAL_HOURS",
        "N_INVERSION_DAYS",
        "N_NORMAL_DAYS",
        "N_INVERSION_MONTHS",
        "N_NORMAL_MONTHS",
    ]
    for column in coverage_count_columns:
        values = pd.to_numeric(coverage_sensitivity[column], errors="raise").to_numpy(float)
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"invalid matched coverage sensitivity count: {column}")
    allowed_coverage_reasons = {
        "eligible",
        "missing_required_state",
        "no_common_month_solar_strata",
        "insufficient_unique_days",
        "insufficient_months",
        "source_payload_missing",
    }
    if not set(coverage_sensitivity["DROP_REASON"].astype(str)).issubset(
        allowed_coverage_reasons
    ):
        raise ValueError("unknown matched coverage sensitivity DROP_REASON")
    raw_columns = [
        "N_COMMON_STRATA",
        "N_INVERSION_HOURS",
        "N_NORMAL_HOURS",
        "N_INVERSION_DAYS",
        "N_NORMAL_DAYS",
        "N_INVERSION_MONTHS",
        "N_NORMAL_MONTHS",
        "DELTA_ABS_BIAS_INVERSION",
        "DELTA_ABS_BIAS_NORMAL",
        "CONTRAST_INV_MINUS_NORMAL",
    ]
    for column in ["MIN_N", "MIN_SPAN_M", "MIN_WEIGHTED_SD_M", "AUX_WEIGHT"]:
        expected = coverage_sensitivity["SETTING_ID"].map(
            {setting_id: registry[column] for setting_id, registry in setting_registry.items()}
        )
        if not coverage_sensitivity[column].eq(expected).all():
            raise ValueError("matched coverage setting columns disagree with SETTING_ID")
    coverage_keys = ["STN", "SETTING_ID"]
    raw_nunique = coverage_sensitivity.groupby(coverage_keys, sort=False)[raw_columns].nunique(
        dropna=False
    )
    if (raw_nunique > 1).any().any():
        raise ValueError("coverage thresholds changed the underlying matched estimand")
    expected_coverage_id = (
        "d"
        + coverage_sensitivity["MIN_DAYS_PER_STATE"].astype(int).astype(str)
        + "_m"
        + coverage_sensitivity["MIN_MONTHS_PER_STATE"].astype(int).astype(str)
    )
    if not coverage_sensitivity["COVERAGE_ID"].eq(expected_coverage_id).all():
        raise ValueError("coverage ID disagrees with its day/month thresholds")
    base_available = coverage_sensitivity["BASE_MATCHED_DATA_AVAILABLE"].astype(bool).to_numpy()
    day_fail = base_available & (
        (coverage_sensitivity["N_INVERSION_DAYS"].to_numpy(float) < coverage_sensitivity["MIN_DAYS_PER_STATE"].to_numpy(float))
        | (coverage_sensitivity["N_NORMAL_DAYS"].to_numpy(float) < coverage_sensitivity["MIN_DAYS_PER_STATE"].to_numpy(float))
    )
    month_fail = base_available & ~day_fail & (
        (coverage_sensitivity["N_INVERSION_MONTHS"].to_numpy(float) < coverage_sensitivity["MIN_MONTHS_PER_STATE"].to_numpy(float))
        | (coverage_sensitivity["N_NORMAL_MONTHS"].to_numpy(float) < coverage_sensitivity["MIN_MONTHS_PER_STATE"].to_numpy(float))
    )
    expected_eligible = base_available & ~day_fail & ~month_fail
    actual_eligible = coverage_sensitivity["COVERAGE_ELIGIBLE"].astype(bool).to_numpy()
    if not np.array_equal(actual_eligible, expected_eligible):
        raise ValueError("matched coverage eligibility disagrees with exact thresholds")
    reasons = coverage_sensitivity["DROP_REASON"].astype(str).to_numpy()
    available_expected_reason = np.where(
        day_fail,
        "insufficient_unique_days",
        np.where(month_fail, "insufficient_months", "eligible"),
    )
    if np.any(base_available & (reasons != available_expected_reason)):
        raise ValueError("matched coverage gate/drop reason mismatch")
    unavailable_allowed = np.isin(
        reasons,
        ["missing_required_state", "no_common_month_solar_strata", "source_payload_missing"],
    )
    if np.any(~base_available & ~unavailable_allowed):
        raise ValueError("unavailable matched data use an invalid drop reason")
    primary_coverage = coverage_sensitivity.loc[
        coverage_sensitivity["SETTING_ID"].eq(PRIMARY_SETTING_ID)
        & coverage_sensitivity["COVERAGE_ID"].eq(PRIMARY_COVERAGE_ID)
    ].set_index("STN")
    primary_matched = matched.set_index("STN")
    if set(primary_coverage.index) != set(primary_matched.index):
        raise ValueError("primary matched/sensitivity station skeleton mismatch")
    primary_coverage = primary_coverage.loc[primary_matched.index]
    coverage_expected_source_missing = coverage_sensitivity["STN"].map(
        pd.Series(source_missing.to_numpy(bool), index=matched["STN"])
    )
    coverage_reports_source_missing = coverage_sensitivity["DROP_REASON"].astype(
        str
    ).eq("source_payload_missing")
    if (
        coverage_expected_source_missing.isna().any()
        or not np.array_equal(
            coverage_reports_source_missing.to_numpy(bool),
            coverage_expected_source_missing.to_numpy(bool),
        )
        or coverage_sensitivity.loc[
            coverage_expected_source_missing,
            ["BASE_MATCHED_DATA_AVAILABLE", "COVERAGE_ELIGIBLE"],
        ]
        .astype(bool)
        .any()
        .any()
    ):
        raise ValueError(
            "source_payload_missing must agree across matched and coverage skeletons"
        )
    primary_coverage_reason = primary_coverage["DROP_REASON"].astype(str)
    primary_matched_reason = primary_matched["DROP_REASON"].astype(str)
    early_coverage_reasons = {
        "missing_required_state",
        "no_common_month_solar_strata",
        "insufficient_unique_days",
        "insufficient_months",
        "source_payload_missing",
    }
    post_coverage_reasons = {
        "eligible",
        "insufficient_calendar_blocks",
        "uncovered_analysis_dates",
        "resampling_degenerate",
        "no_valid_replicates",
        "insufficient_valid_replicates",
    }
    no_reproduced_primary = primary_matched_reason.eq("no_reproduced_hours")
    coverage_stops_early = primary_coverage_reason.isin(early_coverage_reasons)
    early_reason_mismatch = (
        ~no_reproduced_primary
        & coverage_stops_early
        & ~primary_matched_reason.eq(primary_coverage_reason)
    )
    post_coverage_mismatch = (
        ~no_reproduced_primary
        & primary_coverage_reason.eq("eligible")
        & ~primary_matched_reason.isin(post_coverage_reasons)
    )
    if early_reason_mismatch.any() or post_coverage_mismatch.any():
        raise ValueError(
            "matched DROP_REASON disagrees with the primary coverage-stage result"
        )
    exact_pairs = {
        "N_STRATA": "N_COMMON_STRATA",
        "DELTA_ABS_BIAS_INVERSION": "DELTA_ABS_BIAS_INVERSION",
        "DELTA_ABS_BIAS_NORMAL": "DELTA_ABS_BIAS_NORMAL",
        "CONTRAST_INV_MINUS_NORMAL": "CONTRAST_INV_MINUS_NORMAL",
    }
    for matched_column, sensitivity_column in exact_pairs.items():
        if not np.allclose(
            pd.to_numeric(primary_matched[matched_column], errors="coerce"),
            pd.to_numeric(primary_coverage[sensitivity_column], errors="coerce"),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("primary matched estimate differs from its fixed sensitivity row")
    expected_primary_days = primary_coverage[
        ["N_INVERSION_DAYS", "N_NORMAL_DAYS"]
    ].min(axis=1)
    expected_primary_months = primary_coverage[
        ["N_INVERSION_MONTHS", "N_NORMAL_MONTHS"]
    ].min(axis=1)
    if not np.array_equal(
        pd.to_numeric(primary_matched["N_DAYS"], errors="raise").to_numpy(float),
        expected_primary_days.to_numpy(float),
    ) or not np.array_equal(
        pd.to_numeric(primary_matched["N_MONTHS"], errors="raise").to_numpy(float),
        expected_primary_months.to_numpy(float),
    ):
        raise ValueError("primary matched coverage counts differ from sensitivity")
    comparison_eligible = primary_matched["COMPARISON_ELIGIBLE"].astype(bool)
    coverage_eligible = primary_coverage["COVERAGE_ELIGIBLE"].astype(bool)
    diagnostic_eligible = (
        coverage_eligible
        & primary_matched["ANALYSIS_SUPPORT_CHECKED"].astype(bool)
        & pd.to_numeric(
            primary_matched["UNCOVERED_ANALYSIS_DATE_COUNT"], errors="raise"
        ).eq(0)
        & np.isclose(
            pd.to_numeric(
                primary_matched["ANALYSIS_DATE_SUPPORT_FRACTION"], errors="coerce"
            ),
            1.0,
            rtol=0,
            atol=1e-12,
        )
        & (
            pd.to_numeric(
                primary_matched["BLOCK7_VALID_FRACTION"], errors="coerce"
            )
            >= 0.95
        )
        & (
            pd.to_numeric(
                primary_matched["BLOCK7_VALID_REPLICATES"], errors="raise"
            )
            >= 1900
        )
    )
    if not np.array_equal(
        comparison_eligible.to_numpy(bool),
        np.asarray(diagnostic_eligible, dtype=bool),
    ):
        raise ValueError(
            "primary matched eligibility differs from coverage/support/replicate gates"
        )

    _validate_primary_eligibility_ledger_contract(
        primary_ledger,
        matched,
        coverage_sensitivity,
        station_ids=expected_stations,
    )

    exact_contrasts = {
        "H1_ALL_HIGH_V_CONTROL",
        "H1_POS_HIGH_V_CONTROL",
        "H1_NEG_HIGH_V_CONTROL",
        "H2_ALL_HIGH_V_CONTROL",
        "H2_POS_HIGH_V_CONTROL",
        "H2_NEG_HIGH_V_CONTROL",
    }
    if set(physical["CONTRAST_ID"]) != exact_contrasts or len(physical) != 6:
        raise ValueError("physical hypothesis table must contain the exact six contrasts")
    physical_domains = {
        "HYPOTHESIS_ID": {"H1_LOCAL_TOPOGRAPHY", "H2_COAST_MISMATCH"},
        "ROW_KIND": {"group_comparison"},
        "ANALYSIS_ROLE": {"primary", "sign_stratified_descriptive"},
        "SIGN_STRATUM": {"all", "positive", "negative"},
        "ROW_UNIT": {"station_group_contrast"},
        "VARIABLE": {"DARK_INV_FRACTION", "ABS_DCOAST_KM"},
        "VARIABLE_UNIT": {"fraction", "km"},
        "EFFECT_SIZE_NAME": {"rank_biserial"},
        "OBSERVED_EFFECT_DIRECTION": {"greater", "lower", "tie", "not_estimable"},
        "INFERENCE_MODE": {"descriptive_only", "spatial_block_bootstrap"},
        "SPATIAL_BLOCK_METHOD": {"none", "EPSG5179_fixed_square_cluster_bootstrap"},
        "FAMILY_ID": {"NONE_DESCRIPTIVE", "PHYSICAL_H2_PRIMARY_50KM"},
        "Q_METHOD": {"none", "BH_eligible_primary_rows_only"},
        "ANALYSIS_POPULATION": {"practically_equivalent_115_after_bias_level"},
    }
    for column, domain in physical_domains.items():
        if not set(physical[column].dropna()).issubset(domain):
            raise ValueError(f"physical hypothesis {column} domain mismatch")
    probability_columns = ["P_VALUE_30KM", "P_VALUE", "P_VALUE_100KM", "Q_VALUE"]
    for column in probability_columns:
        numeric = pd.to_numeric(physical[column], errors="coerce")
        supplied = physical[column].notna()
        if (supplied & numeric.isna()).any() or (
            supplied & ((numeric < 0) | (numeric > 1))
        ).any():
            raise ValueError("physical probability values must lie in [0, 1]")
    for block in (30, 50, 100):
        low = pd.to_numeric(physical[f"CI_LOW_{block}KM"], errors="coerce")
        high = pd.to_numeric(physical[f"CI_HIGH_{block}KM"], errors="coerce")
        paired = low.notna() & high.notna()
        if (low.notna() != high.notna()).any() or (paired & (low > high)).any():
            raise ValueError("physical spatial confidence interval is invalid")
    for row in physical.itertuples(index=False):
        expected_hypothesis = (
            "H1_LOCAL_TOPOGRAPHY"
            if str(row.CONTRAST_ID).startswith("H1_")
            else "H2_COAST_MISMATCH"
        )
        expected_sign = (
            "all"
            if "_ALL_" in str(row.CONTRAST_ID)
            else "positive"
            if "_POS_" in str(row.CONTRAST_ID)
            else "negative"
        )
        if row.HYPOTHESIS_ID != expected_hypothesis or row.SIGN_STRATUM != expected_sign:
            raise ValueError("physical contrast ID disagrees with hypothesis/sign stratum")
        if int(row.TARGET_EXCLUDED_N) != int(row.TARGET_CANONICAL_N) - int(
            row.TARGET_ELIGIBLE_N
        ) or int(row.REFERENCE_EXCLUDED_N) != int(row.REFERENCE_CANONICAL_N) - int(
            row.REFERENCE_ELIGIBLE_N
        ):
            raise ValueError("physical eligible/excluded denominators are inconsistent")

    # 66열 표 안에 적힌 분모·효과·방향을 그대로 신뢰하지 않고, 게시된
    # equivalent/station/coast 원표에서 다시 계산한다.
    equivalent_for_physical = tables["equivalent_diagnostics.csv"].copy()
    equivalent_for_physical["STN"] = (
        equivalent_for_physical["STN"].astype(str).str.strip()
    )
    equivalent_for_physical["__BIAS_FIXED"] = pd.to_numeric(
        equivalent_for_physical["BIAS_fixed"], errors="raise"
    )
    equivalent_for_physical["__ABS_FIXED"] = pd.to_numeric(
        equivalent_for_physical["ABS_BIAS_FIXED"], errors="raise"
    )
    equivalent_for_physical["__GROUP"] = np.where(
        equivalent_for_physical["__ABS_FIXED"] > 0.5, "high_after", "control"
    )
    equivalent_for_physical["__SIGN"] = np.where(
        equivalent_for_physical["__BIAS_FIXED"] > 0,
        "positive",
        np.where(equivalent_for_physical["__BIAS_FIXED"] < 0, "negative", "zero"),
    )

    structure_for_physical = station.copy()
    structure_for_physical["STN"] = structure_for_physical["STN"].astype(str).str.strip()
    structure_for_physical["__H1_VALUE"] = pd.to_numeric(
        structure_for_physical["DARK_INV_FRACTION"], errors="coerce"
    )
    structure_for_physical["__H2_VALUE"] = pd.to_numeric(
        structure_for_physical["MEDIAN_ABS_DELTA_COAST_KM"], errors="coerce"
    )
    dark_gate = (
        pd.to_numeric(
            structure_for_physical["DARK_PRIMARY_ELIGIBLE_DAYS"], errors="coerce"
        )
        >= 30
    ) & (
        pd.to_numeric(
            structure_for_physical["DARK_PRIMARY_ELIGIBLE_MONTHS"], errors="coerce"
        )
        >= 6
    )
    structure_for_physical.loc[~dark_gate, "__H1_VALUE"] = np.nan

    coast_for_physical = tables["coast_time_signed_bias.csv"].copy()
    if len(coast_for_physical):
        coast_for_physical["STN"] = coast_for_physical["STN"].astype(str).str.strip()
        for coast_row in coast_for_physical.itertuples(index=False):
            expected_sign = _h2_expected_sign(
                float(coast_row.DELTA_COAST_KM),
                int(coast_row.MONTH),
                str(coast_row.SOLAR_STATE),
            )
            observed_sign = (
                "positive"
                if float(coast_row.BIAS_fixed) > 0
                else "negative"
                if float(coast_row.BIAS_fixed) < 0
                else "zero"
            )
            expected_match = bool(
                expected_sign in {"positive", "negative"}
                and observed_sign == expected_sign
            )
            expected_grade = (
                "문헌과 일치하나 직접 미측정"
                if expected_match
                else "현재 자료에서 지지되지 않음"
            )
            if (
                coast_row.H2_EXPECTED_BIAS_SIGN != expected_sign
                or coast_row.H2_OBSERVED_BIAS_SIGN != observed_sign
                or bool(coast_row.H2_DIRECTION_MATCH) != expected_match
                or coast_row.EVIDENCE_GRADE != expected_grade
            ):
                raise ValueError(
                    "coast H2 direction fields differ from signed BIAS and prespecified rule"
                )
    else:
        coast_for_physical = pd.DataFrame(columns=COAST_TIME_COLUMNS)
    direction_for_physical = _direction_station_summary(coast_for_physical)
    if len(direction_for_physical):
        direction_for_physical["STN"] = (
            direction_for_physical["STN"].astype(str).str.strip()
        )

    physical_source = equivalent_for_physical.merge(
        structure_for_physical[
            [
                "STN",
                "__H1_VALUE",
                "__H2_VALUE",
                "TARGET_LATITUDE",
                "TARGET_LONGITUDE",
            ]
        ],
        on="STN",
        how="left",
    )
    physical_source = physical_source.merge(
        direction_for_physical, on="STN", how="left"
    )
    for column in ["MATCH_N", "TESTABLE_N"]:
        physical_source[column] = pd.to_numeric(
            physical_source.get(column, 0), errors="coerce"
        ).fillna(0).astype(int)
    physical_source["DIRECTION_GATE"] = physical_source.get(
        "DIRECTION_GATE", False
    ).fillna(False).astype(bool)

    def same_optional_float(left: Any, right: Any) -> bool:
        if pd.isna(left) and pd.isna(right):
            return True
        if pd.isna(left) or pd.isna(right):
            return False
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))

    for row in physical.itertuples(index=False):
        sign_subset = (
            physical_source
            if row.SIGN_STRATUM == "all"
            else physical_source.loc[physical_source["__SIGN"].eq(row.SIGN_STRATUM)]
        )
        target_all = sign_subset.loc[sign_subset["__GROUP"].eq("high_after")]
        reference_all = sign_subset.loc[sign_subset["__GROUP"].eq("control")]
        value_column = (
            "__H1_VALUE"
            if row.HYPOTHESIS_ID == "H1_LOCAL_TOPOGRAPHY"
            else "__H2_VALUE"
        )
        target = target_all.loc[pd.to_numeric(target_all[value_column], errors="coerce").notna()]
        reference = reference_all.loc[
            pd.to_numeric(reference_all[value_column], errors="coerce").notna()
        ]
        expected_counts = (
            len(target_all),
            len(reference_all),
            len(target),
            len(reference),
            len(target_all) - len(target),
            len(reference_all) - len(reference),
        )
        actual_counts = (
            int(row.TARGET_CANONICAL_N),
            int(row.REFERENCE_CANONICAL_N),
            int(row.TARGET_ELIGIBLE_N),
            int(row.REFERENCE_ELIGIBLE_N),
            int(row.TARGET_EXCLUDED_N),
            int(row.REFERENCE_EXCLUDED_N),
        )
        if actual_counts != expected_counts:
            raise ValueError("physical canonical denominators differ from source tables")
        expected_coverage = bool(
            len(target_all) > 0
            and len(reference_all) > 0
            and len(target) >= math.ceil(0.8 * len(target_all))
            and len(reference) >= math.ceil(0.8 * len(reference_all))
        )
        if bool(row.COVERAGE_ELIGIBLE) != expected_coverage:
            raise ValueError("physical coverage gate differs from source denominators")

        target_values = pd.to_numeric(target[value_column], errors="raise").to_numpy(float)
        reference_values = pd.to_numeric(reference[value_column], errors="raise").to_numpy(float)
        target_median = float(np.median(target_values)) if len(target_values) else np.nan
        reference_median = (
            float(np.median(reference_values)) if len(reference_values) else np.nan
        )
        median_difference = (
            target_median - reference_median
            if np.isfinite(target_median) and np.isfinite(reference_median)
            else np.nan
        )
        effect = (
            rank_biserial_effect(target_values, reference_values)
            if len(target_values) and len(reference_values)
            else np.nan
        )
        expected_observed = (
            "greater"
            if effect > 0
            else "lower"
            if effect < 0
            else "tie"
            if effect == 0
            else "not_estimable"
        )
        expected_effect_match: bool | float = (
            bool(effect > 0) if np.isfinite(effect) else np.nan
        )
        if not all(
            [
                same_optional_float(row.TARGET_MEDIAN, target_median),
                same_optional_float(row.REFERENCE_MEDIAN, reference_median),
                same_optional_float(row.MEDIAN_DIFFERENCE, median_difference),
                same_optional_float(row.EFFECT_ESTIMATE, effect),
            ]
        ) or row.OBSERVED_EFFECT_DIRECTION != expected_observed or not same_optional_float(
            row.EFFECT_DIRECTION_MATCH, expected_effect_match
        ):
            raise ValueError("physical effect estimate/direction differs from source tables")

        if row.HYPOTHESIS_ID == "H2_COAST_MISMATCH":
            target_gate = target_all["DIRECTION_GATE"].astype(bool)
            reference_gate = reference_all["DIRECTION_GATE"].astype(bool)
            expected_target_match = int(target_all.loc[target_gate, "MATCH_N"].sum())
            expected_target_testable = int(
                target_all.loc[target_gate, "TESTABLE_N"].sum()
            )
            expected_reference_match = int(
                reference_all.loc[reference_gate, "MATCH_N"].sum()
            )
            expected_reference_testable = int(
                reference_all.loc[reference_gate, "TESTABLE_N"].sum()
            )
            expected_target_rate = (
                expected_target_match / expected_target_testable
                if expected_target_testable
                else np.nan
            )
            expected_reference_rate = (
                expected_reference_match / expected_reference_testable
                if expected_reference_testable
                else np.nan
            )
            direction_coverage = bool(
                len(target_all) > 0
                and len(reference_all) > 0
                and int(target_gate.sum()) >= math.ceil(0.8 * len(target_all))
                and int(reference_gate.sum()) >= math.ceil(0.8 * len(reference_all))
            )
            expected_direction_match = bool(
                direction_coverage
                and np.isfinite(expected_target_rate)
                and np.isfinite(expected_reference_rate)
                and expected_target_rate > 0.5
                and expected_target_rate > expected_reference_rate
            )
            actual_direction_counts = (
                int(row.TARGET_DIRECTION_MATCH_N),
                int(row.TARGET_DIRECTION_TESTABLE_N),
                int(row.REFERENCE_DIRECTION_MATCH_N),
                int(row.REFERENCE_DIRECTION_TESTABLE_N),
            )
            expected_direction_counts = (
                expected_target_match,
                expected_target_testable,
                expected_reference_match,
                expected_reference_testable,
            )
            if actual_direction_counts != expected_direction_counts or not all(
                [
                    same_optional_float(
                        row.TARGET_DIRECTION_MATCH_RATE, expected_target_rate
                    ),
                    same_optional_float(
                        row.REFERENCE_DIRECTION_MATCH_RATE, expected_reference_rate
                    ),
                ]
            ) or bool(row.BIAS_DIRECTION_MATCH) != expected_direction_match:
                raise ValueError("physical direction counts/rates differ from coast table")
            if row.SIGN_STRATUM != "all":
                expected_descriptive_grade = (
                    "문헌과 일치하나 직접 미측정"
                    if expected_coverage
                    and np.isfinite(median_difference)
                    and np.isfinite(effect)
                    and median_difference > 0
                    and effect > 0
                    and expected_direction_match
                    else "현재 자료에서 지지되지 않음"
                )
                if row.EVIDENCE_GRADE != expected_descriptive_grade:
                    raise ValueError("descriptive H2 evidence grade violates its fixed gate")
    h1_or_descriptive = physical["INFERENCE_MODE"].eq("descriptive_only")
    inferential_columns = [
        "PRIMARY_BLOCK_KM",
        "N_BLOCKS_30KM",
        "TARGET_BLOCKS_30KM",
        "REFERENCE_BLOCKS_30KM",
        "VALID_REPLICATES_30KM",
        "CI_LOW_30KM",
        "CI_HIGH_30KM",
        "P_VALUE_30KM",
        "N_BLOCKS_50KM",
        "TARGET_BLOCKS_50KM",
        "REFERENCE_BLOCKS_50KM",
        "VALID_REPLICATES_50KM",
        "CI_LOW_50KM",
        "CI_HIGH_50KM",
        "P_VALUE",
        "N_BLOCKS_100KM",
        "TARGET_BLOCKS_100KM",
        "REFERENCE_BLOCKS_100KM",
        "VALID_REPLICATES_100KM",
        "CI_LOW_100KM",
        "CI_HIGH_100KM",
        "P_VALUE_100KM",
        "Q_VALUE",
        "SPATIAL_SENSITIVITY_DIRECTION_STABLE",
    ]
    if physical.loc[h1_or_descriptive, inferential_columns].notna().any().any():
        raise ValueError("descriptive physical contrasts must not publish inferential values")
    if not physical.loc[h1_or_descriptive, "FAMILY_SIZE"].eq(0).all():
        raise ValueError("descriptive physical contrasts must stay outside the test family")
    supported_h1 = physical["HYPOTHESIS_ID"].eq("H1_LOCAL_TOPOGRAPHY") & physical[
        "EVIDENCE_GRADE"
    ].eq("자료로 지지")
    if supported_h1.any():
        raise ValueError("H1 cannot be upgraded to direct support without TPI")
    for _, row in physical.loc[
        physical["HYPOTHESIS_ID"].eq("H1_LOCAL_TOPOGRAPHY")
    ].iterrows():
        if (
            row["VARIABLE"] != "DARK_INV_FRACTION"
            or row["EXPECTED_BIAS_DIRECTION_RULE"]
            != "NOT_TESTABLE_TPI_NOT_MEASURED"
            or pd.notna(row["BIAS_DIRECTION_MATCH"])
        ):
            raise ValueError("H1 must remain an indirect TPI-not-measured diagnostic")
        expected_h1_grade = (
            "문헌과 일치하나 직접 미측정"
            if bool(row["COVERAGE_ELIGIBLE"])
            and pd.notna(row["MEDIAN_DIFFERENCE"])
            and pd.notna(row["EFFECT_ESTIMATE"])
            and row["MEDIAN_DIFFERENCE"] > 0
            and row["EFFECT_ESTIMATE"] > 0
            else "현재 자료에서 지지되지 않음"
        )
        if row["EVIDENCE_GRADE"] != expected_h1_grade:
            raise ValueError("H1 evidence grade violates its descriptive-only gate")
    h2 = physical.loc[physical["CONTRAST_ID"].eq("H2_ALL_HIGH_V_CONTROL")]
    if len(h2):
        row = h2.iloc[0]
        if (
            row["INFERENCE_MODE"] != "spatial_block_bootstrap"
            or row["SPATIAL_BLOCK_METHOD"]
            != "EPSG5179_fixed_square_cluster_bootstrap"
            or row["PRIMARY_BLOCK_KM"] != 50
            or row["FAMILY_ID"] != "PHYSICAL_H2_PRIMARY_50KM"
            or row["FAMILY_SIZE"] != 1
            or row["Q_METHOD"] != "BH_eligible_primary_rows_only"
        ):
            raise ValueError("physical H2 primary inference registry mismatch")
        expected_spatial: dict[int, dict[str, Any]] = {}
        if bool(row["COVERAGE_ELIGIBLE"]):
            spatial_source = physical_source.loc[
                physical_source["__GROUP"].isin(["high_after", "control"])
            ]
            spatial_values = pd.to_numeric(
                spatial_source["__H2_VALUE"], errors="coerce"
            ).to_numpy(float)
            spatial_groups = spatial_source["__GROUP"].to_numpy(str)
            spatial_latitude = pd.to_numeric(
                spatial_source["TARGET_LATITUDE"], errors="coerce"
            ).to_numpy(float)
            spatial_longitude = pd.to_numeric(
                spatial_source["TARGET_LONGITUDE"], errors="coerce"
            ).to_numpy(float)
            for block_index, block in enumerate((30, 50, 100)):
                expected_spatial[block] = _spatial_block_rbc(
                    spatial_values,
                    spatial_groups,
                    spatial_latitude,
                    spatial_longitude,
                    block_km=block,
                    n_replicates=2000,
                    seed=20260827 + block_index * 1009,
                )
        for block in (30, 50, 100):
            expected_block = expected_spatial.get(block)
            expected_values = {
                f"N_BLOCKS_{block}KM": (
                    expected_block["n_blocks"] if expected_block is not None else np.nan
                ),
                f"TARGET_BLOCKS_{block}KM": (
                    expected_block["target_blocks"]
                    if expected_block is not None
                    else np.nan
                ),
                f"REFERENCE_BLOCKS_{block}KM": (
                    expected_block["reference_blocks"]
                    if expected_block is not None
                    else np.nan
                ),
                f"VALID_REPLICATES_{block}KM": (
                    expected_block["valid_replicates"]
                    if expected_block is not None
                    else np.nan
                ),
                f"CI_LOW_{block}KM": (
                    expected_block["ci_low"] if expected_block is not None else np.nan
                ),
                f"CI_HIGH_{block}KM": (
                    expected_block["ci_high"] if expected_block is not None else np.nan
                ),
                ("P_VALUE" if block == 50 else f"P_VALUE_{block}KM"): (
                    expected_block["p_value"] if expected_block is not None else np.nan
                ),
            }
            if any(
                not same_optional_float(row[column], expected)
                for column, expected in expected_values.items()
            ):
                raise ValueError(
                    "physical spatial inference differs from fixed-seed source recomputation"
                )
        expected_q = (
            expected_spatial[50]["p_value"]
            if 50 in expected_spatial and expected_spatial[50]["eligible"]
            else np.nan
        )
        all_spatial_eligible = bool(
            expected_spatial
            and all(expected_spatial[block]["eligible"] for block in (30, 50, 100))
        )
        expected_stability: bool | float = (
            all(expected_spatial[block]["ci_low"] > 0 for block in (30, 50, 100))
            if all_spatial_eligible
            else np.nan
        )
        if not same_optional_float(row["Q_VALUE"], expected_q) or not same_optional_float(
            row["SPATIAL_SENSITIVITY_DIRECTION_STABLE"], expected_stability
        ):
            raise ValueError(
                "physical spatial inference q/stability differs from source recomputation"
            )
        block_eligible: dict[int, bool] = {}
        for block in (30, 50, 100):
            counts = [
                row[f"N_BLOCKS_{block}KM"],
                row[f"TARGET_BLOCKS_{block}KM"],
                row[f"REFERENCE_BLOCKS_{block}KM"],
                row[f"VALID_REPLICATES_{block}KM"],
            ]
            counts_available = all(pd.notna(value) for value in counts)
            block_eligible[block] = bool(
                counts_available
                and int(counts[0]) >= 8
                and int(counts[1]) >= 4
                and int(counts[2]) >= 4
                and int(counts[3]) >= 1900
            )
            inference_values = [
                row[f"CI_LOW_{block}KM"],
                row[f"CI_HIGH_{block}KM"],
                row["P_VALUE" if block == 50 else f"P_VALUE_{block}KM"],
            ]
            if block_eligible[block] != all(
                pd.notna(value) for value in inference_values
            ):
                raise ValueError("physical spatial block gate and CI/p availability disagree")
        if pd.notna(row["P_VALUE"]):
            if pd.isna(row["Q_VALUE"]) or not np.isclose(
                float(row["P_VALUE"]), float(row["Q_VALUE"]), atol=1e-12, rtol=0
            ):
                raise ValueError("physical H2 q must equal the sole eligible-family p")
        elif pd.notna(row["Q_VALUE"]):
            raise ValueError("physical H2 q requires a finite primary p")
        direction_match = bool(row["BIAS_DIRECTION_MATCH"])
        effect_direction = bool(
            pd.notna(row["MEDIAN_DIFFERENCE"])
            and pd.notna(row["EFFECT_ESTIMATE"])
            and row["MEDIAN_DIFFERENCE"] > 0
            and row["EFFECT_ESTIMATE"] > 0
        )
        full_support = bool(
            row["COVERAGE_ELIGIBLE"]
            and effect_direction
            and direction_match
            and pd.notna(row["Q_VALUE"])
            and row["Q_VALUE"] <= 0.05
            and all(block_eligible.values())
            and all(row[f"CI_LOW_{block}KM"] > 0 for block in (30, 50, 100))
        )
        expected_grade = (
            "자료로 지지"
            if full_support
            else "현재 자료에서 지지되지 않음"
            if (
                not bool(row["COVERAGE_ELIGIBLE"])
                or not effect_direction
                or not direction_match
            )
            else "문헌과 일치하나 직접 미측정"
        )
        if row["EVIDENCE_GRADE"] != expected_grade:
            raise ValueError("H2 evidence grade violates spatial/direction gates")
    canonical_class_counts = (
        truth["SIGNIFICANCE_CLASS"].value_counts().to_dict()
        if truth is not None and "SIGNIFICANCE_CLASS" in truth
        else {}
    )
    if truth is not None and len(truth) == 529:
        is_canonical_labels = canonical_class_counts == {
            "meaningful_improvement": 221,
            "meaningful_worsening": 109,
            "practically_equivalent": 115,
            "uncertain": 84,
        }
    else:
        is_canonical_labels = False
    if is_canonical_labels:
        all_h1 = physical.loc[physical["CONTRAST_ID"].eq("H1_ALL_HIGH_V_CONTROL")].iloc[0]
        if int(all_h1["TARGET_CANONICAL_N"]) != 33 or int(all_h1["REFERENCE_CANONICAL_N"]) != 82:
            raise ValueError("canonical equivalent high33/control82 invariant failed")
        positive = physical.loc[physical["CONTRAST_ID"].eq("H1_POS_HIGH_V_CONTROL")].iloc[0]
        negative = physical.loc[physical["CONTRAST_ID"].eq("H1_NEG_HIGH_V_CONTROL")].iloc[0]
        if int(positive["TARGET_CANONICAL_N"]) != 19 or int(negative["TARGET_CANONICAL_N"]) != 14:
            raise ValueError("canonical high-after signed 19/14 invariant failed")

    totals = pd.to_numeric(station["TOTAL_PAIRED_HOURS"], errors="raise")
    reproduced = pd.to_numeric(station["REPRODUCED_HOURS"], errors="raise")
    state_sum = station[
        [
            "PRIMARY_INVERSION_LIKE_HOURS",
            "PRIMARY_NORMAL_NEGATIVE_HOURS",
            "PRIMARY_UNCERTAIN_HOURS",
            "PRIMARY_INELIGIBLE_HOURS",
        ]
    ].apply(pd.to_numeric, errors="raise").sum(axis=1)
    if (reproduced > totals).any() or not np.array_equal(
        state_sum.to_numpy(float), totals.to_numpy(float)
    ):
        raise ValueError("station primary state counts do not sum to paired total")
    all_rows = state.loc[state["CONDITION_VARIABLE"].eq("ALL")]
    setting_totals = (
        all_rows.groupby(["STN", "CONFIG"], sort=False)["N_HOURS"].sum().reset_index()
    )
    expected_by_station = station.set_index("STN")["REPRODUCED_HOURS"]
    for row in setting_totals.itertuples(index=False):
        if int(row.N_HOURS) != int(expected_by_station.loc[row.STN]):
            raise ValueError("sensitivity-setting state counts do not sum to reproduced hours")

    equivalent = tables["equivalent_diagnostics.csv"]
    high_after_ids = set(equivalent.loc[equivalent["ABS_BIAS_FIXED"] > 0.5, "STN"])
    if truth is None:
        worsening_ids = set(tables["worsening_diagnostics.csv"]["STN"])
    else:
        worsening_ids = set(
            truth.loc[truth["SIGNIFICANCE_CLASS"].eq("meaningful_worsening"), "STN"]
        )
    for frame in [state, tables["coast_time_signed_bias.csv"], matched, coverage_sensitivity]:
        if not np.array_equal(
            frame["SUBSET_MEANINGFUL_WORSENING"].astype(bool).to_numpy(),
            frame["STN"].isin(worsening_ids).to_numpy(),
        ):
            raise ValueError("worsening subset flag is inconsistent with canonical labels")
        if not np.array_equal(
            frame["SUBSET_HIGH_AFTER"].astype(bool).to_numpy(),
            frame["STN"].isin(high_after_ids).to_numpy(),
        ):
            raise ValueError("high-after subset flag is inconsistent with canonical labels")


def _create_plots(staging: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    """별도 plot 모듈의 5개 함수를 같은 staging 안에서 실행한다."""

    try:
        from visualization.bias_physical_followup_plots import (
            equivalent_before_after,
            high_residual_factors,
            station_885_case,
            structure_condition_bias,
            worsening_by_dz,
        )
    except ImportError as exc:  # pragma: no cover - 병렬 모듈 통합 전 명시적 실패
        raise RuntimeError("bias_physical_followup_plots helper is unavailable") from exc
    calls = [
        (
            equivalent_before_after,
            "equivalent_diagnostics.csv",
            "equivalent_before_after.png",
        ),
        (worsening_by_dz, "worsening_diagnostics.csv", "worsening_by_dz.png"),
        (
            high_residual_factors,
            "high_residual_group_diagnostic.csv",
            "high_residual_factors.png",
        ),
        (
            structure_condition_bias,
            "state_condition_bias.csv",
            "structure_condition_bias.png",
        ),
        (station_885_case, "station_885_diagnostic.csv", "station_885_case.png"),
    ]
    for function, table_name, output_name in calls:
        returned = Path(function(tables[table_name], staging / output_name)).resolve()
        if returned != (staging / output_name).resolve():
            raise ValueError(f"plot helper returned unexpected path: {returned}")


def _prepare_prewrite_primary_anchor(
    tables: Mapping[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> tuple[dict[str, bytes], str]:
    """CSV write 전에 primary 원표와 파생표를 메모리 bytes로 함께 고정한다."""

    payloads = _canonical_csv_payloads(tables)
    primary_hashes = _primary_inference_table_hashes(payloads)
    metadata["primary_eligibility_ledger_rows"] = int(
        len(tables["primary_eligibility_ledger.csv"])
    )
    metadata["primary_eligibility_ledger_sha256"] = primary_hashes[
        "primary_eligibility_ledger.csv"
    ]
    metadata["primary_inference_table_sha256s"] = primary_hashes
    anchor = _primary_ledger_source_binding_sha256(
        ledger_sha256=metadata["primary_eligibility_ledger_sha256"],
        population_sha256=metadata["analysis_station_ids_sha256"],
        config_sha256=_canonical_json_hash(_analysis_config_payload()),
        sources=metadata["input_sources"],
        primary_table_sha256s=primary_hashes,
    )
    metadata["primary_eligibility_source_binding_sha256"] = anchor
    return payloads, anchor


def _write_and_validate_staging(
    staging: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    metadata: dict[str, Any],
    summary_markdown: str,
    failure_injection: str | None,
    rss_observer: Any | None = None,
    canonical_csv_payloads: MutableMapping[str, bytes] | None = None,
    trusted_prewrite_anchor_sha256: str | None = None,
) -> list[dict[str, Any]]:
    _validate_analysis_metadata(dict(metadata), validate_live_sources=False)
    if canonical_csv_payloads is None:
        raise ValueError("staging requires pre-write canonical CSV payloads")
    if set(canonical_csv_payloads) != set(CSV_SCHEMAS):
        raise ValueError("staging canonical CSV payload set mismatch")
    if (
        not _is_sha256_hex(trusted_prewrite_anchor_sha256)
        or metadata.get("primary_eligibility_source_binding_sha256")
        != trusted_prewrite_anchor_sha256
    ):
        raise ValueError("staging requires the trusted pre-write primary anchor")
    for filename, expected_columns in CSV_SCHEMAS.items():
        frame = tables[filename]
        _validate_table(frame, expected_columns, filename)
        payload = canonical_csv_payloads[filename]
        if not isinstance(payload, bytes):
            raise ValueError("canonical CSV payload must be bytes")
        path = staging / filename
        path.write_bytes(payload)
        if path.read_bytes() != payload:
            raise ValueError(f"staged CSV differs from pre-write memory bytes: {filename}")
    primary_hashes = _primary_inference_table_hashes(canonical_csv_payloads)
    if primary_hashes != metadata["primary_inference_table_sha256s"]:
        raise ValueError("staged primary table hashes differ from pre-write anchor")
    # 모든 CSV가 pre-write bytes와 exact 일치하고 anchored hash도 재검증된 뒤에는
    # bytes 묶음을 그림 생성까지 보유할 이유가 없다. caller의 dict도 함께 비운다.
    canonical_csv_payloads.clear()
    gc.collect()
    (staging / "analysis_summary.md").write_text(summary_markdown, encoding="utf-8")
    _create_plots(staging, tables)
    if failure_injection == "figure_save":
        raise RuntimeError("injected figure-save failure")

    import matplotlib.image as mpimg

    for filename in PNG_NAMES:
        path = staging / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty figure: {filename}")
        image = mpimg.imread(path)
        if image.ndim < 2 or min(image.shape[:2]) <= 10:
            raise ValueError(f"undecodable or undersized figure: {filename}")

    # PNG까지 만든 가장 큰 publish 구간 뒤 실제 process RSS를 다시 읽는다.
    # observer는 metadata의 process peak를 갱신하며, 상한 초과 시 manifest와
    # atomic promote 이전에 실패해야 한다.
    if rss_observer is not None:
        rss_observer()
    _validate_analysis_metadata(dict(metadata), validate_live_sources=False)
    (staging / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifacts: list[dict[str, Any]] = []
    listed = [
        *CSV_SCHEMAS,
        *PNG_NAMES,
        "analysis_metadata.json",
        "analysis_summary.md",
    ]
    if len(listed) != len(set(listed)):
        raise RuntimeError("duplicate artifact name in manifest contract")
    for filename in listed:
        path = staging / filename
        entry: dict[str, Any] = {
            "path": filename,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        if filename in CSV_SCHEMAS:
            entry["rows"] = int(len(tables[filename]))
        artifacts.append(entry)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifacts": artifacts,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    exact = set(listed) | {"manifest.json"}
    actual = {path.name for path in staging.iterdir()}
    if actual != exact:
        raise ValueError(f"staging artifact set mismatch: expected={exact}, actual={actual}")
    if failure_injection == "validator":
        raise RuntimeError("injected staging-validator failure")
    return artifacts


def validate_published_artifacts(
    out_dir: os.PathLike[str] | str,
    *,
    allowed_fullnet_root: os.PathLike[str] | str | None = None,
    trusted_primary_eligibility_binding_sha256: str | None = None,
    sources_prevalidated: bool = False,
    receipt_only_restart: bool = False,
) -> dict[str, Any]:
    """게시된 artifact의 exact set·중복·schema·행수·hash·PNG를 재검증한다."""

    if sources_prevalidated and allowed_fullnet_root is not None:
        raise ValueError("injected artifact cannot bypass live source validation")
    if receipt_only_restart and (
        allowed_fullnet_root is not None
        or not sources_prevalidated
        or not _is_sha256_hex(trusted_primary_eligibility_binding_sha256)
    ):
        raise ValueError(
            "receipt-only restart requires a production path, prevalidated receipt, and trusted anchor"
        )

    directory = (
        guard_production_output_directory(out_dir)
        if allowed_fullnet_root is None
        else guard_injected_output_directory(
            out_dir, allowed_fullnet_root=allowed_fullnet_root
        )
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("manifest.json is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "artifacts"}
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version mismatch")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("manifest artifacts must be a list")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(paths) != len(entries) or len(paths) != len(set(paths)):
        raise ValueError("manifest contains invalid or duplicate artifact paths")
    expected = set(CSV_SCHEMAS) | set(PNG_NAMES) | {
        "analysis_metadata.json",
        "analysis_summary.md",
    }
    if set(paths) != expected:
        raise ValueError("manifest artifact set does not match the fixed contract")
    actual = {path.name for path in directory.iterdir()}
    if actual != expected | {"manifest.json"}:
        raise ValueError("published directory contains missing or extra files")

    import matplotlib.image as mpimg

    loaded_tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] | None = None
    for entry in entries:
        expected_entry_keys = {"path", "bytes", "sha256"} | (
            {"rows"} if entry.get("path") in CSV_SCHEMAS else set()
        )
        if set(entry) != expected_entry_keys:
            raise ValueError("manifest artifact entry schema mismatch")
        path = directory / str(entry["path"])
        if not path.is_file() or int(entry.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"artifact size mismatch: {path.name}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"artifact hash mismatch: {path.name}")
        if path.name in CSV_SCHEMAS:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            _validate_table(frame, CSV_SCHEMAS[path.name], path.name)
            loaded_tables[path.name] = frame
            if int(entry.get("rows", -1)) != len(frame):
                raise ValueError(f"artifact row-count mismatch: {path.name}")
        elif path.suffix.lower() == ".png":
            image = mpimg.imread(path)
            if image.ndim < 2 or min(image.shape[:2]) <= 10:
                raise ValueError(f"invalid PNG: {path.name}")
        elif path.name == "analysis_metadata.json":
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("analysis_metadata.json is invalid") from exc
    _validate_analysis_metadata(
        metadata,
        expected_source_mode=(
            "production" if allowed_fullnet_root is None else "injected"
        ),
        validate_live_sources=not sources_prevalidated,
    )
    if allowed_fullnet_root is None:
        _validate_trusted_primary_eligibility_binding(
            metadata,
            trusted_primary_eligibility_binding_sha256,
        )
    elif trusted_primary_eligibility_binding_sha256 is not None:
        raise ValueError("injected artifact validation cannot use a production anchor")
    ledger_path = directory / "primary_eligibility_ledger.csv"
    if (
        int(metadata["primary_eligibility_ledger_rows"])
        != len(loaded_tables["primary_eligibility_ledger.csv"])
        or metadata["primary_eligibility_ledger_sha256"] != _sha256(ledger_path)
    ):
        raise ValueError("primary eligibility ledger identity differs from metadata")
    observed_primary_hashes = {
        name: _sha256(directory / name) for name in PRIMARY_ANCHORED_CSV_NAMES
    }
    if observed_primary_hashes != metadata["primary_inference_table_sha256s"]:
        raise ValueError("primary inference tables differ from the trusted pre-write anchor")
    source_truth = _validate_analysis_population_identity(
        metadata,
        loaded_tables["station_structure_coverage.csv"],
        require_live_source=not receipt_only_restart,
    )
    _validate_cross_table_contracts(loaded_tables, source_truth)
    return manifest


def _validate_published_artifacts_stable_manifest(
    out_dir: os.PathLike[str] | str,
    **validation_kwargs: Any,
) -> str:
    """semantic validator 호출 전후 manifest bytes가 동일할 때만 그 hash를 반환한다."""

    directory = Path(out_dir).expanduser().resolve(strict=True)
    manifest_path = directory / "manifest.json"
    before = _sha256(manifest_path)
    validate_published_artifacts(directory, **validation_kwargs)
    after = _sha256(manifest_path)
    if after != before:
        raise ValueError("manifest changed during post-validation")
    return before


def _replace_with_bounded_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 0.05,
) -> None:
    """OneDrive의 짧은 공유잠금만 제한 재시도하고 영구 권한오류는 드러낸다."""

    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(initial_delay_seconds * (2**attempt))


def _guard_acceptance_receipt_path(
    receipt_path: os.PathLike[str] | str | None,
    *,
    final: Path,
) -> Path:
    """production acceptance receipt를 final tree 밖의 명시 경로로 제한한다."""

    if receipt_path is None:
        raise ValueError("production publication requires an acceptance receipt path")
    candidate = Path(receipt_path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("acceptance receipt path must be absolute")
    parent = candidate.parent.resolve(strict=True)
    resolved = parent / candidate.name
    final_resolved = final.resolve(strict=False)
    try:
        resolved.relative_to(final_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("acceptance receipt must remain outside the final artifact tree")
    if resolved == final_resolved or resolved.is_dir() or candidate.name in {"", ".", ".."}:
        raise ValueError("acceptance receipt path is invalid")
    return resolved


def _build_acceptance_receipt_payload(
    *,
    final: Path,
    trusted_primary_anchor_sha256: str,
    db_snapshot_sha256: str,
) -> dict[str, Any]:
    if not _is_sha256_hex(trusted_primary_anchor_sha256) or not _is_sha256_hex(
        db_snapshot_sha256
    ):
        raise ValueError("acceptance receipt hashes are invalid")
    manifest_path = final / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("accepted output manifest is missing")
    return {
        "receipt_schema_version": 1,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": str(final.resolve(strict=True)),
        "primary_eligibility_source_binding_sha256": trusted_primary_anchor_sha256,
        "output_manifest_sha256": _sha256(manifest_path),
        "source_snapshot_sha256": db_snapshot_sha256,
    }


def _write_acceptance_receipt_atomic(
    receipt_path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path | None, bool]:
    """receipt를 같은 디렉터리의 temp/backup rename으로 원자 교체한다."""

    parent = receipt_path.parent.resolve(strict=True)
    temp = parent / f".{receipt_path.name}.tmp-{uuid.uuid4().hex}"
    backup = parent / f".{receipt_path.name}.backup-{uuid.uuid4().hex}"
    payload_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    moved_old = False
    try:
        with temp.open("xb") as handle:
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if temp.read_bytes() != payload_bytes:
            raise ValueError("acceptance receipt write/read-back mismatch")
        if receipt_path.exists():
            if not receipt_path.is_file():
                raise ValueError("acceptance receipt target must be a file")
            _replace_with_bounded_retry(receipt_path, backup)
            moved_old = True
        _replace_with_bounded_retry(temp, receipt_path)
        return (backup if moved_old else None), True
    except Exception:
        if temp.exists():
            temp.unlink()
        if moved_old and backup.exists() and not receipt_path.exists():
            _replace_with_bounded_retry(backup, receipt_path)
        raise


def _rollback_acceptance_receipt(
    receipt_path: Path,
    backup: Path | None,
    written: bool,
) -> None:
    if written and receipt_path.exists():
        receipt_path.unlink()
    if backup is not None and backup.exists():
        _replace_with_bounded_retry(backup, receipt_path)


def _validate_acceptance_commit_seal(
    final: Path,
    receipt_path: Path,
    expected_payload: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> None:
    """backup 삭제 직전 receipt와 manifest가 post-validation 상태인지 재확인한다."""

    if not _is_sha256_hex(expected_manifest_sha256):
        raise ValueError("post-validation manifest SHA-256 is invalid")
    manifest_path = final / "manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("manifest changed after post-validation")
    expected_receipt_bytes = (
        json.dumps(expected_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if receipt_path.read_bytes() != expected_receipt_bytes:
        raise ValueError("acceptance receipt changed before commit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("post-validation manifest is invalid") from exc
    entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError("post-validation manifest artifact list is invalid")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("post-validation manifest entry is invalid")
        name = entry.get("path")
        if type(name) is not str or Path(name).name != name:
            raise ValueError("post-validation manifest path is invalid")
        path = final / name
        if (
            not path.is_file()
            or isinstance(entry.get("bytes"), bool)
            or not isinstance(entry.get("bytes"), int)
            or path.stat().st_size != entry["bytes"]
            or not _is_sha256_hex(entry.get("sha256"))
            or _sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"artifact changed after post-validation: {name}")


def validate_acceptance_receipt(
    out_dir: os.PathLike[str] | str,
    receipt_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """재시작 뒤 외부 receipt를 신뢰점으로 production bundle을 재검증한다."""

    final = guard_production_output_directory(out_dir)
    receipt = _guard_acceptance_receipt_path(receipt_path, final=final)
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("acceptance receipt is missing or invalid") from exc
    if type(payload) is not dict or set(payload) != {
        "receipt_schema_version",
        "artifact_schema_version",
        "output_dir",
        "primary_eligibility_source_binding_sha256",
        "output_manifest_sha256",
        "source_snapshot_sha256",
    }:
        raise ValueError("acceptance receipt schema mismatch")
    if (
        payload["receipt_schema_version"] != 1
        or payload["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION
        or payload["output_dir"] != str(final.resolve(strict=True))
        or not _is_sha256_hex(
            payload["primary_eligibility_source_binding_sha256"]
        )
        or not _is_sha256_hex(payload["output_manifest_sha256"])
        or not _is_sha256_hex(payload["source_snapshot_sha256"])
        or payload["output_manifest_sha256"] != _sha256(final / "manifest.json")
    ):
        raise ValueError("acceptance receipt identity mismatch")
    metadata = json.loads((final / "analysis_metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("input_sources", {}).get("db_snapshot_sha256")
        != payload["source_snapshot_sha256"]
    ):
        raise ValueError("acceptance receipt DB snapshot identity mismatch")
    validate_published_artifacts(
        final,
        trusted_primary_eligibility_binding_sha256=payload[
            "primary_eligibility_source_binding_sha256"
        ],
        sources_prevalidated=True,
        receipt_only_restart=True,
    )
    return payload


def _atomic_publish(
    staging: Path,
    final: Path,
    *,
    failure_injection: str | None,
    pre_promote_validator: Any | None = None,
    post_promote_validator: Any | None = None,
    acceptance_receipt_path: Path | None = None,
    acceptance_receipt_payload_factory: Any | None = None,
) -> None:
    backup = final.parent / f".{final.name}.backup-{uuid.uuid4().hex}"
    moved_old = False
    promoted_new = False
    receipt_backup: Path | None = None
    receipt_written = False
    post_validation_manifest_sha256: str | None = None
    receipt_payload: Mapping[str, Any] | None = None
    try:
        if pre_promote_validator is not None:
            if not callable(pre_promote_validator):
                raise TypeError("pre_promote_validator must be callable")
            # old final을 backup으로 옮기 *직전* 원본 snapshot을 다시 고정한다.
            # 이 검증이 실패하면 기존 canonical은 건드리지 않는다.
            pre_promote_validator()
        if final.exists():
            _replace_with_bounded_retry(final, backup)
            moved_old = True
        if failure_injection == "after_backup_rename":
            raise RuntimeError("injected failure after backup rename")
        if failure_injection == "promote_staging_rename":
            raise RuntimeError("injected staging promote failure")
        _replace_with_bounded_retry(staging, final)
        promoted_new = True
        if pre_promote_validator is not None:
            # staging이 final이 된 뒤에도 기존 backup을 지우기 전 원본
            # snapshot을 다시 검증한다. 이 검증이 실패하면 except 분기가
            # 새 final을 제거하고 기존 canonical을 backup에서 복구한다.
            pre_promote_validator()
        if post_promote_validator is not None:
            if not callable(post_promote_validator):
                raise TypeError("post_promote_validator must be callable")
            post_validation_result = post_promote_validator()
            if post_validation_result is not None:
                if not _is_sha256_hex(post_validation_result):
                    raise ValueError(
                        "post-promote validator returned an invalid manifest SHA-256"
                    )
                post_validation_manifest_sha256 = post_validation_result
        if acceptance_receipt_path is not None:
            if not callable(acceptance_receipt_payload_factory):
                raise TypeError("acceptance receipt payload factory must be callable")
            receipt_payload = acceptance_receipt_payload_factory()
            if (
                post_validation_manifest_sha256 is not None
                and (
                    not isinstance(receipt_payload, Mapping)
                    or receipt_payload.get("output_manifest_sha256")
                    != post_validation_manifest_sha256
                )
            ):
                raise ValueError(
                    "receipt manifest differs from the post-validation manifest"
                )
            receipt_backup, receipt_written = _write_acceptance_receipt_atomic(
                acceptance_receipt_path,
                receipt_payload,
            )
            if post_validation_manifest_sha256 is not None:
                _validate_acceptance_commit_seal(
                    final,
                    acceptance_receipt_path,
                    receipt_payload,
                    expected_manifest_sha256=post_validation_manifest_sha256,
                )
        elif acceptance_receipt_payload_factory is not None:
            raise ValueError("acceptance receipt path is required for its payload")
        if failure_injection == "after_receipt_before_cleanup":
            raise RuntimeError("injected failure after acceptance receipt")
        if failure_injection == "after_promote_before_cleanup":
            raise RuntimeError("injected failure after promote before cleanup")
        # receipt 설치까지 끝난 시점이 commit point다. 이후 orphan backup
        # 정리는 best-effort이며, 단순 cleanup 실패 때문에 이미 검증된 새
        # final/receipt를 제거하는 rollback 분기로 되돌아가면 안 된다.
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                pass
        if receipt_backup is not None and receipt_backup.exists():
            try:
                receipt_backup.unlink()
            except OSError:
                pass
    except Exception:
        if acceptance_receipt_path is not None:
            _rollback_acceptance_receipt(
                acceptance_receipt_path,
                receipt_backup,
                receipt_written,
            )
        if promoted_new and final.exists():
            shutil.rmtree(final)
        if moved_old and backup.exists():
            _replace_with_bounded_retry(backup, final)
        raise


def run_injected_analysis(
    station_results: pd.DataFrame,
    station_payloads: Mapping[str, Mapping[str, Any]],
    out_dir: os.PathLike[str] | str,
    *,
    allowed_fullnet_root: os.PathLike[str] | str,
    enforce_canonical: bool = False,
    failure_injection: str | None = None,
) -> dict[str, Any]:
    """작은 주입형 자료로 전체 집계·staging·검증·원자 게시 계약을 실행한다."""

    final = guard_injected_output_directory(
        out_dir, allowed_fullnet_root=allowed_fullnet_root
    )
    truth = _prepare_station_results(station_results)
    if enforce_canonical:
        canonical = validate_canonical_station_results(truth)
    else:
        canonical = None
    normalized_payloads = {
        _normalize_station_id(key, field="payload STN"): value
        for key, value in station_payloads.items()
    }
    extra_payloads = set(normalized_payloads) - set(truth["STN"])
    if extra_payloads:
        raise ValueError(f"payload contains stations absent from canonical truth: {sorted(extra_payloads)}")

    station_tables: list[pd.DataFrame] = []
    state_tables: list[pd.DataFrame] = []
    matched_tables: list[pd.DataFrame] = []
    ledger_tables: list[pd.DataFrame] = []
    coast_tables: list[pd.DataFrame] = []
    spatial_sensitivity_tables: list[pd.DataFrame] = []
    coverage_sensitivity_tables: list[pd.DataFrame] = []
    station_885_hourly: pd.DataFrame | None = None
    metrics = Counter(
        {
            "db_query_count": 0,
            "aws_file_read_count": 0,
            "geometry_build_count": 0,
            "unique_geometry_count": 0,
            "processed_paired_hours": 0,
            "reproduced_hour_count": 0,
            "processed_neighbor_values": 0,
            "ols_fit_count": 0,
            "aux_fit_count_none": 0,
            "aux_fit_count_d1": 0,
            "aux_fit_count_d2": 0,
            "loo_fit_count_none": 0,
            "loo_fit_count_d1": 0,
            "loo_fit_count_d2": 0,
            "setting_classification_count": 0,
            "coverage_gate_evaluation_count": 0,
            "peak_chunk_memory_bytes": 0,
        }
    )
    _, source_population_sha256 = _station_population_identity(truth["STN"])
    analysis_config_sha256 = _canonical_json_hash(_analysis_config_payload())

    # 전 station skeleton을 먼저 순회한다. 부분집단 label은 아래에서 후행 결합한다.
    for station_id in truth["STN"]:
        payload = normalized_payloads.get(station_id)
        if payload is None:
            station_tables.append(_empty_station_summary(station_id, "source_payload_missing"))
            state_tables.append(_empty_state_skeleton(station_id, "source_payload_missing"))
            matched_tables.append(_empty_matched_contrast(station_id, "source_payload_missing"))
            ledger_tables.append(
                _empty_primary_eligibility_ledger(
                    station_id,
                    "source_payload_missing",
                    source_population_sha256=source_population_sha256,
                    analysis_config_sha256=analysis_config_sha256,
                )
            )
            spatial_sensitivity_tables.append(
                _empty_spatial_sensitivity(station_id, "source_payload_missing")
            )
            coverage_sensitivity_tables.append(
                _empty_matched_coverage_sensitivity(station_id, "source_payload_missing")
            )
            continue
        result = analyze_station_injected(
            **payload,
            strict_reproduction=True,
            return_hourly=station_id == "885",
            source_population_sha256=source_population_sha256,
            analysis_config_sha256=analysis_config_sha256,
        )
        station_tables.append(result["station_summary"])
        state_tables.append(result["state_condition_bias"])
        matched_tables.append(result["matched_state_contrast"])
        ledger_tables.append(result["primary_eligibility_ledger"])
        spatial_sensitivity_tables.append(
            result["spatial_structure_sensitivity_summary"]
        )
        coverage_sensitivity_tables.append(
            result["matched_state_coverage_sensitivity"]
        )
        coast_tables.append(result["coast_time_signed_bias"])
        if station_id == "885":
            station_885_hourly = result.get("hourly")
        for key in [
            "geometry_build_count",
            "unique_geometry_count",
            "processed_paired_hours",
            "reproduced_hour_count",
            "processed_neighbor_values",
            "ols_fit_count",
            "aux_fit_count_none",
            "aux_fit_count_d1",
            "aux_fit_count_d2",
            "loo_fit_count_none",
            "loo_fit_count_d1",
            "loo_fit_count_d2",
            "setting_classification_count",
            "coverage_gate_evaluation_count",
        ]:
            metrics[key] += int(result["metrics"][key])
        metrics["peak_chunk_memory_bytes"] = max(
            metrics["peak_chunk_memory_bytes"],
            int(result["metrics"]["peak_chunk_memory_bytes"]),
        )

    station_structure = pd.concat(station_tables, ignore_index=True).loc[
        :, STATION_STRUCTURE_COLUMNS
    ]
    state_condition = pd.concat(state_tables, ignore_index=True).loc[
        :, STATE_CONDITION_COLUMNS
    ]
    coast_time = (
        pd.concat(coast_tables, ignore_index=True).loc[:, COAST_TIME_COLUMNS]
        if coast_tables
        else pd.DataFrame(columns=COAST_TIME_COLUMNS)
    )
    matched_contrast = pd.concat(matched_tables, ignore_index=True)
    primary_ledger = pd.concat(ledger_tables, ignore_index=True).loc[
        :, PRIMARY_ELIGIBILITY_LEDGER_COLUMNS
    ]
    spatial_sensitivity = pd.concat(spatial_sensitivity_tables, ignore_index=True).loc[
        :, SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS
    ]
    coverage_sensitivity = pd.concat(
        coverage_sensitivity_tables, ignore_index=True
    ).loc[:, MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS]

    equivalent_result = build_equivalent_diagnostics(truth)
    equivalent = equivalent_result["stations"].loc[:, EQUIVALENT_COLUMNS]
    worsening_result = build_worsening_diagnostics(truth)
    worsening = worsening_result["stations"].loc[:, WORSENING_COLUMNS]
    high_residual = _high_residual_table(equivalent, station_structure)
    station_885 = _station_885_table(station_885_hourly)

    high_after_ids = set(equivalent.loc[equivalent["ABS_BIAS_FIXED"] > 0.5, "STN"])
    worsening_ids = set(
        truth.loc[truth["SIGNIFICANCE_CLASS"].eq("meaningful_worsening"), "STN"]
    )
    for frame in [state_condition, coast_time, matched_contrast, coverage_sensitivity]:
        frame["SUBSET_MEANINGFUL_WORSENING"] = frame["STN"].isin(worsening_ids)
        frame["SUBSET_HIGH_AFTER"] = frame["STN"].isin(high_after_ids)
    _apply_primary_ledger_bh_and_sync(primary_ledger, matched_contrast)
    matched_contrast = matched_contrast.loc[:, MATCHED_STATE_CONTRAST_COLUMNS]
    _apply_station_885_evidence(
        station_885,
        matched_contrast,
        station_structure=station_structure,
        state_condition=state_condition,
        coverage_sensitivity=coverage_sensitivity,
        analysis_station_ids=truth["STN"].tolist(),
    )
    physical = build_physical_hypothesis_comparison(
        equivalent,
        station_structure,
        coast_time,
        state_condition=state_condition,
    )

    tables = {
        "station_structure_coverage.csv": station_structure,
        "state_condition_bias.csv": state_condition,
        "coast_time_signed_bias.csv": coast_time,
        "equivalent_diagnostics.csv": equivalent,
        "worsening_diagnostics.csv": worsening,
        "high_residual_group_diagnostic.csv": high_residual,
        "station_885_diagnostic.csv": station_885,
        "matched_state_contrast.csv": matched_contrast,
        "primary_eligibility_ledger.csv": primary_ledger,
        "spatial_structure_sensitivity_summary.csv": spatial_sensitivity,
        "matched_state_coverage_sensitivity.csv": coverage_sensitivity,
        "physical_hypothesis_comparison.csv": physical,
    }
    metadata = _build_analysis_metadata(
        canonical=canonical,
        metrics={key: int(value) for key, value in metrics.items()},
        analysis_station_ids=truth["STN"],
        input_sources={"source_mode": "injected", "aws_file_inventory": []},
    )
    summary = (
        "# 고도보정 BIAS 물리요인 후속분석\n\n"
        f"- 전체 분석 관측소: {len(truth)}개\n"
        f"- exact none 재현 시각: {metrics['processed_paired_hours']}개\n"
        "- `inversion_like`는 지상 관측망의 공간적 양의 기온-고도 기울기이며 "
        "실제 연직 역전의 직접 관측이 아니다.\n"
        "- 상태별 효과는 같은 시각 mask에서 signed BIAS를 먼저 평균한 뒤 절댓값을 비교했다.\n"
    )

    _validate_cross_table_contracts(tables, truth)
    canonical_csv_payloads, trusted_prewrite_anchor = (
        _prepare_prewrite_primary_anchor(tables, metadata)
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    artifacts: list[dict[str, Any]] = []
    try:
        artifacts = _write_and_validate_staging(
            staging,
            tables=tables,
            metadata=metadata,
            summary_markdown=summary,
            failure_injection=failure_injection,
            canonical_csv_payloads=canonical_csv_payloads,
            trusted_prewrite_anchor_sha256=trusted_prewrite_anchor,
        )
        _atomic_publish(staging, final, failure_injection=failure_injection)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    validate_published_artifacts(final, allowed_fullnet_root=allowed_fullnet_root)
    return {
        "output_dir": str(final),
        "artifacts": artifacts,
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
    }


def _concat_and_release_frame_parts(
    parts: list[pd.DataFrame],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """station별 frame을 합친 즉시 원본 block 참조를 해제한다."""

    if not parts:
        return pd.DataFrame(columns=list(columns or ()))
    combined = pd.concat(parts, ignore_index=True)
    parts.clear()
    gc.collect()
    if columns is not None and list(combined.columns) != list(columns):
        combined = combined.loc[:, list(columns)]
    return combined


def _publish_precomputed_analysis(
    *,
    truth: pd.DataFrame,
    station_outputs: MutableMapping[str, Mapping[str, Any]],
    final: Path,
    canonical: Mapping[str, Any] | None,
    source_metrics: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    private_db_snapshot_identity: Mapping[str, Any] | None = None,
    acceptance_receipt_path: os.PathLike[str] | str | None = None,
    failure_injection: str | None = None,
    rss_reader: Any | None = None,
) -> dict[str, Any]:
    """시간자료를 버린 관측소별 요약만 모아 artifact를 게시한다."""

    if not isinstance(station_outputs, MutableMapping):
        raise TypeError("station_outputs must be mutable for bounded-memory publication")

    station_tables: list[pd.DataFrame] = []
    state_tables: list[pd.DataFrame] = []
    matched_tables: list[pd.DataFrame] = []
    ledger_tables: list[pd.DataFrame] = []
    coast_tables: list[pd.DataFrame] = []
    spatial_sensitivity_tables: list[pd.DataFrame] = []
    coverage_sensitivity_tables: list[pd.DataFrame] = []
    station_885_hourly: pd.DataFrame | None = None
    metrics = Counter(
        {
            "db_query_count": 0,
            "aws_file_read_count": 0,
            "geometry_build_count": 0,
            "unique_geometry_count": 0,
            "processed_paired_hours": 0,
            "reproduced_hour_count": 0,
            "processed_neighbor_values": 0,
            "ols_fit_count": 0,
            "aux_fit_count_none": 0,
            "aux_fit_count_d1": 0,
            "aux_fit_count_d2": 0,
            "loo_fit_count_none": 0,
            "loo_fit_count_d1": 0,
            "loo_fit_count_d2": 0,
            "setting_classification_count": 0,
            "coverage_gate_evaluation_count": 0,
            "peak_chunk_memory_bytes": 0,
        }
    )
    if source_metrics:
        for key, value in source_metrics.items():
            if key in metrics:
                metrics[key] = int(value)
    _, source_population_sha256 = _station_population_identity(truth["STN"])
    analysis_config_sha256 = _canonical_json_hash(_analysis_config_payload())

    for station_id in truth["STN"]:
        result = station_outputs.pop(station_id, None)
        if result is None:
            station_tables.append(_empty_station_summary(station_id, "source_payload_missing"))
            state_tables.append(_empty_state_skeleton(station_id, "source_payload_missing"))
            matched_tables.append(_empty_matched_contrast(station_id, "source_payload_missing"))
            ledger_tables.append(
                _empty_primary_eligibility_ledger(
                    station_id,
                    "source_payload_missing",
                    source_population_sha256=source_population_sha256,
                    analysis_config_sha256=analysis_config_sha256,
                )
            )
            spatial_sensitivity_tables.append(
                _empty_spatial_sensitivity(station_id, "source_payload_missing")
            )
            coverage_sensitivity_tables.append(
                _empty_matched_coverage_sensitivity(station_id, "source_payload_missing")
            )
            continue
        station_tables.append(result["station_summary"])
        state_tables.append(result["state_condition_bias"])
        matched_tables.append(result["matched_state_contrast"])
        station_ledger = result["primary_eligibility_ledger"].copy()
        station_ledger["SOURCE_POPULATION_SHA256"] = source_population_sha256
        station_ledger["ANALYSIS_CONFIG_SHA256"] = analysis_config_sha256
        ledger_tables.append(station_ledger)
        spatial_sensitivity_tables.append(
            result["spatial_structure_sensitivity_summary"]
        )
        coverage_sensitivity_tables.append(
            result["matched_state_coverage_sensitivity"]
        )
        coast_tables.append(result["coast_time_signed_bias"])
        if station_id == "885":
            station_885_hourly = result.get("hourly")
        for key in [
            "geometry_build_count",
            "unique_geometry_count",
            "processed_paired_hours",
            "reproduced_hour_count",
            "processed_neighbor_values",
            "ols_fit_count",
            "aux_fit_count_none",
            "aux_fit_count_d1",
            "aux_fit_count_d2",
            "loo_fit_count_none",
            "loo_fit_count_d1",
            "loo_fit_count_d2",
            "setting_classification_count",
            "coverage_gate_evaluation_count",
        ]:
            metrics[key] += int(result["metrics"][key])
        metrics["peak_chunk_memory_bytes"] = max(
            metrics["peak_chunk_memory_bytes"],
            int(result["metrics"]["peak_chunk_memory_bytes"]),
        )
    # truth 밖의 미사용 payload도 기존과 같이 분석에서 제외하되 참조는 남기지 않는다.
    station_outputs.clear()
    result = None
    gc.collect()

    # 가장 큰 표부터 합치고 각 component list를 즉시 비운다. 따라서 station별 원본과
    # 전체 concat 결과가 모든 표에 대해 동시에 이중 보관되지 않는다.
    coverage_sensitivity = _concat_and_release_frame_parts(
        coverage_sensitivity_tables,
        columns=MATCHED_STATE_COVERAGE_SENSITIVITY_COLUMNS,
    )
    spatial_sensitivity = _concat_and_release_frame_parts(
        spatial_sensitivity_tables,
        columns=SPATIAL_STRUCTURE_SENSITIVITY_COLUMNS,
    )
    state_condition = _concat_and_release_frame_parts(
        state_tables,
        columns=STATE_CONDITION_COLUMNS,
    )
    coast_time = (
        _concat_and_release_frame_parts(coast_tables, columns=COAST_TIME_COLUMNS)
        if coast_tables
        else pd.DataFrame(columns=COAST_TIME_COLUMNS)
    )
    station_structure = _concat_and_release_frame_parts(
        station_tables,
        columns=STATION_STRUCTURE_COLUMNS,
    )
    matched_contrast = _concat_and_release_frame_parts(matched_tables)
    primary_ledger = _concat_and_release_frame_parts(
        ledger_tables,
        columns=PRIMARY_ELIGIBILITY_LEDGER_COLUMNS,
    )
    equivalent = build_equivalent_diagnostics(truth)["stations"].loc[
        :, EQUIVALENT_COLUMNS
    ]
    worsening = build_worsening_diagnostics(truth)["stations"].loc[
        :, WORSENING_COLUMNS
    ]
    high_residual = _high_residual_table(equivalent, station_structure)
    station_885 = _station_885_table(station_885_hourly)

    high_after_ids = set(equivalent.loc[equivalent["ABS_BIAS_FIXED"] > 0.5, "STN"])
    worsening_ids = set(
        truth.loc[truth["SIGNIFICANCE_CLASS"].eq("meaningful_worsening"), "STN"]
    )
    for frame in [state_condition, coast_time, matched_contrast, coverage_sensitivity]:
        frame["SUBSET_MEANINGFUL_WORSENING"] = frame["STN"].isin(worsening_ids)
        frame["SUBSET_HIGH_AFTER"] = frame["STN"].isin(high_after_ids)
    _apply_primary_ledger_bh_and_sync(primary_ledger, matched_contrast)
    matched_contrast = matched_contrast.loc[:, MATCHED_STATE_CONTRAST_COLUMNS]
    _apply_station_885_evidence(
        station_885,
        matched_contrast,
        station_structure=station_structure,
        state_condition=state_condition,
        coverage_sensitivity=coverage_sensitivity,
        analysis_station_ids=truth["STN"].tolist(),
    )
    physical = build_physical_hypothesis_comparison(
        equivalent,
        station_structure,
        coast_time,
        state_condition=state_condition,
    )

    tables = {
        "station_structure_coverage.csv": station_structure,
        "state_condition_bias.csv": state_condition,
        "coast_time_signed_bias.csv": coast_time,
        "equivalent_diagnostics.csv": equivalent,
        "worsening_diagnostics.csv": worsening,
        "high_residual_group_diagnostic.csv": high_residual,
        "station_885_diagnostic.csv": station_885,
        "matched_state_contrast.csv": matched_contrast,
        "primary_eligibility_ledger.csv": primary_ledger,
        "spatial_structure_sensitivity_summary.csv": spatial_sensitivity,
        "matched_state_coverage_sensitivity.csv": coverage_sensitivity,
        "physical_hypothesis_comparison.csv": physical,
    }
    metric_payload = {key: int(value) for key, value in metrics.items()}
    for key, value in dict(source_metrics or {}).items():
        if key not in metric_payload and isinstance(value, (int, float, np.integer, np.floating)):
            metric_payload[key] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    active_rss_reader = _process_rss_bytes if rss_reader is None else rss_reader
    _record_publish_rss(
        metric_payload,
        rss_reader=active_rss_reader,
        phase="post_aggregation",
    )
    metadata = _build_analysis_metadata(
        canonical=canonical,
        metrics=metric_payload,
        analysis_station_ids=truth["STN"],
        input_sources=source_metadata,
    )
    summary = (
        "# 고도보정 BIAS 물리요인 후속분석\n\n"
        f"- 전체 분석 관측소: {len(truth)}개\n"
        f"- exact none 재현 시각: {metrics['processed_paired_hours']}개\n"
        "- `inversion_like`는 지상 관측망의 공간적 양의 기온-고도 기울기이며 "
        "실제 연직 역전의 직접 관측이 아니다.\n"
        "- 상태별 효과는 signed BIAS를 먼저 평균했고, 상태 간 차이는 공통 월×태양상태와 "
        "동일한 7일 날짜 block 가중치로 계산했다.\n"
    )

    _validate_cross_table_contracts(tables, truth)
    canonical_csv_payloads, trusted_prewrite_anchor = (
        _prepare_prewrite_primary_anchor(tables, metadata)
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        artifacts = _write_and_validate_staging(
            staging,
            tables=tables,
            metadata=metadata,
            summary_markdown=summary,
            failure_injection=failure_injection,
            rss_observer=lambda: _record_publish_rss(
                metadata,
                rss_reader=active_rss_reader,
                phase="post_plot_pre_promote",
            ),
            canonical_csv_payloads=canonical_csv_payloads,
            trusted_prewrite_anchor_sha256=trusted_prewrite_anchor,
        )
        pre_promote_validator = None
        production_publication = bool(
            isinstance(source_metadata, Mapping) and source_metadata.get(
            "source_mode"
            ) == "production"
        )
        guarded_receipt: Path | None = None
        if production_publication:
            if private_db_snapshot_identity is None:
                raise ValueError("production publish requires the private DB snapshot identity")
            guarded_receipt = _guard_acceptance_receipt_path(
                acceptance_receipt_path,
                final=final,
            )

            def pre_promote_validator() -> None:
                _validate_live_production_source_identity(
                    source_metadata, verify_db_hash=False
                )
                _validate_private_snapshot_light_identity(
                    private_db_snapshot_identity
                )

        def post_promote_validator() -> str:
            return _validate_published_artifacts_stable_manifest(
                final,
                allowed_fullnet_root=(None if production_publication else final.parent),
                trusted_primary_eligibility_binding_sha256=(
                    trusted_prewrite_anchor if production_publication else None
                ),
                sources_prevalidated=production_publication,
            )

        receipt_factory = None
        if production_publication:
            def receipt_factory() -> dict[str, Any]:
                return _build_acceptance_receipt_payload(
                    final=final,
                    trusted_primary_anchor_sha256=trusted_prewrite_anchor,
                    db_snapshot_sha256=str(source_metadata["db_snapshot_sha256"]),
                )
        _atomic_publish(
            staging,
            final,
            failure_injection=failure_injection,
            pre_promote_validator=pre_promote_validator,
            post_promote_validator=post_promote_validator,
            acceptance_receipt_path=guarded_receipt,
            acceptance_receipt_payload_factory=receipt_factory,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    trusted_primary_binding = trusted_prewrite_anchor
    return {
        "output_dir": str(final),
        "artifacts": artifacts,
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "trusted_primary_eligibility_binding_sha256": trusted_primary_binding,
        "acceptance_receipt_path": (
            str(guarded_receipt) if guarded_receipt is not None else None
        ),
    }


_AWS_THIN_COLUMNS = ("TA", "WS10", "WS1", "RN_60m", "HM", "TD")


def _windows_peak_working_set_bytes(counters: Any) -> int:
    """Windows PROCESS_MEMORY_COUNTERS에서 현재값이 아닌 누적 peak를 선택한다."""

    current = int(counters.WorkingSetSize)
    peak = int(counters.PeakWorkingSetSize)
    return max(current, peak)


def _parse_proc_peak_rss_bytes(status_text: str) -> int | None:
    """Linux /proc status의 VmHWM을 우선하고 없을 때만 VmRSS를 사용한다."""

    values: dict[str, int] = {}
    for line in str(status_text).splitlines():
        if ":" not in line:
            continue
        key, payload = line.split(":", 1)
        if key not in {"VmHWM", "VmRSS"}:
            continue
        parts = payload.split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) < 2 or parts[1].lower() == "kb" else 1
        values[key] = amount * multiplier
    return values.get("VmHWM", values.get("VmRSS"))


def _process_rss_bytes() -> int:
    """process 누적 peak working set/RSS를 표준 라이브러리만으로 읽는다."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return _windows_peak_working_set_bytes(counters)
    status = Path("/proc/self/status")
    proc_peak: int | None = None
    if status.is_file():
        proc_peak = _parse_proc_peak_rss_bytes(status.read_text(encoding="utf-8"))
    import resource

    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    resource_peak = usage if usage > 10_000_000 else usage * 1024
    return max(int(proc_peak or 0), int(resource_peak))


def _record_publish_rss(
    metrics: dict[str, Any],
    *,
    rss_reader: Any,
    phase: str,
) -> int:
    """집계·그림 단계의 실제 RSS를 기록하고 상한 초과 게시를 차단한다."""

    if not callable(rss_reader):
        raise TypeError("rss_reader must be callable")
    rss = int(rss_reader())
    limit = int(metrics.get("rss_limit_bytes", PROCESS_RSS_LIMIT_BYTES))
    previous_peak = int(metrics.get("process_peak_rss_bytes", 0))
    if rss < 0 or limit <= 0 or previous_peak < 0:
        raise ValueError("publish RSS metrics must be non-negative with a positive limit")
    metrics["rss_limit_bytes"] = limit
    metrics["process_peak_rss_bytes"] = max(previous_peak, rss)
    if rss > limit:
        raise MemoryError(
            f"production RSS {rss} exceeds acceptance limit {limit} during {phase}"
        )
    return rss


class ProductionAcceptanceMonitor:
    """station boundary의 실제 RSS와 first-10 runtime acceptance를 계측한다."""

    def __init__(self, *, total_stations: int, rss_limit_bytes: int = PROCESS_RSS_LIMIT_BYTES):
        if total_stations <= 0 or rss_limit_bytes <= 0:
            raise ValueError("monitor totals and RSS limit must be positive")
        self.total_stations = int(total_stations)
        self.rss_limit_bytes = int(rss_limit_bytes)
        self.elapsed: list[float] = []
        self.process_peak_rss_bytes = 0

    def record_station(self, *, elapsed_seconds: float, rss_bytes: int) -> None:
        elapsed = float(elapsed_seconds)
        rss = int(rss_bytes)
        if elapsed < 0 or rss < 0:
            raise ValueError("elapsed/RSS must be non-negative")
        self.elapsed.append(elapsed)
        self.process_peak_rss_bytes = max(self.process_peak_rss_bytes, rss)
        if rss > self.rss_limit_bytes:
            raise MemoryError(
                f"production RSS {rss} exceeds acceptance limit {self.rss_limit_bytes}"
            )

    def summary(self) -> dict[str, float | int]:
        first = self.elapsed[:10]
        first_total = float(sum(first))
        estimate = (
            first_total / len(first) * self.total_stations if first else np.nan
        )
        return {
            "process_peak_rss_bytes": int(self.process_peak_rss_bytes),
            "rss_limit_bytes": int(self.rss_limit_bytes),
            "first_10_station_elapsed_seconds": first_total,
            "estimated_total_runtime_seconds_from_first_10": float(estimate),
            "observed_station_elapsed_seconds": float(sum(self.elapsed)),
            "stations_timed": len(self.elapsed),
        }


class BoundedStationSourceCache:
    """station-year thin source를 제한된 수만 유지하는 결정론적 LRU cache."""

    def __init__(self, *, max_entries: int, loader: Any):
        if max_entries <= 0 or not callable(loader):
            raise ValueError("cache requires a positive bound and callable loader")
        self.max_entries = int(max_entries)
        self.loader = loader
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._peak = 0

    def get(self, key: Any) -> Any:
        normalized = str(key).strip()
        if normalized in self._entries:
            self._hits += 1
            value = self._entries.pop(normalized)
            self._entries[normalized] = value
            return value
        self._misses += 1
        value = self.loader(normalized)
        self._entries[normalized] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        self._peak = max(self._peak, len(self._entries))
        return value

    def metrics(self) -> dict[str, int]:
        return {
            "aws_cache_hit_count": self._hits,
            "aws_cache_miss_count": self._misses,
            "aws_cache_peak_entries": self._peak,
            "aws_cache_eviction_count": self._evictions,
            "aws_cache_max_entries": self.max_entries,
        }

    def clear(self) -> None:
        """누적 hit/miss 계수는 보존하고 보유 중인 station-year frame만 해제한다."""

        self._entries.clear()


def _load_aws_station_year_thin(
    *,
    station_id: str,
    months: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_folder: os.PathLike[str] | str,
) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    """한 station-month CSV를 한 번만 읽어 필요한 열의 얇은 frame을 만든다.

    기존 ``load_aws``의 폴더·월·[start,end) 규칙을 그대로 따르되, 단일 열 public
    loader를 여섯 번 반복해 같은 파일을 재독하는 비용만 제거한다.
    """

    station = _normalize_station_id(station_id, field="AWS station ID")
    folder = Path(base_folder) / f"Station_{station}"
    pieces: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    reads = 0
    base = Path(base_folder).expanduser().resolve()
    for month in months:
        path = folder / f"AWS_{month}.csv"
        if not path.is_file():
            continue
        source = pd.read_csv(path)
        reads += 1
        resolved_path = path.resolve()
        inventory.append(
            {
                "relative_path": resolved_path.relative_to(base).as_posix(),
                "bytes": int(resolved_path.stat().st_size),
                "sha256": _sha256(resolved_path),
                "station_id": station,
                "month": str(month),
            }
        )
        if "TM" not in source.columns:
            continue
        source["TM"] = pd.to_datetime(source["TM"], errors="raise")
        if source["TM"].duplicated(keep=False).any():
            duplicate_tm = source.loc[source["TM"].duplicated(keep=False), "TM"].iloc[0]
            raise ValueError(
                f"AWS station {station} duplicate TM {duplicate_tm} in {path.name}"
            )
        source = source.loc[(source["TM"] >= start) & (source["TM"] < end)].copy()
        for column in _AWS_THIN_COLUMNS:
            if column not in source.columns:
                source[column] = np.nan
            source[column] = pd.to_numeric(source[column], errors="coerce")
        source["__SOURCE_FILE"] = path.name
        pieces.append(source.loc[:, ["TM", *_AWS_THIN_COLUMNS, "__SOURCE_FILE"]])
    if not pieces:
        empty = pd.DataFrame(columns=_AWS_THIN_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="TM")
        return empty, reads, inventory
    frame = pd.concat(pieces, ignore_index=True)
    duplicate = frame["TM"].duplicated(keep=False)
    if duplicate.any():
        conflict = frame.loc[duplicate].sort_values(["TM", "__SOURCE_FILE"])
        timestamp = conflict["TM"].iloc[0]
        sources = conflict.loc[conflict["TM"].eq(timestamp), "__SOURCE_FILE"].astype(str).drop_duplicates().tolist()
        raise ValueError(
            f"AWS station {station} duplicate TM {timestamp} across {' -> '.join(sources)}"
        )
    frame = frame.drop(columns="__SOURCE_FILE").set_index("TM").sort_index()
    return frame, reads, inventory


def run_production_analysis(
    *,
    out_dir: os.PathLike[str] | str,
    db_path: os.PathLike[str] | str | None = None,
    station_results_path: os.PathLike[str] | str | None = None,
    metadata_path: os.PathLike[str] | str | None = None,
    terrain_path: os.PathLike[str] | str | None = None,
    aws_base_folder: os.PathLike[str] | str = str(AWS_DIR),
    acceptance_receipt_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """실제 529개 원자료 runner 진입점.

    Task3에서는 API와 안전 guard만 제공하며 실제 전체 실행은 Task4에서 한다.
    모든 외부 정본은 read-only로 열고 public loader를 재사용한다.
    """

    final = guard_production_output_directory(out_dir)  # loader보다 반드시 먼저 검증한다.
    guarded_receipt = _guard_acceptance_receipt_path(
        acceptance_receipt_path,
        final=final,
    )
    if None in {db_path, station_results_path, metadata_path, terrain_path}:
        raise ValueError("production paths must be supplied explicitly")

    source_metrics: dict[str, Any] = {
        "db_query_count": 0,
        "db_full_hash_verification_count": 0,
        "db_full_hash_verification_elapsed_seconds": 0.0,
        "aws_file_read_count": 0,
    }
    source_metadata = _capture_production_source_metadata(
        db_path=db_path,
        station_results_path=station_results_path,
        metadata_path=metadata_path,
        terrain_path=terrain_path,
        aws_base_folder=aws_base_folder,
        aws_file_inventory=[],
        metrics=source_metrics,
    )

    # 지연 import로 안전 guard 실패 시 DB/파일 loader가 호출되지 않게 한다.
    from analysis import bias_data
    from weather import find_stations
    from weather import load_aws

    truth = _prepare_station_results(
        pd.read_csv(source_metadata["station_results_path"], encoding="utf-8-sig")
    )
    canonical = validate_canonical_station_results(truth)
    _, source_population_sha256 = _station_population_identity(truth["STN"])
    analysis_config_sha256 = _canonical_json_hash(_analysis_config_payload())
    meta = find_stations.load_station_meta(source_metadata["metadata_path"])
    terrain = pd.read_csv(source_metadata["terrain_path"], encoding="utf-8-sig")
    station_outputs: dict[str, dict[str, Any]] = {}
    inventory_by_path: dict[str, dict[str, Any]] = {}
    snapshot_identity: dict[str, Any] | None = None
    source_connection: sqlite3.Connection | None = None
    connection: sqlite3.Connection | None = None
    analysis_failed = False
    try:
        source_connection = bias_data.open_readonly_db(source_metadata["db_path"])
        snapshot_identity = _create_private_sqlite_snapshot(
            source_metadata["db_path"],
            temp_parent=final.parent,
            source_identity=source_metadata,
            metrics=source_metrics,
            source_connection=source_connection,
        )
        source_connection.close()
        source_connection = None
        source_metadata.update(
            {
                key: snapshot_identity[key]
                for key in (
                    "db_snapshot_sha256",
                    "db_snapshot_bytes",
                    "db_snapshot_creation_elapsed_seconds",
                    "db_snapshot_hash_elapsed_seconds",
                )
            }
        )

        connection = _open_immutable_sqlite_snapshot(snapshot_identity)
        connection.execute("BEGIN")
        connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        _validate_private_snapshot_after_query(snapshot_identity, connection)
        connection.set_trace_callback(
            lambda statement: source_metrics.__setitem__(
                "db_query_count",
                source_metrics["db_query_count"]
                + int(statement.lstrip().upper().startswith("SELECT")),
            )
        )
        # ID 모집단은 한 번 조회하고, paired는 아래에서 관측소당 정확히 한 번 읽는다.
        station_ids = pd.read_sql(
            """
            SELECT DISTINCT STN
            FROM raw
            WHERE interpolable=1 AND METHOD IN ('none', 'fixed_lapse')
            ORDER BY CAST(STN AS INTEGER), STN
            """,
            connection,
        )["STN"].astype(str).tolist()
        _validate_private_snapshot_after_query(snapshot_identity, connection)
        neighbor_map = bias_data.load_neighbor_map(connection)
        _validate_private_snapshot_after_query(snapshot_identity, connection)
        bounds = pd.read_sql(
            """
            SELECT MIN(TM) AS START_TM, MAX(TM) AS END_TM
            FROM raw
            WHERE interpolable=1 AND METHOD IN ('none', 'fixed_lapse')
            """,
            connection,
        ).iloc[0]
        _validate_private_snapshot_after_query(snapshot_identity, connection)
        overall_start = pd.to_datetime(bounds["START_TM"], errors="raise")
        overall_end = pd.to_datetime(bounds["END_TM"], errors="raise") + pd.Timedelta(hours=1)
        months = [str(value) for value in load_aws.get_months(overall_start, overall_end)]
        if set(station_ids) != set(truth["STN"]):
            raise ValueError("production DB station population differs from canonical 529")

        def load_station_source(station_id: str) -> pd.DataFrame:
            station_id = _normalize_station_id(station_id, field="production AWS STN")
            frame, reads, inventory = _load_aws_station_year_thin(
                station_id=station_id,
                months=months,
                start=overall_start,
                end=overall_end,
                base_folder=source_metadata["aws_base_folder"],
            )
            source_metrics["aws_file_read_count"] += reads
            for entry in inventory:
                previous = inventory_by_path.get(entry["relative_path"])
                if previous is not None and previous != entry:
                    raise ValueError(
                        f"AWS source changed during analysis: {entry['relative_path']}"
                    )
                inventory_by_path[entry["relative_path"]] = entry
            return frame

        source_cache = BoundedStationSourceCache(
            max_entries=64, loader=load_station_source
        )
        monitor = ProductionAcceptanceMonitor(total_stations=len(station_ids))

        def station_source(station_id: str) -> pd.DataFrame:
            return source_cache.get(station_id)

        # 한 관측소를 분석하고 시간별 frame은 즉시 버린다. 전 station hourly를
        # 누적하지 않으며, TA는 station-year별 최초 한 번만 public loader로 읽는다.
        for station_id in station_ids:
            station_started = time.perf_counter()
            pairs = bias_data.load_station_pairs(connection, station_id)
            _validate_private_snapshot_after_query(snapshot_identity, connection)
            pairs["NEIGHBOR_COUNT"] = pairs["NEIGHBOR_SET_ID"].map(
                lambda signature: len(
                    validate_exact_neighbor_ids(
                        station_id,
                        neighbor_map[str(signature)],
                    )
                )
            )
            signatures = pairs["NEIGHBOR_SET_ID"].astype(str).unique().tolist()
            required_neighbors: set[str] = set()
            for signature in signatures:
                if signature not in neighbor_map:
                    raise ValueError(f"unknown production neighbor signature: {signature}")
                required_neighbors.update(
                    validate_exact_neighbor_ids(station_id, neighbor_map[signature])
                )
            for neighbor_id in sorted(required_neighbors, key=lambda value: (int(value), value)):
                source = station_source(neighbor_id)
                if source.empty or source["TA"].notna().sum() == 0:
                    raise ValueError(f"TA source missing for exact neighbor {neighbor_id}")
            ta_data = pd.DataFrame(
                {
                    neighbor_id: station_source(neighbor_id)["TA"]
                    for neighbor_id in required_neighbors
                }
            )

            target_source = station_source(station_id)
            weather = target_source.loc[:, ["WS10", "WS1", "RN_60m", "HM", "TD"]].copy()
            weather.index.name = "TM"
            weather = weather.reset_index()
            weather["STN"] = station_id
            station_outputs[station_id] = analyze_station_injected(
                pairs=pairs,
                neighbor_map=neighbor_map,
                meta=meta,
                ta_data=ta_data,
                weather=weather,
                terrain=terrain,
                strict_reproduction=True,
                return_hourly=station_id == "885",
                source_population_sha256=source_population_sha256,
                analysis_config_sha256=analysis_config_sha256,
            )
            monitor.record_station(
                elapsed_seconds=time.perf_counter() - station_started,
                rss_bytes=_process_rss_bytes(),
            )
        source_metrics.update(source_cache.metrics())
        # station 계산이 끝난 뒤에는 원자료 cache가 게시 단계에서 쓰이지 않는다.
        # 마지막 반복의 지역 변수도 함께 끊어 대형 집계/직렬화 전에 조기 해제한다.
        source_cache.clear()
        pairs = ta_data = weather = target_source = source = None
        gc.collect()
        source_metrics.update(monitor.summary())
    except Exception:
        analysis_failed = True
        raise
    finally:
        if connection is not None:
            connection.set_trace_callback(None)
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        if source_connection is not None:
            if source_connection.in_transaction:
                source_connection.rollback()
            source_connection.close()
        if analysis_failed and snapshot_identity is not None:
            _cleanup_private_sqlite_snapshot(snapshot_identity)
            snapshot_identity = None

    try:
        if snapshot_identity is None:
            raise RuntimeError("production DB snapshot was not created")
        source_metadata["aws_file_inventory"] = [
            inventory_by_path[key] for key in sorted(inventory_by_path)
        ]
        _validate_original_db_full_hash(source_metadata, metrics=source_metrics)
        _validate_live_production_source_identity(
            source_metadata, verify_db_hash=False
        )
        _validate_private_snapshot_full_hash(
            snapshot_identity, metrics=source_metrics
        )
        source_metadata["db_snapshot_hash_elapsed_seconds"] = snapshot_identity[
            "db_snapshot_hash_elapsed_seconds"
        ]
        if int(source_metrics["db_full_hash_verification_count"]) != 4:
            raise ValueError("production DB full-hash count must be exactly four")
        return _publish_precomputed_analysis(
            truth=truth,
            station_outputs=station_outputs,
            final=final,
            canonical=canonical,
            source_metrics=source_metrics,
            source_metadata=source_metadata,
            private_db_snapshot_identity=snapshot_identity,
            acceptance_receipt_path=guarded_receipt,
        )
    finally:
        if snapshot_identity is not None:
            _cleanup_private_sqlite_snapshot(snapshot_identity)


__all__ = [
    "analyze_station_injected",
    "build_matched_state_contrast",
    "build_state_condition_bias",
    "classify_solar_state",
    "guard_output_directory",
    "merge_weather_conditions",
    "reconstruct_signature_geometry",
    "run_injected_analysis",
    "run_production_analysis",
    "solar_elevation_noaa",
    "validate_canonical_station_results",
    "validate_exact_neighbor_ids",
    "validate_published_artifacts",
    "validate_acceptance_receipt",
    "vectorized_structure_diagnostics",
]
