"""BIAS 발표 피드백 후속 분석의 표·그림 생성 helper.

기존 ``bias_feedback_analysis`` 정본은 읽기 전용으로 사용한다. 이 파일은 후속
분석 전용이며, Task 1에서는 표와 명시적으로 지정한 staging 그림만 만든다.
"""

from __future__ import annotations

from project_paths import WINDOWS_FONT_DIR

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
import warnings

import numpy as np
import pandas as pd

from analyze_bias_feedback import (
    build_daily_matrices,
    load_daily_paired_errors,
    validate_annual_against_existing,
)
from bias_significance import bootstrap_bias_effects, make_month_stratified_block_weights


FULLNET = Path("WeatherData_AWS/interpolation_batch/fullnet")
SOURCE_STATION_RESULTS = FULLNET / "bias_feedback_analysis" / "bias_feedback_station_results.csv"
FOLLOWUP_OUT_DIR = FULLNET / "bias_feedback_followup"
DEFAULT_LITERATURE_SOURCE = Path(".omc/sdd/bias-feedback-followup/literature_claim_map.md")
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260818

CLASS_ORDER = [
    "meaningful_improvement",
    "meaningful_worsening",
    "practically_equivalent",
    "uncertain",
]
EQUIVALENT_LEVEL_ORDER = ["≤0.1°C", "(0.1, 0.5]°C", ">0.5°C"]
DELTA_Z_BIN_ORDER = ["≤50m", "50–100m", "100–200m", "200–300m", ">300m"]
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]
QUARTER_CALENDAR_DAYS = {"Q1": 91, "Q2": 91, "Q3": 92, "Q4": 92}
POINT_CLASS_ORDER = ["improvement", "small_change", "worsening"]
PRACTICAL_DELTA_C = 0.10
QUARTER_COVERAGE_FRACTION = 0.85
QUARTER_LABEL = "2024년 달력상 분기"
REVERSAL_COLUMNS = [
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
LEAVE_ONE_QUARTER_COLUMNS = [
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

EQUIVALENT_TRANSITION_COLUMNS = (
    "BEFORE_ABS_BIAS_LEVEL",
    "AFTER_ABS_BIAS_LEVEL",
    "N",
    "FRACTION_OF_EQUIVALENT",
)
DELTA_Z_CATEGORY_COLUMNS = (
    "DELTA_Z_BIN",
    "SIGNIFICANCE_CLASS",
    "N",
    "BIN_TOTAL",
    "FRACTION_WITHIN_BIN",
)
QUARTERLY_STATION_COLUMNS = (
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
    "CI95_LOW",
    "CI95_HIGH",
    "CI_POSITION",
    "BOOT_BLOCK_DAYS",
    "BOOT_REPLICATES",
    "BOOT_SEED",
)
QUARTERLY_NETWORK_COLUMNS = (
    "QUARTER",
    "N_ELIGIBLE",
    "MEDIAN_DELTA_ABS_BIAS",
    "Q25_DELTA_ABS_BIAS",
    "Q75_DELTA_ABS_BIAS",
    "N_POINT_IMPROVEMENT",
    "FRAC_POINT_IMPROVEMENT",
    "N_POINT_SMALL_CHANGE",
    "FRAC_POINT_SMALL_CHANGE",
    "N_POINT_WORSENING",
    "FRAC_POINT_WORSENING",
    "COMMON_COHORT_N",
    "COMMON_MEDIAN_DELTA_ABS_BIAS",
    "COMMON_Q25_DELTA_ABS_BIAS",
    "COMMON_Q75_DELTA_ABS_BIAS",
    "COMMON_N_POINT_IMPROVEMENT",
    "COMMON_FRAC_POINT_IMPROVEMENT",
    "COMMON_N_POINT_SMALL_CHANGE",
    "COMMON_FRAC_POINT_SMALL_CHANGE",
    "COMMON_N_POINT_WORSENING",
    "COMMON_FRAC_POINT_WORSENING",
)
QUARTERLY_SENSITIVITY_COLUMNS = (
    "QUARTER",
    "BLOCK_DAYS",
    "N_BOOT",
    "N_ELIGIBLE",
    "MEDIAN_CI95_LOW",
    "MEDIAN_CI95_HIGH",
    "MEDIAN_CI95_WIDTH",
    "CI_POSITION_AGREE_N",
    "CI_POSITION_AGREE_FRACTION",
)

CSV_COLUMN_ORDERS = {
    "equivalent_bias_transition.csv": EQUIVALENT_TRANSITION_COLUMNS,
    "delta_z_category_rates.csv": DELTA_Z_CATEGORY_COLUMNS,
    "quarterly_station_effects.csv": QUARTERLY_STATION_COLUMNS,
    "quarterly_network_summary.csv": QUARTERLY_NETWORK_COLUMNS,
    "quarterly_station_reversals.csv": tuple(REVERSAL_COLUMNS),
    "leave_one_quarter_out.csv": tuple(LEAVE_ONE_QUARTER_COLUMNS),
    "quarterly_block_sensitivity.csv": QUARTERLY_SENSITIVITY_COLUMNS,
}
PNG_ARTIFACTS = (
    "bias_equivalent_before_after.png",
    "bias_quarterly_effect.png",
    "bias_quarterly_heatmap.png",
    "bias_leave_one_quarter_out.png",
)
ANALYSIS_ARTIFACTS = (
    *CSV_COLUMN_ORDERS.keys(),
    *PNG_ARTIFACTS,
    "literature_claim_map.md",
    "bias_feedback_followup_report.md",
    "bias_feedback_followup_summary.json",
)
SUMMARY_JSON_KEYS = (
    "status",
    "schema_version",
    "analysis_year",
    "station_count",
    "paired_hours",
    "practical_delta_c",
    "quarter_coverage_fraction",
    "primary_block_days",
    "sensitivity_block_days",
    "bootstrap_replicates",
    "seed",
    "official_class_counts",
    "equivalent_station_count",
    "annual_validation",
    "csv_rows",
    "artifact_names",
    "artifact_hashes",
    "literature_source_sha256",
)


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"station results lack required column: {column}")
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError(f"station results contain non-finite values: {column}")
    return values


def _absolute_bias_level(values: pd.Series) -> pd.Categorical:
    absolute = values.abs()
    labels = np.select(
        [absolute <= 0.1, absolute <= 0.5],
        EQUIVALENT_LEVEL_ORDER[:2],
        default=EQUIVALENT_LEVEL_ORDER[2],
    )
    return pd.Categorical(labels, categories=EQUIVALENT_LEVEL_ORDER, ordered=True)


def _absolute_delta_z_km(frame: pd.DataFrame) -> pd.Series:
    for column in ("mean_absdz_geom_km", "mean_abs_delta_z_km"):
        if column in frame:
            return _finite_numeric(frame, column).abs()
    raise ValueError("station results lack mean absolute ΔZ column")


def _validate_significance_classes(frame: pd.DataFrame) -> None:
    if "SIGNIFICANCE_CLASS" not in frame:
        raise ValueError("station results lack required column: SIGNIFICANCE_CLASS")
    if frame["SIGNIFICANCE_CLASS"].isna().any():
        raise ValueError("station results contain missing SIGNIFICANCE_CLASS")
    unexpected = set(frame["SIGNIFICANCE_CLASS"].dropna()) - set(CLASS_ORDER)
    if unexpected:
        raise ValueError(f"unexpected significance classes: {sorted(unexpected)}")


def build_equivalent_bias_transition(station_results: pd.DataFrame) -> pd.DataFrame:
    """‘차이 거의 없음’ station의 보정 전·후 절대 BIAS 전이 3×3 표를 만든다."""
    _validate_significance_classes(station_results)
    frame = station_results.loc[
        station_results["SIGNIFICANCE_CLASS"].eq("practically_equivalent")
    ].copy()
    before = _absolute_bias_level(_finite_numeric(frame, "BIAS_none"))
    after = _absolute_bias_level(_finite_numeric(frame, "BIAS_fixed"))
    denominator = len(frame)
    rows = []
    for before_level in EQUIVALENT_LEVEL_ORDER:
        for after_level in EQUIVALENT_LEVEL_ORDER:
            count = int(((before == before_level) & (after == after_level)).sum())
            rows.append(
                {
                    "BEFORE_ABS_BIAS_LEVEL": before_level,
                    "AFTER_ABS_BIAS_LEVEL": after_level,
                    "N": count,
                    "FRACTION_OF_EQUIVALENT": count / denominator if denominator else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_delta_z_category_rates(station_results: pd.DataFrame) -> pd.DataFrame:
    """평균 절대 ΔZ 구간별 연간 네 분류의 count·구간 내 비율을 long 형식으로 만든다."""
    _validate_significance_classes(station_results)
    frame = station_results.copy()
    delta_z_km = _absolute_delta_z_km(frame)
    frame["DELTA_Z_BIN"] = pd.cut(
        delta_z_km,
        bins=[-np.inf, 0.05, 0.10, 0.20, 0.30, np.inf],
        labels=DELTA_Z_BIN_ORDER,
        include_lowest=True,
        right=True,
    )
    rows = []
    for delta_z_bin in DELTA_Z_BIN_ORDER:
        in_bin = frame["DELTA_Z_BIN"].eq(delta_z_bin)
        bin_total = int(in_bin.sum())
        for significance_class in CLASS_ORDER:
            count = int((in_bin & frame["SIGNIFICANCE_CLASS"].eq(significance_class)).sum())
            rows.append(
                {
                    "DELTA_Z_BIN": delta_z_bin,
                    "SIGNIFICANCE_CLASS": significance_class,
                    "N": count,
                    "BIN_TOTAL": bin_total,
                    "FRACTION_WITHIN_BIN": count / bin_total if bin_total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_equivalent_before_after(station_results: pd.DataFrame, output_path) -> Path:
    """‘차이 거의 없음’ station의 보정 전·후 절대 BIAS를 지정한 단일 경로에 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    _validate_significance_classes(station_results)
    frame = station_results.loc[
        station_results["SIGNIFICANCE_CLASS"].eq("practically_equivalent")
    ].copy()
    before = _finite_numeric(frame, "BIAS_none").abs().to_numpy(float)
    after = _finite_numeric(frame, "BIAS_fixed").abs().to_numpy(float)
    levels = _absolute_bias_level(pd.Series(before))
    colors = {"≤0.1°C": "#2878B5", "(0.1, 0.5]°C": "#F0AD4E", ">0.5°C": "#D9534F"}
    korean_font_path = (WINDOWS_FONT_DIR / "malgun.ttf")
    korean_font = FontProperties(fname=korean_font_path) if korean_font_path.is_file() else None

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for index, (before_value, after_value, level) in enumerate(zip(before, after, levels)):
        ax.plot([0, 1], [before_value, after_value], color=colors[str(level)], alpha=0.45, lw=1)
        ax.scatter([0, 1], [before_value, after_value], color=colors[str(level)], s=18, zorder=3)
    ax.axhline(0.1, color="#777777", lw=0.8, ls="--")
    ax.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["보정 전", "보정 후"])
    if korean_font is not None:
        for label in ax.get_xticklabels():
            label.set_fontproperties(korean_font)
    ax.set_ylabel("절대 BIAS (°C)", fontproperties=korean_font)
    ax.set_title("차이 거의 없음 관측소의 보정 전·후 절대 BIAS", fontproperties=korean_font)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def assign_calendar_quarter(dates) -> pd.Categorical:
    """2024년 날짜를 서로 겹치지 않는 달력상 Q1–Q4에 배정한다."""
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise")).normalize()
    if len(index) and not (index.year == 2024).all():
        raise ValueError("calendar quarter dates must all belong to 2024")
    labels = np.asarray([f"Q{((month - 1) // 3) + 1}" for month in index.month], dtype=object)
    return pd.Categorical(labels, categories=QUARTER_ORDER, ordered=True)


def _station_universe(frame: pd.DataFrame, stations=None) -> list[str]:
    if stations is None:
        station_ids = sorted(frame["STN"].unique().tolist())
        if not station_ids:
            raise ValueError("stations and daily cannot both be empty")
        return station_ids
    station_ids = [str(stn) for stn in stations]
    if not station_ids:
        raise ValueError("stations must be a non-empty unique list")
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("stations must remain unique after string conversion")
    unknown = set(frame["STN"]) - set(station_ids)
    if unknown:
        raise ValueError(f"daily table contains stations outside canonical universe: {sorted(unknown)}")
    return station_ids


def _prepare_quarterly_daily(daily: pd.DataFrame | None, stations=None) -> pd.DataFrame:
    required = {"STN", "DATE", "sum_none", "sum_fixed", "count"}
    if daily is None:
        daily = pd.DataFrame(columns=sorted(required))
    if not isinstance(daily, pd.DataFrame):
        raise ValueError("daily must be a pandas DataFrame or None")
    if daily.empty and not required.issubset(daily.columns) and stations is not None:
        daily = pd.DataFrame(columns=sorted(required))
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily table missing columns: {sorted(missing)}")
    frame = daily[["STN", "DATE", "sum_none", "sum_fixed", "count"]].copy()
    frame["STN"] = frame["STN"].astype(str)
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise").dt.normalize()
    if frame.duplicated(["STN", "DATE"]).any():
        raise ValueError("daily table contains duplicate station-date rows")
    if not (frame["DATE"].dt.year == 2024).all():
        raise ValueError("daily table dates must all belong to 2024")
    for column in ["sum_none", "sum_fixed", "count"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["sum_none", "sum_fixed", "count"]].to_numpy(float)).all():
        raise ValueError("daily sums and counts must be finite")
    if (frame["count"] < 0).any():
        raise ValueError("daily count must not be negative")
    frame["QUARTER"] = assign_calendar_quarter(frame["DATE"])
    _station_universe(frame, stations)
    return frame.sort_values(["STN", "DATE"], kind="stable").reset_index(drop=True)


def _point_class(value: float, delta: float = PRACTICAL_DELTA_C) -> str:
    value = float(value)
    delta = float(delta)
    if value < -delta and not np.isclose(value, -delta, rtol=0.0, atol=1e-12):
        return "improvement"
    if value > delta and not np.isclose(value, delta, rtol=0.0, atol=1e-12):
        return "worsening"
    return "small_change"


def classify_ci_position(low: float, high: float, delta: float = PRACTICAL_DELTA_C) -> str:
    """95% CI의 실질 경계 ±delta 대비 위치를 분류한다."""
    low = float(low)
    high = float(high)
    delta = float(delta)
    if not np.isfinite([low, high, delta]).all() or low > high or delta <= 0:
        raise ValueError("CI bounds must be finite and ordered, and delta must be positive")
    if high < -delta:
        return "below_minus_delta"
    if low > delta:
        return "above_plus_delta"
    return "overlaps_practical_band"


def build_quarterly_point_effects(
    daily: pd.DataFrame | None = None,
    stations=None,
) -> pd.DataFrame:
    """station×Q1–Q4 골격에서 일별 원합계로 분기 Δ|BIAS|를 계산한다."""
    frame = _prepare_quarterly_daily(daily, stations=stations)
    station_ids = _station_universe(frame, stations)
    rows = []
    for stn in station_ids:
        station = frame.loc[frame["STN"].eq(stn)]
        for quarter in QUARTER_ORDER:
            calendar_days = QUARTER_CALENDAR_DAYS[quarter]
            minimum_days = int(math.ceil(QUARTER_COVERAGE_FRACTION * calendar_days))
            group = station.loc[station["QUARTER"].eq(quarter)]
            valid = group.loc[group["count"].gt(0)]
            valid_days = int(valid["DATE"].nunique())
            months_present = int(valid["DATE"].dt.month.nunique())
            paired_hours = float(valid["count"].sum())
            enough_days = valid_days >= minimum_days
            all_months = months_present == 3
            eligible = bool(enough_days and all_months)
            if valid_days == 0:
                reason = "no_data"
            elif not all_months and not enough_days:
                reason = "missing_months_and_below_85pct"
            elif not all_months:
                reason = "missing_months"
            elif not enough_days:
                reason = "below_85pct"
            else:
                reason = "eligible"

            bias_none = np.nan
            bias_fixed = np.nan
            effect = np.nan
            point_class = pd.NA
            if eligible:
                bias_none = float(valid["sum_none"].sum() / paired_hours)
                bias_fixed = float(valid["sum_fixed"].sum() / paired_hours)
                effect = abs(bias_fixed) - abs(bias_none)
                point_class = _point_class(effect)
            rows.append(
                {
                    "STN": str(stn),
                    "QUARTER": quarter,
                    "CALENDAR_DAYS": calendar_days,
                    "MIN_VALID_DAYS": minimum_days,
                    "VALID_DAYS": valid_days,
                    "COVERAGE": valid_days / calendar_days,
                    "MONTHS_PRESENT": months_present,
                    "PAIRED_HOURS": int(paired_hours) if paired_hours.is_integer() else paired_hours,
                    "COVERAGE_ELIGIBLE": eligible,
                    "COVERAGE_REASON": reason,
                    "BIAS_none": bias_none,
                    "BIAS_fixed": bias_fixed,
                    "DELTA_ABS_BIAS": effect,
                    "POINT_CLASS": point_class,
                }
            )
    return pd.DataFrame(rows)


def _stable_quarter_seed(seed: int, stn: str, quarter: str) -> int:
    payload = f"{int(seed)}\x1f{str(stn)}\x1f{quarter}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _strict_boolean_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"table missing required column: {column}")
    valid = frame[column].map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        raise ValueError(f"{column} values must be actual boolean values")
    return frame[column].astype(bool)


def bootstrap_quarterly_effects(
    daily: pd.DataFrame | None,
    n_boot: int,
    block_days: int,
    seed: int,
    stations=None,
) -> pd.DataFrame:
    """가용 station·분기에 월별 층화 원형 블록 bootstrap 95% CI를 붙인다."""
    n_boot = _positive_integer(n_boot, "n_boot")
    block_days = _positive_integer(block_days, "block_days")
    frame = _prepare_quarterly_daily(daily, stations=stations)
    result = build_quarterly_point_effects(frame, stations=stations)
    result["CI95_LOW"] = np.nan
    result["CI95_HIGH"] = np.nan
    result["CI_POSITION"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["BOOT_BLOCK_DAYS"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["BOOT_REPLICATES"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["BOOT_SEED"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    for row_index in result.index[result["COVERAGE_ELIGIBLE"]]:
        stn = str(result.at[row_index, "STN"])
        quarter = str(result.at[row_index, "QUARTER"])
        valid = frame.loc[
            frame["STN"].eq(stn) & frame["QUARTER"].eq(quarter) & frame["count"].gt(0)
        ].sort_values("DATE", kind="stable")
        quarter_seed = _stable_quarter_seed(seed, stn, quarter)
        weights = make_month_stratified_block_weights(
            valid["DATE"], n_boot=n_boot, block_days=block_days, seed=quarter_seed
        )
        station_daily = {
            "sum_none": valid["sum_none"].to_numpy(float)[None, :],
            "sum_fixed": valid["sum_fixed"].to_numpy(float)[None, :],
            "count": valid["count"].to_numpy(float)[None, :],
        }
        samples = bootstrap_bias_effects(station_daily, weights)["delta_abs_bias"][:, 0]
        low, high = np.quantile(samples, [0.025, 0.975])
        result.at[row_index, "CI95_LOW"] = float(low)
        result.at[row_index, "CI95_HIGH"] = float(high)
        result.at[row_index, "CI_POSITION"] = classify_ci_position(low, high)
        result.at[row_index, "BOOT_BLOCK_DAYS"] = int(block_days)
        result.at[row_index, "BOOT_REPLICATES"] = int(n_boot)
        result.at[row_index, "BOOT_SEED"] = int(quarter_seed)
    return result


def _point_summary(values: pd.Series, prefix: str = "") -> dict:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
    n = len(numeric)
    classes = pd.Series([_point_class(value) for value in numeric], dtype="object")
    output = {
        f"{prefix}MEDIAN_DELTA_ABS_BIAS": float(np.median(numeric)) if n else np.nan,
        f"{prefix}Q25_DELTA_ABS_BIAS": float(np.quantile(numeric, 0.25)) if n else np.nan,
        f"{prefix}Q75_DELTA_ABS_BIAS": float(np.quantile(numeric, 0.75)) if n else np.nan,
    }
    for point_class in POINT_CLASS_ORDER:
        count = int(classes.eq(point_class).sum())
        suffix = point_class.upper()
        output[f"{prefix}N_POINT_{suffix}"] = count
        output[f"{prefix}FRAC_POINT_{suffix}"] = count / n if n else np.nan
    return output


def summarize_quarterly_network(quarterly: pd.DataFrame) -> pd.DataFrame:
    """분기별 가용 집합과 4분기 공통 cohort의 station-balanced 요약을 만든다."""
    required = {
        "STN",
        "QUARTER",
        "COVERAGE_ELIGIBLE",
        "DELTA_ABS_BIAS",
        "CI95_LOW",
        "CI95_HIGH",
    }
    missing = required.difference(quarterly.columns)
    if missing:
        raise ValueError(f"quarterly table missing columns: {sorted(missing)}")
    frame = quarterly.copy()
    frame["STN"] = frame["STN"].astype(str)
    if frame.duplicated(["STN", "QUARTER"]).any():
        raise ValueError("quarterly table contains duplicate station-quarter rows")
    if not set(frame["QUARTER"]).issubset(QUARTER_ORDER):
        raise ValueError("quarter must use the Q1–Q4 domain")
    incomplete = [
        stn
        for stn, group in frame.groupby("STN", sort=False)
        if len(group) != 4 or set(group["QUARTER"]) != set(QUARTER_ORDER)
    ]
    if incomplete:
        raise ValueError(f"complete station-quarter Q1–Q4 skeleton required: {incomplete[:5]}")
    eligible_mask = _strict_boolean_mask(frame, "COVERAGE_ELIGIBLE")
    numeric = frame[["DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if eligible_mask.any() and not np.isfinite(numeric.loc[eligible_mask].to_numpy(float)).all():
        raise ValueError("eligible rows require finite effect and CI")
    if (~eligible_mask).any() and not frame.loc[
        ~eligible_mask, ["DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"]
    ].isna().all().all():
        raise ValueError("ineligible rows require effect and CI to be NA")
    if eligible_mask.any() and (
        numeric.loc[eligible_mask, "CI95_LOW"] > numeric.loc[eligible_mask, "CI95_HIGH"]
    ).any():
        raise ValueError("eligible CI bounds must be ordered")
    eligible = frame.loc[eligible_mask].copy()
    counts = eligible.groupby("STN")["QUARTER"].nunique()
    common_stations = set(counts.index[counts.eq(4)])
    rows = []
    for quarter in QUARTER_ORDER:
        group = eligible.loc[eligible["QUARTER"].eq(quarter)]
        common = group.loc[group["STN"].isin(common_stations)]
        rows.append(
            {
                "QUARTER": quarter,
                "N_ELIGIBLE": len(group),
                **_point_summary(group["DELTA_ABS_BIAS"]),
                "COMMON_COHORT_N": len(common),
                **_point_summary(common["DELTA_ABS_BIAS"], prefix="COMMON_"),
            }
        )
    return pd.DataFrame(rows)


def build_quarterly_reversals(
    quarterly: pd.DataFrame,
    annual_results: pd.DataFrame,
    minimum_eligible_quarters: int = 2,
) -> pd.DataFrame:
    """분기 원부호·실질 반전과 연간–분기 점추정 class 불일치를 분리한다."""
    q_required = {"STN", "QUARTER", "COVERAGE_ELIGIBLE", "DELTA_ABS_BIAS"}
    a_required = {"STN", "DELTA_ABS_BIAS"}
    if q_required.difference(quarterly.columns):
        raise ValueError(f"quarterly table missing columns: {sorted(q_required.difference(quarterly.columns))}")
    if a_required.difference(annual_results.columns):
        raise ValueError(f"annual results missing columns: {sorted(a_required.difference(annual_results.columns))}")
    if int(minimum_eligible_quarters) != minimum_eligible_quarters or minimum_eligible_quarters < 2:
        raise ValueError("minimum_eligible_quarters must be an integer of at least 2")
    q = quarterly.copy()
    a = annual_results.copy()
    q["STN"] = q["STN"].astype(str)
    a["STN"] = a["STN"].astype(str)
    if q.duplicated(["STN", "QUARTER"]).any() or a["STN"].duplicated().any():
        raise ValueError("station identifiers must be unique at their table grain")
    unexpected = set(q["STN"]) - set(a["STN"])
    if unexpected:
        raise ValueError(f"quarterly table contains stations absent from annual results: {sorted(unexpected)}")
    q["_ELIGIBLE"] = _strict_boolean_mask(q, "COVERAGE_ELIGIBLE")

    rows = []
    for annual in a.sort_values("STN", kind="stable").itertuples(index=False):
        stn = str(annual.STN)
        station = q.loc[q["STN"].eq(stn) & q["_ELIGIBLE"]]
        effects = pd.to_numeric(station["DELTA_ABS_BIAS"], errors="raise").to_numpy(float)
        n_eligible = len(effects)
        sufficient = n_eligible >= int(minimum_eligible_quarters)
        annual_effect = float(getattr(annual, "DELTA_ABS_BIAS"))
        annual_point_class = _point_class(annual_effect)
        raw_crossing = pd.NA
        practical_crossing = pd.NA
        mismatch = pd.NA
        if sufficient:
            raw_crossing = bool(np.any(effects < 0) and np.any(effects > 0))
            practical_crossing = bool(
                np.any(effects < -PRACTICAL_DELTA_C) and np.any(effects > PRACTICAL_DELTA_C)
            )
            mismatch = bool(any(_point_class(value) != annual_point_class for value in effects))
        rows.append(
            {
                "STN": stn,
                "N_ELIGIBLE_QUARTERS": n_eligible,
                "MIN_ELIGIBLE_QUARTERS": int(minimum_eligible_quarters),
                "REVERSAL_STATUS": "sufficient" if sufficient else "insufficient_eligible_quarters",
                "RAW_SIGN_CROSSING": raw_crossing,
                "PRACTICAL_CROSSING": practical_crossing,
                "ANNUAL_QUARTER_POINT_CLASS_MISMATCH": mismatch,
                "ANNUAL_DELTA_ABS_BIAS": annual_effect,
                "ANNUAL_POINT_CLASS": annual_point_class,
                "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE": getattr(
                    annual, "SIGNIFICANCE_CLASS", pd.NA
                ),
            }
        )
    result = pd.DataFrame(rows, columns=REVERSAL_COLUMNS)
    for column in [
        "RAW_SIGN_CROSSING",
        "PRACTICAL_CROSSING",
        "ANNUAL_QUARTER_POINT_CLASS_MISMATCH",
    ]:
        result[column] = pd.array(result[column], dtype="boolean")
    return result


def _validate_primary_quarterly_result(
    primary_result: pd.DataFrame,
    daily: pd.DataFrame | None,
    stations,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required = {
        "STN",
        "QUARTER",
        "COVERAGE_ELIGIBLE",
        "DELTA_ABS_BIAS",
        "CI95_LOW",
        "CI95_HIGH",
        "CI_POSITION",
        "BOOT_BLOCK_DAYS",
        "BOOT_REPLICATES",
        "BOOT_SEED",
    }
    if not isinstance(primary_result, pd.DataFrame):
        raise ValueError("primary_result must be a pandas DataFrame")
    missing = required.difference(primary_result.columns)
    if missing:
        raise ValueError(f"primary_result missing columns: {sorted(missing)}")
    primary = primary_result.copy(deep=True)
    primary["STN"] = primary["STN"].astype(str)
    if primary.duplicated(["STN", "QUARTER"]).any():
        raise ValueError("primary_result contains duplicate station-quarter rows")
    if not set(primary["QUARTER"]).issubset(QUARTER_ORDER):
        raise ValueError("primary_result quarter must use the Q1–Q4 domain")

    expected = build_quarterly_point_effects(daily, stations=stations)
    expected_keys = pd.MultiIndex.from_frame(expected[["STN", "QUARTER"]])
    primary_keys = pd.MultiIndex.from_frame(primary[["STN", "QUARTER"]])
    if len(primary) != len(expected) or set(primary_keys) != set(expected_keys):
        raise ValueError("primary_result station-quarter keys must match the expected coverage skeleton")
    order = expected[["STN", "QUARTER"]].merge(
        primary,
        on=["STN", "QUARTER"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    primary_eligible = _strict_boolean_mask(order, "COVERAGE_ELIGIBLE")
    expected_eligible = _strict_boolean_mask(expected, "COVERAGE_ELIGIBLE")
    if not np.array_equal(primary_eligible.to_numpy(), expected_eligible.to_numpy()):
        raise ValueError("primary_result coverage eligibility does not match daily coverage skeleton")

    metadata = ["BOOT_BLOCK_DAYS", "BOOT_REPLICATES", "BOOT_SEED"]
    eligible_metadata = order.loc[primary_eligible, metadata].apply(pd.to_numeric, errors="coerce")
    if primary_eligible.any() and not np.isfinite(eligible_metadata.to_numpy(float)).all():
        raise ValueError("eligible primary_result bootstrap metadata must be finite")
    if (~primary_eligible).any() and not order.loc[~primary_eligible, metadata].isna().all().all():
        raise ValueError("ineligible primary_result bootstrap metadata must be NA")
    if primary_eligible.any() and not eligible_metadata["BOOT_BLOCK_DAYS"].eq(7).all():
        raise ValueError("eligible primary_result BOOT_BLOCK_DAYS must equal 7")
    if primary_eligible.any() and not eligible_metadata["BOOT_REPLICATES"].eq(n_boot).all():
        raise ValueError("eligible primary_result BOOT_REPLICATES must equal n_boot")
    if primary_eligible.any():
        expected_seeds = np.asarray(
            [
                _stable_quarter_seed(seed, stn, quarter)
                for stn, quarter in order.loc[primary_eligible, ["STN", "QUARTER"]].itertuples(
                    index=False, name=None
                )
            ],
            dtype=np.uint64,
        )
        actual_seeds = eligible_metadata["BOOT_SEED"].to_numpy(dtype=np.uint64)
        if not np.array_equal(actual_seeds, expected_seeds):
            raise ValueError("eligible primary_result BOOT_SEED does not match stable station-quarter seed")
    return order


def build_quarterly_block_sensitivity(
    daily: pd.DataFrame | None = None,
    n_boot: int = 2000,
    seed: int = 20260818,
    results_by_block: dict[int, pd.DataFrame] | None = None,
    primary_result: pd.DataFrame | None = None,
    stations=None,
) -> pd.DataFrame:
    """3·7·14일 블록의 station CI 요약과 7일 CI 위치 일치율을 계산한다."""
    n_boot = _positive_integer(n_boot, "n_boot")
    blocks = (3, 7, 14)
    if results_by_block is not None and primary_result is not None:
        raise ValueError("primary_result cannot be combined with results_by_block")
    if results_by_block is None:
        if daily is None:
            raise ValueError("daily is required when results_by_block is not supplied")
        if primary_result is not None:
            primary_result = _validate_primary_quarterly_result(
                primary_result,
                daily=daily,
                stations=stations,
                n_boot=n_boot,
                seed=seed,
            )
        results_by_block = {}
        for block in blocks:
            if block == 7 and primary_result is not None:
                results_by_block[block] = primary_result.copy(deep=True)
            else:
                results_by_block[block] = bootstrap_quarterly_effects(
                    daily,
                    n_boot=n_boot,
                    block_days=block,
                    seed=seed,
                    stations=stations,
                )
    if set(results_by_block) != set(blocks):
        raise ValueError("results_by_block must contain exactly block lengths 3, 7, and 14")
    reference = results_by_block[7].copy()
    reference["STN"] = reference["STN"].astype(str)
    reference = reference[["STN", "QUARTER", "CI_POSITION"]].rename(
        columns={"CI_POSITION": "CI_POSITION_7D"}
    )
    rows = []
    for quarter in QUARTER_ORDER:
        for block in blocks:
            frame = results_by_block[block].copy()
            frame["STN"] = frame["STN"].astype(str)
            eligible = frame.loc[
                frame["QUARTER"].eq(quarter) & frame["COVERAGE_ELIGIBLE"].astype(bool)
            ].copy()
            merged = eligible.merge(reference, on=["STN", "QUARTER"], how="left", validate="one_to_one")
            agree = merged["CI_POSITION"].eq(merged["CI_POSITION_7D"])
            widths = pd.to_numeric(eligible["CI95_HIGH"], errors="raise") - pd.to_numeric(
                eligible["CI95_LOW"], errors="raise"
            )
            n_eligible = len(eligible)
            rows.append(
                {
                    "QUARTER": quarter,
                    "BLOCK_DAYS": block,
                    "N_BOOT": int(n_boot),
                    "N_ELIGIBLE": n_eligible,
                    "MEDIAN_CI95_LOW": float(eligible["CI95_LOW"].median()) if n_eligible else np.nan,
                    "MEDIAN_CI95_HIGH": float(eligible["CI95_HIGH"].median()) if n_eligible else np.nan,
                    "MEDIAN_CI95_WIDTH": float(widths.median()) if n_eligible else np.nan,
                    "CI_POSITION_AGREE_N": int(agree.sum()),
                    "CI_POSITION_AGREE_FRACTION": float(agree.mean()) if n_eligible else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _quarter_plot_font():
    from matplotlib.font_manager import FontProperties

    path = (WINDOWS_FONT_DIR / "malgun.ttf")
    return FontProperties(fname=path) if path.is_file() else None


def plot_quarterly_effect(quarterly: pd.DataFrame, output_path) -> Path:
    """station별 분기 효과와 95% CI를 지정한 staging PNG에 그린다."""
    import matplotlib.pyplot as plt

    required = {"QUARTER", "COVERAGE_ELIGIBLE", "DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"}
    missing = required.difference(quarterly.columns)
    if missing:
        raise ValueError(f"quarterly table missing columns: {sorted(missing)}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _quarter_plot_font()
    colors = {"Q1": "#2878B5", "Q2": "#43A047", "Q3": "#F0AD4E", "Q4": "#D9534F"}
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    try:
        frame = quarterly.loc[quarterly["COVERAGE_ELIGIBLE"].astype(bool)]
        for x, quarter in enumerate(QUARTER_ORDER):
            group = frame.loc[frame["QUARTER"].eq(quarter)].reset_index(drop=True)
            if group.empty:
                continue
            offsets = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else np.array([0.0])
            x_values = x + offsets
            ax.vlines(
                x_values,
                group["CI95_LOW"].to_numpy(float),
                group["CI95_HIGH"].to_numpy(float),
                color=colors[quarter],
                alpha=0.35,
                lw=1.0,
            )
            ax.scatter(x_values, group["DELTA_ABS_BIAS"], color=colors[quarter], s=22, label=quarter)
        ax.axhspan(-PRACTICAL_DELTA_C, PRACTICAL_DELTA_C, color="#9E9E9E", alpha=0.15)
        ax.axhline(0.0, color="#444444", lw=0.9)
        ax.set_xticks(range(4), QUARTER_ORDER)
        ax.set_xlabel(QUARTER_LABEL, fontproperties=font)
        ax.set_ylabel("Δ|BIAS| (°C)", fontproperties=font)
        ax.set_title("2024년 달력상 분기별 관측소 BIAS 보정 효과", fontproperties=font)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_quarterly_heatmap(quarterly: pd.DataFrame, output_path) -> Path:
    """station×분기 Δ|BIAS| color heatmap을 지정한 staging PNG에 그린다."""
    import matplotlib.pyplot as plt

    required = {"STN", "QUARTER", "COVERAGE_ELIGIBLE", "DELTA_ABS_BIAS"}
    missing = required.difference(quarterly.columns)
    if missing:
        raise ValueError(f"quarterly table missing columns: {sorted(missing)}")
    frame = quarterly.copy()
    frame["STN"] = frame["STN"].astype(str)
    frame.loc[~frame["COVERAGE_ELIGIBLE"].astype(bool), "DELTA_ABS_BIAS"] = np.nan
    matrix = frame.pivot(index="STN", columns="QUARTER", values="DELTA_ABS_BIAS").reindex(
        columns=QUARTER_ORDER
    ).sort_index()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _quarter_plot_font()
    fig, ax = plt.subplots(figsize=(8.2, max(3.6, min(12.0, 1.5 + len(matrix) * 0.12))))
    try:
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#D9D9D9")
        finite = np.abs(matrix.to_numpy(float)[np.isfinite(matrix.to_numpy(float))])
        limit = max(float(finite.max()) if len(finite) else PRACTICAL_DELTA_C, PRACTICAL_DELTA_C)
        image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(4), QUARTER_ORDER)
        if len(matrix) <= 20:
            tick_positions = np.arange(len(matrix), dtype=int)
        else:
            tick_positions = np.linspace(0, len(matrix) - 1, num=20, dtype=int)
        ax.set_yticks(tick_positions, matrix.index[tick_positions].tolist())
        ax.set_xlabel(QUARTER_LABEL, fontproperties=font)
        ax.set_ylabel("관측소", fontproperties=font)
        ax.set_title("2024년 달력상 분기별 관측소 Δ|BIAS|", fontproperties=font)
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Δ|BIAS| (°C)", fontproperties=font)
        fig.text(
            0.5,
            0.01,
            "개별 관측소 값의 조회 정본: quarterly_station_effects.csv",
            ha="center",
            fontproperties=font,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _canonical_annual_results(
    annual_results: pd.DataFrame,
    stations=None,
) -> tuple[pd.DataFrame, list[str]]:
    required = {"STN", "DELTA_ABS_BIAS", "SIGNIFICANCE_CLASS"}
    if not isinstance(annual_results, pd.DataFrame):
        raise ValueError("annual_results must be a pandas DataFrame")
    missing = required.difference(annual_results.columns)
    if missing:
        raise ValueError(f"annual results missing columns: {sorted(missing)}")
    annual = annual_results[["STN", "DELTA_ABS_BIAS", "SIGNIFICANCE_CLASS"]].copy()
    annual["STN"] = annual["STN"].astype(str)
    if annual["STN"].duplicated().any():
        raise ValueError("annual results contain duplicate STN rows")
    if annual.empty:
        raise ValueError("annual results must contain at least one canonical station")
    _validate_significance_classes(annual)
    annual["DELTA_ABS_BIAS"] = pd.to_numeric(annual["DELTA_ABS_BIAS"], errors="raise")
    if not np.isfinite(annual["DELTA_ABS_BIAS"].to_numpy(float)).all():
        raise ValueError("annual DELTA_ABS_BIAS must be finite")

    annual_stations = annual["STN"].tolist()
    if stations is None:
        station_ids = annual_stations
    else:
        station_ids = [str(stn) for stn in stations]
        if not station_ids or len(station_ids) != len(set(station_ids)):
            raise ValueError("stations must be a non-empty unique canonical list")
        if set(station_ids) != set(annual_stations) or len(station_ids) != len(annual_stations):
            raise ValueError("stations must exactly match annual STN universe")
        annual = annual.set_index("STN").loc[station_ids].reset_index()
    return annual.reset_index(drop=True), station_ids


def build_leave_one_quarter_out(
    daily: pd.DataFrame | None,
    annual_results: pd.DataFrame,
    stations=None,
) -> pd.DataFrame:
    """제외 분기를 뺀 나머지 원시 오차 합으로 station별 9개월 효과를 재계산한다."""
    annual, station_ids = _canonical_annual_results(annual_results, stations=stations)
    frame = _prepare_quarterly_daily(daily, stations=station_ids)
    rows = []
    for annual_row in annual.itertuples(index=False):
        stn = str(annual_row.STN)
        annual_effect = float(annual_row.DELTA_ABS_BIAS)
        annual_point_class = _point_class(annual_effect)
        station_daily = frame.loc[frame["STN"].eq(stn) & frame["count"].gt(0)]
        for excluded_quarter in QUARTER_ORDER:
            remaining = station_daily.loc[
                ~station_daily["QUARTER"].eq(excluded_quarter)
            ]
            paired_hours = float(remaining["count"].sum())
            row = {
                "STN": stn,
                "EXCLUDED_QUARTER": excluded_quarter,
                "LOQ_STATUS": "no_remaining_data",
                "REMAINING_PAIRED_HOURS": (
                    int(paired_hours) if paired_hours.is_integer() else paired_hours
                ),
                "ANNUAL_DELTA_ABS_BIAS": annual_effect,
                "ANNUAL_POINT_CLASS": annual_point_class,
                "ANNUAL_SIGNIFICANCE_CLASS_REFERENCE": annual_row.SIGNIFICANCE_CLASS,
                "REMAINING_BIAS_none": np.nan,
                "REMAINING_BIAS_fixed": np.nan,
                "REMAINING_DELTA_ABS_BIAS": np.nan,
                "REMAINING_POINT_CLASS": pd.NA,
                "SHIFT_FROM_ANNUAL": np.nan,
                "ABS_SHIFT_FROM_ANNUAL": np.nan,
                "RAW_SIGN_CHANGED": pd.NA,
                "POINT_CLASS_CHANGED": pd.NA,
                "IS_MAX_ABS_INFLUENCE_QUARTER": pd.NA,
            }
            if paired_hours > 0:
                bias_none = float(remaining["sum_none"].sum() / paired_hours)
                bias_fixed = float(remaining["sum_fixed"].sum() / paired_hours)
                remaining_effect = abs(bias_fixed) - abs(bias_none)
                remaining_point_class = _point_class(remaining_effect)
                shift = remaining_effect - annual_effect
                row.update(
                    {
                        "LOQ_STATUS": "eligible",
                        "REMAINING_BIAS_none": bias_none,
                        "REMAINING_BIAS_fixed": bias_fixed,
                        "REMAINING_DELTA_ABS_BIAS": remaining_effect,
                        "REMAINING_POINT_CLASS": remaining_point_class,
                        "SHIFT_FROM_ANNUAL": shift,
                        "ABS_SHIFT_FROM_ANNUAL": abs(shift),
                        "RAW_SIGN_CHANGED": bool(
                            np.sign(remaining_effect) != np.sign(annual_effect)
                        ),
                        "POINT_CLASS_CHANGED": bool(
                            remaining_point_class != annual_point_class
                        ),
                    }
                )
            rows.append(row)

    result = pd.DataFrame(rows, columns=LEAVE_ONE_QUARTER_COLUMNS)
    for stn, indices in result.groupby("STN", sort=False).groups.items():
        station_indices = list(indices)
        eligible_indices = [
            index
            for index in station_indices
            if result.at[index, "LOQ_STATUS"] == "eligible"
            and np.isfinite(result.at[index, "ABS_SHIFT_FROM_ANNUAL"])
        ]
        if eligible_indices:
            maximum = max(
                float(result.at[index, "ABS_SHIFT_FROM_ANNUAL"])
                for index in eligible_indices
            )
            for index in eligible_indices:
                result.at[index, "IS_MAX_ABS_INFLUENCE_QUARTER"] = bool(
                    float(result.at[index, "ABS_SHIFT_FROM_ANNUAL"]) == maximum
                )
    for column in [
        "RAW_SIGN_CHANGED",
        "POINT_CLASS_CHANGED",
        "IS_MAX_ABS_INFLUENCE_QUARTER",
    ]:
        result[column] = pd.array(result[column], dtype="boolean")
    return result


def plot_leave_one_quarter_out(loq: pd.DataFrame, output_path) -> Path:
    """분기 제외에 따른 annual Δ|BIAS| shift 분포를 color overview로 그린다."""
    import matplotlib.pyplot as plt

    required = {
        "STN",
        "EXCLUDED_QUARTER",
        "LOQ_STATUS",
        "SHIFT_FROM_ANNUAL",
        "ABS_SHIFT_FROM_ANNUAL",
    }
    missing = required.difference(loq.columns)
    if missing:
        raise ValueError(f"leave-one-quarter-out table missing columns: {sorted(missing)}")
    frame = loq.loc[loq["LOQ_STATUS"].eq("eligible")].copy()
    for column in ["SHIFT_FROM_ANNUAL", "ABS_SHIFT_FROM_ANNUAL"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["SHIFT_FROM_ANNUAL", "ABS_SHIFT_FROM_ANNUAL"]].to_numpy(float)).all():
        raise ValueError("eligible leave-one-quarter-out shifts must be finite")
    if not set(frame["EXCLUDED_QUARTER"]).issubset(QUARTER_ORDER):
        raise ValueError("excluded quarter must use the Q1–Q4 domain")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _quarter_plot_font()
    colors = ["#2878B5", "#43A047", "#F0AD4E", "#D9534F"]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    try:
        groups = [
            frame.loc[frame["EXCLUDED_QUARTER"].eq(quarter), "SHIFT_FROM_ANNUAL"].to_numpy(float)
            for quarter in QUARTER_ORDER
        ]
        nonempty = [(index, values) for index, values in enumerate(groups) if len(values)]
        if nonempty:
            positions = [index + 1 for index, _ in nonempty]
            boxes = ax.boxplot(
                [values for _, values in nonempty],
                positions=positions,
                widths=0.48,
                patch_artist=True,
                showfliers=False,
            )
            for box, (index, _) in zip(boxes["boxes"], nonempty):
                box.set_facecolor(colors[index])
                box.set_alpha(0.42)
            for index, values in nonempty:
                offsets = (
                    np.linspace(-0.18, 0.18, len(values))
                    if len(values) > 1
                    else np.array([0.0])
                )
                ax.scatter(
                    index + 1 + offsets,
                    values,
                    s=8,
                    alpha=0.20,
                    color=colors[index],
                    rasterized=True,
                )
        ax.axhline(0.0, color="#444444", lw=0.9)
        ax.set_xticks(range(1, 5), QUARTER_ORDER)
        ax.set_xlabel("제외한 2024년 달력상 분기", fontproperties=font)
        ax.set_ylabel("9개월 Δ|BIAS| - 연간 Δ|BIAS| (°C)", fontproperties=font)
        ax.set_title("분기 제외 영향도 분포", fontproperties=font)
        ax.grid(axis="y", alpha=0.2)
        fig.text(
            0.5,
            0.01,
            "개별 관측소 값의 조회 정본: leave_one_quarter_out.csv",
            ha="center",
            fontproperties=font,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(path, dpi=190, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_runner_inputs(
    annual_results: pd.DataFrame,
    daily: pd.DataFrame,
    expected_station_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    expected_station_count = _positive_integer(expected_station_count, "expected_station_count")
    required = {
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "SIGNIFICANCE_CLASS",
    }
    if not isinstance(annual_results, pd.DataFrame):
        raise ValueError("annual_results must be a pandas DataFrame")
    missing = required.difference(annual_results.columns)
    if missing:
        raise ValueError(f"annual results missing columns: {sorted(missing)}")
    annual = annual_results.copy(deep=True).reset_index(drop=True)
    annual["STN"] = annual["STN"].astype(str)
    if annual["STN"].duplicated().any():
        raise ValueError("annual results contain duplicate STN rows")
    if len(annual) != expected_station_count:
        raise AssertionError(
            f"expected {expected_station_count} annual stations, received {len(annual)}"
        )
    _validate_significance_classes(annual)
    bias_none = _finite_numeric(annual, "BIAS_none").to_numpy(float)
    bias_fixed = _finite_numeric(annual, "BIAS_fixed").to_numpy(float)
    effect = _finite_numeric(annual, "DELTA_ABS_BIAS").to_numpy(float)
    expected_effect = np.abs(bias_fixed) - np.abs(bias_none)
    if not np.allclose(effect, expected_effect, rtol=0.0, atol=1e-12):
        raise AssertionError("annual DELTA_ABS_BIAS does not match the two annual BIAS values")

    station_ids = annual["STN"].tolist()
    prepared_daily = _prepare_quarterly_daily(daily, stations=station_ids)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    matrices = build_daily_matrices(prepared_daily, station_ids, dates)
    annual_validation = validate_annual_against_existing(matrices, annual)
    return annual, prepared_daily, matrices, annual_validation


def _build_followup_tables(
    annual: pd.DataFrame,
    daily: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    station_ids = annual["STN"].astype(str).tolist()
    quarterly_7 = bootstrap_quarterly_effects(
        daily,
        n_boot=n_boot,
        block_days=7,
        seed=seed,
        stations=station_ids,
    )
    tables = {
        "equivalent_bias_transition.csv": build_equivalent_bias_transition(annual),
        "delta_z_category_rates.csv": build_delta_z_category_rates(annual),
        "quarterly_station_effects.csv": quarterly_7,
        "quarterly_network_summary.csv": summarize_quarterly_network(quarterly_7),
        "quarterly_station_reversals.csv": build_quarterly_reversals(quarterly_7, annual),
        "leave_one_quarter_out.csv": build_leave_one_quarter_out(
            daily, annual, stations=station_ids
        ),
        "quarterly_block_sensitivity.csv": build_quarterly_block_sensitivity(
            daily=daily,
            n_boot=n_boot,
            seed=seed,
            primary_result=quarterly_7,
            stations=station_ids,
        ),
    }
    return _order_followup_tables(tables, station_ids)


def _order_followup_tables(
    tables: dict[str, pd.DataFrame],
    station_ids: list[str],
) -> dict[str, pd.DataFrame]:
    if set(tables) != set(CSV_COLUMN_ORDERS):
        raise AssertionError("analysis table names do not match the seven CSV contracts")
    station_rank = {str(stn): index for index, stn in enumerate(station_ids)}
    quarter_rank = {quarter: index for index, quarter in enumerate(QUARTER_ORDER)}
    ordered = {}
    for name, frame in tables.items():
        expected_columns = list(CSV_COLUMN_ORDERS[name])
        if set(frame.columns) != set(expected_columns) or len(frame.columns) != len(expected_columns):
            raise AssertionError(f"{name} columns do not match its exact schema")
        result = frame.loc[:, expected_columns].copy()
        if name == "equivalent_bias_transition.csv":
            before_rank = {value: index for index, value in enumerate(EQUIVALENT_LEVEL_ORDER)}
            result["_A"] = result["BEFORE_ABS_BIAS_LEVEL"].map(before_rank)
            result["_B"] = result["AFTER_ABS_BIAS_LEVEL"].map(before_rank)
            result = result.sort_values(["_A", "_B"], kind="stable").drop(columns=["_A", "_B"])
        elif name == "delta_z_category_rates.csv":
            bin_rank = {value: index for index, value in enumerate(DELTA_Z_BIN_ORDER)}
            class_rank = {value: index for index, value in enumerate(CLASS_ORDER)}
            result["_A"] = result["DELTA_Z_BIN"].map(bin_rank)
            result["_B"] = result["SIGNIFICANCE_CLASS"].map(class_rank)
            result = result.sort_values(["_A", "_B"], kind="stable").drop(columns=["_A", "_B"])
        elif name in {
            "quarterly_station_effects.csv",
            "leave_one_quarter_out.csv",
        }:
            quarter_column = "QUARTER" if "QUARTER" in result else "EXCLUDED_QUARTER"
            result["_A"] = result["STN"].astype(str).map(station_rank)
            result["_B"] = result[quarter_column].map(quarter_rank)
            if result[["_A", "_B"]].isna().any().any():
                raise AssertionError(f"{name} contains a key outside the canonical universe")
            result = result.sort_values(["_A", "_B"], kind="stable").drop(columns=["_A", "_B"])
        elif name == "quarterly_station_reversals.csv":
            result["_A"] = result["STN"].astype(str).map(station_rank)
            if result["_A"].isna().any():
                raise AssertionError(f"{name} contains a station outside the canonical universe")
            result = result.sort_values("_A", kind="stable").drop(columns="_A")
        elif name == "quarterly_network_summary.csv":
            result["_A"] = result["QUARTER"].map(quarter_rank)
            result = result.sort_values("_A", kind="stable").drop(columns="_A")
        elif name == "quarterly_block_sensitivity.csv":
            block_rank = {3: 0, 7: 1, 14: 2}
            result["_A"] = result["QUARTER"].map(quarter_rank)
            result["_B"] = result["BLOCK_DAYS"].map(block_rank)
            result = result.sort_values(["_A", "_B"], kind="stable").drop(columns=["_A", "_B"])
        ordered[name] = result.reset_index(drop=True).loc[:, expected_columns]
    return ordered


def _assert_close(actual, expected, message: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(message)


def _finite_validation_float(value, message: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(message) from error
    if not np.isfinite(number):
        raise AssertionError(message)
    return number


def _validate_followup_tables(
    annual: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    n_boot: int,
) -> None:
    if set(tables) != set(CSV_COLUMN_ORDERS):
        raise AssertionError("analysis tables must contain exactly the seven contracted CSVs")
    for name, columns in CSV_COLUMN_ORDERS.items():
        if tables[name].columns.tolist() != list(columns):
            raise AssertionError(f"{name} column order changed")

    station_ids = annual["STN"].astype(str).tolist()
    station_set = set(station_ids)
    n_stations = len(station_ids)
    official_counts = (
        annual["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    )

    transition = tables["equivalent_bias_transition.csv"]
    expected_transition_keys = {
        (before, after)
        for before in EQUIVALENT_LEVEL_ORDER
        for after in EQUIVALENT_LEVEL_ORDER
    }
    if len(transition) != 9 or set(
        map(tuple, transition[["BEFORE_ABS_BIAS_LEVEL", "AFTER_ABS_BIAS_LEVEL"]].to_numpy())
    ) != expected_transition_keys:
        raise AssertionError("equivalent transition must preserve the complete 3×3 grid")
    equivalent_n = int(official_counts["practically_equivalent"])
    if int(transition["N"].sum()) != equivalent_n:
        raise AssertionError("equivalent transition denominator does not match the official class")
    expected_fraction = (
        transition["N"].to_numpy(float) / equivalent_n
        if equivalent_n
        else np.zeros(len(transition), dtype=float)
    )
    if not np.allclose(
        transition["FRACTION_OF_EQUIVALENT"].to_numpy(float),
        expected_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("equivalent transition percentages use the wrong denominator")

    delta_z = tables["delta_z_category_rates.csv"]
    expected_delta_keys = {
        (delta_bin, label) for delta_bin in DELTA_Z_BIN_ORDER for label in CLASS_ORDER
    }
    if len(delta_z) != 20 or set(
        map(tuple, delta_z[["DELTA_Z_BIN", "SIGNIFICANCE_CLASS"]].to_numpy())
    ) != expected_delta_keys:
        raise AssertionError("ΔZ category table must preserve the complete 5×4 grid")
    if int(delta_z["N"].sum()) != n_stations:
        raise AssertionError("ΔZ category counts do not sum to the annual station count")
    for delta_bin in DELTA_Z_BIN_ORDER:
        group = delta_z.loc[delta_z["DELTA_Z_BIN"].eq(delta_bin)]
        totals = group["BIN_TOTAL"].unique()
        if len(totals) != 1 or int(group["N"].sum()) != int(totals[0]):
            raise AssertionError("ΔZ within-bin counts do not match BIN_TOTAL")
        denominator = int(totals[0])
        expected = group["N"].to_numpy(float) / denominator if denominator else np.zeros(4)
        if not np.allclose(group["FRACTION_WITHIN_BIN"], expected, rtol=0.0, atol=1e-12):
            raise AssertionError("ΔZ within-bin percentages use the wrong denominator")
    delta_class_counts = delta_z.groupby("SIGNIFICANCE_CLASS", observed=False)["N"].sum()
    for label in CLASS_ORDER:
        if int(delta_class_counts.get(label, 0)) != int(official_counts[label]):
            raise AssertionError("ΔZ class totals do not match the annual official classes")

    quarterly = tables["quarterly_station_effects.csv"]
    expected_keys = {(stn, quarter) for stn in station_ids for quarter in QUARTER_ORDER}
    actual_keys = set(map(tuple, quarterly[["STN", "QUARTER"]].astype(str).to_numpy()))
    if len(quarterly) != n_stations * 4 or actual_keys != expected_keys:
        raise AssertionError("quarterly station table must be the complete annual STN×Q1–Q4 skeleton")
    eligible = _strict_boolean_mask(quarterly, "COVERAGE_ELIGIBLE")
    allowed_reasons = {
        "eligible",
        "no_data",
        "missing_months",
        "below_85pct",
        "missing_months_and_below_85pct",
    }
    if not set(quarterly["COVERAGE_REASON"]).issubset(allowed_reasons):
        raise AssertionError("quarterly coverage reason is outside the fixed domain")
    if not quarterly.loc[eligible, "COVERAGE_REASON"].eq("eligible").all():
        raise AssertionError("eligible quarterly rows must use COVERAGE_REASON=eligible")
    if quarterly.loc[~eligible, "COVERAGE_REASON"].eq("eligible").any():
        raise AssertionError("ineligible quarterly rows cannot use COVERAGE_REASON=eligible")
    eligible_values = [
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "CI95_LOW",
        "CI95_HIGH",
        "BOOT_BLOCK_DAYS",
        "BOOT_REPLICATES",
        "BOOT_SEED",
    ]
    if eligible.any() and not np.isfinite(
        quarterly.loc[eligible, eligible_values].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ).all():
        raise AssertionError("eligible quarterly rows must have finite BIAS, CI, and bootstrap metadata")
    if eligible.any():
        if not quarterly.loc[eligible, "BOOT_BLOCK_DAYS"].eq(7).all():
            raise AssertionError("primary quarterly rows must use a 7-day block")
        if not quarterly.loc[eligible, "BOOT_REPLICATES"].eq(n_boot).all():
            raise AssertionError("primary quarterly rows must use the requested bootstrap count")
        if quarterly.loc[eligible, ["POINT_CLASS", "CI_POSITION"]].isna().any().any():
            raise AssertionError("eligible quarterly rows require point and CI classes")
    derived_columns = [
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
        "POINT_CLASS",
        "CI95_LOW",
        "CI95_HIGH",
        "CI_POSITION",
        "BOOT_BLOCK_DAYS",
        "BOOT_REPLICATES",
        "BOOT_SEED",
    ]
    if (~eligible).any() and not quarterly.loc[~eligible, derived_columns].isna().all().all():
        raise AssertionError("ineligible quarterly rows must keep all derived values as NA")

    network = tables["quarterly_network_summary.csv"]
    if len(network) != 4 or network["QUARTER"].tolist() != QUARTER_ORDER:
        raise AssertionError("quarterly network summary must contain Q1–Q4 exactly once")
    eligible_pairs = quarterly.loc[eligible, ["STN", "QUARTER"]]
    eligible_counts = eligible_pairs.groupby("QUARTER", observed=False).size()
    common_counts = eligible_pairs.groupby("STN")["QUARTER"].nunique()
    common_stations = set(common_counts.index[common_counts.eq(4)])
    network_by_quarter = network.set_index("QUARTER")
    if not network_by_quarter.index.is_unique:
        raise AssertionError("quarterly network summary requires unique quarter keys")
    for quarter in QUARTER_ORDER:
        row = network_by_quarter.loc[quarter]
        quarter_mask = eligible & quarterly["QUARTER"].eq(quarter)
        group = quarterly.loc[quarter_mask]
        common = group.loc[group["STN"].astype(str).isin(common_stations)]
        expected_n = int(eligible_counts.get(quarter, 0))
        for field, expected in [
            ("N_ELIGIBLE", expected_n),
            ("COMMON_COHORT_N", len(common_stations)),
        ]:
            try:
                actual = float(row[field])
            except (TypeError, ValueError) as error:
                raise AssertionError(f"network {field} must be numeric") from error
            if not np.isfinite(actual) or not actual.is_integer() or int(actual) != expected:
                raise AssertionError(f"network {field} does not match the quarterly source")

        for prefix, source in [("", group), ("COMMON_", common)]:
            effects = pd.to_numeric(source["DELTA_ABS_BIAS"], errors="raise").to_numpy(float)
            denominator = len(effects)
            expected_statistics = {
                f"{prefix}MEDIAN_DELTA_ABS_BIAS": (
                    float(np.median(effects)) if denominator else np.nan
                ),
                f"{prefix}Q25_DELTA_ABS_BIAS": (
                    float(np.quantile(effects, 0.25)) if denominator else np.nan
                ),
                f"{prefix}Q75_DELTA_ABS_BIAS": (
                    float(np.quantile(effects, 0.75)) if denominator else np.nan
                ),
            }
            for field, expected in expected_statistics.items():
                actual = row[field]
                if denominator:
                    actual_number = _finite_validation_float(
                        actual,
                        f"network {field} must be finite for a non-empty cohort",
                    )
                    if not np.isclose(
                        actual_number, expected, rtol=0.0, atol=1e-12
                    ):
                        raise AssertionError(
                            f"network {field} does not match the quarterly source"
                        )
                elif not pd.isna(actual):
                    raise AssertionError(f"network empty cohort requires {field}=NA")

            classes = pd.Series(
                [_point_class(value) for value in effects], dtype="object"
            )
            for point_class in POINT_CLASS_ORDER:
                suffix = point_class.upper()
                expected_count = int(classes.eq(point_class).sum())
                count_field = f"{prefix}N_POINT_{suffix}"
                fraction_field = f"{prefix}FRAC_POINT_{suffix}"
                try:
                    actual_count = float(row[count_field])
                except (TypeError, ValueError) as error:
                    raise AssertionError(f"network {count_field} must be numeric") from error
                if (
                    not np.isfinite(actual_count)
                    or not actual_count.is_integer()
                    or int(actual_count) != expected_count
                ):
                    raise AssertionError(
                        f"network {count_field} does not match the quarterly source"
                    )
                actual_fraction = row[fraction_field]
                if denominator:
                    expected_fraction = expected_count / denominator
                    actual_fraction_number = _finite_validation_float(
                        actual_fraction,
                        f"network {fraction_field} must be finite for a non-empty cohort",
                    )
                    if not np.isclose(
                        actual_fraction_number,
                        expected_fraction,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise AssertionError(
                            f"network {fraction_field} does not equal count/denominator"
                        )
                elif not pd.isna(actual_fraction):
                    raise AssertionError(
                        f"network empty cohort requires {fraction_field}=NA"
                    )

    reversals = tables["quarterly_station_reversals.csv"]
    if len(reversals) != n_stations or set(reversals["STN"].astype(str)) != station_set:
        raise AssertionError("quarterly reversal table must retain every annual station")
    loq = tables["leave_one_quarter_out.csv"]
    loq_keys = set(map(tuple, loq[["STN", "EXCLUDED_QUARTER"]].astype(str).to_numpy()))
    if len(loq) != n_stations * 4 or loq_keys != expected_keys:
        raise AssertionError("leave-one-quarter-out table must be the complete annual STN×Q1–Q4 skeleton")
    if not set(loq["LOQ_STATUS"]).issubset({"eligible", "no_remaining_data"}):
        raise AssertionError("leave-one-quarter-out status is outside the fixed domain")
    loq_eligible = loq["LOQ_STATUS"].eq("eligible")
    loq_numeric = [
        "REMAINING_BIAS_none",
        "REMAINING_BIAS_fixed",
        "REMAINING_DELTA_ABS_BIAS",
        "SHIFT_FROM_ANNUAL",
        "ABS_SHIFT_FROM_ANNUAL",
    ]
    if loq_eligible.any() and not np.isfinite(
        loq.loc[loq_eligible, loq_numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ).all():
        raise AssertionError("eligible leave-one-quarter-out rows must have finite derived values")
    if loq_eligible.any() and loq.loc[
        loq_eligible,
        [
            "REMAINING_POINT_CLASS",
            "RAW_SIGN_CHANGED",
            "POINT_CLASS_CHANGED",
            "IS_MAX_ABS_INFLUENCE_QUARTER",
        ],
    ].isna().any().any():
        raise AssertionError("eligible leave-one-quarter-out rows require all class and flag values")
    loq_derived = loq_numeric + [
        "REMAINING_POINT_CLASS",
        "RAW_SIGN_CHANGED",
        "POINT_CLASS_CHANGED",
        "IS_MAX_ABS_INFLUENCE_QUARTER",
    ]
    if (~loq_eligible).any() and not loq.loc[~loq_eligible, loq_derived].isna().all().all():
        raise AssertionError("leave-one-quarter-out rows without remaining data require derived NA")

    sensitivity = tables["quarterly_block_sensitivity.csv"]
    expected_sensitivity_keys = {
        (quarter, block) for quarter in QUARTER_ORDER for block in (3, 7, 14)
    }
    if len(sensitivity) != 12 or set(
        map(tuple, sensitivity[["QUARTER", "BLOCK_DAYS"]].to_numpy())
    ) != expected_sensitivity_keys:
        raise AssertionError("quarterly block sensitivity must preserve the 4×3 grid")
    if not sensitivity["N_BOOT"].eq(n_boot).all():
        raise AssertionError("quarterly block sensitivity uses the wrong bootstrap count")
    for row in sensitivity.itertuples(index=False):
        expected_n = int(eligible_counts.get(row.QUARTER, 0))
        try:
            actual_n = float(row.N_ELIGIBLE)
            agree_n = float(row.CI_POSITION_AGREE_N)
        except (TypeError, ValueError) as error:
            raise AssertionError("sensitivity counts must be numeric") from error
        if (
            not np.isfinite(actual_n)
            or not actual_n.is_integer()
            or int(actual_n) != expected_n
        ):
            raise AssertionError(
                "sensitivity eligible set changed across block lengths"
            )
        if (
            not np.isfinite(agree_n)
            or not agree_n.is_integer()
            or not 0 <= int(agree_n) <= expected_n
        ):
            raise AssertionError(
                "sensitivity agreement count must stay between zero and N_ELIGIBLE"
            )

        summary_values = {
            "MEDIAN_CI95_LOW": row.MEDIAN_CI95_LOW,
            "MEDIAN_CI95_HIGH": row.MEDIAN_CI95_HIGH,
            "MEDIAN_CI95_WIDTH": row.MEDIAN_CI95_WIDTH,
            "CI_POSITION_AGREE_FRACTION": row.CI_POSITION_AGREE_FRACTION,
        }
        if expected_n:
            finite_summaries = {
                name: _finite_validation_float(
                    value,
                    "sensitivity non-empty rows require finite CI and agreement summaries",
                )
                for name, value in summary_values.items()
            }
            if finite_summaries["MEDIAN_CI95_LOW"] > finite_summaries["MEDIAN_CI95_HIGH"]:
                raise AssertionError("sensitivity median CI low cannot exceed CI high")
            if finite_summaries["MEDIAN_CI95_WIDTH"] < 0:
                raise AssertionError("sensitivity median CI width cannot be negative")
            expected_fraction = int(agree_n) / expected_n
            if not np.isclose(
                finite_summaries["CI_POSITION_AGREE_FRACTION"],
                expected_fraction,
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(
                    "sensitivity agreement fraction does not equal count/denominator"
                )
        else:
            if int(agree_n) != 0:
                raise AssertionError(
                    "sensitivity empty rows require zero agreement count"
                )
            if not all(pd.isna(value) for value in summary_values.values()):
                raise AssertionError(
                    "sensitivity empty rows require NA CI and agreement summaries"
                )

        if int(row.BLOCK_DAYS) == 7:
            if int(agree_n) != expected_n:
                raise AssertionError(
                    "7-day sensitivity row does not reuse the primary eligible set"
                )
            if expected_n and not np.isclose(
                float(row.CI_POSITION_AGREE_FRACTION),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(
                    "7-day sensitivity row must agree exactly with the primary result"
                )
            primary_group = quarterly.loc[
                eligible & quarterly["QUARTER"].eq(row.QUARTER)
            ]
            expected_low = float(primary_group["CI95_LOW"].median()) if expected_n else np.nan
            expected_high = float(primary_group["CI95_HIGH"].median()) if expected_n else np.nan
            expected_width = (
                float((primary_group["CI95_HIGH"] - primary_group["CI95_LOW"]).median())
                if expected_n
                else np.nan
            )
            for actual, expected, name in [
                (row.MEDIAN_CI95_LOW, expected_low, "CI low"),
                (row.MEDIAN_CI95_HIGH, expected_high, "CI high"),
                (row.MEDIAN_CI95_WIDTH, expected_width, "CI width"),
            ]:
                if expected_n and not np.isclose(
                    actual, expected, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        f"7-day sensitivity {name} changed from the primary table"
                    )


def validate_production_acceptance(
    annual_results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    annual_validation: dict[str, float],
) -> bool:
    """실제 529개 정본 실행에서만 적용하는 고정 acceptance 계약을 검사한다."""
    expected_counts = {
        "meaningful_improvement": 221,
        "meaningful_worsening": 109,
        "practically_equivalent": 115,
        "uncertain": 84,
    }
    annual = annual_results.copy()
    annual["STN"] = annual["STN"].astype(str)
    actual_counts = (
        annual["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0).astype(int).to_dict()
    )
    if len(annual) != 529 or actual_counts != expected_counts:
        raise AssertionError(
            f"production official class counts changed: expected={expected_counts}, actual={actual_counts}"
        )
    sensitivity_boot = pd.to_numeric(
        tables["quarterly_block_sensitivity.csv"]["N_BOOT"], errors="raise"
    ).unique()
    if len(sensitivity_boot) != 1 or int(sensitivity_boot[0]) != DEFAULT_N_BOOT:
        raise AssertionError(
            f"production sensitivity must use exactly {DEFAULT_N_BOOT} bootstrap replicates"
        )
    _validate_followup_tables(annual, tables, n_boot=int(sensitivity_boot[0]))
    expected_rows = {
        "equivalent_bias_transition.csv": 9,
        "delta_z_category_rates.csv": 20,
        "quarterly_station_effects.csv": 2116,
        "quarterly_network_summary.csv": 4,
        "quarterly_station_reversals.csv": 529,
        "leave_one_quarter_out.csv": 2116,
        "quarterly_block_sensitivity.csv": 12,
    }
    actual_rows = {name: len(frame) for name, frame in tables.items()}
    if actual_rows != expected_rows:
        raise AssertionError(f"production CSV row counts changed: {actual_rows}")
    if int(tables["equivalent_bias_transition.csv"]["N"].sum()) != 115:
        raise AssertionError("production equivalent denominator changed from 115")
    if int(tables["delta_z_category_rates.csv"]["N"].sum()) != 529:
        raise AssertionError("production ΔZ category counts do not sum to 529")
    required_residuals = {"max_abs_bias_none", "max_abs_bias_fixed"}
    if set(annual_validation) != required_residuals:
        raise AssertionError("annual validation keys changed")
    residuals = np.asarray([annual_validation[key] for key in sorted(required_residuals)], dtype=float)
    if not np.isfinite(residuals).all() or np.any(residuals < 0) or np.any(residuals > 1e-10):
        raise AssertionError(f"annual BIAS validation residual is too large: {annual_validation}")
    return True


def _format_value(value: float) -> str:
    return "자료 없음" if not np.isfinite(value) else f"{value:+.3f}°C"


def write_followup_report(
    annual: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_path,
) -> Path:
    """승인된 해석 범위 안에서 실제 계산값을 쉬운 한국어 Markdown으로 요약한다."""
    counts = annual["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    korean_labels = {
        "meaningful_improvement": "유의미한 개선",
        "meaningful_worsening": "유의미한 악화",
        "practically_equivalent": "차이 거의 없음",
        "uncertain": "판단 불확실",
    }
    class_lines = [
        f"- {korean_labels[label]}: {int(counts[label])}개 ({100 * counts[label] / len(annual):.1f}%)"
        for label in CLASS_ORDER
    ]

    transition = tables["equivalent_bias_transition.csv"]
    transition_counts = transition.set_index(
        ["BEFORE_ABS_BIAS_LEVEL", "AFTER_ABS_BIAS_LEVEL"]
    )["N"]
    transition_lines = [
        "| 보정 전 \\ 보정 후 | ≤0.1°C | (0.1, 0.5]°C | >0.5°C |",
        "|---|---:|---:|---:|",
    ]
    for before in EQUIVALENT_LEVEL_ORDER:
        row_counts = [
            int(transition_counts.loc[(before, after)])
            for after in EQUIVALENT_LEVEL_ORDER
        ]
        transition_lines.append(
            f"| {before} | {row_counts[0]} | {row_counts[1]} | {row_counts[2]} |"
        )
    low_low_n = int(transition_counts.loc[("≤0.1°C", "≤0.1°C")])
    high_high_n = int(transition_counts.loc[(">0.5°C", ">0.5°C")])
    same_level_n = sum(
        int(transition_counts.loc[(level, level)]) for level in EQUIVALENT_LEVEL_ORDER
    )
    equivalent_n = int(transition["N"].sum())

    delta_z = tables["delta_z_category_rates.csv"]
    delta_lines = [
        "| 평균 절대 ΔZ | 유의미한 개선 | 유의미한 악화 | 차이 거의 없음 | 판단 불확실 |",
        "|---|---:|---:|---:|---:|",
    ]
    for delta_bin in DELTA_Z_BIN_ORDER:
        group = delta_z.loc[delta_z["DELTA_Z_BIN"].eq(delta_bin)]
        cells = []
        for label in CLASS_ORDER:
            row = group.loc[group["SIGNIFICANCE_CLASS"].eq(label)].iloc[0]
            cells.append(
                f"{int(row['N'])}개 "
                f"({100 * float(row['FRACTION_WITHIN_BIN']):.1f}%)"
            )
        delta_lines.append(
            f"| {delta_bin} | " + " | ".join(cells) + " |"
        )

    network = tables["quarterly_network_summary.csv"]
    quarter_lines = [
        "| 분기 | 공통 관측소 수 | 중앙값 | 가운데 50% 범위 | 개선 | 작은 변화 | 악화 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in network.itertuples(index=False):
        point_cells = []
        for point_class in POINT_CLASS_ORDER:
            suffix = point_class.upper()
            count = int(getattr(row, f"COMMON_N_POINT_{suffix}"))
            fraction = getattr(row, f"COMMON_FRAC_POINT_{suffix}")
            fraction_display = (
                "자료 없음"
                if pd.isna(fraction)
                else f"{100 * float(fraction):.1f}%"
            )
            point_cells.append(f"{count}개 ({fraction_display})")
        quarter_lines.append(
            f"| {row.QUARTER} | {int(row.COMMON_COHORT_N)} | "
            f"{_format_value(float(row.COMMON_MEDIAN_DELTA_ABS_BIAS))} | "
            f"{_format_value(float(row.COMMON_Q25_DELTA_ABS_BIAS))} ~ "
            f"{_format_value(float(row.COMMON_Q75_DELTA_ABS_BIAS))} | "
            + " | ".join(point_cells)
            + " |"
        )
    loq = tables["leave_one_quarter_out.csv"]
    loq_eligible = loq.loc[loq["LOQ_STATUS"].eq("eligible")]
    class_changed = int(loq_eligible["POINT_CLASS_CHANGED"].fillna(False).astype(bool).sum())
    sensitivity = tables["quarterly_block_sensitivity.csv"]
    sensitivity_lines = []
    for block in (3, 14):
        values = sensitivity.loc[
            sensitivity["BLOCK_DAYS"].eq(block), "CI_POSITION_AGREE_FRACTION"
        ].dropna()
        display = "자료 없음" if values.empty else f"{100 * float(values.min()):.1f}% 이상"
        sensitivity_lines.append(f"- {block}일 블록: 7일 결과와 분기별 최소 일치율 {display}")

    text = f"""# 고도보정 BIAS 후속 분석 결과

## 1. 이번에 보완한 질문

- ‘차이 거의 없음’을 보정 전후 BIAS 크기로 다시 나누었다.
- ΔZ 구간마다 공식 네 분류의 개수와 비율을 모두 계산했다.
- 2024년을 서로 겹치지 않는 달력상 분기 Q1–Q4로 나눠 시기별 효과를 탐색했다.
- 한 분기씩 제외한 나머지 9개월 결과도 다시 계산해 특정 시기의 영향이 큰지 확인했다.
- 외부 문헌은 우리 결과와 물리적으로 맞는 설명을 고르는 데 사용했으며, 우리 자료의 인과를 증명하지는 않는다.

## 2. 연간 공식 분류

{chr(10).join(class_lines)}

공식 네 분류는 7일 블록 재표집의 불확실성까지 반영한 연간 판단이다. 반면 분기 비교와 분기 제외 분석의
점추정 세 구간은 효과값이 -0.10°C보다 작은지, -0.10~+0.10°C인지, +0.10°C보다 큰지만 나눈 진단이다.
두 분류를 같은 의미로 바꾸어 쓰지 않았다.

## 3. ‘차이 거의 없음’과 ΔZ 결과

### 차이 거의 없음의 3×3 전체 전이표

{chr(10).join(transition_lines)}

- 낮음→낮음 {low_low_n}개는 보정 전에도 |BIAS|가 0.1°C 이하였고 보정 후에도 낮게 유지됐다.
- 높음→높음 {high_high_n}개는 보정 전후 모두 |BIAS|가 0.5°C보다 커서, 변화량이 작다는 이유만으로 정확하다고 볼 수 없다.
- 대각선, 즉 같은 구간에 남은 경우는 {same_level_n}/{equivalent_n}개였다. 따라서 ‘차이 거의 없음’은 원래 정확한 경우와 원래 오차가 큰 채 남은 경우를 구분해서 봐야 한다.

### ΔZ 5개 구간과 공식 네 분류

{chr(10).join(delta_lines)}

ΔZ가 크면 고도보정이 이동시키는 기온도 커져 개선 가능성이 커질 수 있다. 하지만 실제 기온과 고도의 관계가
고정 감률과 맞지 않으면 큰 ΔZ가 자동으로 개선을 보장하지는 않는다.

## 4. 2024년 달력상 분기 탐색

공통 관측소에 대해 각 분기의 효과 중앙값과 가운데 50% 범위, 점추정 세 구간을 함께 나타냈다.

{chr(10).join(quarter_lines)}

이 결과는 여러 해의 계절 평균이 아니라 2024년 한 해 안의 네 구간을 비교한 탐색 결과다. 분기 자료의 85%와
세 달이 모두 있는 경우만 수치 해석에 포함했다. 85%는 통계적 유의성 기준이나 선행연구의 보편 기준이 아니라,
자료가 부족한 분기를 충분한 분기처럼 다루지 않기 위해 이번 분석에서 정한 기술 기준이다.

## 5. 7일 재표집과 분기 제외 진단

- 유효한 분기 제외 계산 {len(loq_eligible)}건 가운데 점추정 세 구간이 연간 결과와 달라진 경우는 {class_changed}건이었다.
{chr(10).join(sensitivity_lines)}

7일은 가까운 날짜의 연속성을 완전히 깨지 않기 위한 월별 층화 고정길이 원형 블록 재표집 부트스트랩의 길이다.
계절을 나누는 기간이 아니며, 최적이라고 확인된 값은 아니다. 그래서 3일과 14일 결과를 함께 확인했다.

## 6. 낮과 밤의 물리적 해석 범위

기존 시간대 분석의 09–17시와 21–05시는 일출·일몰을 날짜별로 계산한 구간이 아니라 고정된 시계 시간이다.
산악지역 선행관측은 낮의 혼합과 밤의 복사냉각·찬 공기 배수·역전 때문에 기온과 고도의 관계가 달라질 수 있음을
보여준다. 따라서 우리 시간대 패턴은 알려진 물리 과정과 일치한다고 설명할 수 있다. 다만 이번 자료에는 일사량,
운량, 풍속, 연직 기온이 없으므로 각 관측소의 원인이 그 과정이었다고 확정하지 않는다.

## 7. 해석 결론

- 고도보정 효과의 차이는 ΔZ의 크기뿐 아니라 기존 BIAS 방향, 시기, 실제 감률과 고정 감률의 차이에 함께 좌우된다.
- 연간 공식 네 분류, 분기 점추정 세 구간, 분기 제외 영향도는 서로 다른 질문에 답하므로 따로 제시해야 한다.
- 문헌은 물리적으로 일관된 설명을 제공하지만, 현재 분석만으로 직접 인과를 확정하지 않는다.
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_csv_deterministic(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        na_rep="",
    )


def _validate_literature_source(literature_source, out_dir) -> tuple[Path, str]:
    source = Path(literature_source).resolve()
    output = Path(out_dir).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("literature source must be a non-empty regular file")
    if source == output or output in source.parents:
        raise ValueError("literature source must remain outside the replaceable output directory")
    return source, _sha256_file(source)


def _make_staging_directory(out_dir) -> Path:
    output = Path(out_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    return staging


def _publish_staging(staging, out_dir) -> None:
    """검증된 same-parent staging을 rollback 가능한 directory replacement로 게시한다."""
    staging_path = Path(staging).resolve()
    output = Path(out_dir).resolve()
    if not staging_path.is_dir() or staging_path.parent != output.parent:
        raise ValueError("staging must be an existing sibling directory of out_dir")
    if output.exists() and not output.is_dir():
        raise ValueError("out_dir exists but is not a directory")
    backup = None
    if output.exists():
        backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
        os.replace(output, backup)
    try:
        os.replace(staging_path, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as error:
            warnings.warn(
                f"new output was published, but backup cleanup failed: {backup}: {error}",
                RuntimeWarning,
            )


def _validate_staging_artifacts(staging: Path) -> None:
    import matplotlib.image as mpimg

    children = list(staging.iterdir())
    actual = {path.name for path in children if path.is_file()}
    if len(children) != len(actual) or actual != set(ANALYSIS_ARTIFACTS):
        raise AssertionError(
            f"artifact mismatch: expected={sorted(ANALYSIS_ARTIFACTS)}, actual={sorted(actual)}"
        )
    if any(path.stat().st_size == 0 for path in children):
        raise AssertionError("empty analysis artifact generated")
    for filename in PNG_ARTIFACTS:
        image = mpimg.imread(staging / filename)
        if image.ndim < 2 or image.size == 0:
            raise AssertionError(f"PNG cannot be decoded: {filename}")
    report = (staging / "bias_feedback_followup_report.md").read_text(encoding="utf-8").lower()
    for forbidden in ("dz_geom", "stationary bootstrap", "iqr"):
        if forbidden in report:
            raise AssertionError(f"forbidden user-facing label in report: {forbidden}")


def run_followup_analysis(
    annual_results: pd.DataFrame,
    daily: pd.DataFrame,
    literature_source,
    out_dir,
    expected_station_count: int,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    enforce_production_acceptance: bool = False,
) -> dict:
    """주입된 정본 자료만 사용해 14개 후속 산출물을 검증·게시한다."""
    n_boot = _positive_integer(n_boot, "n_boot")
    seed = _positive_integer(seed, "seed")
    source, source_hash = _validate_literature_source(literature_source, out_dir)
    annual, prepared_daily, matrices, annual_validation = _canonical_runner_inputs(
        annual_results,
        daily,
        expected_station_count=expected_station_count,
    )
    tables = _build_followup_tables(annual, prepared_daily, n_boot=n_boot, seed=seed)
    _validate_followup_tables(annual, tables, n_boot=n_boot)
    if enforce_production_acceptance:
        validate_production_acceptance(annual, tables, annual_validation)

    staging = _make_staging_directory(out_dir)
    try:
        for filename, frame in tables.items():
            _write_csv_deterministic(frame, staging / filename)
        plot_equivalent_before_after(annual, staging / "bias_equivalent_before_after.png")
        plot_quarterly_effect(
            tables["quarterly_station_effects.csv"], staging / "bias_quarterly_effect.png"
        )
        plot_quarterly_heatmap(
            tables["quarterly_station_effects.csv"], staging / "bias_quarterly_heatmap.png"
        )
        plot_leave_one_quarter_out(
            tables["leave_one_quarter_out.csv"], staging / "bias_leave_one_quarter_out.png"
        )
        literature_copy = staging / "literature_claim_map.md"
        shutil.copyfile(source, literature_copy)
        if _sha256_file(literature_copy) != source_hash:
            raise AssertionError("literature map copy hash does not match its approved source")
        write_followup_report(
            annual,
            tables,
            staging / "bias_feedback_followup_report.md",
        )

        csv_hashes = {
            filename: _sha256_file(staging / filename)
            for filename in sorted(CSV_COLUMN_ORDERS)
        }
        official_counts = (
            annual["SIGNIFICANCE_CLASS"]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
            .astype(int)
        )
        summary = {
            "status": "complete",
            "schema_version": 1,
            "analysis_year": 2024,
            "station_count": int(len(annual)),
            "paired_hours": int(np.asarray(matrices["count"], dtype=float).sum()),
            "practical_delta_c": PRACTICAL_DELTA_C,
            "quarter_coverage_fraction": QUARTER_COVERAGE_FRACTION,
            "primary_block_days": 7,
            "sensitivity_block_days": [3, 7, 14],
            "bootstrap_replicates": int(n_boot),
            "seed": int(seed),
            "official_class_counts": {
                label: int(official_counts[label]) for label in CLASS_ORDER
            },
            "equivalent_station_count": int(official_counts["practically_equivalent"]),
            "annual_validation": {
                key: float(annual_validation[key]) for key in sorted(annual_validation)
            },
            "csv_rows": {filename: int(len(tables[filename])) for filename in sorted(tables)},
            "artifact_names": sorted(ANALYSIS_ARTIFACTS),
            "artifact_hashes": csv_hashes,
            "literature_source_sha256": source_hash,
        }
        if set(summary) != set(SUMMARY_JSON_KEYS):
            raise AssertionError("summary JSON keys changed")
        (staging / "bias_feedback_followup_summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _validate_staging_artifacts(staging)
        _publish_staging(staging, out_dir)
        staging = None
        return summary
    finally:
        if staging is not None and Path(staging).exists():
            shutil.rmtree(staging)


def load_production_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """기존 연간 정본과 read-only SQLite에서 실제 529개 입력을 로드한다."""
    annual = pd.read_csv(
        SOURCE_STATION_RESULTS,
        encoding="utf-8-sig",
        dtype={"STN": str},
    )
    annual["STN"] = annual["STN"].astype(str)
    if len(annual) != 529 or annual["STN"].duplicated().any():
        raise AssertionError("production source must contain exactly 529 unique stations")
    daily = load_daily_paired_errors(annual["STN"].tolist())
    return annual, daily


def _validate_production_out_dir(out_dir) -> Path:
    """실제 실행 출력은 FULLNET 바로 아래의 승인된 leaf 이름만 허용한다."""
    output = Path(out_dir).resolve()
    fullnet = FULLNET.resolve()
    final_name = "bias_feedback_followup"
    acceptance_pattern = re.compile(
        r"bias_feedback_followup_accept_[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
    )
    name_allowed = output.name == final_name or acceptance_pattern.fullmatch(output.name)
    if output.parent != fullnet or not name_allowed:
        raise ValueError(
            "production out_dir must be an approved direct child of FULLNET: "
            "bias_feedback_followup or bias_feedback_followup_accept_<token>"
        )
    if output.exists() and not output.is_dir():
        raise ValueError("production out_dir leaf exists but is not a directory")
    return output


def run_production_analysis(
    out_dir=FOLLOWUP_OUT_DIR,
    literature_source=DEFAULT_LITERATURE_SOURCE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """실제 529개 정본용 진입점. 빠른 주입 시험과 분리한다."""
    if _positive_integer(n_boot, "n_boot") != DEFAULT_N_BOOT:
        raise ValueError(
            f"production acceptance requires exactly {DEFAULT_N_BOOT} bootstrap replicates"
        )
    safe_out_dir = _validate_production_out_dir(out_dir)
    annual, daily = load_production_inputs()
    return run_followup_analysis(
        annual_results=annual,
        daily=daily,
        literature_source=literature_source,
        out_dir=safe_out_dir,
        expected_station_count=529,
        n_boot=n_boot,
        seed=seed,
        enforce_production_acceptance=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="고도보정 BIAS 후속 분석 실행")
    parser.add_argument("--out-dir", type=Path, default=FOLLOWUP_OUT_DIR)
    parser.add_argument("--literature-source", type=Path, default=DEFAULT_LITERATURE_SOURCE)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    summary = run_production_analysis(
        out_dir=args.out_dir,
        literature_source=args.literature_source,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
