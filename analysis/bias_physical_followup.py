"""BIAS 물리요인 후속 분석에 쓰는 부작용 없는 계산 함수.

이 모듈은 파일을 읽거나 쓰지 않는다. 기존 연간 정본의 분류는 입력으로만 받고,
BIAS·기온-고도 기울기·IDW 평균을 명시된 부호와 단위로 다시 계산한다.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from typing import Any

import numpy as np
import pandas as pd


BIAS_LEVELS = ("low", "medium", "high")
DZ_BINS = ("≤50m", "50–100m", "100–200m", "200–300m", ">300m")
WORSENING_MECHANISMS = ("same_direction_worsened", "overshoot_worsened")
SIGNIFICANCE_CLASSES = (
    "meaningful_improvement",
    "meaningful_worsening",
    "practically_equivalent",
    "uncertain",
)

# Student t 0.975 quantiles. 실제 보간 이웃은 최대 10개이므로 df=n-2는 1~8이다.
# 값은 표준 t 분포표의 양측 95% 임계값이다.
_T_975 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118487916806,
    7: 2.3646242510102993,
    8: 2.3060041350333704,
}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"required columns missing: {missing}")


def _finite_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{column} must be finite")
    return values


def _prepare_station_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "SIGNIFICANCE_CLASS",
        "mean_absdz_geom_km",
    ]
    _require_columns(frame, required)
    if frame["STN"].isna().any():
        raise ValueError("STN must not contain missing values")
    if frame["SIGNIFICANCE_CLASS"].isna().any():
        raise ValueError("SIGNIFICANCE_CLASS must not contain missing values")

    out = frame.copy()
    out["STN"] = out["STN"].map(str).str.strip()
    if out["STN"].eq("").any():
        raise ValueError("STN must not contain empty values")
    if out["STN"].duplicated().any():
        raise ValueError("STN must be unique after string normalization")
    invalid_classes = sorted(
        set(out["SIGNIFICANCE_CLASS"].astype(str)) - set(SIGNIFICANCE_CLASSES)
    )
    if invalid_classes:
        raise ValueError(f"unknown SIGNIFICANCE_CLASS values: {invalid_classes}")
    for column in ["BIAS_none", "BIAS_fixed", "mean_absdz_geom_km"]:
        out[column] = _finite_numeric_series(out, column)
    if (out["mean_absdz_geom_km"] < 0).any():
        raise ValueError("mean_absdz_geom_km must be non-negative")
    return out


def _bias_level(values: pd.Series) -> pd.Series:
    absolute = values.abs().to_numpy(float)
    labels = np.select(
        [absolute <= 0.1, absolute <= 0.5],
        ["low", "medium"],
        default="high",
    )
    return pd.Series(labels, index=values.index, dtype="object")


def _median_or_nan(values: pd.Series) -> float:
    return float(values.median()) if len(values) else float("nan")


def build_equivalent_diagnostics(station_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """공식 `practically_equivalent` 관측소의 전후 BIAS 수준을 분리한다.

    반환값은 관측소별 표, 완전한 3×3 전이표, 보정 전·후를 분리한 그룹 요약이다.
    경계는 low ≤0.1°C, medium (0.1, 0.5]°C, high >0.5°C이다.
    """

    frame = _prepare_station_frame(station_results)
    subset = frame.loc[
        frame["SIGNIFICANCE_CLASS"].eq("practically_equivalent")
    ].copy()
    subset["ABS_BIAS_NONE"] = subset["BIAS_none"].abs()
    subset["ABS_BIAS_FIXED"] = subset["BIAS_fixed"].abs()
    subset["CORRECTION_SHIFT"] = subset["BIAS_fixed"] - subset["BIAS_none"]
    subset["ABS_CORRECTION_SHIFT"] = subset["CORRECTION_SHIFT"].abs()
    subset["DELTA_ABS_BIAS"] = subset["ABS_BIAS_FIXED"] - subset["ABS_BIAS_NONE"]
    subset["ABS_DZ_M"] = subset["mean_absdz_geom_km"] * 1000.0
    subset["BIAS_NONE_LEVEL"] = _bias_level(subset["BIAS_none"])
    subset["BIAS_FIXED_LEVEL"] = _bias_level(subset["BIAS_fixed"])

    total = len(subset)
    transition_rows: list[dict[str, Any]] = []
    for before in BIAS_LEVELS:
        for after in BIAS_LEVELS:
            count = int(
                (
                    subset["BIAS_NONE_LEVEL"].eq(before)
                    & subset["BIAS_FIXED_LEVEL"].eq(after)
                ).sum()
            )
            transition_rows.append(
                {
                    "BIAS_NONE_LEVEL": before,
                    "BIAS_FIXED_LEVEL": after,
                    "N": count,
                    "FRACTION_OF_EQUIVALENT": count / total if total else np.nan,
                }
            )
    transition = pd.DataFrame(transition_rows)

    summary_rows: list[dict[str, Any]] = []
    for grouping, level_column in [
        ("before", "BIAS_NONE_LEVEL"),
        ("after", "BIAS_FIXED_LEVEL"),
    ]:
        for level in BIAS_LEVELS:
            group = subset.loc[subset[level_column].eq(level)]
            summary_rows.append(
                {
                    "GROUPING": grouping,
                    "LEVEL": level,
                    "N": int(len(group)),
                    "MEDIAN_ABS_DZ_M": _median_or_nan(group["ABS_DZ_M"]),
                    "MEDIAN_ABS_CORRECTION_C": _median_or_nan(
                        group["ABS_CORRECTION_SHIFT"]
                    ),
                    "MEDIAN_ABS_BIAS_NONE_C": _median_or_nan(group["ABS_BIAS_NONE"]),
                    "MEDIAN_ABS_BIAS_FIXED_C": _median_or_nan(
                        group["ABS_BIAS_FIXED"]
                    ),
                }
            )
    group_summary = pd.DataFrame(summary_rows)

    return {
        "stations": subset.reset_index(drop=True),
        "transition": transition,
        "group_summary": group_summary,
    }


def _delta_z_bin(abs_dz_m: pd.Series) -> pd.Series:
    values = abs_dz_m.to_numpy(float)
    labels = np.select(
        [values <= 50.0, values <= 100.0, values <= 200.0, values <= 300.0],
        DZ_BINS[:4],
        default=DZ_BINS[4],
    )
    return pd.Series(labels, index=abs_dz_m.index, dtype="object")


def _expected_worsening_mechanism(bias_none: float, bias_fixed: float) -> str:
    correction = bias_fixed - bias_none
    product = bias_none * correction
    if product > 0:
        return "same_direction_worsened"
    if product < 0 and abs(correction) > 2.0 * abs(bias_none):
        return "overshoot_worsened"
    raise ValueError("meaningful worsening row does not satisfy an approved mechanism")


def build_worsening_diagnostics(station_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """공식 `meaningful_worsening`을 대수적 기전과 |ΔZ| 구간으로 요약한다."""

    frame = _prepare_station_frame(station_results)
    _require_columns(frame, ["MECHANISM"])
    subset = frame.loc[
        frame["SIGNIFICANCE_CLASS"].eq("meaningful_worsening")
    ].copy()
    if subset["MECHANISM"].isna().any():
        raise ValueError("MECHANISM must not be missing for meaningful worsening")
    invalid = sorted(set(subset["MECHANISM"]) - set(WORSENING_MECHANISMS))
    if invalid:
        raise ValueError(f"unknown worsening mechanisms: {invalid}")

    subset["CORRECTION_SHIFT"] = subset["BIAS_fixed"] - subset["BIAS_none"]
    subset["DELTA_ABS_BIAS"] = subset["BIAS_fixed"].abs() - subset["BIAS_none"].abs()
    if (subset["DELTA_ABS_BIAS"] <= 0).any():
        raise ValueError("meaningful worsening must have positive DELTA_ABS_BIAS")
    expected = [
        _expected_worsening_mechanism(before, after)
        for before, after in zip(subset["BIAS_none"], subset["BIAS_fixed"])
    ]
    if not np.array_equal(np.asarray(expected, dtype=object), subset["MECHANISM"].to_numpy()):
        raise ValueError("MECHANISM is inconsistent with BIAS and correction shift")

    subset["ABS_DZ_M"] = subset["mean_absdz_geom_km"] * 1000.0
    subset["DZ_BIN"] = _delta_z_bin(subset["ABS_DZ_M"])

    rows: list[dict[str, Any]] = []
    for label in DZ_BINS:
        group = subset.loc[subset["DZ_BIN"].eq(label)]
        effects = group["DELTA_ABS_BIAS"]
        rows.append(
            {
                "DZ_BIN": label,
                "N": int(len(group)),
                "SAME_DIRECTION_N": int(
                    group["MECHANISM"].eq("same_direction_worsened").sum()
                ),
                "OVERSHOOT_N": int(
                    group["MECHANISM"].eq("overshoot_worsened").sum()
                ),
                "MEDIAN_DELTA_ABS_BIAS": _median_or_nan(effects),
                "Q25_DELTA_ABS_BIAS": (
                    float(effects.quantile(0.25)) if len(effects) else np.nan
                ),
                "Q75_DELTA_ABS_BIAS": (
                    float(effects.quantile(0.75)) if len(effects) else np.nan
                ),
                "MAX_DELTA_ABS_BIAS": float(effects.max()) if len(effects) else np.nan,
            }
        )
    return {
        "stations": subset.reset_index(drop=True),
        "by_dz": pd.DataFrame(rows),
    }


def _numeric_vector(name: str, values, *, minimum_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < minimum_length:
        raise ValueError(f"{name} must be a one-dimensional vector with length >= {minimum_length}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate_parallel_vectors(*arrays: np.ndarray) -> int:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("input vectors must have equal length")
    return len(arrays[0])


def _validate_target_exclusion(
    n: int,
    *,
    neighbor_ids: Sequence | None,
    target_id: Any | None,
) -> str | None:
    if neighbor_ids is None:
        if target_id is not None:
            raise ValueError("neighbor_ids are required when target_id is provided")
        return None
    ids = list(neighbor_ids)
    if len(ids) != n:
        raise ValueError("neighbor_ids length must match data vectors")
    if any(value is None for value in ids):
        raise ValueError("neighbor_ids must not contain missing values")
    normalized = [str(value).strip() for value in ids]
    if any(value == "" for value in normalized):
        raise ValueError("neighbor_ids must not contain empty values")
    if len(set(normalized)) != n:
        raise ValueError("neighbor_ids must be unique")
    if target_id is not None and str(target_id) in set(normalized):
        raise ValueError("target_id must not appear among neighbors")
    canonical = "".join(
        f"{len(value)}:{value};" for value in sorted(normalized)
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def ols_temperature_elevation_slope(
    elevations_m,
    temperatures_c,
    *,
    neighbor_ids: Sequence | None = None,
    target_id: Any | None = None,
) -> dict[str, float | int]:
    """비가중 OLS 근지면 기온-고도 기울기와 95%·HC3 구간을 계산한다.

    기울기 단위는 °C/km이다. 현재 production 이웃 상한에 맞춰 3~10개만
    허용하며, `n>=6` 연구 품질조건은 분류 함수에서 별도로 적용한다.
    """

    elevation = _numeric_vector("elevations_m", elevations_m, minimum_length=3)
    temperature = _numeric_vector("temperatures_c", temperatures_c, minimum_length=3)
    n = _validate_parallel_vectors(elevation, temperature)
    if n > 10:
        raise ValueError("OLS supports at most 10 exact interpolation neighbors")
    neighbor_fingerprint = _validate_target_exclusion(
        n, neighbor_ids=neighbor_ids, target_id=target_id
    )
    if neighbor_fingerprint is None:
        raise ValueError("neighbor_ids are required to fingerprint the OLS neighbor set")

    if float(elevation.max() - elevation.min()) <= 0:
        raise ValueError("elevation spread must be positive")
    x = elevation / 1000.0
    x_centered = x - x.mean()
    y_centered = temperature - temperature.mean()
    sxx = float(np.dot(x_centered, x_centered))
    if not np.isfinite(sxx) or sxx <= 0:
        raise ValueError("elevation spread must be positive")
    slope = float(np.dot(x_centered, y_centered) / sxx)
    intercept = float(temperature.mean() - slope * x.mean())
    fitted = intercept + slope * x
    residual = temperature - fitted
    sse = float(np.dot(residual, residual))
    degrees_freedom = n - 2
    t_critical = _T_975[degrees_freedom]

    mse = max(sse / degrees_freedom, 0.0)
    standard_error = float(np.sqrt(mse / sxx))
    ci_low = slope - t_critical * standard_error
    ci_high = slope + t_critical * standard_error

    sst = float(np.dot(y_centered, y_centered))
    r_squared = float(1.0 - sse / sst) if sst > 0 else float("nan")

    leverage = 1.0 / n + (x_centered**2) / sxx
    if np.any(leverage >= 1.0):
        raise ValueError("HC3 leverage is not identifiable")
    adjusted_residual = residual / (1.0 - leverage)
    hc3_variance_slope = float(
        np.sum((x_centered**2) * (adjusted_residual**2)) / (sxx**2)
    )
    hc3_standard_error = float(np.sqrt(max(hc3_variance_slope, 0.0)))

    return {
        "n": n,
        "neighbor_fingerprint": neighbor_fingerprint,
        "slope_c_per_km": slope,
        "intercept_c": intercept,
        "standard_error_c_per_km": standard_error,
        "ci95_low_c_per_km": float(ci_low),
        "ci95_high_c_per_km": float(ci_high),
        "r_squared": r_squared,
        "hc3_standard_error_c_per_km": hc3_standard_error,
        "hc3_ci95_low_c_per_km": float(slope - t_critical * hc3_standard_error),
        "hc3_ci95_high_c_per_km": float(slope + t_critical * hc3_standard_error),
        "elevation_span_m": float(elevation.max() - elevation.min()),
    }


def weighted_temperature_elevation_slope(
    elevations_m,
    temperatures_c,
    *,
    distances_km,
    power: float = 2.0,
    neighbor_ids: Sequence | None = None,
    target_id: Any | None = None,
) -> dict[str, Any]:
    """거리 가중 기온-고도 기울기, n_eff, 가중 고도 SD를 계산한다."""

    elevation = _numeric_vector("elevations_m", elevations_m, minimum_length=2)
    temperature = _numeric_vector("temperatures_c", temperatures_c, minimum_length=2)
    n = _validate_parallel_vectors(elevation, temperature)
    neighbor_fingerprint = _validate_target_exclusion(
        n, neighbor_ids=neighbor_ids, target_id=target_id
    )
    if neighbor_fingerprint is None:
        raise ValueError(
            "neighbor_ids are required to fingerprint the weighted neighbor set"
        )
    distance = _numeric_vector("distances_km", distances_km, minimum_length=2)
    _validate_parallel_vectors(elevation, distance)
    if np.any(distance <= 0):
        raise ValueError("weighted slope requires strictly positive distances")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be a finite positive number")
    min_distance = float(distance.min())
    raw_weight = (min_distance / distance) ** float(power)

    weight_sum = float(raw_weight.sum())
    normalized = raw_weight / weight_sum
    x = elevation / 1000.0
    x_mean = float(np.dot(normalized, x))
    y_mean = float(np.dot(normalized, temperature))
    x_centered = x - x_mean
    denominator = float(np.dot(normalized, x_centered**2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("weighted elevation spread must be positive")
    slope = float(np.dot(normalized, x_centered * (temperature - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    n_eff = float(weight_sum**2 / np.dot(raw_weight, raw_weight))
    weighted_sd_m = float(
        np.sqrt(np.dot(normalized, (elevation - np.dot(normalized, elevation)) ** 2))
    )

    return {
        "n": n,
        "neighbor_fingerprint": neighbor_fingerprint,
        "slope_c_per_km": slope,
        "intercept_c": intercept,
        "n_eff": n_eff,
        "weighted_elevation_sd_m": weighted_sd_m,
        "elevation_span_m": float(elevation.max() - elevation.min()),
        "weights_normalized": normalized.copy(),
    }


def _finite_metric(result: dict[str, Any], key: str, source: str) -> float:
    if key not in result:
        raise ValueError(f"{source} result missing: {key}")
    try:
        value = float(result[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"{source}.{key} must be finite")
    return value


def classify_spatial_temperature_structure(
    ols_result: dict[str, Any],
    weighted_result: dict[str, Any],
    *,
    min_n: int = 6,
    min_elevation_span_m: float = 100.0,
    min_weighted_elevation_sd_m: float = 50.0,
    min_n_eff: float = 3.0,
) -> dict[str, Any]:
    """품질조건과 두 회귀 부호로 시간별 공간 기온구조를 4분류한다."""

    if not isinstance(ols_result, dict) or not isinstance(weighted_result, dict):
        raise TypeError("regression results must be dictionaries")
    if min_n < 2:
        raise ValueError("min_n must be at least 2")
    for name, value in [
        ("min_elevation_span_m", min_elevation_span_m),
        ("min_weighted_elevation_sd_m", min_weighted_elevation_sd_m),
        ("min_n_eff", min_n_eff),
    ]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    ols_n = _finite_metric(ols_result, "n", "ols")
    weighted_n = _finite_metric(weighted_result, "n", "weighted")
    ols_span = _finite_metric(ols_result, "elevation_span_m", "ols")
    weighted_span = _finite_metric(weighted_result, "elevation_span_m", "weighted")
    n_eff = _finite_metric(weighted_result, "n_eff", "weighted")
    weighted_sd = _finite_metric(
        weighted_result, "weighted_elevation_sd_m", "weighted"
    )
    ols_slope = _finite_metric(ols_result, "slope_c_per_km", "ols")
    ci_low = _finite_metric(ols_result, "ci95_low_c_per_km", "ols")
    ci_high = _finite_metric(ols_result, "ci95_high_c_per_km", "ols")
    weighted_slope = _finite_metric(
        weighted_result, "slope_c_per_km", "weighted"
    )
    ols_fingerprint = ols_result.get("neighbor_fingerprint")
    weighted_fingerprint = weighted_result.get("neighbor_fingerprint")
    if not isinstance(ols_fingerprint, str) or not ols_fingerprint:
        raise ValueError("ols.neighbor_fingerprint must be a non-empty string")
    if not isinstance(weighted_fingerprint, str) or not weighted_fingerprint:
        raise ValueError("weighted.neighbor_fingerprint must be a non-empty string")
    if ols_fingerprint != weighted_fingerprint:
        raise ValueError("OLS and weighted regressions must use the same neighbor IDs")
    if not ols_n.is_integer() or not weighted_n.is_integer():
        raise ValueError("regression sample sizes must be integers")
    if ols_n < 2 or weighted_n < 2:
        raise ValueError("regression sample sizes must be at least 2")
    if int(ols_n) != int(weighted_n):
        raise ValueError("OLS and weighted regressions must use the same neighbors")
    if ols_span < 0 or weighted_span < 0:
        raise ValueError("elevation spans must be non-negative")
    if not np.isclose(ols_span, weighted_span, rtol=0.0, atol=1e-9):
        raise ValueError("OLS and weighted elevation spans must match")
    metric_tolerance = 1e-9 * max(1.0, weighted_n, weighted_span)
    if n_eff < 1.0 - metric_tolerance or n_eff > weighted_n + metric_tolerance:
        raise ValueError("weighted.n_eff must lie in [1, n]")
    if weighted_sd < -metric_tolerance:
        raise ValueError("weighted elevation SD must be non-negative")
    if weighted_sd > weighted_span / 2.0 + metric_tolerance:
        raise ValueError("weighted elevation SD exceeds the range-based maximum")
    if ci_low > ci_high:
        raise ValueError("OLS confidence interval bounds are reversed")

    reasons = []
    if ols_n < min_n or weighted_n < min_n:
        reasons.append("n_below_threshold")
    if min(ols_span, weighted_span) < min_elevation_span_m:
        reasons.append("elevation_span_below_threshold")
    if weighted_sd < min_weighted_elevation_sd_m:
        reasons.append("weighted_elevation_sd_below_threshold")
    if n_eff < min_n_eff:
        reasons.append("n_eff_below_threshold")
    if reasons:
        return {
            "classification": "ineligible",
            "eligible": False,
            "reason": ";".join(reasons),
        }

    if ols_slope > 0 and ci_low > 0 and weighted_slope > 0:
        classification = "inversion_like"
    elif ols_slope < 0 and ci_high < 0 and weighted_slope < 0:
        classification = "normal_negative"
    else:
        classification = "uncertain"
    return {"classification": classification, "eligible": True, "reason": "eligible"}


def exact_idw_none_weighted_mean(
    temperatures_c,
    distances_km,
    *,
    power: float = 2.0,
    neighbor_ids: Sequence | None = None,
    target_id: Any | None = None,
) -> float:
    """기존 `none` IDW와 같은 거리 역가중 평균을 순수 계산한다."""

    temperature = _numeric_vector("temperatures_c", temperatures_c, minimum_length=1)
    distance = _numeric_vector("distances_km", distances_km, minimum_length=1)
    n = _validate_parallel_vectors(temperature, distance)
    _validate_target_exclusion(n, neighbor_ids=neighbor_ids, target_id=target_id)
    if np.any(distance < 0):
        raise ValueError("distances_km must be non-negative")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be a finite positive number")

    order = np.argsort(distance, kind="stable")
    distance = distance[order]
    temperature = temperature[order]

    zero = np.flatnonzero(distance == 0)
    if len(zero):
        # 기존 IDW와 같이 거리 정렬 뒤 처음 만나는 거리 0 이웃을 사용한다.
        # 같은 거리끼리는 stable sort가 호출자가 제공한 metadata 순서를 보존한다.
        return float(temperature[int(zero[0])])

    # 최소거리로 스케일하면 모든 weight가 [0, 1]에 있어 극소거리에서도
    # overflow하지 않는다. 정렬된 원본 IDW처럼 scalar 순서로 누적한다.
    min_distance = float(distance[0])
    weighted_sum = 0.0
    weight_sum = 0.0
    for value, current_distance in zip(temperature, distance):
        weight = (min_distance / float(current_distance)) ** float(power)
        weighted_sum += weight * float(value)
        weight_sum += weight
    return float(weighted_sum / weight_sum)


def summarize_condition_bias(
    frame: pd.DataFrame,
    *,
    group_by: str | Sequence[str],
) -> pd.DataFrame:
    """조건 안에서 signed BIAS를 먼저 평균한 뒤 Δ|BIAS|를 계산한다.

    `DELTA_MAE`는 혼동 방지를 위한 별도 보조열이며 `DELTA_ABS_BIAS`를 대체하지 않는다.
    """

    if isinstance(group_by, str):
        group_columns = [group_by]
    else:
        group_columns = list(group_by)
    if not group_columns or len(set(group_columns)) != len(group_columns):
        raise ValueError("group_by must contain one or more unique columns")
    _require_columns(frame, [*group_columns, "error_none", "error_fixed"])
    if frame.empty:
        raise ValueError("frame must not be empty")
    if frame[group_columns].isna().any().any():
        raise ValueError("group columns must not contain missing values")

    work = frame.copy()
    work["error_none"] = _finite_numeric_series(work, "error_none")
    work["error_fixed"] = _finite_numeric_series(work, "error_fixed")

    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in work.groupby(grouper, sort=False, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        row = dict(zip(group_columns, keys))
        bias_none = float(group["error_none"].mean())
        bias_fixed = float(group["error_fixed"].mean())
        mae_none = float(group["error_none"].abs().mean())
        mae_fixed = float(group["error_fixed"].abs().mean())
        row.update(
            {
                "N": int(len(group)),
                "BIAS_none": bias_none,
                "BIAS_fixed": bias_fixed,
                "DELTA_ABS_BIAS": abs(bias_fixed) - abs(bias_none),
                "MAE_none": mae_none,
                "MAE_fixed": mae_fixed,
                "DELTA_MAE": mae_fixed - mae_none,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def rank_biserial_effect(group_x, group_y) -> float:
    """독립 두 집단의 rank-biserial 효과크기 P(X>Y)-P(X<Y)를 반환한다."""

    x = _numeric_vector("group_x", group_x, minimum_length=1)
    y = _numeric_vector("group_y", group_y, minimum_length=1)
    differences = x[:, None] - y[None, :]
    wins = int(np.count_nonzero(differences > 0))
    losses = int(np.count_nonzero(differences < 0))
    return float((wins - losses) / differences.size)


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR q값을 입력 순서로 반환한다."""

    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("pvalues must be a non-empty finite one-dimensional array")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


__all__ = [
    "benjamini_hochberg",
    "build_equivalent_diagnostics",
    "build_worsening_diagnostics",
    "classify_spatial_temperature_structure",
    "exact_idw_none_weighted_mean",
    "ols_temperature_elevation_slope",
    "rank_biserial_effect",
    "summarize_condition_bias",
    "weighted_temperature_elevation_slope",
]
