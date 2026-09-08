"""BIAS 변화의 실질적 크기와 시간 의존성 기반 불확실성 계산."""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNIFICANCE_CLASSES = (
    "meaningful_improvement",
    "meaningful_worsening",
    "practically_equivalent",
    "uncertain",
)


def _validated_dates(dates) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise")).normalize()
    if len(index) == 0:
        raise ValueError("dates must not be empty")
    if not index.is_unique:
        raise ValueError("dates must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("dates must be strictly increasing")
    return index


def make_month_stratified_block_weights(
    dates,
    n_boot: int,
    block_days: int,
    seed: int,
) -> np.ndarray:
    """월별 표본수를 유지하는 원형 이동 일블록 bootstrap 빈도행렬을 만든다."""
    index = _validated_dates(dates)
    if int(n_boot) != n_boot or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if int(block_days) != block_days or block_days <= 0:
        raise ValueError("block_days must be a positive integer")
    n_boot = int(n_boot)
    block_days = int(block_days)
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, len(index)), dtype=np.int16)

    month_codes = index.to_period("M")
    for month in month_codes.unique():
        positions = np.flatnonzero(month_codes == month)
        n_days = len(positions)
        blocks_needed = int(np.ceil(n_days / block_days))
        starts = rng.integers(0, n_days, size=(n_boot, blocks_needed))
        offsets = np.arange(block_days, dtype=int)
        local_draws = (starts[:, :, None] + offsets[None, None, :]) % n_days
        local_draws = local_draws.reshape(n_boot, -1)[:, :n_days]
        for row in range(n_boot):
            weights[row, positions] = np.bincount(
                local_draws[row], minlength=n_days
            ).astype(np.int16)
    return weights


def _numeric_matrix(value, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return matrix


def bootstrap_bias_effects(daily: dict, weights) -> dict[str, np.ndarray]:
    """일별 오차 합과 짝 bootstrap 빈도로 BIAS와 Δ|BIAS| 표본을 계산한다."""
    required = {"sum_none", "sum_fixed", "count"}
    missing = required.difference(daily)
    if missing:
        raise ValueError(f"daily matrices missing: {sorted(missing)}")
    sum_none = _numeric_matrix(daily["sum_none"], "sum_none")
    sum_fixed = _numeric_matrix(daily["sum_fixed"], "sum_fixed")
    count = _numeric_matrix(daily["count"], "count")
    if sum_none.shape != sum_fixed.shape or sum_none.shape != count.shape:
        raise ValueError("daily matrices must have identical shapes")
    if np.any(count < 0):
        raise ValueError("count must not be negative")
    boot_weights = _numeric_matrix(weights, "weights")
    if boot_weights.shape[1] != sum_none.shape[1]:
        raise ValueError("weights day dimension must match daily matrices")
    if np.any(boot_weights < 0):
        raise ValueError("weights must not be negative")

    denominators = boot_weights @ count.T
    if np.any(denominators <= 0):
        raise ValueError("bootstrap replica has zero valid paired hours")
    bias_none = (boot_weights @ sum_none.T) / denominators
    bias_fixed = (boot_weights @ sum_fixed.T) / denominators
    return {
        "bias_none": bias_none,
        "bias_fixed": bias_fixed,
        "delta_abs_bias": np.abs(bias_fixed) - np.abs(bias_none),
    }


def bootstrap_stationwise_bias_effects(
    daily: dict,
    dates,
    n_boot: int,
    block_days: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """각 station의 실제 유효일 안에서 월별 블록 bootstrap을 수행한다."""
    date_index = _validated_dates(dates)
    sum_none = _numeric_matrix(daily.get("sum_none"), "sum_none")
    sum_fixed = _numeric_matrix(daily.get("sum_fixed"), "sum_fixed")
    count = _numeric_matrix(daily.get("count"), "count")
    if sum_none.shape != sum_fixed.shape or sum_none.shape != count.shape:
        raise ValueError("daily matrices must have identical shapes")
    if sum_none.shape[1] != len(date_index):
        raise ValueError("dates length must match daily matrix day dimension")
    if np.any(count < 0):
        raise ValueError("count must not be negative")

    n_stations = sum_none.shape[0]
    bias_none = np.empty((int(n_boot), n_stations), dtype=float)
    bias_fixed = np.empty_like(bias_none)
    delta_abs = np.empty_like(bias_none)
    coverage_days = np.zeros(n_stations, dtype=int)
    coverage_months = np.zeros(n_stations, dtype=int)
    coverage_hours = count.sum(axis=1).astype(int)

    for station_index in range(n_stations):
        valid = count[station_index] > 0
        if not valid.any():
            raise ValueError(f"station {station_index} has no valid paired days")
        valid_dates = date_index[valid]
        coverage_days[station_index] = int(valid.sum())
        coverage_months[station_index] = int(valid_dates.to_period("M").nunique())
        weights = make_month_stratified_block_weights(
            valid_dates,
            n_boot=n_boot,
            block_days=block_days,
            seed=int(seed) + station_index * 1009,
        )
        station_daily = {
            "sum_none": sum_none[station_index : station_index + 1, valid],
            "sum_fixed": sum_fixed[station_index : station_index + 1, valid],
            "count": count[station_index : station_index + 1, valid],
        }
        result = bootstrap_bias_effects(station_daily, weights)
        bias_none[:, station_index] = result["bias_none"][:, 0]
        bias_fixed[:, station_index] = result["bias_fixed"][:, 0]
        delta_abs[:, station_index] = result["delta_abs_bias"][:, 0]
    return {
        "bias_none": bias_none,
        "bias_fixed": bias_fixed,
        "delta_abs_bias": delta_abs,
        "coverage_days": coverage_days,
        "coverage_months": coverage_months,
        "coverage_hours": coverage_hours,
    }


def _tail_probability(mask: np.ndarray) -> float:
    return float((int(np.count_nonzero(mask)) + 1) / (len(mask) + 1))


def bootstrap_boundary_pvalues(
    theta_hat: float,
    samples,
    delta: float,
) -> dict[str, float]:
    """실질 경계 ±delta에 대한 중심화 bootstrap 단측·동등성 p값을 계산한다."""
    theta_hat = float(theta_hat)
    delta = float(delta)
    boot = np.asarray(samples, dtype=float)
    if not np.isfinite(theta_hat) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("theta_hat must be finite and delta must be positive")
    if boot.ndim != 1 or len(boot) < 20 or not np.isfinite(boot).all():
        raise ValueError("samples must be a finite one-dimensional array of length >=20")

    centered = boot - theta_hat
    p_improve = _tail_probability(centered <= theta_hat + delta)
    p_worsen = _tail_probability(centered >= theta_hat - delta)
    p_equiv_lower = _tail_probability(centered >= theta_hat + delta)
    p_equiv_upper = _tail_probability(centered <= theta_hat - delta)
    return {
        "p_improve": p_improve,
        "p_worsen": p_worsen,
        "p_equivalent": max(p_equiv_lower, p_equiv_upper),
    }


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini–Hochberg 방식의 FDR 보정 q값을 원래 순서로 반환한다."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("pvalues must be a non-empty finite one-dimensional array")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    scale = len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def classify_station_effects(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """FDR q값으로 station을 개선·악화·동등·불확실 중 하나로 판정한다."""
    required = ["q_improve", "q_worsen", "q_equivalent"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"classification columns missing: {missing}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    out = frame.copy()
    q = out[required].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("classification q-values must be finite and in [0, 1]")
    claims = q < alpha
    if np.any(claims.sum(axis=1) > 1):
        raise ValueError("conflicting station significance claims")
    labels = np.full(len(out), "uncertain", dtype=object)
    labels[claims[:, 0]] = "meaningful_improvement"
    labels[claims[:, 1]] = "meaningful_worsening"
    labels[claims[:, 2]] = "practically_equivalent"
    out["SIGNIFICANCE_CLASS"] = labels
    return out
