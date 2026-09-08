"""발표 피드백을 반영한 station별 BIAS 유의성·물리 해석 분석."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path.cwd() / ".omc" / "matplotlib").resolve())
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bias_data import load_station_pairs, open_readonly_db
from bias_significance import (
    benjamini_hochberg,
    bootstrap_boundary_pvalues,
    bootstrap_stationwise_bias_effects,
    classify_station_effects,
)
from bias_statistics import spearman_effect


FULLNET = Path("WeatherData_AWS/interpolation_batch/fullnet")
SOURCE_DIR = FULLNET / "bias_analysis"
OUT_DIR = FULLNET / "bias_feedback_analysis"
PRIMARY_DELTA_C = 0.10
PRIMARY_BLOCK_DAYS = 7
PRIMARY_N_BOOT = 2000
SEED = 20260812
DELTA_SENSITIVITY = (0.05, 0.10, 0.20)
BLOCK_SENSITIVITY = (1, 3, 7, 14)
COVERAGE_MIN_DAYS = 180
COVERAGE_MIN_MONTHS = 9

CLASS_ORDER = [
    "meaningful_improvement",
    "meaningful_worsening",
    "practically_equivalent",
    "uncertain",
]
CLASS_LABELS = {
    "meaningful_improvement": "유의미한 개선",
    "meaningful_worsening": "유의미한 악화",
    "practically_equivalent": "차이 거의 없음",
    "uncertain": "판단 불확실",
}
CLASS_COLORS = {
    "meaningful_improvement": "#2878B5",
    "meaningful_worsening": "#D9534F",
    "practically_equivalent": "#777777",
    "uncertain": "#E5A000",
}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def aggregate_pairs_to_daily(pairs: pd.DataFrame) -> pd.DataFrame:
    """동일 시각 paired 오차를 station·날짜별 합과 표본 수로 축약한다."""
    required = {"TM", "STN", "error_none", "error_fixed"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"paired rows missing columns: {sorted(missing)}")
    frame = pairs[list(required)].copy()
    frame["TM"] = pd.to_datetime(frame["TM"], errors="raise")
    frame["STN"] = frame["STN"].astype(str)
    frame["error_none"] = pd.to_numeric(frame["error_none"], errors="raise")
    frame["error_fixed"] = pd.to_numeric(frame["error_fixed"], errors="raise")
    if not np.isfinite(frame[["error_none", "error_fixed"]].to_numpy(float)).all():
        raise ValueError("paired errors must be finite")
    frame["DATE"] = frame["TM"].dt.normalize()
    return (
        frame.groupby(["STN", "DATE"], sort=True, as_index=False)
        .agg(
            sum_none=("error_none", "sum"),
            sum_fixed=("error_fixed", "sum"),
            count=("error_none", "size"),
        )
        .sort_values(["STN", "DATE"])
        .reset_index(drop=True)
    )


def load_daily_paired_errors(stations) -> pd.DataFrame:
    """읽기 전용 SQLite에서 station별 paired 오차를 읽어 일별로 축약한다."""
    station_ids = [str(stn) for stn in stations]
    if not station_ids or len(station_ids) != len(set(station_ids)):
        raise ValueError("stations must be a non-empty unique list")
    parts = []
    con = open_readonly_db()
    try:
        for index, stn in enumerate(station_ids, start=1):
            parts.append(aggregate_pairs_to_daily(load_station_pairs(con, stn)))
            if index % 50 == 0 or index == len(station_ids):
                print(f"daily paired load: {index}/{len(station_ids)} stations")
    finally:
        con.close()
    return pd.concat(parts, ignore_index=True)


def build_daily_matrices(daily, stations, dates) -> dict[str, np.ndarray]:
    """일별 long 표를 station×달력일 행렬로 변환한다."""
    station_ids = [str(stn) for stn in stations]
    if not station_ids or len(station_ids) != len(set(station_ids)):
        raise ValueError("stations must be non-empty and unique")
    date_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise")).normalize()
    if len(date_index) == 0 or not date_index.is_unique or not date_index.is_monotonic_increasing:
        raise ValueError("dates must be non-empty, unique, and increasing")
    required = {"STN", "DATE", "sum_none", "sum_fixed", "count"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily table missing columns: {sorted(missing)}")
    frame = daily[list(required)].copy()
    frame["STN"] = frame["STN"].astype(str)
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise").dt.normalize()
    if frame.duplicated(["STN", "DATE"]).any():
        raise ValueError("daily table contains duplicate station-date rows")
    station_lut = {stn: i for i, stn in enumerate(station_ids)}
    date_lut = {date: i for i, date in enumerate(date_index)}
    if not set(frame["STN"]).issubset(station_lut):
        raise ValueError("daily table contains an unknown station")
    if not set(frame["DATE"]).issubset(date_lut):
        raise ValueError("daily table contains a date outside calendar")
    shape = (len(station_ids), len(date_index))
    matrices = {
        "sum_none": np.zeros(shape, dtype=float),
        "sum_fixed": np.zeros(shape, dtype=float),
        "count": np.zeros(shape, dtype=float),
    }
    for row in frame.itertuples(index=False):
        i = station_lut[str(row.STN)]
        j = date_lut[pd.Timestamp(row.DATE)]
        matrices["sum_none"][i, j] = float(row.sum_none)
        matrices["sum_fixed"][i, j] = float(row.sum_fixed)
        matrices["count"][i, j] = float(row.count)
    matrices["stations"] = np.asarray(station_ids, dtype=object)
    matrices["dates"] = date_index.to_numpy()
    return matrices


def validate_annual_against_existing(matrices, existing, atol=1e-10):
    """일별 행렬의 연간 BIAS가 기존 station 정본과 같은지 검산한다."""
    counts = np.asarray(matrices["count"], dtype=float).sum(axis=1)
    if np.any(counts <= 0):
        raise AssertionError("station with zero annual paired hours")
    bias_none = np.asarray(matrices["sum_none"], dtype=float).sum(axis=1) / counts
    bias_fixed = np.asarray(matrices["sum_fixed"], dtype=float).sum(axis=1) / counts
    if len(existing) != len(bias_none):
        raise AssertionError("existing station rows do not match daily matrices")
    expected_none = pd.to_numeric(existing["BIAS_none"], errors="raise").to_numpy(float)
    expected_fixed = pd.to_numeric(existing["BIAS_fixed"], errors="raise").to_numpy(float)
    residuals = {
        "max_abs_bias_none": float(np.max(np.abs(bias_none - expected_none))),
        "max_abs_bias_fixed": float(np.max(np.abs(bias_fixed - expected_fixed))),
    }
    if residuals["max_abs_bias_none"] > atol or residuals["max_abs_bias_fixed"] > atol:
        raise AssertionError(f"annual BIAS validation failed: {residuals}")
    return residuals


def summarize_station_bootstrap(stations, bootstrap_delta, delta=PRIMARY_DELTA_C):
    """station별 bootstrap 효과표에서 CI·p/q값과 네 범주를 만든다."""
    frame = stations.reset_index(drop=True).copy()
    boot = np.asarray(bootstrap_delta, dtype=float)
    if boot.ndim != 2 or boot.shape[1] != len(frame) or not np.isfinite(boot).all():
        raise ValueError("bootstrap_delta must be finite n_boot × n_station")
    theta = pd.to_numeric(frame["DELTA_ABS_BIAS"], errors="raise").to_numpy(float)
    p_rows = [
        bootstrap_boundary_pvalues(theta[i], boot[:, i], delta)
        for i in range(len(frame))
    ]
    p = pd.DataFrame(p_rows)
    frame["DELTA_THRESHOLD_C"] = float(delta)
    frame["CI95_LOW"] = np.quantile(boot, 0.025, axis=0)
    frame["CI95_HIGH"] = np.quantile(boot, 0.975, axis=0)
    frame["CI90_LOW"] = np.quantile(boot, 0.05, axis=0)
    frame["CI90_HIGH"] = np.quantile(boot, 0.95, axis=0)
    eligible = (
        frame["COVERAGE_ELIGIBLE"].astype(bool).to_numpy()
        if "COVERAGE_ELIGIBLE" in frame
        else np.ones(len(frame), dtype=bool)
    )
    for column in p.columns:
        frame[column] = p[column].to_numpy(float)
        q = np.ones(len(frame), dtype=float)
        if eligible.any():
            q[eligible] = benjamini_hochberg(p.loc[eligible, column])
        frame[column.replace("p_", "q_")] = q
    frame = classify_station_effects(frame)
    return frame


def run_station_inference(stations, matrices, dates, block_days, n_boot, delta, seed):
    boot_result = bootstrap_stationwise_bias_effects(
        matrices, dates, n_boot=n_boot, block_days=block_days, seed=seed
    )
    prepared = stations.copy()
    prepared["COVERAGE_DAYS"] = boot_result["coverage_days"]
    prepared["COVERAGE_MONTHS"] = boot_result["coverage_months"]
    prepared["COVERAGE_HOURS"] = boot_result["coverage_hours"]
    prepared["COVERAGE_ELIGIBLE"] = (
        (prepared["COVERAGE_DAYS"] >= COVERAGE_MIN_DAYS)
        & (prepared["COVERAGE_MONTHS"] >= COVERAGE_MIN_MONTHS)
    )
    prepared["COVERAGE_NOTE"] = np.where(
        prepared["COVERAGE_ELIGIBLE"],
        "annual inference eligible",
        "insufficient annual coverage; retained as uncertain",
    )
    boot = boot_result["delta_abs_bias"]
    result = summarize_station_bootstrap(prepared, boot, delta=delta)
    result["BOOT_BLOCK_DAYS"] = int(block_days)
    result["BOOT_REPLICATES"] = int(n_boot)
    result["BOOT_SEED"] = int(seed)
    return result, boot


def _coverage_from_matrices(matrices, dates):
    count = np.asarray(matrices["count"], dtype=float)
    valid = count > 0
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    months = date_index.to_period("M")
    coverage_months = np.array(
        [len(pd.unique(months[row])) for row in valid], dtype=int
    )
    return valid.sum(axis=1), coverage_months, count.sum(axis=1).astype(int)


def run_sensitivity(
    stations,
    matrices,
    dates,
    n_boot,
    primary_labels,
    primary_bootstrap=None,
):
    rows = []
    station_labels = {}
    for block_days in BLOCK_SENSITIVITY:
        if block_days == PRIMARY_BLOCK_DAYS and primary_bootstrap is not None:
            coverage_days, coverage_months, coverage_hours = _coverage_from_matrices(
                matrices, dates
            )
            boot_result = {
                "delta_abs_bias": np.asarray(primary_bootstrap, dtype=float),
                "coverage_days": coverage_days,
                "coverage_months": coverage_months,
                "coverage_hours": coverage_hours,
            }
        else:
            boot_result = bootstrap_stationwise_bias_effects(
                matrices, dates, n_boot=n_boot, block_days=block_days, seed=SEED
            )
        prepared = stations.copy()
        prepared["COVERAGE_ELIGIBLE"] = (
            (boot_result["coverage_days"] >= COVERAGE_MIN_DAYS)
            & (boot_result["coverage_months"] >= COVERAGE_MIN_MONTHS)
        )
        boot = boot_result["delta_abs_bias"]
        for delta in DELTA_SENSITIVITY:
            result = summarize_station_bootstrap(prepared, boot, delta=delta)
            key = (float(delta), int(block_days))
            labels = result["SIGNIFICANCE_CLASS"].to_numpy(object)
            station_labels[key] = labels
            counts = result["SIGNIFICANCE_CLASS"].value_counts()
            rows.append(
                {
                    "delta_c": float(delta),
                    "block_days": int(block_days),
                    "n_boot": int(n_boot),
                    "agreement_with_primary": float(np.mean(labels == primary_labels)),
                    **{name: int(counts.get(name, 0)) for name in CLASS_ORDER},
                }
            )
    label_matrix = np.vstack(list(station_labels.values()))
    stability = np.mean(label_matrix == primary_labels[None, :], axis=0)
    return pd.DataFrame(rows), stability


def build_temporal_summary(monthly, diurnal):
    """월별·시간대별 station 중앙값·평균과 가운데 50% 범위를 만든다."""
    rows = []
    for period_type, frame, period in [
        ("month", monthly, "MONTH"),
        ("hour", diurnal, "HOUR"),
    ]:
        for key, group in frame.groupby(period, sort=True):
            record = {"PERIOD_TYPE": period_type, "PERIOD": key, "N": len(group)}
            for metric in ["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                record[f"{metric}_median"] = float(values.median())
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_q25"] = float(values.quantile(0.25))
                record[f"{metric}_q75"] = float(values.quantile(0.75))
            rows.append(record)
    return pd.DataFrame(rows)


def _summarize_physical_group(frame, check, group_col):
    rows = []
    for group_name, group in frame.groupby(group_col, observed=True, dropna=False):
        counts = group["SIGNIFICANCE_CLASS"].value_counts()
        rows.append(
            {
                "CHECK": check,
                "GROUP": str(group_name),
                "N": len(group),
                "MEDIAN_DELTA_ABS_BIAS": float(group["DELTA_ABS_BIAS"].median()),
                "MEDIAN_BASELINE_ABS_BIAS": float(group["BIAS_none"].abs().median()),
                **{f"FRAC_{name}": float(counts.get(name, 0) / len(group)) for name in CLASS_ORDER},
            }
        )
    return rows


def build_physical_checks(station_results):
    """수치 패턴의 물리적 해석을 뒷받침할 층화 요약을 만든다."""
    frame = station_results.copy()
    mean_abs_col = (
        "mean_absdz_geom_km"
        if "mean_absdz_geom_km" in frame
        else "mean_abs_delta_z_km"
    )
    frame["ABS_DZ_M"] = pd.to_numeric(frame[mean_abs_col], errors="raise") * 1000
    frame["DZ_BIN"] = pd.cut(
        frame["ABS_DZ_M"],
        [-np.inf, 50, 100, 200, 300, np.inf],
        labels=["≤50m", "50–100m", "100–200m", "200–300m", ">300m"],
    )
    frame["BASELINE_BIN"] = pd.cut(
        frame["BIAS_none"].abs(),
        [-np.inf, 0.1, 0.5, np.inf],
        labels=["|기존 BIAS|≤0.1°C", "0.1–0.5°C", ">0.5°C"],
    )
    rows = []
    rows.extend(_summarize_physical_group(frame, "absolute_delta_z", "DZ_BIN"))
    rows.extend(_summarize_physical_group(frame, "baseline_accuracy", "BASELINE_BIN"))
    if "terrain_group" in frame:
        rows.extend(_summarize_physical_group(frame, "terrain_group", "terrain_group"))
        frame["DZ_TERRAIN_GROUP"] = (
            frame["DZ_BIN"].astype(str) + " | " + frame["terrain_group"].astype(str)
        )
        rows.extend(
            _summarize_physical_group(
                frame, "terrain_within_delta_z", "DZ_TERRAIN_GROUP"
            )
        )
    for source, label in [
        ("mean_neighbor_elev_sd_m", "neighbor_elevation_spread"),
        ("mean_weighted_distance_km", "neighbor_distance"),
        ("relief_1km_m", "terrain_relief"),
    ]:
        if source in frame and frame[source].nunique(dropna=True) >= 4:
            group_col = f"{source}_quartile"
            frame[group_col] = pd.qcut(
                frame[source], 4, labels=["Q1 낮음", "Q2", "Q3", "Q4 높음"], duplicates="drop"
            )
            rows.extend(_summarize_physical_group(frame, label, group_col))
            combined_col = f"DZ_{group_col}"
            frame[combined_col] = (
                frame["DZ_BIN"].astype(str) + " | " + frame[group_col].astype(str)
            )
            rows.extend(
                _summarize_physical_group(
                    frame, f"{label}_within_delta_z", combined_col
                )
            )
    return pd.DataFrame(rows)


def _delta_z_column(frame):
    for name in ["mean_delta_z_km", "mean_dz_geom_km"]:
        if name in frame:
            return name
    raise ValueError("station table lacks mean ΔZ column")


def create_dz_before_after_figure(station_results):
    """같은 축에서 ΔZ–보정 전·후 BIAS를 나란히 비교하는 그림을 만든다."""
    frame = station_results
    x = pd.to_numeric(frame[_delta_z_column(frame)], errors="raise").to_numpy(float) * 1000
    y_values = [
        pd.to_numeric(frame["BIAS_none"], errors="raise").to_numpy(float),
        pd.to_numeric(frame["BIAS_fixed"], errors="raise").to_numpy(float),
    ]
    x_pad = max(np.ptp(x) * 0.04, 10.0)
    combined_y = np.concatenate(y_values)
    y_pad = max(np.ptp(combined_y) * 0.06, 0.2)
    x_lim = (float(x.min() - x_pad), float(x.max() + x_pad))
    y_lim = (float(combined_y.min() - y_pad), float(combined_y.max() + y_pad))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, y, title in zip(axes, y_values, ["고도보정 전", "고도보정 후"]):
        for category in CLASS_ORDER:
            mask = frame["SIGNIFICANCE_CLASS"].eq(category).to_numpy()
            ax.scatter(
                x[mask], y[mask], s=24, alpha=0.72,
                color=CLASS_COLORS[category], edgecolors="none",
                label=CLASS_LABELS[category],
            )
        rho, _ = spearman_effect(x, y)
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("평균 ΔZ (m)")
        ax.set_title(f"{title}\nSpearman ρ={rho:.2f}")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("연평균 BIAS (°C)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("같은 station의 ΔZ와 고도보정 전·후 BIAS", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig, axes


def _save_figure(fig, out_dir, filename):
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dz_effect(frame, out_dir):
    x_col = "mean_absdz_geom_km" if "mean_absdz_geom_km" in frame else "mean_abs_delta_z_km"
    x = frame[x_col].to_numpy(float) * 1000
    y = frame["DELTA_ABS_BIAS"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for category in CLASS_ORDER:
        mask = frame["SIGNIFICANCE_CLASS"].eq(category)
        ax.scatter(x[mask], y[mask], s=25, alpha=0.7, color=CLASS_COLORS[category],
                   label=CLASS_LABELS[category], edgecolors="none")
    bins = [-np.inf, 50, 100, 200, 300, np.inf]
    centers, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x > lo) & (x <= hi)
        if mask.any():
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    ax.plot(centers, medians, color="black", marker="o", lw=1.4, label="ΔZ 구간 중앙값")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("평균 절대 ΔZ (m)")
    ax.set_ylabel("Δ|BIAS| (°C, 음수=개선)")
    ax.set_title("같은 station에서 본 ΔZ 크기와 BIAS 변화")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, ncol=2)
    return _save_figure(fig, out_dir, "bias_dz_effect.png")


def plot_significance_summary(frame, out_dir):
    counts = frame["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(
        [CLASS_LABELS[name] for name in CLASS_ORDER],
        counts.to_numpy(), color=[CLASS_COLORS[name] for name in CLASS_ORDER]
    )
    for i, value in enumerate(counts):
        axes[0].text(i, value + max(counts.max() * 0.02, 1), str(value), ha="center")
    axes[0].set_ylabel("station 수")
    axes[0].set_title("실질 크기+통계적 확실성 분류")
    axes[0].tick_params(axis="x", rotation=20)
    data = [frame.loc[frame["SIGNIFICANCE_CLASS"].eq(name), "DELTA_ABS_BIAS"] for name in CLASS_ORDER]
    axes[1].boxplot(
        data,
        tick_labels=[CLASS_LABELS[name] for name in CLASS_ORDER],
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
    )
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("Δ|BIAS| (°C)")
    axes[1].set_title("분류별 실제 변화량 — 극단값 포함")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "bias_significance_summary.png")


def create_temporal_physical_figure(temporal):
    """월별·시간대별 BIAS 그림을 만든다.

    시간대 패널에는 원인을 미리 단정하는 문구나 배경 음영을 넣지 않는다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, period_type in enumerate(["month", "hour"]):
        sub = temporal.loc[temporal["PERIOD_TYPE"].eq(period_type)].copy()
        if period_type == "month":
            x = sub["PERIOD"].astype(str).str[-2:].astype(int)
            xlabel = "월"
        else:
            x = pd.to_numeric(sub["PERIOD"])
            xlabel = "시각"
        axes[0, col].plot(x, sub["BIAS_none_median"], marker="o", label="보정 전 BIAS")
        axes[0, col].plot(x, sub["BIAS_fixed_median"], marker="o", label="보정 후 BIAS")
        axes[0, col].axhline(0, color="black", lw=0.8)
        axes[0, col].set_ylabel("관측소 BIAS 중앙값 (°C)")
        axes[0, col].set_title("월별" if period_type == "month" else "시간대별")
        axes[0, col].grid(alpha=0.18)
        axes[1, col].plot(x, sub["DELTA_ABS_BIAS_median"], color="#7A5195", marker="o")
        axes[1, col].axhline(0, color="black", lw=0.8)
        axes[1, col].set_ylabel("관측소 Δ|BIAS| 중앙값 (°C)")
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].grid(alpha=0.18)
    axes[0, 0].legend()
    fig.suptitle("월·시간대에 따라 달라지는 고도보정의 BIAS 효과", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def plot_temporal_physical(temporal, out_dir):
    fig, _ = create_temporal_physical_figure(temporal)
    return _save_figure(fig, out_dir, "bias_temporal_physical.png")


def select_cases(station_results):
    """분류 후 사전 규칙으로 대표 사례를 선택한다."""
    frame = station_results.copy()
    selected = []
    rules = {
        "meaningful_improvement": ("DELTA_ABS_BIAS", True),
        "meaningful_worsening": ("DELTA_ABS_BIAS", False),
        "practically_equivalent": ("ABS_BASELINE", True),
        "uncertain": ("CI_WIDTH", False),
    }
    frame["ABS_BASELINE"] = frame["BIAS_none"].abs()
    frame["CI_WIDTH"] = frame["CI95_HIGH"] - frame["CI95_LOW"]
    for category in CLASS_ORDER:
        group = frame.loc[frame["SIGNIFICANCE_CLASS"].eq(category)].copy()
        if group.empty:
            continue
        column, ascending = rules[category]
        selected.append(group.sort_values([column, "STN"], ascending=[ascending, True]).iloc[0])
    if not selected:
        raise ValueError("no representative cases available")
    return pd.DataFrame(selected).drop(columns=["ABS_BASELINE", "CI_WIDTH"]).reset_index(drop=True)


def plot_case_studies(cases, monthly, out_dir):
    n = len(cases)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, case in zip(axes.ravel(), cases.itertuples(index=False)):
        ax.set_visible(True)
        sub = monthly.loc[monthly["STN"].astype(str).eq(str(case.STN))].copy()
        x = sub["MONTH"].astype(str).str[-2:].astype(int)
        ax.plot(x, sub["BIAS_none"], marker="o", label="보정 전")
        ax.plot(x, sub["BIAS_fixed"], marker="o", label="보정 후")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{case.STN} {getattr(case, '지점명')} — {CLASS_LABELS[case.SIGNIFICANCE_CLASS]}")
        ax.set_xlabel("월")
        ax.set_ylabel("BIAS (°C)")
        ax.grid(alpha=0.18)
        values = case._asdict()
        dz_value = values.get("mean_dz_geom_km", values.get("mean_delta_z_km", np.nan))
        coverage_days = values.get("COVERAGE_DAYS", np.nan)
        coverage_months = values.get("COVERAGE_MONTHS", np.nan)
        note = f"평균 ΔZ {dz_value * 1000:.0f}m"
        if np.isfinite(coverage_days) and np.isfinite(coverage_months):
            note += f"\n자료 {int(coverage_days)}일·{int(coverage_months)}개월"
        ax.text(
            0.02, 0.96, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#BBBBBB"}
        )
    axes[0, 0].legend()
    fig.tight_layout()
    return _save_figure(fig, out_dir, "bias_case_studies_feedback.png")


def plot_sensitivity(sensitivity, out_dir):
    pivot = sensitivity.pivot(index="block_days", columns="delta_c", values="agreement_with_primary")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    image = axes[0].imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)), [f"{v:.2f}" for v in pivot.columns])
    axes[0].set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
    axes[0].set_xlabel("실질 기준 δ (°C)")
    axes[0].set_ylabel("블록 길이 (일)")
    axes[0].set_title("기본 분류와 station 일치율")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            axes[0].text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0], label="일치율")
    primary_delta = sensitivity.loc[sensitivity["block_days"].eq(PRIMARY_BLOCK_DAYS)].copy()
    bottom = np.zeros(len(primary_delta))
    x = np.arange(len(primary_delta))
    for category in CLASS_ORDER:
        values = primary_delta[category].to_numpy()
        axes[1].bar(x, values, bottom=bottom, color=CLASS_COLORS[category], label=CLASS_LABELS[category])
        bottom += values
    axes[1].set_xticks(x, [f"δ={v:.2f}" for v in primary_delta["delta_c"]])
    axes[1].set_ylabel("station 수")
    axes[1].set_title("7일 블록에서 기준별 분류")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "bias_sensitivity.png")


def _markdown_table(frame, columns=None):
    table = frame if columns is None else frame[columns]
    if table.empty:
        return "(해당 사례 없음)"
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []
    for row in table.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + rows)


def write_feedback_report(results, temporal, sensitivity, physical, cases, out_dir):
    """수치·물리 과정·근거 수준·한계를 함께 담은 Markdown 보고서를 쓴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = results["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    dz_col = _delta_z_column(results)
    rho_before, _ = spearman_effect(results[dz_col], results["BIAS_none"])
    rho_after, _ = spearman_effect(results[dz_col], results["BIAS_fixed"])
    hour = temporal.loc[temporal["PERIOD_TYPE"].eq("hour")].copy()
    day = hour.loc[pd.to_numeric(hour["PERIOD"]).between(9, 17), "DELTA_ABS_BIAS_median"].median()
    night_hours = pd.to_numeric(hour["PERIOD"])
    night = hour.loc[(night_hours >= 21) | (night_hours <= 5), "DELTA_ABS_BIAS_median"].median()
    min_agreement = float(sensitivity["agreement_with_primary"].min())
    raw_improved = int((results["DELTA_ABS_BIAS"] < 0).sum())
    raw_worsened = int((results["DELTA_ABS_BIAS"] > 0).sum())
    raw_near_zero = int((results["DELTA_ABS_BIAS"].abs() <= PRIMARY_DELTA_C).sum())
    worsening = results.loc[
        results["SIGNIFICANCE_CLASS"].eq("meaningful_worsening"), "DELTA_ABS_BIAS"
    ]
    worsening_median = float(worsening.median()) if len(worsening) else np.nan
    worsening_over_half = int((worsening > 0.5).sum())
    worsening_over_one = int((worsening > 1.0).sum())
    coverage_ineligible = (
        int((~results["COVERAGE_ELIGIBLE"].astype(bool)).sum())
        if "COVERAGE_ELIGIBLE" in results
        else 0
    )
    threshold_rows = sensitivity.loc[
        sensitivity["block_days"].eq(PRIMARY_BLOCK_DAYS)
    ].copy()
    threshold_columns = [
        "delta_c", "meaningful_improvement", "meaningful_worsening",
        "practically_equivalent", "uncertain"
    ]
    threshold_display = threshold_rows.rename(columns={
        "delta_c": "실질 기준 δ(°C)",
        "meaningful_improvement": "유의미한 개선 수",
        "meaningful_worsening": "유의미한 악화 수",
        "practically_equivalent": "차이 거의 없음 수",
        "uncertain": "판단 불확실 수",
    })
    threshold_display_columns = [
        "실질 기준 δ(°C)", "유의미한 개선 수", "유의미한 악화 수",
        "차이 거의 없음 수", "판단 불확실 수"
    ]
    threshold_table = (
        _markdown_table(threshold_display, threshold_display_columns)
        if set(threshold_columns).issubset(threshold_rows.columns)
        else "(합성 테스트에서는 분류수 생략)"
    )
    physical_columns = [
        "GROUP", "N", "MEDIAN_DELTA_ABS_BIAS", "FRAC_meaningful_improvement",
        "FRAC_meaningful_worsening", "FRAC_practically_equivalent", "FRAC_uncertain"
    ]
    dz_physical = physical.loc[physical["CHECK"].eq("absolute_delta_z")]
    baseline_physical = physical.loc[physical["CHECK"].eq("baseline_accuracy")]
    physical_names = {
        "GROUP": "구간",
        "N": "station 수",
        "MEDIAN_DELTA_ABS_BIAS": "중앙 Δ|BIAS| (°C)",
        "FRAC_meaningful_improvement": "유의미한 개선 비율",
        "FRAC_meaningful_worsening": "유의미한 악화 비율",
        "FRAC_practically_equivalent": "차이 거의 없음 비율",
        "FRAC_uncertain": "판단 불확실 비율",
    }
    physical_display_columns = [physical_names[col] for col in physical_columns]
    dz_table = (
        _markdown_table(dz_physical.rename(columns=physical_names), physical_display_columns)
        if set(physical_columns).issubset(dz_physical.columns)
        else "(합성 테스트에서는 상세표 생략)"
    )
    baseline_table = (
        _markdown_table(
            baseline_physical.rename(columns=physical_names), physical_display_columns
        )
        if set(physical_columns).issubset(baseline_physical.columns)
        else "(합성 테스트에서는 상세표 생략)"
    )
    case_cols = [
        col for col in ["STN", "지점명", "SIGNIFICANCE_CLASS", "BIAS_none", "BIAS_fixed",
                        "DELTA_ABS_BIAS", "CI95_LOW", "CI95_HIGH"] if col in cases
    ]
    case_display = cases.loc[:, case_cols].copy()
    if "SIGNIFICANCE_CLASS" in case_display:
        case_display["SIGNIFICANCE_CLASS"] = case_display["SIGNIFICANCE_CLASS"].map(
            CLASS_LABELS
        ).fillna(case_display["SIGNIFICANCE_CLASS"])
    case_display_names = {
        "STN": "station",
        "SIGNIFICANCE_CLASS": "분류",
        "BIAS_none": "보정 전 BIAS (°C)",
        "BIAS_fixed": "보정 후 BIAS (°C)",
        "DELTA_ABS_BIAS": "Δ|BIAS| (°C)",
        "CI95_LOW": "95% 구간 하한 (°C)",
        "CI95_HIGH": "95% 구간 상한 (°C)",
    }
    case_display = case_display.rename(columns=case_display_names)
    text = f"""# BIAS 발표 피드백 반영 분석

## 1. 결론 요약

- 유의미한 개선: **{counts['meaningful_improvement']}개**
- 유의미한 악화: **{counts['meaningful_worsening']}개**
- 차이 거의 없음: **{counts['practically_equivalent']}개**
- 판단 불확실: **{counts['uncertain']}개**

기준은 `δ=0.10°C`, 월별 층화 7일 블록 bootstrap 2,000회, station 간 FDR 5%다.
기존의 단순 개선/악화 개수는 방향만 보여주므로 최종 판정으로 사용하지 않는다.

## 2. 변화의 의미 구분

### 관찰

연속형 `Δ|BIAS|`는 station마다 크기가 다르다. 부호만 보면 개선 {raw_improved}개·악화
{raw_worsened}개지만, **{raw_near_zero}개는 변화량 자체가 ±0.10°C 이내**다.

### 수치 근거

위 네 범주는 변화가 ±0.10°C 경계를 넘는지와 시간 자료의 bootstrap 불확실성을 함께 반영했다.
유의미한 악화 {len(worsening)}개의 중앙 악화량은 **{worsening_median:.3f}°C**이고, 0.5°C 초과는
**{worsening_over_half}개**, 1°C 초과는 **{worsening_over_one}개**다. 즉 유의미한 악화는 존재하지만
큰 악화는 일부에 집중된다.

### 물리적 의미

보정 전 BIAS가 이미 0에 가까운 station은 작은 고도보정만 더해져도 수학적으로는 악화가 될 수 있다.
그러나 그 변화가 0.10°C보다 작다면 실제 대기 현상의 큰 실패라기보다 이미 정확한 예측에 작은 이동이
추가된 경우로 해석하는 편이 타당하다.

### 검증 범위와 한계

0.10°C는 유일한 자연법칙이 아니므로 0.05·0.20°C 민감도를 함께 보고한다. 자료가 180일·9개월보다
적은 **{coverage_ineligible}개 station**은 전체 그래프에는 유지하되 연간 판정은 `판단 불확실`로 두었다.
bootstrap은 2024년 내부 시간 변동을 반영하지만 다른 연도 일반화까지 보장하지 않는다.

## 3. ΔZ와 보정 전·후 BIAS

### 관찰

평균 ΔZ와 BIAS의 순위상관은 보정 전 `ρ={rho_before:.3f}`, 보정 후 `ρ={rho_after:.3f}`다.

### 수치 근거

두 패널은 같은 529개 station, 같은 축, 같은 분류색을 사용한다. 따라서 별도 히스토그램의 빈을
간접 대응시키지 않고 각 station의 ΔZ와 BIAS를 직접 비교한다.

평균 절대 ΔZ 구간별 결과는 다음과 같다. 비율 열은 0~1 범위다.

{dz_table}

### 물리적 의미

ΔZ는 target 고도와 가중 이웃 고도의 차이이므로 고정 감률이 이동시키는 온도의 크기와 방향을 정한다.
보정 후 ΔZ–BIAS 관계가 약해진다면 고도에서 비롯된 계통 편향이 제거된 것이다. 다만 같은 ΔZ에서도
기존 BIAS 방향과 실제 기온–고도 관계가 다르면 개선폭은 달라진다.

### 검증 범위와 한계

보정 이동량과 ΔZ의 관계는 구현 항등식이지만, BIAS 개선과 ΔZ의 상관은 인과효과 자체가 아니다.
지형·기온감률·이웃 대표성이 함께 들어간 결과다.

## 4. 월별·시간대별 차이

### 관찰

station 중앙 `Δ|BIAS|`은 주간(09–17시) 중앙 **{day:.3f}°C**, 야간(21–05시) 중앙
**{night:.3f}°C**다.

### 수치 근거

`bias_temporal_physical.png`에 월별·시간대별 보정 전후 BIAS와 BIAS 절댓값 변화를 함께 표시했다.

### 물리적 의미

낮에는 일사로 지표가 데워지고 대류 혼합이 발달해 기온이 고도에 따라 비교적 일관되게 변할 수 있다.
밤에는 복사냉각과 안정층, 계곡의 찬 공기 정체, 역전으로 기온–고도 관계가 약해지거나 뒤집힐 수 있다.
따라서 고정 -6.5°C/km가 낮에는 상대적으로 잘 맞고 밤에는 효과가 작아지는 패턴은 경계층의 일변화와
물리적으로 정합적이다. 월별 차이도 일사·대기 안정도·강수와 감률의 계절변화가 누적된 결과일 수 있다.

### 검증 범위와 한계

현재 자료에는 복사·운량·바람이 없으므로 혼합층이나 역전을 직접 관측한 것은 아니다. 시간 패턴과 기존
감률 진단이 해당 설명에 부합한다는 수준이며 인과 증명으로 표현하지 않는다.

## 5. 지형·이웃 조건

### 관찰

ΔZ 구간, 기존 BIAS 크기, 지형, 이웃 고도 산포·거리·기복별 요약을 같은 station 결과에 연결했다.

### 수치 근거

층화 결과 정본은 `bias_feedback_physical_checks.csv`다. ΔZ가 비슷한 station 안에서 중앙 변화량과
네 분류 비율을 비교하므로 ΔZ 크기와 지형 조건을 혼동하는 문제를 줄였다.

기존 BIAS 크기별 결과는 다음과 같다.

{baseline_table}

지형·이웃 조건은 `terrain_within_delta_z`, `neighbor_elevation_spread_within_delta_z`,
`neighbor_distance_within_delta_z`, `terrain_relief_within_delta_z` 행에서 같은 ΔZ 구간 안으로 제한해 비교했다.

### 물리적 의미

산악·계곡에서는 냉기 배수와 사면 일사 차이가 생기고, 해안에서는 해양의 열용량이 기온 일변화를
완화한다. 이웃이 멀거나 고도가 제각각이면 가중 이웃 기온이 target의 국지 환경을 덜 대표할 수 있다.
이런 과정은 고도 하나로 제거되지 않는 잔여 BIAS를 만든다.

### 검증 범위와 한계

지형과 이웃 변수는 관측 결과와 독립적으로 만든 설명변수지만, 여기서는 층화 연관성만 제시한다.
관측망 배치가 무작위가 아니므로 지형의 순수 인과효과로 해석하지 않는다.

## 6. 대표 station

### 관찰

사례는 결과를 본 뒤 임의로 고르지 않고 각 분류의 사전 규칙으로 선택했다.

### 수치 근거

{_markdown_table(case_display, list(case_display.columns))}

### 물리적 의미

큰 개선 사례는 기존 고도 편향과 보정 방향이 잘 맞은 경우, 큰 악화 사례는 실제 국지 기온–고도 관계가
고정 감률과 맞지 않거나 기존 BIAS와 보정 방향이 겹친 경우다. 차이 없음 사례는 원래 BIAS와 ΔZ가 작아
고도보정이 개입할 여지가 적은 경우를 보여준다.

### 검증 범위와 한계

사례는 전체 분포를 이해하기 위한 예시이며 모집단 결론의 근거를 대신하지 않는다.

## 7. 민감도와 비판적 셀프 리뷰

### 관찰

실질 기준과 블록 길이를 바꿨을 때 기본 분류와의 최소 station 일치율은 **{min_agreement:.3f}**다.

### 수치 근거

`bias_feedback_sensitivity.csv`와 `bias_sensitivity.png`에 12개 조합의 분류수와 일치율을 기록했다.

7일 블록에서 실질 기준만 바꾼 결과는 다음과 같다.

{threshold_table}

### 물리적 의미

임계값에 따라 분류가 바뀌는 station은 효과가 작거나 자연 변동과 비슷한 크기라는 뜻이다. 이는 고도보정의
물리 효과가 전혀 없다는 뜻이 아니라, 현재 1년 자료로 실용적 크기를 안정적으로 구분하기 어렵다는 뜻이다.

### 검증 범위와 한계

월별 층화는 계절 구성 왜곡을 막지만 월 경계를 넘는 장기 기상 지속성은 완전히 보존하지 않는다.
블록 1–14일 민감도로 이 선택의 영향을 제시한다. 529개 station 다중 판정은 FDR로 보정했지만,
공간적으로 가까운 station 간 의존성까지 명시적으로 모형화하지는 않았다.

또한 180일·9개월 coverage 기준은 단기 신규 station을 연간 결과로 오해하지 않기 위한 보호장치다.
자료 분포상 제외된 15개 다음 station은 270일·9개월로 간격이 크지만, 이 기준 자체가 물리 법칙은 아니다.

## 8. 최종 해석

ΔZ는 고도보정량의 크기와 방향을 결정한다. 그러나 BIAS가 실제로 개선되는 정도는 기존 BIAS의 방향,
시간·계절에 따라 달라지는 실제 기온–고도 관계, 그리고 target을 둘러싼 지형·이웃 대표성에 의해 달라진다.
따라서 결과는 단순 개선/악화 개수가 아니라 실질 크기, 통계 불확실성, 물리적 의미, 인과 한계를 함께
제시해야 한다.
"""
    path = out_dir / "bias_feedback_analysis_report.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _staging_directory(out_dir):
    out_dir = Path(out_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    path = out_dir.parent / f".{out_dir.name}.staging-{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _publish(staging, out_dir):
    staging = Path(staging)
    out_dir = Path(out_dir).resolve()
    backup = None
    if out_dir.exists():
        backup = out_dir.parent / f".{out_dir.name}.backup-{uuid.uuid4().hex}"
        os.replace(out_dir, backup)
    try:
        os.replace(staging, out_dir)
    except BaseException:
        if backup is not None and not out_dir.exists():
            os.replace(backup, out_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def run_analysis(out_dir=OUT_DIR, n_boot=PRIMARY_N_BOOT):
    """실제 529 station 분석을 실행하고 완전 산출물을 원자 게시한다."""
    station = pd.read_csv(
        SOURCE_DIR / "bias_station_factors.csv", encoding="utf-8-sig", dtype={"STN": str}
    ).sort_values("STN", key=lambda x: x.astype(int)).reset_index(drop=True)
    if len(station) != 529 or station["STN"].duplicated().any():
        raise AssertionError("expected 529 unique source stations")
    monthly = pd.read_csv(SOURCE_DIR / "bias_monthly.csv", encoding="utf-8-sig", dtype={"STN": str})
    diurnal = pd.read_csv(SOURCE_DIR / "bias_diurnal.csv", encoding="utf-8-sig", dtype={"STN": str})
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    daily = load_daily_paired_errors(station["STN"])
    matrices = build_daily_matrices(daily, station["STN"], dates)
    residuals = validate_annual_against_existing(matrices, station)

    primary, primary_bootstrap = run_station_inference(
        station, matrices, dates, PRIMARY_BLOCK_DAYS, n_boot,
        PRIMARY_DELTA_C, SEED
    )
    sensitivity, stability = run_sensitivity(
        station,
        matrices,
        dates,
        n_boot,
        primary["SIGNIFICANCE_CLASS"].to_numpy(object),
        primary_bootstrap=primary_bootstrap,
    )
    primary["SENSITIVITY_STABILITY"] = stability
    temporal = build_temporal_summary(monthly, diurnal)
    physical = build_physical_checks(primary)
    cases = select_cases(primary)

    staging = _staging_directory(out_dir)
    try:
        _write_csv(primary, staging / "bias_feedback_station_results.csv")
        _write_csv(sensitivity, staging / "bias_feedback_sensitivity.csv")
        _write_csv(temporal, staging / "bias_feedback_temporal.csv")
        _write_csv(physical, staging / "bias_feedback_physical_checks.csv")
        _write_csv(cases, staging / "bias_feedback_cases.csv")
        fig, _ = create_dz_before_after_figure(primary)
        _save_figure(fig, staging, "bias_dz_before_after.png")
        plot_dz_effect(primary, staging)
        plot_significance_summary(primary, staging)
        plot_temporal_physical(temporal, staging)
        plot_case_studies(cases, monthly, staging)
        plot_sensitivity(sensitivity, staging)
        write_feedback_report(primary, temporal, sensitivity, physical, cases, staging)

        counts = primary["SIGNIFICANCE_CLASS"].value_counts().reindex(CLASS_ORDER, fill_value=0)
        summary = {
            "status": "complete",
            "analyzed_stations": int(len(primary)),
            "paired_hours": int(matrices["count"].sum()),
            "delta_c": PRIMARY_DELTA_C,
            "block_days": PRIMARY_BLOCK_DAYS,
            "bootstrap_replicates": int(n_boot),
            "seed": SEED,
            "class_counts": {name: int(counts[name]) for name in CLASS_ORDER},
            "annual_validation": residuals,
            "minimum_sensitivity_agreement": float(sensitivity["agreement_with_primary"].min()),
            "median_sensitivity_stability": float(np.median(stability)),
            "artifacts": sorted(path.name for path in staging.iterdir()) + ["bias_feedback_run_summary.json"],
        }
        (staging / "bias_feedback_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        expected = set(summary["artifacts"])
        actual = {path.name for path in staging.iterdir()}
        if actual != expected:
            raise AssertionError(f"artifact mismatch: expected={expected}, actual={actual}")
        if any((staging / name).stat().st_size == 0 for name in expected):
            raise AssertionError("empty artifact generated")
        if sum(summary["class_counts"].values()) != 529:
            raise AssertionError("significance classes do not sum to 529")
        _publish(staging, out_dir)
        staging = None
        return summary
    finally:
        if staging is not None and Path(staging).exists():
            shutil.rmtree(staging)


def main():
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
