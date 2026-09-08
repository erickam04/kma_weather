"""논문용 station 해발고도 연속 분석.

기존 station×hour 표준지표를 읽어 시간대·방법별로 지표와 해발고도의
OLS 기울기, R², Spearman 순위상관을 계산한다. 별도 개선량은 만들지 않는다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect/standard_metrics_station_hour.csv"
OUT_DIR = ROOT / "WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect"
PLOT_DIR = OUT_DIR / "plots"
METHODS = (("none", "고도보정 미적용", "#2878B5"), ("fixed_lapse", "고정감률 적용", "#D9534F"))
METRICS = (("MAE_C", "MAE"), ("BIAS_C", "BIAS"), ("RMSE_C", "RMSE"))


def rank_corr(x, y):
    xr = pd.Series(x).rank(method="average").to_numpy(float)
    yr = pd.Series(y).rank(method="average").to_numpy(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def fit_relation(frame, metric):
    x = frame["elev_m"].to_numpy(float)
    y = frame[metric].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "N_STATIONS": int(len(frame)),
        "SLOPE_C_PER_100M": float(slope * 100.0),
        "INTERCEPT_C": float(intercept),
        "R2": float(1.0 - residual / total) if total > 0 else np.nan,
        "SPEARMAN_RHO": rank_corr(x, y),
        "MEDIAN_C": float(np.median(y)),
    }


def build_summary(data):
    records = []
    for hour in range(24):
        for method, _, _ in METHODS:
            part = data.loc[(data["HOUR"] == hour) & (data["METHOD"] == method)]
            for metric, metric_label in METRICS:
                record = {"HOUR": hour, "METHOD": method, "METRIC": metric_label}
                record.update(fit_relation(part, metric))
                records.append(record)
    return pd.DataFrame(records)


def plot_metric(data, summary, metric, metric_label, output_path):
    fig = plt.figure(figsize=(12.5, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 0.85), hspace=0.34, wspace=0.20)
    x_line = np.linspace(float(data["elev_m"].min()), float(data["elev_m"].max()), 200)

    for column, hour in enumerate((6, 15)):
        ax = fig.add_subplot(grid[0, column])
        for method, label, color in METHODS:
            part = data.loc[(data["HOUR"] == hour) & (data["METHOD"] == method)]
            fit = summary.loc[
                (summary["HOUR"] == hour)
                & (summary["METHOD"] == method)
                & (summary["METRIC"] == metric_label)
            ].iloc[0]
            ax.scatter(part["elev_m"], part[metric], s=11, alpha=0.25, color=color, edgecolors="none")
            ax.plot(
                x_line,
                fit["INTERCEPT_C"] + fit["SLOPE_C_PER_100M"] / 100.0 * x_line,
                color=color,
                linewidth=2.2,
                label=f"{label}: {fit['SLOPE_C_PER_100M']:+.3f}°C/100m",
            )
        ax.set_title(f"{hour:02d}시", fontsize=13, fontweight="bold")
        ax.set_xlabel("관측소 해발고도 (m)")
        ax.set_ylabel(f"{metric_label} (°C)")
        ax.grid(alpha=0.25)
        if metric_label == "BIAS":
            ax.axhline(0, color="#555555", linewidth=1)
        ax.legend(fontsize=9, frameon=False)

    ax = fig.add_subplot(grid[1, :])
    for method, label, color in METHODS:
        part = summary.loc[(summary["METHOD"] == method) & (summary["METRIC"] == metric_label)]
        ax.plot(part["HOUR"], part["SLOPE_C_PER_100M"], marker="o", markersize=4, linewidth=2.2, color=color, label=label)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set(xlabel="시각 (KST)", ylabel=f"{metric_label}–고도 기울기 (°C/100m)", xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.suptitle(f"관측소 해발고도에 따른 시간대별 {metric_label}: 고도보정 전·후", fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    data = pd.read_csv(INPUT, encoding="utf-8-sig")
    required = {"STN", "elev_m", "HOUR", "METHOD", "BIAS_C", "MAE_C", "RMSE_C"}
    if missing := required.difference(data.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if data["STN"].nunique() != 513 or len(data) != 513 * 24 * 2:
        raise ValueError("expected 513 stations × 24 hours × 2 methods")
    if data[list(required - {"STN", "METHOD"})].isna().any().any():
        raise ValueError("continuous analysis input contains missing values")

    summary = build_summary(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "continuous_elevation_hourly_relations.csv", index=False, encoding="utf-8-sig")
    for metric, label in METRICS:
        number = {"MAE": "07", "BIAS": "08", "RMSE": "09"}[label]
        plot_metric(data, summary, metric, label, PLOT_DIR / f"{number}_continuous_elevation_{label.lower()}.png")

    selected = summary.loc[summary["HOUR"].isin([6, 15])].copy()
    print(selected[["HOUR", "METHOD", "METRIC", "SLOPE_C_PER_100M", "R2", "SPEARMAN_RHO"]].to_string(index=False))


if __name__ == "__main__":
    main()
