"""보정 후(fixed_lapse) 절대 MAE가 station마다 다른 이유를 세 가설로 귀인하는 분석.

가설:
  H1 고도보정 잔차   — target-이웃 고도차(Δelev), 이웃 고도 산포
  H2 공간 분포       — 이웃 수·거리·방위 커버리지·weight 집중도
  H3 IDW 고유 한계   — 랜덤 오차 성분, 이웃수-오차 상관, 이웃 기온 산포

기존 로드/필터/IDW 모듈은 호출만 하고 알고리즘은 수정하지 않는다 (CLAUDE.md).

산출물:
  WeatherData_AWS/interpolation_batch/station_mae_factors.csv
  WeatherData_AWS/interpolation_visualization/mae_analysis/mae_decomposition.png
  WeatherData_AWS/interpolation_visualization/mae_analysis/mae_vs_factors.png
  WeatherData_AWS/interpolation_visualization/mae_analysis/mae_month_heatmap.png

실행: python analyze_station_mae.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from find_stations import (
    DEFAULT_MAX_DISTANCE_KM,
    DEFAULT_MAX_NEIGHBORS,
    filter_valid_station_meta,
    load_station_meta,
)
from idw import (
    ELEVATION_COLUMN,
    MIN_NEIGHBORS,
    POWER,
    TARGET_COLUMN,
    idw_interpolate_with_trace,
)
from load_aws import prepare_evaluation_data
import pipeline_config as cfg

BATCH_DIR = Path("WeatherData_AWS/interpolation_batch")
VIZ_DIR = Path(cfg.viz_dir("mae_analysis"))
BATCH_RAW = BATCH_DIR / "batch_raw_all.csv"
TARGETS_CSV = BATCH_DIR / "target_stations.csv"
SUMMARY_CSV = BATCH_DIR / "analysis_summary.csv"
FACTORS_CSV = BATCH_DIR / "station_mae_factors.csv"

# 보간 불가(반경30km 내 이웃 1개)로 batch 결과가 없는 고립 station → 분석 제외.
EXCLUDE_STATIONS = {"530"}

# 이웃 기하 표본: 각 월 15일 정오(관측 결측 시 같은 날 유효시각으로 대체).
SAMPLE_MONTHS = [f"2024-{m:02d}-15" for m in range(1, 13)]
SAMPLE_HOUR = "12:00"
METHOD = "fixed_lapse"


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def bearing_deg(lat1, lon1, lat2, lon2):
    """target(1)에서 이웃(2)을 바라보는 방위각(0~360, 북=0, 시계방향)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def max_angular_gap(bearings):
    """이웃 방위각들을 원 둘레에 배치했을 때 가장 큰 빈 각도(도).

    360에 가까울수록 한쪽으로 완전히 쏠린 것, 작을수록 사방에 고루 분포.
    """
    b = sorted(bearings)
    if len(b) == 0:
        return float("nan")
    if len(b) == 1:
        return 360.0
    gaps = [b[i + 1] - b[i] for i in range(len(b) - 1)]
    gaps.append(360.0 - b[-1] + b[0])  # wrap-around 구간
    return max(gaps)


def load_targets():
    df = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig")
    df["지점"] = df["지점"].astype(str)
    return df[~df["지점"].isin(EXCLUDE_STATIONS)].reset_index(drop=True)


def mae_profile(raw):
    """batch_raw_all에서 fixed_lapse의 station별 오차 프로파일과 분해."""
    sub = raw[raw["METHOD"] == METHOD].copy()
    rows = []
    for stn, g in sub.groupby("STN"):
        err = g["ERROR"].to_numpy()
        mae = np.mean(np.abs(err))
        bias = np.mean(err)
        rmse = math.sqrt(np.mean(err ** 2))
        std = math.sqrt(max(rmse ** 2 - bias ** 2, 0.0))  # 랜덤(분산) 성분
        # RMSE^2 = BIAS^2 + VAR (정확 분해) → 계통 성분 비중
        systematic_share = (bias ** 2) / (rmse ** 2) if rmse > 0 else float("nan")
        # 이웃수-오차 상관 (시간축; H2/H3 근거). 음(-)이면 이웃 적을수록 오차↑.
        if g["NEIGHBOR_COUNT"].nunique() > 1:
            corr = np.corrcoef(g["NEIGHBOR_COUNT"], g["ABS_ERROR"])[0, 1]
        else:
            corr = float("nan")
        rows.append(
            {
                "STN": str(stn),
                "samples": len(g),
                "MAE": mae,
                "RMSE": rmse,
                "BIAS": bias,
                "ERR_STD": std,
                "systematic_share": systematic_share,
                "corr_ncount_abserr": corr,
            }
        )
    return pd.DataFrame(rows)


def monthly_mae(raw):
    sub = raw[raw["METHOD"] == METHOD].copy()
    sub["STN"] = sub["STN"].astype(str)
    piv = (
        sub.assign(ABS=sub["ERROR"].abs())
        .groupby(["STN", "MONTH"])["ABS"]
        .mean()
        .unstack("MONTH")
    )
    return piv


def geometry_metrics(meta, targets):
    """월별 대표 시각의 trace를 재계산해 station별 이웃 기하 지표(평균)를 산출."""
    target_ids = list(targets["지점"])
    per_sample = {sid: [] for sid in target_ids}

    for day in SAMPLE_MONTHS:
        start = pd.to_datetime(day)
        end = start + pd.Timedelta(days=1)
        print(f"  기하 표본 로드: {day}")
        data, eval_meta = prepare_evaluation_data(meta, start, end, TARGET_COLUMN)
        preferred = pd.to_datetime(f"{day} {SAMPLE_HOUR}")

        for sid in target_ids:
            if sid not in data.columns:
                continue
            col = data[sid]
            valid_idx = col[col.notna()].index
            if len(valid_idx) == 0:
                continue
            t = preferred if preferred in valid_idx else valid_idx[
                np.abs((valid_idx - preferred).values).argmin()
            ]
            values_at_time = data.loc[t]
            valid_meta = filter_valid_station_meta(eval_meta, t)
            trow = valid_meta[valid_meta["지점"] == sid]
            if len(trow) == 0:
                continue
            trow = trow.iloc[0]
            _, trace = idw_interpolate_with_trace(
                target_station_id=sid,
                values_at_time=values_at_time,
                meta=valid_meta,
                power=POWER,
                max_distance_km=DEFAULT_MAX_DISTANCE_KM,
                min_neighbors=MIN_NEIGHBORS,
                max_neighbors=DEFAULT_MAX_NEIGHBORS,
                elevation_method=METHOD,
            )
            if len(trace) == 0:
                continue

            w = trace["weight_normalized"].to_numpy()
            dist = trace["distance_km"].to_numpy()
            elev = pd.to_numeric(trace["elevation_m"], errors="coerce").to_numpy()
            raw_vals = pd.to_numeric(trace["raw_value"], errors="coerce").to_numpy()
            target_elev = pd.to_numeric(trow.get(ELEVATION_COLUMN), errors="coerce")

            bearings = [
                bearing_deg(trow["위도"], trow["경도"], r["latitude"], r["longitude"])
                for _, r in trace.iterrows()
            ]
            elev_mask = ~np.isnan(elev)
            w_elev = (
                float(np.sum(w[elev_mask] * elev[elev_mask]) / np.sum(w[elev_mask]))
                if elev_mask.any()
                else float("nan")
            )

            per_sample[sid].append(
                {
                    "n_neighbors": len(trace),
                    "eff_neighbors": 1.0 / float(np.sum(w ** 2)),
                    "max_weight": float(np.max(w)),
                    "w_mean_dist": float(np.sum(w * dist)),
                    "nearest_dist": float(np.min(dist)),
                    "max_angular_gap": max_angular_gap(bearings),
                    "delta_elev": float(target_elev - w_elev)
                    if not np.isnan(w_elev) and pd.notna(target_elev)
                    else float("nan"),
                    "neighbor_elev_std": float(np.nanstd(elev)),
                    "neighbor_val_std": float(np.nanstd(raw_vals)),
                }
            )

    rows = []
    for sid, samples in per_sample.items():
        if not samples:
            continue
        df = pd.DataFrame(samples)
        agg = {"STN": sid, "geom_samples": len(df)}
        for c in df.columns:
            agg[c] = float(np.nanmean(df[c]))
        rows.append(agg)
    return pd.DataFrame(rows)


def verify_against_summary(profile):
    """재계산 MAE가 analysis_summary.csv(fixed_lapse)와 일치하는지 확인."""
    s = pd.read_csv(SUMMARY_CSV)
    st = s[(s["axis"] == "station") & (s["METHOD"] == METHOD)].copy()
    st["STN"] = st["STN"].astype(int).astype(str)
    ref = st.set_index("STN")["MAE"]
    print("\n=== 재현성 검증 (재계산 MAE vs analysis_summary) ===")
    ok = True
    for _, r in profile.iterrows():
        expected = ref.get(r["STN"])
        if expected is None:
            continue
        diff = abs(r["MAE"] - expected)
        flag = "OK" if diff < 0.001 else "MISMATCH"
        if diff >= 0.001:
            ok = False
        print(f"  {r['STN']}: recomputed={r['MAE']:.4f} summary={expected:.4f} diff={diff:.5f} [{flag}]")
    print("  => 전부 일치" if ok else "  => 불일치 존재 (조사 필요)")
    return ok


def plot_decomposition(factors, path):
    setup_korean_font()
    f = factors.sort_values("노장해발고도(m)")
    labels = [f"{r.STN}\n{r.지점명}" for r in f.itertuples()]
    x = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, f["ERR_STD"], label="랜덤 성분 (오차 표준편차 → H3)", color="#4aa564")
    ax.bar(x, f["BIAS"].abs(), bottom=f["ERR_STD"], label="계통 성분 |BIAS| (→ H1/H2)", color="#d1913c")
    ax.plot(x, f["MAE"], "o-", color="#d62728", label="MAE (실측)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("°C")
    ax.set_title("보정 후 오차 분해: 계통(|BIAS|) vs 랜덤(표준편차) — 고도 오름차순", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mae_vs_factors(factors, path):
    setup_korean_font()
    specs = [
        ("delta_elev", "Δ고도 = target - 이웃가중평균 (m)  [H1]"),
        ("max_angular_gap", "최대 방위 공백 (도)  [H2 쏠림]"),
        ("w_mean_dist", "가중평균 이웃거리 (km)  [H2]"),
        ("eff_neighbors", "유효 이웃수 1/Σw²  [H2]"),
        ("neighbor_val_std", "이웃 기온 산포 std (°C)  [H3]"),
        ("corr_ncount_abserr", "이웃수-오차 상관  [H2/H3]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, title) in zip(axes.ravel(), specs):
        if col not in factors:
            ax.set_visible(False)
            continue
        ax.scatter(factors[col], factors["MAE"], s=60, c="#2d6cdf")
        for r in factors.itertuples():
            ax.annotate(f"{r.STN} {r.지점명}", (getattr(r, col), r.MAE),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel(title)
        ax.set_ylabel("보정 후 MAE (°C)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("보정 후 MAE vs 설명변수 (station 8개, 상관은 참고용 — 인과 단정 아님)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_month_heatmap(piv, factors, path):
    setup_korean_font()
    order = factors.sort_values("노장해발고도(m)")["STN"].tolist()
    piv = piv.reindex(order)
    labels = [f"{r.STN} {r.지점명}" for r in factors.sort_values("노장해발고도(m)").itertuples()]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(int(c))[-2:] + "월" for c in piv.columns], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="MAE (°C)")
    ax.set_title("station × 월 보정 후 MAE (고도 오름차순)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_station_meta()
    targets = load_targets()
    print(f"분석 대상 {len(targets)} station (530 태하 제외): {list(targets['지점'])}")

    raw = pd.read_csv(BATCH_RAW, encoding="utf-8-sig")
    raw["STN"] = raw["STN"].astype(str)

    print("\n[1/3] 오차 프로파일·분해 ...")
    profile = mae_profile(raw)
    piv = monthly_mae(raw)

    print("[2/3] 이웃 기하 지표 (월별 표본) ...")
    geom = geometry_metrics(meta, targets)

    factors = (
        targets.rename(columns={"지점": "STN"})
        .merge(profile, on="STN", how="left")
        .merge(geom, on="STN", how="left")
    )

    verify_against_summary(profile)

    factors.to_csv(FACTORS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n요인 표 저장: {FACTORS_CSV}")

    print("[3/3] 그림 생성 ...")
    plot_decomposition(factors, VIZ_DIR / "mae_decomposition.png")
    plot_mae_vs_factors(factors, VIZ_DIR / "mae_vs_factors.png")
    plot_month_heatmap(piv, factors, VIZ_DIR / "mae_month_heatmap.png")

    cols = ["STN", "지점명", "노장해발고도(m)", "MAE", "BIAS", "ERR_STD",
            "systematic_share", "delta_elev", "max_angular_gap", "w_mean_dist",
            "eff_neighbors", "neighbor_val_std", "corr_ncount_abserr"]
    show = factors[[c for c in cols if c in factors]].sort_values("노장해발고도(m)")
    print("\n=== station별 요인 요약 (고도 오름차순) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
