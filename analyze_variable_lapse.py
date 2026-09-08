"""유동적 lapse rate batch 결과 분석 (4종 종합).

baseline batch(none/fixed_lapse; batch_raw_all.csv)와
유동 batch(monthly/band; batch_variable_raw.csv)를 합쳐:
  - method × (overall/season/station) MAE·RMSE·BIAS·ERR_STD
  - 계통(|BIAS|) vs 랜덤(ERR_STD) 분해
  - station별 4종 MAE 그림, 계절 그림
  - 성공기준 대조 (overall 비악화, 160 봄·554 겨울 개선 여부)

실행: python analyze_variable_lapse.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import math

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

import pipeline_config as cfg

BATCH_DIR = "WeatherData_AWS/interpolation_batch"
VIZ_DIR = cfg.viz_dir("variable_lapse")
METHOD_ORDER = ["none", "fixed_lapse", "monthly_lapse", "band_lapse"]
TARGETS_CSV = f"{BATCH_DIR}/target_stations.csv"


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def season_of(ym):
    m = int(str(ym)[4:])
    return {12: "1_winter", 1: "1_winter", 2: "1_winter",
            3: "2_spring", 4: "2_spring", 5: "2_spring",
            6: "3_summer", 7: "3_summer", 8: "3_summer",
            9: "4_fall", 10: "4_fall", 11: "4_fall"}[m]


def decompose(g):
    err = g["ERROR"].to_numpy()
    mae = np.mean(np.abs(err))
    bias = float(np.mean(err))
    rmse = math.sqrt(float(np.mean(err ** 2)))
    std = math.sqrt(max(rmse ** 2 - bias ** 2, 0.0))
    return pd.Series({"n": len(g), "MAE": mae, "RMSE": rmse,
                      "BIAS": bias, "ERR_STD": std})


def main():
    base = pd.read_csv(f"{BATCH_DIR}/batch_raw_all.csv", encoding="utf-8-sig")
    var = pd.read_csv(f"{BATCH_DIR}/batch_variable_raw.csv", encoding="utf-8-sig")
    df = pd.concat([base, var], ignore_index=True)
    df["STN"] = df["STN"].astype(str)
    df["season"] = df["MONTH"].apply(season_of)

    tg = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig")
    tg["지점"] = tg["지점"].astype(str)
    name_of = dict(zip(tg["지점"], tg["지점명"]))
    elev_of = dict(zip(tg["지점"], tg["노장해발고도(m)"]))

    # overall
    print("===== overall (method별) =====")
    ov = df.groupby("METHOD").apply(decompose).reindex(METHOD_ORDER)
    print(ov.round(4).to_string())

    # station × method MAE
    print("\n===== station별 MAE (method별) =====")
    st = df.groupby(["STN", "METHOD"]).apply(decompose).reset_index()
    piv = st.pivot(index="STN", columns="METHOD", values="MAE").reindex(columns=METHOD_ORDER)
    piv = piv.reindex(sorted(piv.index, key=lambda s: elev_of.get(s, 0)))
    piv.index = [f"{s} {name_of.get(s,'')}({elev_of.get(s,0):.0f}m)" for s in piv.index]
    print(piv.round(4).to_string())

    # season × method MAE
    print("\n===== 계절별 MAE (method별) =====")
    se = df.groupby(["season", "METHOD"]).apply(decompose).reset_index()
    spiv = se.pivot(index="season", columns="METHOD", values="MAE").reindex(columns=METHOD_ORDER)
    print(spiv.round(4).to_string())

    # 성공기준 대조
    print("\n===== 성공기준 대조 =====")
    fixed_ov = ov.loc["fixed_lapse", "MAE"]
    for meth in ["monthly_lapse", "band_lapse"]:
        d = ov.loc[meth, "MAE"] - fixed_ov
        print(f"  overall {meth}: {ov.loc[meth,'MAE']:.4f} (fixed {fixed_ov:.4f}, Δ{d:+.4f}) "
              f"{'비악화' if d <= 0.001 else '악화'}")
    # 160 봄 / 554 겨울
    def cell(stn, season, meth):
        s = df[(df.STN == stn) & (df.season == season) & (df.METHOD == meth)]
        return decompose(s)["MAE"] if len(s) else float("nan")
    for stn, season, label in [("160", "2_spring", "160 봄"), ("554", "1_winter", "554 겨울")]:
        f = cell(stn, season, "fixed_lapse")
        for meth in ["monthly_lapse", "band_lapse"]:
            v = cell(stn, season, meth)
            print(f"  {label} {meth}: {v:.4f} (fixed {f:.4f}, Δ{v-f:+.4f}) "
                  f"{'개선' if v < f else '악화/무변'}")

    # 저장
    st.to_csv(f"{BATCH_DIR}/variable_lapse_station_scores.csv", index=False, encoding="utf-8-sig")
    ov.to_csv(f"{BATCH_DIR}/variable_lapse_overall_scores.csv", encoding="utf-8-sig")

    # 그림 1: station별 4종 MAE
    setup_korean_font()
    os.makedirs(VIZ_DIR, exist_ok=True)
    order = sorted(df["STN"].unique(), key=lambda s: elev_of.get(s, 0))
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, meth in enumerate(METHOD_ORDER):
        vals = [decompose(df[(df.STN == s) & (df.METHOD == meth)])["MAE"]
                if len(df[(df.STN == s) & (df.METHOD == meth)]) else np.nan for s in order]
        ax.bar(x + (i - 1.5) * w, vals, w, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{name_of.get(s,'')}" for s in order], fontsize=9)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("고도보정 method별 보정 후 MAE (2024, 고도 오름차순)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{VIZ_DIR}/variable_lapse_mae_by_station.png", dpi=150)
    plt.close(fig)

    # 그림 2: 계절별 overall MAE
    fig, ax = plt.subplots(figsize=(10, 6))
    seasons = ["1_winter", "2_spring", "3_summer", "4_fall"]
    x = np.arange(len(seasons))
    for i, meth in enumerate(METHOD_ORDER):
        vals = [spiv.loc[s, meth] if s in spiv.index else np.nan for s in seasons]
        ax.plot(x, vals, "o-", label=meth)
    ax.set_xticks(x); ax.set_xticklabels(["겨울", "봄", "여름", "가을"])
    ax.set_ylabel("MAE (°C)")
    ax.set_title("계절별 보정 후 MAE (method별, 2024 전 target)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{VIZ_DIR}/variable_lapse_mae_by_season.png", dpi=150)
    plt.close(fig)

    print(f"\n저장: variable_lapse_station_scores.csv, overall_scores.csv, "
          f"그림 2종({VIZ_DIR}/variable_lapse_mae_by_*.png)")


if __name__ == "__main__":
    main()
