import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


import pipeline_config as cfg

BATCH_FOLDER = cfg.OUT_BATCH
RAW_PATH = os.path.join(BATCH_FOLDER, "batch_raw_all.csv")
TARGETS_PATH = os.path.join(BATCH_FOLDER, "target_stations.csv")
OUT_FOLDER = cfg.viz_dir("batch_performance")
ELEVATION_COLUMN = "노장해발고도(m)"
YEAR = cfg.YEAR

METHOD_LABEL = {
    "none": "무보정",
    "fixed_lapse": "고정 lapse rate",
    "monthly_lapse": "월별 lapse rate",
    "band_lapse": "고도구간 lapse rate",
}
COLORS = {
    "none": "#c0504d",
    "fixed_lapse": "#4472c4",
    "monthly_lapse": "#9bbb59",
    "band_lapse": "#8064a2",
}


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def season_of(month):
    return {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
            6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}[month]


def metrics(df):
    return pd.Series({
        "MAE": df["ABS_ERROR"].mean(),
        "RMSE": (df["ERROR"] ** 2).mean() ** 0.5,
        "BIAS": df["ERROR"].mean(),
    })


def elevation_band(elev):
    if elev < 100:
        return "0-100m"
    if elev < 500:
        return "100-500m"
    return "500m+"


def grouped_bar(ax, categories, none_vals, fixed_vals, ylabel, title):
    x = np.arange(len(categories))
    w = 0.38
    ax.bar(x - w / 2, none_vals, w, label=METHOD_LABEL["none"], color=COLORS["none"])
    ax.bar(x + w / 2, fixed_vals, w, label=METHOD_LABEL["fixed_lapse"], color=COLORS["fixed_lapse"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (n, f) in enumerate(zip(none_vals, fixed_vals)):
        ax.text(i - w / 2, n, f"{n:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, f, f"{f:.2f}", ha="center", va="bottom", fontsize=8)


def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    setup_korean_font()

    df = pd.read_csv(RAW_PATH)
    df["STN"] = df["STN"].astype(str)
    df["TM"] = pd.to_datetime(df["TM"])
    df["month_num"] = df["TM"].dt.month
    df["season"] = df["month_num"].apply(season_of)

    targets = pd.read_csv(TARGETS_PATH)
    targets["지점"] = targets["지점"].astype(str)
    elev_map = dict(zip(targets["지점"], targets[ELEVATION_COLUMN]))
    df["target_elev"] = df["STN"].map(elev_map)
    df["elev_band"] = df["target_elev"].apply(elevation_band)

    def summary(keys):
        return df.groupby(keys + ["METHOD"])[["ERROR", "ABS_ERROR"]].apply(metrics).reset_index()

    # ---- 1. 고도구간별 MAE ----
    s = summary(["elev_band"])
    order = ["0-100m", "100-500m", "500m+"]
    none = [s[(s.elev_band == b) & (s.METHOD == "none")]["MAE"].values[0] for b in order]
    fixed = [s[(s.elev_band == b) & (s.METHOD == "fixed_lapse")]["MAE"].values[0] for b in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, order, none, fixed, "MAE (°C)", f"고도구간별 기온 보간 MAE ({YEAR}년 12개월)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FOLDER, "batch_mae_by_elevation.png"), dpi=130); plt.close(fig)

    # ---- 2. 계절별 MAE ----
    s = summary(["season"])
    order = ["봄", "여름", "가을", "겨울"]
    none = [s[(s.season == b) & (s.METHOD == "none")]["MAE"].values[0] for b in order]
    fixed = [s[(s.season == b) & (s.METHOD == "fixed_lapse")]["MAE"].values[0] for b in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped_bar(ax, order, none, fixed, "MAE (°C)", f"계절별 기온 보간 MAE ({YEAR}년)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FOLDER, "batch_mae_by_season.png"), dpi=130); plt.close(fig)

    # ---- 3. 월별 MAE 추이 (선 그래프) ----
    s = summary(["month_num"])
    months = sorted(df["month_num"].unique())
    none = [s[(s.month_num == m) & (s.METHOD == "none")]["MAE"].values[0] for m in months]
    fixed = [s[(s.month_num == m) & (s.METHOD == "fixed_lapse")]["MAE"].values[0] for m in months]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(months, none, "o-", color=COLORS["none"], label=METHOD_LABEL["none"])
    ax.plot(months, fixed, "s-", color=COLORS["fixed_lapse"], label=METHOD_LABEL["fixed_lapse"])
    ax.set_xticks(months); ax.set_xlabel("월"); ax.set_ylabel("MAE (°C)")
    ax.set_title(f"월별 기온 보간 MAE 추이 ({YEAR}년)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FOLDER, "batch_mae_by_month.png"), dpi=130); plt.close(fig)

    # ---- 4. 고도 vs 개선폭(ΔMAE) 산점도 ----
    s = summary(["STN"])
    piv = s.pivot_table(index="STN", columns="METHOD", values="MAE")
    piv["dMAE"] = piv["none"] - piv["fixed_lapse"]
    piv["elev"] = piv.index.map(elev_map)
    piv = piv.sort_values("elev")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(piv["elev"], piv["dMAE"], s=70, color="#4472c4", zorder=3)
    for stn, row in piv.iterrows():
        ax.annotate(stn, (row["elev"], row["dMAE"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("target station 해발고도 (m)")
    ax.set_ylabel("ΔMAE = MAE(무보정) - MAE(보정)  (°C)")
    ax.set_title(f"고도 vs 고도보정 개선폭 (station별, {YEAR}년)", fontsize=13, fontweight="bold")
    ax.text(0.02, 0.95, "위쪽일수록 보정 효과 큼", transform=ax.transAxes,
            fontsize=9, color="#555", va="top")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FOLDER, "batch_elevation_vs_improvement.png"), dpi=130); plt.close(fig)

    print("저장 완료:", OUT_FOLDER)
    for f in ["batch_mae_by_elevation.png", "batch_mae_by_season.png",
              "batch_mae_by_month.png", "batch_elevation_vs_improvement.png"]:
        print("  -", f)


if __name__ == "__main__":
    main()
