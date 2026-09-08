"""고정 lapse rate 최적값 탐색 (TODO #2 실험).

근거 판단(계획서 `.omc/plans/todo2-optimal-fixed-lapse.md`): 선행연구(Dodson ΔZ-편향0 방법)
+ 현재 데이터(BIAS_none~ΔZ 기울기 6.60, 고ΔZ 지점 연평균 γ −6.35~−7.05) → 후보 −6.6~−6.7.
이 스크립트는 그 판단을 **IDW 재실행 없이** 기존 batch 산출물로 확인한다.

핵심 항등식 (알고리즘 무수정, 기존 CSV만 읽음):
    ΔZ_시각 = (ERROR_fixed − ERROR_none) / (−6.5)     # 시각별 ΔZ 역산
    ERROR(γ) = ERROR_none + γ · ΔZ_시각                # 임의 γ의 시각별 오차
γ=−6.5에서 ERROR(γ)=ERROR_fixed는 구성상 자명 → 검증은 "재구성 overall MAE가
analysis_summary와 일치하는가"로 한다.

산출: MAE(γ) 곡선 그림 + station별 최적 γ 표 + 콘솔 리포트.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

import pipeline_config as cfg

BATCH = cfg.OUT_BATCH
CURRENT_LAPSE = -6.5  # 현행 fixed_lapse (°C/km 아님 주의: elevation_correction는 °C/m=-0.0065)
GAMMA_GRID = np.arange(-9.0, -3.999, 0.05)  # °C/km 격자
SUMMARY = os.path.join(BATCH, "analysis_summary.csv")
OUT_CSV = os.path.join(BATCH, "optimal_fixed_lapse.csv")


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_paired():
    """월별 fixed_lapse·none raw를 로드해 (TM,STN)로 병합, ΔZ_시각 역산."""
    fixed = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig")
         for p in sorted(glob.glob(os.path.join(BATCH, "raw_2024??_fixed_lapse.csv")))],
        ignore_index=True,
    )
    none = pd.concat(
        [pd.read_csv(p, encoding="utf-8-sig")
         for p in sorted(glob.glob(os.path.join(BATCH, "raw_2024??_none.csv")))],
        ignore_index=True,
    )
    for df in (fixed, none):
        df["STN"] = df["STN"].astype(str)
    m = fixed[["TM", "STN", "ERROR"]].merge(
        none[["TM", "STN", "ERROR"]], on=["TM", "STN"], suffixes=("_fixed", "_none")
    )
    # ΔZ_시각 = (ERROR_fixed − ERROR_none) / γ_현행.  γ=CURRENT_LAPSE(°C/km), ΔZ 단위 km.
    m["dZ_km"] = (m["ERROR_fixed"] - m["ERROR_none"]) / CURRENT_LAPSE
    return m


def error_at(m, gamma):
    """임의 γ(°C/km)에서 시각별 오차."""
    return m["ERROR_none"] + gamma * m["dZ_km"]


def verify(m):
    """재구성 overall MAE(γ=−6.5)가 analysis_summary(fixed) overall과 일치하는지."""
    recon = np.abs(error_at(m, CURRENT_LAPSE)).mean()
    s = pd.read_csv(SUMMARY)
    ov = s[(s["axis"] == "overall") & (s["METHOD"] == "fixed_lapse")]["MAE"]
    ref = float(ov.iloc[0]) if len(ov) else float("nan")
    ok = abs(recon - ref) < 0.005
    print(f"[검증] 재구성 overall MAE(γ=-6.5)={recon:.4f} vs summary={ref:.4f} "
          f"diff={abs(recon-ref):.5f} [{'OK' if ok else 'MISMATCH'}]")
    return ok


def main():
    setup_korean_font()
    m = load_paired()
    print(f"병합 행: {len(m):,} | station: {sorted(m['STN'].unique())}")
    verify(m)

    # 전체 MAE(γ) 곡선 (pooled)
    overall = np.array([np.abs(error_at(m, g)).mean() for g in GAMMA_GRID])
    best_overall = GAMMA_GRID[int(np.argmin(overall))]
    mae_at_current = np.abs(error_at(m, CURRENT_LAPSE)).mean()
    mae_at_best = overall.min()

    # station별 MAE(γ)와 최적 γ
    rows = []
    per_station_curves = {}
    for stn, g in m.groupby("STN"):
        curve = np.array([np.abs(g["ERROR_none"] + gm * g["dZ_km"]).mean() for gm in GAMMA_GRID])
        per_station_curves[stn] = curve
        best = GAMMA_GRID[int(np.argmin(curve))]
        rows.append({
            "STN": stn, "n": len(g),
            "MAE_curr(-6.5)": np.abs(g["ERROR_none"] + CURRENT_LAPSE * g["dZ_km"]).mean(),
            "best_gamma": best,
            "MAE_best": curve.min(),
            "mean_dZ_km": g["dZ_km"].mean(),
        })
    st = pd.DataFrame(rows).sort_values("mean_dZ_km")
    st["dMAE(curr-best)"] = st["MAE_curr(-6.5)"] - st["MAE_best"]

    print("\n=== 전체(pooled) ===")
    print(f"  현행 -6.5 MAE = {mae_at_current:.4f}")
    print(f"  최적 γ = {best_overall:.2f}  → MAE = {mae_at_best:.4f}  "
          f"(개선 {mae_at_current - mae_at_best:.4f}, {100*(mae_at_current-mae_at_best)/mae_at_current:.2f}%)")

    print("\n=== station별 (ΔZ 오름차순) ===")
    print(st.round(4).to_string(index=False))

    st.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n표 저장: {OUT_CSV}")

    # 그림
    viz = cfg.viz_dir("mae_analysis")
    os.makedirs(viz, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(GAMMA_GRID, overall, color="#2d6cdf", lw=2)
    ax1.axvline(CURRENT_LAPSE, color="#888", ls="--", label=f"현행 -6.5 ({mae_at_current:.4f})")
    ax1.axvline(best_overall, color="#d62728", ls="-",
                label=f"최적 {best_overall:.2f} ({mae_at_best:.4f})")
    ax1.set_xlabel("고정 lapse rate γ (°C/km)")
    ax1.set_ylabel("전체 MAE (°C)")
    ax1.set_title("전체(pooled) MAE(γ) 곡선", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    for stn in st["STN"]:
        lbl = f"{stn} (dZ={st.set_index('STN').loc[stn,'mean_dZ_km']:.2f})"
        ax2.plot(GAMMA_GRID, per_station_curves[stn], label=lbl, lw=1.3)
    ax2.axvline(CURRENT_LAPSE, color="#888", ls="--")
    ax2.set_xlabel("고정 lapse rate γ (°C/km)")
    ax2.set_ylabel("station별 MAE (°C)")
    ax2.set_title("station별 MAE(γ) — 최적 γ가 지점마다 다름", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(viz, "optimal_fixed_lapse_curve.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"그림 저장: {out}")


if __name__ == "__main__":
    main()
