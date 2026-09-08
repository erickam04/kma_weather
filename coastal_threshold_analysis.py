"""해안 임계거리(10km)의 데이터 기반 근거 산출.

문제: 지형 분류의 '해안 ≤10km' 임계는 민숙주·최영은(2019, 웹·원문 미검증) 단일 출처에
의존해 근거가 약했다. → **우리 549지점 실측으로 임계를 직접 도출**한다.

방법(순환성 없음 — MAE와 무관한 물리량 사용):
  - **일교차 DTR**(daily max−min 관측기온의 연평균)은 해양성/대륙성의 고전적 지표.
    바다는 열관성이 커 해안 지점의 일교차를 억제한다.
  - DTR을 해안거리에 대해 **조각선형(broken-stick) 회귀**: DTR = a + b·min(dist, c) + e·elev
    → SSE 최소가 되는 전이점 c 를 탐색. c 이후로는 해안 영향이 포화(기울기 0).
  - 고도는 공변량으로 통제(고지대는 DTR이 다름). 산지 혼입 방지를 위해 저지대만 사용.

실행: python coastal_threshold_analysis.py
출력: 콘솔 리포트 + fullnet/coastal_threshold.csv (구간별 DTR) + 그림
"""

from project_paths import LOO_DB

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

DB = str(LOO_DB)
FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
TERRAIN = os.path.join(FULLNET, "station_terrain.csv")
ELEV_MAX = 200.0        # 저지대만(산지 고도효과 배제)
MIN_HOURS_PER_DAY = 20  # 일교차 산출에 필요한 최소 시간 수


def setup_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_dtr(con):
    """지점별 연평균 일교차(DTR). 관측값만 사용(method 무관)."""
    q = ("SELECT STN, substr(TM,1,10) d, MAX(OBSERVED)-MIN(OBSERVED) dtr, COUNT(*) n "
         "FROM raw WHERE METHOD='none' AND interpolable=1 "
         f"GROUP BY STN, d HAVING n >= {MIN_HOURS_PER_DAY}")
    df = pd.read_sql(q, con)
    df["STN"] = df["STN"].astype(str)
    return df.groupby("STN")["dtr"].mean().rename("DTR")


def breakpoint_fit(x, y, z, grid=np.arange(2, 40.01, 0.5)):
    """DTR = a + b·min(dist,c) + e·elev 의 SSE 최소 c 탐색. 반환 (c, beta, R²)."""
    best = None
    sst = ((y - y.mean()) ** 2).sum()
    for c in grid:
        X = np.column_stack([np.ones_like(x), np.minimum(x, c), z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((y - X @ beta) ** 2).sum()
        if best is None or sse < best[1]:
            best = (float(c), sse, beta)
    c, sse, beta = best
    return c, beta, 1 - sse / sst


def main():
    setup_font()
    os.makedirs(FULLNET, exist_ok=True)
    con = sqlite3.connect(DB)
    dtr = load_dtr(con)
    con.close()

    terr = pd.read_csv(TERRAIN, encoding="utf-8-sig", dtype={"STN": str}).set_index("STN")
    m = terr.join(dtr, how="inner").dropna(subset=["DTR", "coast_dist_km", "elev_m"])
    low = m[m["elev_m"] < ELEV_MAX]
    x = low["coast_dist_km"].to_numpy(); y = low["DTR"].to_numpy(); z = low["elev_m"].to_numpy()

    c, beta, r2 = breakpoint_fit(x, y, z)
    coastal = y[x <= 2].mean(); plateau = y[x >= 20].mean()

    print("=" * 66)
    print("해안 임계거리 — 데이터 기반 도출 (일교차 DTR 기반)")
    print("=" * 66)
    print(f"대상: 저지대(<{ELEV_MAX:.0f}m) {len(low)}지점 (전체 {len(m)})")
    print(f"★ 조각선형 전이점(breakpoint) = {c:.1f} km   (R²={r2:.3f})")
    print(f"   전이점 이전 기울기 +{beta[1]:.3f} °C/km, 이후 포화(0)")
    print(f"   해안(≤2km) DTR {coastal:.2f}°C  vs  내륙(≥20km) {plateau:.2f}°C "
          f"→ 해양 억제폭 {plateau-coastal:.2f}°C")
    for p in (0.80, 0.90, 0.95):
        tgt = coastal + p * (plateau - coastal)
        d = (tgt - beta[0] - beta[2] * z.mean()) / beta[1]
        print(f"   내륙값의 {int(p*100)}% 도달 거리 ≈ {d:.1f} km")

    # 민감도: 고도컷·통계량·격자
    print("\n[민감도]")
    for ecut in (100, 200, 300, 1e9):
        sub = m[m["elev_m"] < ecut]
        cc, _, rr = breakpoint_fit(sub["coast_dist_km"].to_numpy(),
                                   sub["DTR"].to_numpy(), sub["elev_m"].to_numpy())
        print(f"  고도컷 <{'전체' if ecut > 1e8 else int(ecut)}m: 전이점 {cc:.1f} km (n={len(sub)}, R²={rr:.3f})")

    # 구간별 표 저장
    bins = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 200]
    lb = low.copy(); lb["bin"] = pd.cut(lb["coast_dist_km"], bins)
    tbl = lb.groupby("bin", observed=True).agg(
        n=("DTR", "size"), DTR=("DTR", "mean"), elev_mean=("elev_m", "mean")).round(3).reset_index()
    tbl["bin"] = tbl["bin"].astype(str)
    tbl.to_csv(os.path.join(FULLNET, "coastal_threshold.csv"),
               index=False, encoding="utf-8-sig")
    print("\n구간별 평균 DTR:")
    print(tbl.to_string(index=False))

    # 그림
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, s=16, alpha=0.45, color="#2d6cdf", label=f"저지대 지점 (n={len(low)})")
    xs = np.linspace(0, max(40, x.max()), 300)
    ax.plot(xs, beta[0] + beta[1] * np.minimum(xs, c) + beta[2] * z.mean(),
            color="#d62728", lw=2.5, label=f"조각선형 적합 (전이점 {c:.1f} km)")
    ax.axvline(c, color="#d62728", ls="--", alpha=0.6)
    ax.axvline(10, color="#2ca02c", ls=":", lw=2, label="현행 기준 10 km")
    ax.set_xlabel("해안선까지 거리 (km)"); ax.set_ylabel("연평균 일교차 DTR (°C)")
    ax.set_title("해양 영향은 해안 ~8–9km에서 포화 — '해안' 임계의 데이터 근거",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 60); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FULLNET, "coastal_threshold.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\n저장: {out}, coastal_threshold.csv")


if __name__ == "__main__":
    main()
