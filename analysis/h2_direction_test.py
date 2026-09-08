"""H2 방위검정 — 이웃을 방향으로 쪼개 재보간 (사수 조언 방식).

가설: "이웃이 target 주변 어느 방향에 있는지는 보간 오차와 무관하다"(n=8 R²=0.054).
검정: 각 지점을 (a) 전체 이웃 (b) 북쪽 이웃만 (c) 남쪽 이웃만 으로 각각 fixed_lapse 보간.
  방향에 따라 보간값이 안 달라지면(=north≈south≈all) 방향 무관 결론.
  달라지면 방향(이웃 편중)이 오차원.

알고리즘 무수정 — fast_idw(direction=...)로 이웃 반구 필터만 적용. 전체 549, 2024.
공정 비교: 세 방향 모두 성공한 시각(inner join)에서만 대조 + 반구별 이웃≥3 필요.
출력: interpolation_batch/fullnet/h2_direction_*.csv + 요약 print.
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN
from weather.load_aws import BASE_FOLDER, prepare_evaluation_data
from weather.fast_idw import evaluate_idw_fast

OUT = "WeatherData_AWS/interpolation_batch/fullnet"
YEAR = 2024


def discover_months():
    import glob
    ms = set()
    for p in glob.glob(os.path.join(BASE_FOLDER, "Station_*", f"AWS_{YEAR}*.csv")):
        ms.add(os.path.basename(p).replace("AWS_", "").replace(".csv", ""))
    return sorted(ms)


def _run_month(ym):
    """한 달: all/north/south fixed_lapse 보간 후 (TM,STN) inner merge."""
    import contextlib, io
    meta = load_station_meta()
    y, mo = int(ym[:4]), int(ym[4:])
    start = pd.Timestamp(year=y, month=mo, day=1)
    end = start + pd.offsets.MonthBegin(1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)
        res = {}
        for tag, d in [("all", None), ("N", "north"), ("S", "south")]:
            r = evaluate_idw_fast(data, emeta, target_stations=None,
                                  elevation_method="fixed_lapse", direction=d)
            res[tag] = r[["TM", "STN", "OBSERVED", "PREDICTED"]].rename(
                columns={"PREDICTED": f"P_{tag}"})
    m = res["all"].merge(res["N"][["TM", "STN", "P_N"]], on=["TM", "STN"])
    m = m.merge(res["S"][["TM", "STN", "P_S"]], on=["TM", "STN"])
    m["STN"] = m["STN"].astype(str)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    months = discover_months()
    nproc = min(len(months), os.cpu_count() or 1)
    print(f"H2 방위검정: {len(months)}개월 × 3방향(all/N/S) fixed_lapse, procs={nproc}")
    with Pool(nproc) as pool:
        parts = pool.map(_run_month, months)
    df = pd.concat(parts, ignore_index=True)
    print(f"세 방향 모두 성공한 시각(공통): {len(df):,} (지점 {df.STN.nunique()})")

    # 오차
    df["e_all"] = df["P_all"] - df["OBSERVED"]
    df["e_N"] = df["P_N"] - df["OBSERVED"]
    df["e_S"] = df["P_S"] - df["OBSERVED"]
    df["diff_NS"] = df["P_N"] - df["P_S"]

    print("\n=== 전체(공통 시각) MAE 대조 ===")
    print(f"  전체이웃  MAE={df.e_all.abs().mean():.4f}")
    print(f"  북쪽만    MAE={df.e_N.abs().mean():.4f}")
    print(f"  남쪽만    MAE={df.e_S.abs().mean():.4f}")
    print(f"\n  북-남 예측차 |P_N - P_S|: 평균={df.diff_NS.abs().mean():.4f}  "
          f"중앙={df.diff_NS.abs().median():.4f}  90%={df.diff_NS.abs().quantile(0.9):.4f}")
    print(f"  (참고: 전체 MAE ~0.85 대비 이 차이가 작으면 방향 영향 미미)")

    # 지점별
    g = df.groupby("STN").agg(
        n=("OBSERVED", "size"),
        MAE_all=("e_all", lambda x: x.abs().mean()),
        MAE_N=("e_N", lambda x: x.abs().mean()),
        MAE_S=("e_S", lambda x: x.abs().mean()),
        mean_absdiff_NS=("diff_NS", lambda x: x.abs().mean()),
    ).reset_index()
    g["MAE_N_minus_all"] = g["MAE_N"] - g["MAE_all"]
    g["MAE_S_minus_all"] = g["MAE_S"] - g["MAE_all"]
    g.to_csv(os.path.join(OUT, "h2_direction_by_station.csv"),
             index=False, encoding="utf-8-sig")

    # H2 핵심: 방향 예측차가 큰 지점이 실제로 MAE도 높은가? (상관)
    def r2(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
        if len(x) < 3:
            return np.nan, 0
        b, a = np.polyfit(x, y, 1); p = a + b * x
        ssr = np.sum((y - p) ** 2); sst = np.sum((y - y.mean()) ** 2)
        return (1 - ssr / sst if sst > 0 else np.nan), len(x)

    r2_v, n_v = r2(g["mean_absdiff_NS"], g["MAE_all"])
    print("\n" + "=" * 56)
    print("H2 결론 (전체 549)")
    print("=" * 56)
    print(f"지점 수(공통 시각 보유): {len(g)}")
    print(f"북쪽만 MAE − 전체 MAE : 중앙 {g.MAE_N_minus_all.median():+.4f} "
          f"(악화 지점 {(g.MAE_N_minus_all>0).mean():.0%})")
    print(f"남쪽만 MAE − 전체 MAE : 중앙 {g.MAE_S_minus_all.median():+.4f} "
          f"(악화 지점 {(g.MAE_S_minus_all>0).mean():.0%})")
    print(f"방향 예측차(|P_N−P_S|) ~ MAE 상관 R²={r2_v:.3f} (n={n_v})")
    print(f"  → R²이 낮고 MAE_N≈MAE_S≈MAE_all이면 '방향 무관' 확증(n=8 R²=0.054와 정합)")


if __name__ == "__main__":
    main()
