"""fast_idw 를 golden(batch_raw_all.csv) 전체와 대조 (8 target × 12개월 × 2method).

evaluate_idw_fast 가 원본과 수치적으로 동일한지 135,958행 전량 검증 + 속도 측정.
합격 기준(계획서 §10): 상대 1e-6. (실측은 ~1e-15 기계정밀도 기대.)
"""

import time

import numpy as np
import pandas as pd

from find_stations import load_station_meta
from idw import TARGET_COLUMN
from load_aws import prepare_evaluation_data
from fast_idw import evaluate_idw_fast

GOLDEN = "WeatherData_AWS/interpolation_batch/batch_raw_all.csv"


def main():
    golden = pd.read_csv(GOLDEN, encoding="utf-8-sig", dtype={"STN": str, "MONTH": str})
    targets = sorted(golden["STN"].unique())
    months = sorted(golden["MONTH"].unique())
    print(f"golden: {len(golden)} rows, {len(targets)} target, {len(months)} month")

    meta = load_station_meta()
    out = []
    t_load = t_compute = 0.0
    for ym in months:
        y, mo = int(ym[:4]), int(ym[4:])
        start = pd.Timestamp(year=y, month=mo, day=1)
        end = start + pd.offsets.MonthBegin(1)
        t0 = time.perf_counter()
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)
        t_load += time.perf_counter() - t0
        for method in ["none", "fixed_lapse"]:
            t0 = time.perf_counter()
            r = evaluate_idw_fast(data, emeta, target_stations=targets,
                                  elevation_method=method)
            t_compute += time.perf_counter() - t0
            r = r.copy(); r["MONTH"] = ym; r["METHOD"] = method
            out.append(r)

    new = pd.concat(out, ignore_index=True)
    new["TM"] = new["TM"].astype(str); new = new.astype({"STN": str})
    print(f"\nfast_idw: {len(new)} rows | 로드 {t_load:.1f}s + 계산 {t_compute:.1f}s "
          f"= {t_load + t_compute:.1f}s (원본 단일코어 추정 ~380h)")

    g = golden.copy(); g["TM"] = g["TM"].astype(str)
    merged = g.merge(new, on=["TM", "STN", "METHOD"], how="outer",
                     suffixes=("_g", "_n"), indicator=True)
    only_g = (merged["_merge"] == "left_only").sum()
    only_n = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()
    print(f"\n행 대조: both={both}, golden전용={only_g}, new전용={only_n}")

    b = merged[merged["_merge"] == "both"]
    print("\n=== 컬럼별 최대 절대차 ===")
    ok = (only_g == 0 and only_n == 0)
    for col in ["OBSERVED", "PREDICTED", "ERROR", "ABS_ERROR", "NEIGHBOR_COUNT"]:
        d = (b[f"{col}_g"] - b[f"{col}_n"]).abs().max()
        print(f"  {col:14s}: max|Δ|={d:.3e}")

    print("\n=== method별 PREDICTED (상대오차) ===")
    for method in ["none", "fixed_lapse"]:
        s = b[b["METHOD"] == method]
        d = (s["PREDICTED_g"] - s["PREDICTED_n"]).abs()
        denom = s["PREDICTED_g"].abs().replace(0, np.nan)
        rel = (d / denom).max()
        verdict = "OK" if rel < 1e-6 else "FAIL"
        if rel >= 1e-6:
            ok = False
        print(f"  {method:12s}: max|Δ|={d.max():.3e}  max_rel={rel:.3e}  [{verdict}]")

    print("\n" + ("[PASS] fast_idw ≡ golden (1e-6 이내)" if ok else "[FAIL]"))


if __name__ == "__main__":
    main()
