"""Stage 0 불변식 검증 (계획서 §10 Stage 0 AC).

신러너(new_batch_loo.run_loo)가 기존 golden(batch_raw_all.csv)을
(TM,STN,METHOD) 단위로 재현하는지 대조한다.
  - 불변식: method='none' 은 baseline과 완전 동일(exact, diff==0).
  - 러너대조: 같은 evaluate_idw 호출이므로 fixed_lapse 도 사실상 bit-identical
    기대. 계획서 허용오차는 상대 1e-6 (부동소수 합산순서 차이 흡수).

golden 전체(8 target × 12개월 × 2 method)를 병렬 재생산 → 완전 대조.
출력: KMA_WORK_DIR/loo_validate (OneDrive 밖).
"""

from project_paths import WORK_DIR

import numpy as np
import pandas as pd

from new_batch_loo import run_loo

GOLDEN = "WeatherData_AWS/interpolation_batch/batch_raw_all.csv"
OUT_DIR = str(WORK_DIR / "loo_validate")
COMBINE = str(WORK_DIR / "loo_validate" / "combined.csv")
KEYS = ["TM", "STN", "METHOD"]
VALCOLS = ["OBSERVED", "PREDICTED", "ERROR", "ABS_ERROR", "NEIGHBOR_COUNT"]


def main():
    golden = pd.read_csv(GOLDEN, encoding="utf-8-sig", dtype={"STN": str, "MONTH": str})
    targets = sorted(golden["STN"].unique())
    months = sorted(golden["MONTH"].unique())
    print(f"golden: {len(golden)} rows, {len(targets)} target, {len(months)} month")

    new = run_loo(
        targets=targets, months=months,
        methods=["none", "fixed_lapse"],
        out_dir=OUT_DIR, combine_to=COMBINE, processes=12,
    )
    if new is None:
        print("[FAIL] 신러너 결과 없음")
        return

    new = new.astype({"STN": str, "MONTH": str})
    print(f"new: {len(new)} rows")

    # (TM,STN,METHOD) 로 merge (TM은 문자열로 통일)
    g = golden.copy(); g["TM"] = g["TM"].astype(str)
    n = new.copy(); n["TM"] = n["TM"].astype(str)
    merged = g.merge(n, on=KEYS, how="outer", suffixes=("_g", "_n"), indicator=True)

    only_g = (merged["_merge"] == "left_only").sum()
    only_n = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()
    print(f"\n행 대조: both={both}, golden전용={only_g}, new전용={only_n}")

    ok = (only_g == 0 and only_n == 0)
    b = merged[merged["_merge"] == "both"]
    print("\n=== 컬럼별 최대 절대차 ===")
    for col in VALCOLS:
        diff = (b[f"{col}_g"] - b[f"{col}_n"]).abs()
        print(f"  {col:14s}: max|Δ|={diff.max():.3e}")

    # 불변식: none exact
    print("\n=== 불변식 (method별 PREDICTED max|Δ|) ===")
    for method in ["none", "fixed_lapse"]:
        sub = b[b["METHOD"] == method]
        d = (sub["PREDICTED_g"] - sub["PREDICTED_n"]).abs()
        # 상대오차
        denom = sub["PREDICTED_g"].abs().replace(0, np.nan)
        rel = (d / denom).max()
        tag = "exact 요구" if method == "none" else "rel<1e-6 요구"
        verdict = "OK" if (d.max() == 0 if method == "none" else d.max() < 1e-6) else "FAIL"
        print(f"  {method:12s}: max|Δ|={d.max():.3e}  max_rel={rel:.3e}  [{verdict}] ({tag})")
        if method == "none" and d.max() != 0:
            ok = False
        if method == "fixed_lapse" and d.max() >= 1e-6:
            ok = False

    print("\n" + ("[PASS] Stage 0 불변식 통과" if ok else "[FAIL] 불일치 존재"))


if __name__ == "__main__":
    main()
