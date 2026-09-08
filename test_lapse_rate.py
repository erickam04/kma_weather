"""lapse_rate.py 단위 테스트 (Phase B 계획의 검증 요건).

pytest 없이 실행 가능: python test_lapse_rate.py
전부 통과하면 'ALL TESTS PASSED'를 출력한다.
"""

import math
import sys

try:  # Windows 콘솔(cp949)에서도 한글·기호 출력 보장
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from elevation_correction import (
    FIXED_LAPSE_RATE,
    from_reference_elevation,
    to_reference_elevation,
)
from lapse_rate import (
    GAMMA_MAX,
    GAMMA_MIN,
    band_from_reference,
    band_to_reference,
    build_band_table,
    build_monthly_table,
    clamp_gamma,
    cumulative_offset,
    estimate_gamma_at_time,
    load_band_table,
    load_monthly_table,
    resolve_monthly_gamma,
    save_table,
)

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("  OK " if cond else "  FAIL"), name)
    return cond


def make_synthetic(gamma_per_m, n=200, intercept=15.0, seed=0, noise=0.0):
    """T = intercept + gamma*z (+noise) 인 합성 시각 데이터."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1500, n)
    ta = intercept + gamma_per_m * z + (rng.normal(0, noise, n) if noise else 0.0)
    ids = [str(100 + i) for i in range(n)]
    values = pd.Series(ta, index=ids)
    elev = pd.Series(z, index=ids)
    return values, elev


def test_regression_recovery():
    print("[1] 시각별 회귀가 알려진 γ를 복원하는가")
    values, elev = make_synthetic(-0.0065)
    reg, bands = estimate_gamma_at_time(values, elev)
    check("γ 복원 (오차 <1e-9)", reg is not None and abs(reg["gamma"] - (-0.0065)) < 1e-9)
    check("corr ≈ -1", reg["corr"] < -0.999)
    check("구간 기울기도 γ와 동일", all(abs(b["gamma"] - (-0.0065)) < 1e-9 for b in bands))
    check("구간 pair 수 = 3", len(bands) == 3)


def test_min_stations_gate():
    print("[2] 표본 부족 게이트")
    values, elev = make_synthetic(-0.0065, n=50)  # < MIN_REGRESSION_STATIONS
    reg, _ = estimate_gamma_at_time(values, elev)
    check("n<100 이면 회귀 없음", reg is None)


def test_exclude_stations():
    print("[3] target 제외")
    values, elev = make_synthetic(-0.0065, n=120)
    reg_all, _ = estimate_gamma_at_time(values, elev)
    exclude = [str(100 + i) for i in range(25)]  # 120-25=95 < 100
    reg_ex, _ = estimate_gamma_at_time(values, elev, exclude_stations=exclude)
    check("제외 후 표본 미달 → None", reg_all is not None and reg_ex is None)


def test_clamp():
    print("[4] γ 클램프")
    check("하한", clamp_gamma(-0.05) == GAMMA_MIN)
    check("상한", clamp_gamma(+0.05) == GAMMA_MAX)
    check("범위 내 통과", clamp_gamma(-0.0065) == -0.0065)
    values, elev = make_synthetic(-0.03)  # 비물리적으로 가파름
    reg, bands = estimate_gamma_at_time(values, elev)
    check("회귀 결과 클램프", reg["gamma"] == GAMMA_MIN)
    check("구간 기울기 클램프", all(b["gamma"] == GAMMA_MIN for b in bands))


def test_monthly_table():
    print("[5] 월별 테이블 집계·조회")
    tms = (list(pd.date_range("2024-01-01", periods=5, freq="h"))
           + list(pd.date_range("2024-07-01", periods=5, freq="h")))
    gammas = [-0.005, -0.005, -0.006, -0.004, -0.005,
              -0.008, -0.007, -0.008, -0.009, -0.008]
    df = pd.DataFrame({"TM": tms, "gamma": gammas, "n": 200, "corr": -0.9})
    table = build_monthly_table(df)
    jan = float(table[table["month"] == 1]["gamma"].iloc[0])
    jul = float(table[table["month"] == 7]["gamma"].iloc[0])
    feb = float(table[table["month"] == 2]["gamma"].iloc[0])
    check("1월 중앙값", jan == -0.005)
    check("7월 중앙값", jul == -0.008)
    check("누락 월은 고정 γ fallback", feb == FIXED_LAPSE_RATE)

    lookup = {int(r["month"]): float(r["gamma"]) for _, r in table.iterrows()}
    check("resolve: 1월", resolve_monthly_gamma("2024-01-15 12:00", lookup) == -0.005)
    check("resolve: 테이블 None → 고정", resolve_monthly_gamma("2024-01-15", None) == FIXED_LAPSE_RATE)


def test_band_constant_equals_fixed():
    print("[6] 불변식: 모든 구간 γ 동일 → fixed_lapse와 완전 동일")
    g = -0.0065
    band_df = pd.DataFrame([
        {"TM": "t", "pair": 0, "z_lo": 50.0, "z_hi": 180.0, "gamma": g},
        {"TM": "t", "pair": 1, "z_lo": 180.0, "z_hi": 420.0, "gamma": g},
        {"TM": "t", "pair": 2, "z_lo": 420.0, "z_hi": 900.0, "gamma": g},
    ])
    table = build_band_table(band_df)
    ok = True
    for z in [0.0, 10.0, 99.9, 100.0, 250.0, 512.0, 771.8, 1675.8, 3000.0]:
        for v in [-15.0, 0.0, 27.3]:
            a = band_to_reference(v, z, table)
            b = to_reference_elevation(v, z, method="fixed_lapse", lapse_rate=g)
            c = band_from_reference(v, z, table)
            d = from_reference_elevation(v, z, method="fixed_lapse", lapse_rate=g)
            if not (math.isclose(a, b, abs_tol=1e-12) and math.isclose(c, d, abs_tol=1e-12)):
                ok = False
    check("to/from_reference 완전 일치", ok)


def test_band_piecewise_math():
    print("[7] 경로적분 수치 검증 (수기 계산 대조)")
    table = pd.DataFrame([
        {"z_lo": 0.0, "z_hi": 200.0, "gamma": -0.010},
        {"z_lo": 200.0, "z_hi": 500.0, "gamma": -0.006},
        {"z_lo": 500.0, "z_hi": float("inf"), "gamma": -0.002},
    ])
    # C(100)=-1.0 / C(200)=-2.0 / C(350)=-2.0-0.9=-2.9 / C(800)=-2.0-1.8-0.6=-4.4
    cases = {100.0: -1.0, 200.0: -2.0, 350.0: -2.9, 500.0: -3.8, 800.0: -4.4}
    ok = all(math.isclose(cumulative_offset(z, table), c, abs_tol=1e-12)
             for z, c in cases.items())
    check("C(z) 수기 대조 5건", ok)
    check("경계 연속성 C(200-)≈C(200+)",
          math.isclose(cumulative_offset(199.999999, table),
                       cumulative_offset(200.000001, table), abs_tol=1e-4))
    check("z<0 연장", math.isclose(cumulative_offset(-50.0, table), 0.5, abs_tol=1e-12))


def test_band_identity_and_nan():
    print("[8] 항등성·NaN fallback")
    table = pd.DataFrame([
        {"z_lo": 0.0, "z_hi": 300.0, "gamma": -0.008},
        {"z_lo": 300.0, "z_hi": float("inf"), "gamma": -0.003},
    ])
    v, z = 21.7, 512.0
    check("from(to(v)) == v",
          math.isclose(band_from_reference(band_to_reference(v, z, table), z, table), v,
                       abs_tol=1e-12))
    check("고도 NaN → 원본", band_to_reference(v, float("nan"), table) == v)
    check("값 NaN → NaN 유지", pd.isna(band_to_reference(float("nan"), z, table)))
    check("table None → 원본", band_to_reference(v, z, None) == v)


def test_band_table_build_and_io(tmpdir=None):
    if tmpdir is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            return test_band_table_build_and_io(temp_dir)
    print("[9] band 테이블 구성·저장·로드 round-trip")
    band_df = pd.DataFrame([
        # pair 0: centroid 50→180 구간 기울기 (시각 3개, 중앙값 -0.009)
        {"TM": 1, "pair": 0, "z_lo": 48.0, "z_hi": 178.0, "gamma": -0.010},
        {"TM": 2, "pair": 0, "z_lo": 50.0, "z_hi": 180.0, "gamma": -0.009},
        {"TM": 3, "pair": 0, "z_lo": 52.0, "z_hi": 182.0, "gamma": -0.008},
        # pair 1
        {"TM": 1, "pair": 1, "z_lo": 178.0, "z_hi": 420.0, "gamma": -0.006},
        {"TM": 2, "pair": 1, "z_lo": 180.0, "z_hi": 421.0, "gamma": -0.005},
        {"TM": 3, "pair": 1, "z_lo": 182.0, "z_hi": 422.0, "gamma": -0.007},
        # pair 2
        {"TM": 1, "pair": 2, "z_lo": 420.0, "z_hi": 900.0, "gamma": -0.002},
        {"TM": 2, "pair": 2, "z_lo": 421.0, "z_hi": 905.0, "gamma": -0.003},
        {"TM": 3, "pair": 2, "z_lo": 422.0, "z_hi": 910.0, "gamma": -0.001},
    ])
    table = build_band_table(band_df)
    check("구간 3개, [0,inf) 커버",
          len(table) == 3 and table.iloc[0]["z_lo"] == 0.0
          and math.isinf(table.iloc[-1]["z_hi"]))
    check("경계 = centroid 중앙값(180, 421)",
          table.iloc[0]["z_hi"] == 180.0 and table.iloc[1]["z_hi"] == 421.0)
    check("γ = pair 중앙값", list(table["gamma"]) == [-0.009, -0.006, -0.002])
    check("구간 연속(빈틈 없음)",
          all(table.iloc[i]["z_hi"] == table.iloc[i + 1]["z_lo"]
              for i in range(len(table) - 1)))

    import os
    path = os.path.join(tmpdir, "_test_band_table.csv")
    save_table(table, path)
    loaded = load_band_table(path)
    check("저장→로드 γ 일치",
          np.allclose(loaded["gamma"], table["gamma"])
          and math.isinf(loaded.iloc[-1]["z_hi"]))
    os.remove(path)

    mpath = os.path.join(tmpdir, "_test_monthly_table.csv")
    mt = build_monthly_table(pd.DataFrame({
        "TM": pd.date_range("2024-03-01", periods=3, freq="h"),
        "gamma": [-0.004, -0.003, -0.005], "n": 200, "corr": -0.9,
    }))
    save_table(mt, mpath)
    lookup = load_monthly_table(mpath)
    check("monthly 저장→로드 (3월)", lookup[3] == -0.004)
    os.remove(mpath)


def main():
    for fn in [test_regression_recovery, test_min_stations_gate, test_exclude_stations,
               test_clamp, test_monthly_table, test_band_constant_equals_fixed,
               test_band_piecewise_math, test_band_identity_and_nan,
               test_band_table_build_and_io]:
        fn()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print()
    if n_fail == 0:
        print(f"ALL TESTS PASSED ({len(PASS)} checks)")
    else:
        print(f"{n_fail}/{len(PASS)} FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
