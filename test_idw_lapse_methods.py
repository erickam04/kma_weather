"""IDW와 유동적 lapse rate method 통합 테스트.

pytest 없이 실행 가능: python test_idw_lapse_methods.py
"""

import math

import pandas as pd

from idw import evaluate_idw, idw_interpolate_with_trace


def make_meta():
    return pd.DataFrame([
        {
            "지점": "100", "지점명": "target", "위도": 37.0, "경도": 127.0,
            "노장해발고도(m)": 100.0, "시작일": pd.Timestamp("2020-01-01"),
            "종료일": pd.NaT,
        },
        {
            "지점": "200", "지점명": "neighbor", "위도": 37.0, "경도": 127.01,
            "노장해발고도(m)": 0.0, "시작일": pd.Timestamp("2020-01-01"),
            "종료일": pd.NaT,
        },
    ])


def check(name, cond):
    print(("  OK " if cond else "  FAIL"), name)
    if not cond:
        raise AssertionError(name)


def test_monthly_lapse_uses_timestamp_month_gamma():
    meta = make_meta()
    values = pd.Series({"100": 999.0, "200": 10.0})
    monthly_table = {4: -0.010}

    predicted, trace = idw_interpolate_with_trace(
        "100",
        values,
        meta,
        min_neighbors=1,
        max_neighbors=1,
        elevation_method="monthly_lapse",
        timestamp="2024-04-15 12:00",
        monthly_lapse_table=monthly_table,
    )

    check("monthly_lapse 예측값", math.isclose(predicted, 9.0, abs_tol=1e-12))
    check("trace에 해석된 γ 기록", trace.iloc[0]["lapse_rate"] == -0.010)


def test_band_lapse_uses_piecewise_path_integral():
    meta = make_meta()
    values = pd.Series({"100": 999.0, "200": 10.0})
    band_table = pd.DataFrame([
        {"z_lo": 0.0, "z_hi": 50.0, "gamma": -0.010},
        {"z_lo": 50.0, "z_hi": float("inf"), "gamma": -0.002},
    ])

    predicted, trace = idw_interpolate_with_trace(
        "100",
        values,
        meta,
        min_neighbors=1,
        max_neighbors=1,
        elevation_method="band_lapse",
        band_lapse_table=band_table,
    )

    check("band_lapse 예측값", math.isclose(predicted, 9.4, abs_tol=1e-12))
    check("trace에 구간 method 표시", trace.iloc[0]["lapse_rate"] == "band_lapse")


def test_evaluate_idw_passes_monthly_context_per_timestamp():
    meta = make_meta()
    data = pd.DataFrame(
        [{"100": 999.0, "200": 10.0}],
        index=[pd.Timestamp("2024-04-15 12:00")],
    )

    result = evaluate_idw(
        data,
        meta,
        target_stations=["100"],
        min_neighbors=1,
        max_neighbors=1,
        elevation_method="monthly_lapse",
        monthly_lapse_table={4: -0.010},
    )

    check("evaluate_idw monthly_lapse 예측값",
          len(result) == 1 and math.isclose(result.iloc[0]["PREDICTED"], 9.0, abs_tol=1e-12))


def main():
    test_monthly_lapse_uses_timestamp_month_gamma()
    test_band_lapse_uses_piecewise_path_integral()
    test_evaluate_idw_passes_monthly_context_per_timestamp()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
