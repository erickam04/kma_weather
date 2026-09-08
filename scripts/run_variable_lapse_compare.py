"""유동적 lapse rate 4종 비교 (none / fixed_lapse / monthly_lapse / band_lapse).

MAE 분석이 지목한 대표 station·월에서 방향성을 확인한다:
  160 부산(레)·554 미시령(고지대 실패) + 871 윗세오름(양호 고지대) + 537 임계(Δ고도≈0).
  4월(봄, 감률 완만) / 12월(겨울, 감률 가파름).

실행: python -m scripts.run_variable_lapse_compare
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN, evaluate_idw
from weather.lapse_rate import BAND_TABLE_PATH, MONTHLY_TABLE_PATH, load_band_table, load_monthly_table
from weather.load_aws import prepare_evaluation_data

STATIONS = ["160", "554", "871", "537", "904", "768"]
MONTHS = ["202404", "202412"]
METHODS = ["none", "fixed_lapse", "monthly_lapse", "band_lapse"]


def score(result):
    if len(result) == 0:
        return None
    return {
        "n": len(result),
        "MAE": round(result["ABS_ERROR"].mean(), 4),
        "RMSE": round((result["ERROR"] ** 2).mean() ** 0.5, 4),
        "BIAS": round(result["ERROR"].mean(), 4),
    }


def main():
    meta = load_station_meta()
    monthly_table = load_monthly_table(MONTHLY_TABLE_PATH)
    band_table = load_band_table(BAND_TABLE_PATH)

    rows = []
    for ym in MONTHS:
        y, m = int(ym[:4]), int(ym[4:])
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthBegin(1)
        print(f"\n===== {ym} 로드 중 =====")
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)

        for method in METHODS:
            result = evaluate_idw(
                data, emeta,
                target_stations=STATIONS,
                elevation_method=method,
                monthly_lapse_table=monthly_table,
                band_lapse_table=band_table,
            )
            if len(result) == 0:
                continue
            result["STN"] = result["STN"].astype(str)
            for stn, g in result.groupby("STN"):
                s = score(g)
                rows.append({"month": ym, "STN": stn, "method": method, **s})

    df = pd.DataFrame(rows)
    # station×월별로 method 비교 (MAE 피벗)
    print("\n\n===== 보정 후 MAE 비교 (°C) =====")
    for ym in MONTHS:
        sub = df[df["month"] == ym]
        piv = sub.pivot(index="STN", columns="method", values="MAE").reindex(
            index=[s for s in STATIONS if s in sub["STN"].values], columns=METHODS
        )
        print(f"\n[{ym}] MAE")
        print(piv.to_string())
    print("\n===== BIAS 비교 (°C) =====")
    for ym in MONTHS:
        sub = df[df["month"] == ym]
        piv = sub.pivot(index="STN", columns="method", values="BIAS").reindex(
            index=[s for s in STATIONS if s in sub["STN"].values], columns=METHODS
        )
        print(f"\n[{ym}] BIAS")
        print(piv.to_string())

    df.to_csv("WeatherData_AWS/interpolation_batch/variable_lapse_compare.csv",
              index=False, encoding="utf-8-sig")
    print("\n저장: WeatherData_AWS/interpolation_batch/variable_lapse_compare.csv")


if __name__ == "__main__":
    main()
