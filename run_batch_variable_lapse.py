"""2024 전체 batch — 유동적 lapse rate(monthly_lapse, band_lapse) 평가.

baseline(none/fixed_lapse)의 batch_raw_all.csv·target_stations.csv는 건드리지 않는다.
동일한 target station(pick_target_stations)에 대해 새 method 2종만 실행하여
별도 파일(raw_{ym}_{method}.csv, batch_variable_raw.csv)에 저장한다.

실행: python run_batch_variable_lapse.py
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from find_stations import load_station_meta
from idw import TARGET_COLUMN, evaluate_idw
from lapse_rate import BAND_TABLE_PATH, MONTHLY_TABLE_PATH, load_band_table, load_monthly_table
from load_aws import prepare_evaluation_data
from run_batch_test import discover_months, pick_target_stations

YEAR = 2024
METHODS = ["monthly_lapse", "band_lapse"]
OUTPUT_FOLDER = "WeatherData_AWS/interpolation_batch"


def main():
    meta = load_station_meta()
    monthly_table = load_monthly_table(MONTHLY_TABLE_PATH)
    band_table = load_band_table(BAND_TABLE_PATH)

    picks = pick_target_stations(meta)
    targets = picks["지점"].tolist()
    months = discover_months(YEAR)
    print("target:", targets)
    print("월:", months)

    all_results = []
    for ym in months:
        y, m = int(ym[:4]), int(ym[4:])
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthBegin(1)
        print(f"\n===== {ym} 로드 =====")
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)

        for method in METHODS:
            result = evaluate_idw(
                data, emeta,
                target_stations=targets,
                elevation_method=method,
                monthly_lapse_table=monthly_table,
                band_lapse_table=band_table,
            )
            if len(result) == 0:
                continue
            result = result.copy()
            result["MONTH"] = ym
            result["METHOD"] = method
            all_results.append(result)
            part = os.path.join(OUTPUT_FOLDER, f"raw_{ym}_{method}.csv")
            result.to_csv(part, index=False, encoding="utf-8-sig")
            print(f"  {method}: {len(result)} rows -> {part}")

    if not all_results:
        print("결과 없음")
        return
    combined = pd.concat(all_results, ignore_index=True)
    out = os.path.join(OUTPUT_FOLDER, "batch_variable_raw.csv")
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n전체 저장: {out}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
