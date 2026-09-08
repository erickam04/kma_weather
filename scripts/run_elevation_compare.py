import os

import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN, evaluate_idw, print_score
from weather.lapse_rate import load_band_table, load_monthly_table
from weather.load_aws import prepare_evaluation_data


# 현재 확보된 데이터 범위(2024-02)에 맞춘 고정 비교 설정
START_DATE = "2024-02-01"
END_DATE = "2024-02-29"  # inclusive (내부적으로 +1일 처리)
TARGET_STATIONS = ["116"]
METHODS = ["none", "fixed_lapse", "monthly_lapse", "band_lapse"]
OUTPUT_FOLDER = "WeatherData_AWS/interpolation_test"


def score_dict(result):
    if len(result) == 0:
        return {"samples": 0, "MAE": None, "RMSE": None, "BIAS": None}

    return {
        "samples": len(result),
        "MAE": round(result["ABS_ERROR"].mean(), 4),
        "RMSE": round((result["ERROR"] ** 2).mean() ** 0.5, 4),
        "BIAS": round(result["ERROR"].mean(), 4),
    }


def load_lapse_tables(methods):
    monthly_table = load_monthly_table() if "monthly_lapse" in methods else None
    band_table = load_band_table() if "band_lapse" in methods else None
    return monthly_table, band_table


def main():
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)

    meta = load_station_meta()
    data, evaluation_meta = prepare_evaluation_data(
        meta,
        start,
        end,
        target_column=TARGET_COLUMN,
    )

    print()
    print("평가 기간:", START_DATE, "~", END_DATE)
    print("대상 관측소:", TARGET_STATIONS)
    print("사용 가능한 관측소 수:", len(data.columns))
    print("평가 기간 row 수:", len(data))
    print("보간 대상 변수:", TARGET_COLUMN)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    summary = {}
    monthly_table, band_table = load_lapse_tables(METHODS)

    for method in METHODS:
        print()
        print("=" * 40)
        print("고도보정 방식:", method)

        result = evaluate_idw(
            data,
            evaluation_meta,
            target_stations=TARGET_STATIONS,
            elevation_method=method,
            monthly_lapse_table=monthly_table,
            band_lapse_table=band_table,
        )
        print_score(result)

        save_path = os.path.join(
            OUTPUT_FOLDER, f"idw_result_{TARGET_COLUMN}_{method}.csv"
        )
        result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print("Saved:", save_path)

        summary[method] = score_dict(result)

    print()
    print("=" * 40)
    print("===== 고도보정 방식별 성능 비교 =====")
    comparison = pd.DataFrame(summary).T
    print(comparison.to_string())


if __name__ == "__main__":
    main()
