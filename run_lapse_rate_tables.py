import os

import pandas as pd

from find_stations import load_station_meta
from idw import TARGET_COLUMN
from lapse_rate import (
    BAND_TABLE_PATH,
    HOURLY_GAMMA_PATH,
    MONTHLY_TABLE_PATH,
    build_band_table,
    build_monthly_table,
    estimate_gamma_series,
    save_table,
)
from load_aws import prepare_evaluation_data


YEAR = 2024
OUTPUT_FOLDER = "WeatherData_AWS/interpolation_batch"
TARGET_STATIONS_PATH = os.path.join(OUTPUT_FOLDER, "target_stations.csv")
BAND_HOURLY_PATH = os.path.join(OUTPUT_FOLDER, "hourly_band_gamma_2024.csv")


def load_excluded_targets(path=TARGET_STATIONS_PATH):
    """lapse rate 추정에서 제외할 batch target station 목록."""
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "지점" not in df.columns:
        return []
    return df["지점"].astype(str).tolist()


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    start = pd.Timestamp(year=YEAR, month=1, day=1)
    end = pd.Timestamp(year=YEAR + 1, month=1, day=1)
    excluded_targets = load_excluded_targets()

    print("lapse rate 테이블 생성")
    print("기간:", start, "~", end, "[end exclusive]")
    print("보간 대상 변수:", TARGET_COLUMN)
    print("회귀 제외 target:", excluded_targets if excluded_targets else "(없음)")

    meta = load_station_meta()
    data, evaluation_meta = prepare_evaluation_data(
        meta,
        start,
        end,
        target_column=TARGET_COLUMN,
    )
    print("사용 가능 station:", len(data.columns), "| row:", len(data))

    gamma_df, band_df = estimate_gamma_series(
        data,
        evaluation_meta,
        exclude_stations=excluded_targets,
    )
    print("전국 회귀 성공 시각:", len(gamma_df))
    print("구간 pair 추정 row:", len(band_df))

    gamma_df.to_csv(HOURLY_GAMMA_PATH, index=False, encoding="utf-8-sig")
    band_df.to_csv(BAND_HOURLY_PATH, index=False, encoding="utf-8-sig")

    monthly_table = build_monthly_table(gamma_df)
    band_table = build_band_table(band_df)
    save_table(monthly_table, MONTHLY_TABLE_PATH)
    save_table(band_table, BAND_TABLE_PATH)

    print()
    print("월별 γ 테이블 저장:", MONTHLY_TABLE_PATH)
    print(monthly_table.assign(gamma_C_per_km=monthly_table["gamma"] * 1000).to_string(index=False))
    print()
    print("고도구간 γ 테이블 저장:", BAND_TABLE_PATH)
    print(band_table.assign(gamma_C_per_km=band_table["gamma"] * 1000).to_string(index=False))
    print()
    print("시각별 γ 저장:", HOURLY_GAMMA_PATH)
    print("시각별 구간 pair 저장:", BAND_HOURLY_PATH)


if __name__ == "__main__":
    main()
