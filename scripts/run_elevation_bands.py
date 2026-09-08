import os

import pandas as pd

from weather.find_stations import load_station_meta, filter_valid_station_meta
from weather.idw import TARGET_COLUMN, ELEVATION_COLUMN, evaluate_idw
from weather.load_aws import prepare_evaluation_data


START_DATE = "2024-02-01"
END_DATE = "2024-02-29"
METHODS = ["none", "fixed_lapse"]
OUTPUT_FOLDER = "WeatherData_AWS/interpolation_test"
REF_TIME = "2024-02-15"


def score_dict(result):
    if len(result) == 0:
        return {"samples": 0, "MAE": None, "RMSE": None, "BIAS": None}
    return {
        "samples": len(result),
        "MAE": round(result["ABS_ERROR"].mean(), 4),
        "RMSE": round((result["ERROR"] ** 2).mean() ** 0.5, 4),
        "BIAS": round(result["ERROR"].mean(), 4),
    }


def pick_target_stations(meta, data):
    """데이터가 있는 station 중 고도 스펙트럼을 대표하도록 몇 개 선정."""
    valid = filter_valid_station_meta(meta, REF_TIME)
    valid = valid[valid["지점"].isin(data.columns)]
    valid = valid[valid[ELEVATION_COLUMN].notna()].sort_values(ELEVATION_COLUMN)

    n = len(valid)
    # 최저 / 25% / 중앙 / 75% / 최고 위치의 station
    idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    picks = valid.iloc[idxs][["지점", "지점명", ELEVATION_COLUMN]]
    return picks


def main():
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)

    meta = load_station_meta()
    data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)

    picks = pick_target_stations(meta, data)
    print("선정된 target station (고도순):")
    print(picks.to_string(index=False))

    rows = []
    for _, p in picks.iterrows():
        stn = p["지점"]
        elev = p[ELEVATION_COLUMN]
        rec = {"STN": stn, "name": p["지점명"], "elev_m": round(elev, 1)}
        for method in METHODS:
            result = evaluate_idw(
                data, emeta, target_stations=[stn], elevation_method=method
            )
            s = score_dict(result)
            rec[f"MAE_{method}"] = s["MAE"]
            rec[f"RMSE_{method}"] = s["RMSE"]
            rec[f"BIAS_{method}"] = s["BIAS"]
        rows.append(rec)
        print("done:", stn, p["지점명"], round(elev, 1))

    table = pd.DataFrame(rows)
    print()
    print("===== 고도대별 고도보정 효과 =====")
    print(table.to_string(index=False))

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_FOLDER, "elevation_bands_summary.csv")
    table.to_csv(save_path, index=False, encoding="utf-8-sig")
    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
