
from project_paths import AWS_DIR
import os

import pandas as pd

from find_stations import filter_station_meta_for_period


BASE_FOLDER = str(AWS_DIR)


def get_months(start, end):
    expected_times = pd.date_range(
        start=start,
        end=end - pd.Timedelta(hours=1),
        freq="h",
    )
    return expected_times.strftime("%Y%m").unique()


def load_station_series(
    station_id,
    months,
    start,
    end,
    target_column,
    base_folder=BASE_FOLDER,
):
    station_folder = os.path.join(base_folder, f"Station_{station_id}")

    if not os.path.isdir(station_folder):
        return None

    data_list = []

    for month in months:
        file_path = os.path.join(station_folder, f"AWS_{month}.csv")

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            continue

        df["TM"] = pd.to_datetime(df["TM"])
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        df = df[(df["TM"] >= start) & (df["TM"] < end)]

        data_list.append(df[["TM", target_column]])

    if len(data_list) == 0:
        return None

    data = pd.concat(data_list, ignore_index=True)
    data = data.drop_duplicates(subset=["TM"], keep="last")
    data = data.set_index("TM").sort_index()

    return data[target_column]


def load_all_station_data(
    meta,
    start,
    end,
    target_column,
    base_folder=BASE_FOLDER,
):
    months = get_months(start, end)
    station_data = {}

    for station_id in meta["지점"].drop_duplicates():
        series = load_station_series(
            station_id,
            months,
            start,
            end,
            target_column,
            base_folder=base_folder,
        )

        if series is not None:
            station_data[station_id] = series

    if len(station_data) == 0:
        return pd.DataFrame()

    return pd.DataFrame(station_data)


def prepare_evaluation_data(
    meta,
    start,
    end,
    target_column,
    base_folder=BASE_FOLDER,
):
    period_meta = filter_station_meta_for_period(meta, start, end)
    data = load_all_station_data(
        period_meta,
        start,
        end,
        target_column,
        base_folder=base_folder,
    )

    available_stations = set(data.columns)
    evaluation_meta = meta[meta["지점"].isin(available_stations)].copy()
    station_order = [
        station_id
        for station_id in period_meta["지점"].drop_duplicates()
        if station_id in data.columns
    ]
    data = data[station_order]

    return data, evaluation_meta
