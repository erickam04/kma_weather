
from project_paths import META_PATH
from math import atan2, cos, radians, sin, sqrt

import pandas as pd


META_FILE = str(META_PATH)
DEFAULT_MAX_DISTANCE_KM = 30
DEFAULT_MAX_NEIGHBORS = 10


def get_distance_km(lat1, lon1, lat2, lon2):
    """위도/경도로 두 지점 사이의 거리(km)를 계산한다."""
    earth_radius = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius * c


def load_station_meta(meta_file=META_FILE):
    meta = pd.read_csv(meta_file, encoding="cp949")

    meta["시작일"] = pd.to_datetime(meta["시작일"])
    meta["종료일"] = pd.to_datetime(meta["종료일"], errors="coerce")
    meta["위도"] = pd.to_numeric(meta["위도"], errors="coerce")
    meta["경도"] = pd.to_numeric(meta["경도"], errors="coerce")
    meta["노장해발고도(m)"] = pd.to_numeric(meta["노장해발고도(m)"], errors="coerce")
    meta["지점"] = meta["지점"].astype(str)

    meta = meta[
        meta["시작일"].notna()
        & meta["위도"].notna()
        & meta["경도"].notna()
    ].copy()

    return meta


def filter_valid_station_meta(meta, current_time):
    now = pd.to_datetime(current_time)

    valid = meta[
        (meta["시작일"] <= now)
        & ((meta["종료일"].isna()) | (now < meta["종료일"]))
    ].copy()

    valid = valid.sort_values("시작일")
    valid = valid.drop_duplicates(subset=["지점"], keep="last")

    return valid


def filter_station_meta_for_period(meta, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    return meta[
        (meta["시작일"] < end)
        & ((meta["종료일"].isna()) | (start < meta["종료일"]))
    ].copy()


def get_nearby_valid_stations(
    meta,
    target_station_id,
    current_time,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
):
    valid = filter_valid_station_meta(meta, current_time)
    target_station_id = str(target_station_id)
    target_row = valid[valid["지점"] == target_station_id]

    if len(target_row) == 0:
        return pd.DataFrame(columns=list(valid.columns) + ["거리_km"])

    target_lat = target_row.iloc[0]["위도"]
    target_lon = target_row.iloc[0]["경도"]
    neighbors = []

    for _, row in valid.iterrows():
        station_id = str(row["지점"])

        if station_id == target_station_id:
            continue

        distance = get_distance_km(
            target_lat,
            target_lon,
            row["위도"],
            row["경도"],
        )

        if max_distance_km is not None and distance > max_distance_km:
            continue

        item = row.to_dict()
        item["거리_km"] = distance
        neighbors.append(item)

    result = pd.DataFrame(neighbors)

    if len(result) == 0:
        return pd.DataFrame(columns=list(valid.columns) + ["거리_km"])

    result = result.sort_values("거리_km")

    if max_neighbors is not None:
        result = result.head(max_neighbors)

    return result.reset_index(drop=True)


def main():
    meta = load_station_meta()
    current_time = input("기준 날짜 (예: 2024-02-01): ")
    target_station_id = input("기준 관측소 번호: ").strip()

    result = get_nearby_valid_stations(
        meta,
        target_station_id=target_station_id,
        current_time=current_time,
    )

    print()
    print("근처 유효 관측소 수:", len(result))

    if len(result) == 0:
        print("조건에 맞는 관측소가 없습니다.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
