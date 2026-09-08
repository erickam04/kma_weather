import pandas as pd

from weather.elevation_correction import (
    FIXED_LAPSE_RATE,
    REFERENCE_ELEVATION_M,
    from_reference_elevation,
    to_reference_elevation,
)
from weather.find_stations import (
    DEFAULT_MAX_DISTANCE_KM,
    DEFAULT_MAX_NEIGHBORS,
    filter_valid_station_meta,
    get_distance_km,
)
from weather.lapse_rate import (
    band_from_reference,
    band_to_reference,
    resolve_monthly_gamma,
)


TARGET_COLUMN = "TA"
POWER = 2
MIN_NEIGHBORS = 3
ELEVATION_COLUMN = "노장해발고도(m)"
IDW_RESULT_COLUMNS = [
    "TM",
    "STN",
    "OBSERVED",
    "PREDICTED",
    "ERROR",
    "ABS_ERROR",
    "NEIGHBOR_COUNT",
]
IDW_TRACE_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "elevation_m",
    "source_elevation_m",
    "reference_elevation_m",
    "raw_value",
    "value_at_reference",
    "elevation_adjustment",
    "lapse_rate",
    "value",
    "distance_km",
    "weight",
    "weight_normalized",
    "contribution",
]


def _empty_trace():
    return pd.DataFrame(columns=IDW_TRACE_COLUMNS)


def _resolve_lapse_rate(elevation_method, timestamp, lapse_rate, monthly_lapse_table):
    if elevation_method == "monthly_lapse":
        return resolve_monthly_gamma(timestamp, monthly_lapse_table)
    return lapse_rate


def _to_reference_value(
    value,
    source_elevation,
    elevation_method,
    effective_lapse_rate,
    band_lapse_table,
):
    if elevation_method == "band_lapse":
        return band_to_reference(value, source_elevation, band_lapse_table)

    method = "fixed_lapse" if elevation_method == "monthly_lapse" else elevation_method
    return to_reference_elevation(
        value,
        source_elevation,
        method=method,
        lapse_rate=effective_lapse_rate,
    )


def _from_reference_value(
    value,
    target_elevation,
    elevation_method,
    effective_lapse_rate,
    band_lapse_table,
):
    if elevation_method == "band_lapse":
        return band_from_reference(value, target_elevation, band_lapse_table)

    method = "fixed_lapse" if elevation_method == "monthly_lapse" else elevation_method
    return from_reference_elevation(
        value,
        target_elevation,
        method=method,
        lapse_rate=effective_lapse_rate,
    )


def _trace_lapse_rate(elevation_method, effective_lapse_rate):
    if elevation_method == "none":
        return pd.NA
    if elevation_method == "band_lapse":
        return "band_lapse"
    return effective_lapse_rate


def idw_interpolate(
    target_station_id,
    values_at_time,
    meta,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    elevation_method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    timestamp=None,
    monthly_lapse_table=None,
    band_lapse_table=None,
):
    predicted, trace = idw_interpolate_with_trace(
        target_station_id=target_station_id,
        values_at_time=values_at_time,
        meta=meta,
        power=power,
        max_distance_km=max_distance_km,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
        elevation_method=elevation_method,
        lapse_rate=lapse_rate,
        timestamp=timestamp,
        monthly_lapse_table=monthly_lapse_table,
        band_lapse_table=band_lapse_table,
    )

    return predicted, len(trace)


def idw_interpolate_with_trace(
    target_station_id,
    values_at_time,
    meta,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    elevation_method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    timestamp=None,
    monthly_lapse_table=None,
    band_lapse_table=None,
):
    target_row = meta[meta["지점"] == str(target_station_id)]

    if len(target_row) == 0:
        return None, _empty_trace()

    target_lat = target_row.iloc[0]["위도"]
    target_lon = target_row.iloc[0]["경도"]
    target_elevation = target_row.iloc[0].get(ELEVATION_COLUMN, pd.NA)
    effective_lapse_rate = _resolve_lapse_rate(
        elevation_method, timestamp, lapse_rate, monthly_lapse_table
    )
    neighbors = []

    for station_id, value in values_at_time.items():
        station_id = str(station_id)

        if station_id == str(target_station_id):
            continue

        if pd.isna(value):
            continue

        station_row = meta[meta["지점"] == station_id]

        if len(station_row) == 0:
            continue

        distance = get_distance_km(
            target_lat,
            target_lon,
            station_row.iloc[0]["위도"],
            station_row.iloc[0]["경도"],
        )

        if max_distance_km is not None and distance > max_distance_km:
            continue

        source_elevation = station_row.iloc[0].get(ELEVATION_COLUMN, pd.NA)
        # 1단계: 이웃 기온을 기준면(0m) 값으로 환산 -> 이 값으로 IDW 수행
        value_at_reference = _to_reference_value(
            value,
            source_elevation,
            elevation_method,
            effective_lapse_rate,
            band_lapse_table,
        )

        neighbors.append(
            {
                "station_id": station_id,
                "station_name": station_row.iloc[0].get("지점명", ""),
                "latitude": station_row.iloc[0]["위도"],
                "longitude": station_row.iloc[0]["경도"],
                "elevation_m": source_elevation,
                "source_elevation_m": source_elevation,
                "reference_elevation_m": (
                    REFERENCE_ELEVATION_M if elevation_method != "none" else pd.NA
                ),
                "raw_value": value,
                "value_at_reference": value_at_reference,
                "elevation_adjustment": value_at_reference - value,
                "lapse_rate": _trace_lapse_rate(elevation_method, effective_lapse_rate),
                # IDW 계산에 실제로 사용되는 값 (기준면 환산값)
                "value": value_at_reference,
                "distance_km": distance,
            }
        )

    if len(neighbors) < min_neighbors:
        return None, pd.DataFrame(neighbors, columns=IDW_TRACE_COLUMNS)

    neighbors = sorted(neighbors, key=lambda x: x["distance_km"])

    if max_neighbors is not None:
        neighbors = neighbors[:max_neighbors]

    weighted_sum = 0
    weight_sum = 0

    for item in neighbors:
        distance = item["distance_km"]
        value = item["value"]

        if distance == 0:
            item["weight"] = float("inf")
            item["weight_normalized"] = 1.0
            item["contribution"] = value
            trace = pd.DataFrame(neighbors, columns=IDW_TRACE_COLUMNS)
            trace["weight"] = trace["weight"].fillna(0)
            trace["weight_normalized"] = trace["weight_normalized"].fillna(0)
            trace["contribution"] = trace["contribution"].fillna(0)
            # 3단계: 기준면 값을 target 고도로 되돌림
            predicted = _from_reference_value(
                value,
                target_elevation,
                elevation_method,
                effective_lapse_rate,
                band_lapse_table,
            )
            return predicted, trace

        weight = 1 / (distance ** power)
        item["weight"] = weight
        weighted_sum += weight * value
        weight_sum += weight

    if weight_sum == 0:
        return None, pd.DataFrame(neighbors, columns=IDW_TRACE_COLUMNS)

    # 2단계: 기준면(0m)에서 IDW 가중평균
    predicted_at_reference = weighted_sum / weight_sum

    for item in neighbors:
        item["weight_normalized"] = item["weight"] / weight_sum
        item["contribution"] = item["weight_normalized"] * item["value"]

    # 3단계: 기준면 예측값을 target 고도로 되돌림
    predicted = _from_reference_value(
        predicted_at_reference,
        target_elevation,
        elevation_method,
        effective_lapse_rate,
        band_lapse_table,
    )

    return predicted, pd.DataFrame(neighbors, columns=IDW_TRACE_COLUMNS)


def evaluate_idw(
    data,
    meta,
    target_stations=None,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    elevation_method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    monthly_lapse_table=None,
    band_lapse_table=None,
):
    results = []
    meta_cache = {}

    def get_meta_for_time(tm):
        date_key = pd.Timestamp(tm).normalize()

        if date_key not in meta_cache:
            meta_cache[date_key] = filter_valid_station_meta(meta, date_key)

        return meta_cache[date_key]

    if target_stations is None:
        target_stations = list(data.columns)
    else:
        target_stations = [str(x) for x in target_stations]

    for target_station_id in target_stations:
        if target_station_id not in data.columns:
            print("데이터 없음:", target_station_id)
            continue

        print("Testing station:", target_station_id)

        for tm, values_at_time in data.iterrows():
            observed = values_at_time[target_station_id]

            if pd.isna(observed):
                continue

            valid_meta = get_meta_for_time(tm)
            predicted, neighbor_count = idw_interpolate(
                target_station_id=target_station_id,
                values_at_time=values_at_time,
                meta=valid_meta,
                power=power,
                max_distance_km=max_distance_km,
                min_neighbors=min_neighbors,
                max_neighbors=max_neighbors,
                elevation_method=elevation_method,
                lapse_rate=lapse_rate,
                timestamp=tm,
                monthly_lapse_table=monthly_lapse_table,
                band_lapse_table=band_lapse_table,
            )

            if predicted is None:
                continue

            error = predicted - observed

            results.append(
                {
                    "TM": tm,
                    "STN": target_station_id,
                    "OBSERVED": observed,
                    "PREDICTED": predicted,
                    "ERROR": error,
                    "ABS_ERROR": abs(error),
                    "NEIGHBOR_COUNT": neighbor_count,
                }
            )

    return pd.DataFrame(results, columns=IDW_RESULT_COLUMNS)


def evaluate_station_idw(
    data,
    meta,
    target_stations=None,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
    elevation_method="none",
    lapse_rate=FIXED_LAPSE_RATE,
    monthly_lapse_table=None,
    band_lapse_table=None,
):
    return evaluate_idw(
        data,
        meta,
        target_stations=target_stations,
        power=power,
        max_distance_km=max_distance_km,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
        elevation_method=elevation_method,
        lapse_rate=lapse_rate,
        monthly_lapse_table=monthly_lapse_table,
        band_lapse_table=band_lapse_table,
    )


def print_score(result):
    if len(result) == 0:
        print("평가 결과가 없습니다")
        return

    mae = result["ABS_ERROR"].mean()
    rmse = (result["ERROR"] ** 2).mean() ** 0.5
    bias = result["ERROR"].mean()

    print()
    print("===== IDW 보간 성능 =====")
    print("평가 샘플 수:", len(result))
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("BIAS:", round(bias, 4))
