"""고속 IDW 평가 — evaluate_idw 와 bit-identical, pandas 병목 제거 (계획서 §2-2 축2).

전략(수학 무변경, 오버헤드만 제거):
  1. 거리·기하는 좌표가 바뀔 때만 변함 → 시각별이 아니라 **일별(=coord-epoch) 1회 precompute.**
     거리는 기존 `find_stations.get_distance_km` 를 **그대로 호출**해 값이 동일.
  2. 이웃 후보를 **30km 내로 미리 좁히고 거리순 정렬**(안정정렬로 컬럼순 tie 유지) →
     시각별로는 그 shortlist에서 NaN만 걸러 상위 10개 취함.
  3. 최종 가중합은 원본과 **동일한 순서(거리 오름차순 순차 루프)** → 부동소수까지 동일.
  4. 버려지던 trace DataFrame 생성 안 함(evaluate_idw 는 len(trace)만 사용).

원본 idw.evaluate_idw 와 동일한 9컬럼 결과(TM,STN,OBSERVED,PREDICTED,ERROR,
ABS_ERROR,NEIGHBOR_COUNT)를 반환. none/fixed_lapse 만 지원(이번 확장 범위).

검증: validate_fast_idw.py 가 golden 과 대조.
"""

import hashlib

import numpy as np
import pandas as pd

from weather.elevation_correction import (
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
from weather.idw import (
    ELEVATION_COLUMN,
    IDW_RESULT_COLUMNS,
    MIN_NEIGHBORS,
    POWER,
    TARGET_COLUMN,
)


def _meta_lookup(valid_meta):
    """valid_meta -> {지점: (위도, 경도, 고도)} 및 정렬용 컬럼 판단에 쓰는 dict."""
    lut = {}
    epoch = {}
    for _, r in valid_meta.iterrows():
        sid = str(r["지점"])
        lut[sid] = (r["위도"], r["경도"], r.get(ELEVATION_COLUMN, np.nan))
        # COORD_EPOCH = 그 좌표가 유효해진 시작일(이설 구분자). NaT면 빈문자.
        s = r.get("시작일", pd.NaT)
        epoch[sid] = "" if pd.isna(s) else pd.Timestamp(s).strftime("%Y-%m-%d")
    return lut, epoch


def _build_geometry(target_id, columns, lut, direction=None):
    """target 에 대한 30km 내 후보 shortlist (거리순, 컬럼순 tie).

    반환: list of (col_pos, station_id, distance, elevation). 원본과 동일하게
    - target 자기 제외
    - valid_meta(=lut)에 없는 컬럼 제외
    - 30km 초과 제외
    - 거리 오름차순 안정정렬(동일거리는 컬럼 등장 순서 유지 = 원본 sorted 안정성).

    direction: None(전체) / "north"(위도>target) / "south"(위도<target).
      H2 방위검정용 — 이웃을 반구로 쪼개 재보간. direction=None이면 기존과 완전 동일.
    """
    if target_id not in lut:
        return None
    t_lat, t_lon, _ = lut[target_id]
    cands = []
    for pos, sid in enumerate(columns):        # columns = data 컬럼 순서(원본 iteration 순서)
        sid = str(sid)
        if sid == target_id:
            continue
        if sid not in lut:
            continue
        s_lat, s_lon, s_elev = lut[sid]
        if direction == "north" and not (s_lat > t_lat):
            continue
        if direction == "south" and not (s_lat < t_lat):
            continue
        dist = get_distance_km(t_lat, t_lon, s_lat, s_lon)   # 기존 함수 그대로 = 동일값
        if DEFAULT_MAX_DISTANCE_KM is not None and dist > DEFAULT_MAX_DISTANCE_KM:
            continue
        cands.append((pos, sid, dist, s_elev))
    # 거리 오름차순, 동일거리는 col 순서(안정). Python sorted 안정성과 동일하게 pos를 2차키로.
    cands.sort(key=lambda x: (x[2], x[0]))
    return cands


def _predict(target_id, geom, row_values, target_elev, method, gamma):
    """한 시각 예측. 원본 idw_interpolate_with_trace 의 성공 경로를 값 동일하게 재현.

    row_values: {station_id: value} (그 시각). geom: _build_geometry 결과.
    반환: (predicted, neighbor_ids, n_available).
      - 성공: predicted=값, neighbor_ids=실제 사용 이웃 id(거리순, ≤max), n_available=NaN제거 후 30km내 이웃수.
      - 실패(이웃<MIN): predicted=None, neighbor_ids=[], n_available=가용 이웃수(진단용).
    NEIGHBOR_COUNT(=golden)은 성공 시 len(neighbor_ids)=min(n_available,max) 로 동일.
    """
    # 1) shortlist 에서 NaN 값 이웃 제거 (거리순 유지)
    neighbors = []
    for pos, sid, dist, s_elev in geom:
        v = row_values.get(sid)
        if v is None or (v != v):          # NaN 체크 (pd.isna 대체, float NaN 전용)
            continue
        neighbors.append((sid, dist, s_elev, v))

    n_available = len(neighbors)
    if n_available < MIN_NEIGHBORS:
        return None, [], n_available

    # 2) 상위 max_neighbors (이미 거리순) — 원본과 동일 슬라이스
    if DEFAULT_MAX_NEIGHBORS is not None:
        neighbors = neighbors[:DEFAULT_MAX_NEIGHBORS]
    neighbor_ids = [sid for sid, _d, _e, _v in neighbors]

    # 3) 각 이웃 값을 기준면(0m)으로 환산 (원본과 동일 함수)
    #    method="none"이면 그대로. fixed_lapse 는 to_reference_elevation.
    ref_vals = []
    for sid, dist, s_elev, v in neighbors:
        if method == "none":
            vr = v
        else:
            vr = to_reference_elevation(v, s_elev, method="fixed_lapse", lapse_rate=gamma)
        ref_vals.append((dist, vr))

    # 4) distance==0 분기 — 원본은 정렬된 이웃 순회 중 첫 dist==0에서 즉시 반환.
    #    거리 오름차순이므로 첫 이웃만 검사하면 충분(원본과 동일 결과).
    first_dist, first_vr = ref_vals[0]
    if first_dist == 0:
        predicted = (first_vr if method == "none"
                     else from_reference_elevation(first_vr, target_elev,
                                                    method="fixed_lapse", lapse_rate=gamma))
        return predicted, neighbor_ids, n_available

    # 5) 가중합 — 원본과 **동일 순서(거리 오름차순 순차)** 로 누적 → 부동소수 동일
    weighted_sum = 0
    weight_sum = 0
    for dist, vr in ref_vals:
        weight = 1 / (dist ** POWER)
        weighted_sum += weight * vr
        weight_sum += weight

    if weight_sum == 0:
        return None, [], n_available

    predicted_at_ref = weighted_sum / weight_sum

    # 6) 기준면 -> target 고도
    if method == "none":
        predicted = predicted_at_ref
    else:
        predicted = from_reference_elevation(predicted_at_ref, target_elev,
                                             method="fixed_lapse", lapse_rate=gamma)
    return predicted, neighbor_ids, n_available


EXT_COLUMNS = IDW_RESULT_COLUMNS + ["interpolable", "NEIGHBOR_SET_ID", "COORD_EPOCH"]


def _neighbor_signature(neighbor_ids):
    """정렬된 이웃 id 집합의 결정적 서명(프로세스 무관). 빈 집합은 ''."""
    if not neighbor_ids:
        return ""
    key = "-".join(sorted(neighbor_ids))
    return hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()


def evaluate_idw_fast(data, meta, target_stations=None, elevation_method="none",
                      lapse_rate=None, extended=False, direction=None):
    """evaluate_idw 의 고속 등가물. none/fixed_lapse 만. 결과 스키마·값 동일.

    extended=False: 9컬럼(IDW_RESULT_COLUMNS), 성공 행만 — evaluate_idw 와 동일.
    extended=True : 관측 존재 시각을 전부 방출(interpolable 플래그) + NEIGHBOR_SET_ID +
                    COORD_EPOCH. 반환 (DataFrame[EXT_COLUMNS], neighbor_set_map{sig:[ids]}).
                    interpolable=True 부분집합은 extended=False 및 golden 과 값 동일.
    lapse_rate 기본값은 elevation_correction.FIXED_LAPSE_RATE (idw 기본과 동일).
    """
    from weather.elevation_correction import FIXED_LAPSE_RATE
    gamma = FIXED_LAPSE_RATE if lapse_rate is None else lapse_rate

    if elevation_method not in ("none", "fixed_lapse"):
        raise ValueError(f"fast_idw 는 none/fixed_lapse 만 지원: {elevation_method}")

    if target_stations is None:
        target_stations = list(data.columns)
    else:
        target_stations = [str(x) for x in target_stations]
    target_stations = [t for t in target_stations if t in data.columns]

    columns = [str(c) for c in data.columns]
    meta_by_date = {}
    geom_cache = {}  # (date_key, target) -> geom

    results = []
    nbr_map = {}  # signature -> sorted neighbor ids (확장 모드)
    values_matrix = data.to_numpy()
    index = data.index

    for ri in range(len(index)):
        tm = index[ri]
        row = values_matrix[ri]
        date_key = pd.Timestamp(tm).normalize()

        if date_key not in meta_by_date:
            vm = filter_valid_station_meta(meta, date_key)
            meta_by_date[date_key] = _meta_lookup(vm)
        lut, epoch_lut = meta_by_date[date_key]

        row_values = {columns[i]: row[i] for i in range(len(columns))}

        for target_id in target_stations:
            observed = row_values.get(target_id)
            if observed is None or observed != observed:   # NaN 관측 -> 평가 불가, 제외
                continue
            if target_id not in lut:
                continue

            gkey = (date_key, target_id)
            if gkey not in geom_cache:
                geom_cache[gkey] = _build_geometry(target_id, columns, lut, direction)
            geom = geom_cache[gkey]

            if not geom:
                # 후보 자체가 없음(고립) — 확장 모드에선 interpolable=False 방출
                if extended:
                    results.append({
                        "TM": tm, "STN": target_id, "OBSERVED": observed,
                        "PREDICTED": np.nan, "ERROR": np.nan, "ABS_ERROR": np.nan,
                        "NEIGHBOR_COUNT": 0, "interpolable": False,
                        "NEIGHBOR_SET_ID": "", "COORD_EPOCH": epoch_lut.get(target_id, ""),
                    })
                continue

            target_elev = lut[target_id][2]
            predicted, neighbor_ids, n_available = _predict(
                target_id, geom, row_values, target_elev, elevation_method, gamma
            )

            if predicted is None:
                if extended:
                    results.append({
                        "TM": tm, "STN": target_id, "OBSERVED": observed,
                        "PREDICTED": np.nan, "ERROR": np.nan, "ABS_ERROR": np.nan,
                        "NEIGHBOR_COUNT": n_available, "interpolable": False,
                        "NEIGHBOR_SET_ID": "", "COORD_EPOCH": epoch_lut.get(target_id, ""),
                    })
                continue

            error = predicted - observed
            rec = {
                "TM": tm, "STN": target_id, "OBSERVED": observed,
                "PREDICTED": predicted, "ERROR": error,
                "ABS_ERROR": abs(error), "NEIGHBOR_COUNT": len(neighbor_ids),
            }
            if extended:
                sig = _neighbor_signature(neighbor_ids)
                if sig and sig not in nbr_map:
                    nbr_map[sig] = sorted(neighbor_ids)
                rec["interpolable"] = True
                rec["NEIGHBOR_SET_ID"] = sig
                rec["COORD_EPOCH"] = epoch_lut.get(target_id, "")
            results.append(rec)

    if extended:
        return pd.DataFrame(results, columns=EXT_COLUMNS), nbr_map
    return pd.DataFrame(results, columns=IDW_RESULT_COLUMNS)
