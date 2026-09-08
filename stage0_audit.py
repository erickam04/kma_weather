"""Stage 0 감사 — 전체망 확장의 사전확정 (계획서 §10 Stage 0 ①③).

목적:
  1. 실제 LOO 대상 N 확정 — 메타(좌표 有) ∩ 데이터 폴더 ∩ 2024 실관측.
     ASOS(메타에 좌표 없는 지점)는 자동 배제됨(방침: AWS only).
  2. 엣지케이스 카탈로그 — 불변식 검증셋 구성용
     (고립 / 좌표중복 distance==0 / NaN고도 / 2024 내 이설=좌표epoch 전환).

알고리즘 모듈 무수정 — 기존 로드/필터 함수만 재사용(읽기 전용).
출력: WeatherData_AWS/interpolation_batch/stage0/ (소형 CSV, OneDrive 유지).
      대량 raw는 Stage 1에서 OneDrive 밖(KMA_WORK_DIR)으로.
"""

import glob
import os

import pandas as pd

from find_stations import (
    DEFAULT_MAX_DISTANCE_KM,
    filter_valid_station_meta,
    get_distance_km,
    load_station_meta,
)
from idw import ELEVATION_COLUMN, MIN_NEIGHBORS
from load_aws import BASE_FOLDER

YEAR = 2024
OUT_DIR = "WeatherData_AWS/interpolation_batch/stage0"
# 이설 판정 등에서 쓰는 대표 기준일(연중). 이웃 충분성은 4계절로 점검.
REF_DAYS = [f"{YEAR}-01-15", f"{YEAR}-04-15", f"{YEAR}-07-15", f"{YEAR}-10-15"]


def folder_station_ids(base_folder=BASE_FOLDER):
    """데이터 폴더가 존재하는 지점 id 집합."""
    ids = set()
    for path in glob.glob(os.path.join(base_folder, "Station_*")):
        if os.path.isdir(path):
            ids.add(os.path.basename(path).replace("Station_", "").strip())
    return ids


def has_year_data(station_id, year=YEAR, base_folder=BASE_FOLDER):
    """해당 지점 폴더에 AWS_{year}MM.csv 가 하나라도 있는지 (실관측 존재의 경량 프록시)."""
    pattern = os.path.join(base_folder, f"Station_{station_id}", f"AWS_{year}*.csv")
    return len(glob.glob(pattern)) > 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = load_station_meta()  # NaN 좌표/시작일 행은 이미 제거됨
    meta_ids = set(meta["지점"].unique())
    folders = folder_station_ids()

    # --- 1. N 확정 --------------------------------------------------------
    with_coords = meta_ids                       # 메타에 좌표 있는 지점 (AWS)
    with_folder = meta_ids & folders             # + 데이터 폴더 존재
    with_data = {s for s in with_folder if has_year_data(s)}  # + 2024 파일 존재
    asos_like = folders - meta_ids               # 폴더 있으나 메타 없음 = ASOS 등 배제

    print("=" * 60)
    print("N 확정")
    print("=" * 60)
    print(f"메타 unique 지점 (좌표 有, AWS)     : {len(with_coords)}")
    print(f"  + 데이터 폴더 존재               : {len(with_folder)}")
    print(f"  + 2024 관측파일 존재 (LOO 후보)  : {len(with_data)}")
    print(f"메타 없는 폴더 (ASOS 등, 배제)     : {len(asos_like)}")
    print(f"메타 있으나 폴더 없음 (제외)       : {len(meta_ids - folders)}")

    # --- 2. 엣지케이스 카탈로그 ------------------------------------------
    cand = sorted(with_data)

    # (a) NaN 고도 — 고도보정 fallback 경로 검증용
    nan_elev = sorted(
        s for s in cand
        if meta[meta["지점"] == s][ELEVATION_COLUMN].isna().all()
    )

    # (b) 2024 내 이설 (좌표 epoch 전환) — 시작일 또는 종료일이 2024 안에 있는 지점
    y0, y1 = pd.Timestamp(f"{YEAR}-01-01"), pd.Timestamp(f"{YEAR+1}-01-01")
    reloc = []
    for s in cand:
        rows = meta[meta["지점"] == s]
        starts_in = ((rows["시작일"] >= y0) & (rows["시작일"] < y1)).any()
        ends_in = ((rows["종료일"] >= y0) & (rows["종료일"] < y1)).any()
        if starts_in or ends_in:
            reloc.append(s)
    reloc = sorted(reloc)

    # (c) 좌표 중복 (distance==0 분기) + (d) 고립(이웃<MIN) — 4계절 기준일로 점검
    dup_zero = set()
    isolated_all = None  # 모든 기준일에서 이웃<MIN 인 지점
    neighbor_counts = {}  # 지점 -> 기준일별 유효이웃수(30km내)
    for ref in REF_DAYS:
        valid = filter_valid_station_meta(meta, ref)
        valid = valid[valid["지점"].isin(cand)]
        coords = valid[["지점", "위도", "경도"]].values
        # 이웃수 계산 (30km 내) + distance==0 탐지
        this_isolated = set()
        for i, (sid, lat, lon) in enumerate(coords):
            n = 0
            for j, (sid2, lat2, lon2) in enumerate(coords):
                if i == j:
                    continue
                d = get_distance_km(lat, lon, lat2, lon2)
                if d == 0:
                    dup_zero.add(tuple(sorted([str(sid), str(sid2)])))
                if d <= DEFAULT_MAX_DISTANCE_KM:
                    n += 1
            neighbor_counts.setdefault(str(sid), {})[ref] = n
            if n < MIN_NEIGHBORS:
                this_isolated.add(str(sid))
        isolated_all = this_isolated if isolated_all is None else (isolated_all & this_isolated)

    print()
    print("=" * 60)
    print("엣지케이스 카탈로그")
    print("=" * 60)
    print(f"(a) NaN 고도 지점            : {len(nan_elev)}  예: {nan_elev[:10]}")
    print(f"(b) 2024 내 이설(좌표epoch)  : {len(reloc)}  예: {reloc[:10]}")
    print(f"(c) 좌표중복(distance==0)쌍  : {len(dup_zero)}  {sorted(dup_zero)[:10]}")
    print(f"(d) 상시 고립(4계절 모두<3)  : {len(isolated_all)}  {sorted(isolated_all)}")

    # --- 3. 저장 ----------------------------------------------------------
    # N 요약
    pd.DataFrame([
        {"항목": "meta_unique_coords", "count": len(with_coords)},
        {"항목": "meta_and_folder", "count": len(with_folder)},
        {"항목": "meta_folder_data_LOO후보", "count": len(with_data)},
        {"항목": "asos_like_excluded", "count": len(asos_like)},
        {"항목": "meta_no_folder", "count": len(meta_ids - folders)},
    ]).to_csv(os.path.join(OUT_DIR, "n_summary.csv"), index=False, encoding="utf-8-sig")

    # LOO 후보 전체 + 이웃수
    rows = []
    for s in cand:
        r = meta[meta["지점"] == s]
        nc = neighbor_counts.get(s, {})
        rows.append({
            "STN": s,
            "n_meta_rows": len(r),
            "elev_m": r[ELEVATION_COLUMN].iloc[-1],
            "elev_nan": s in set(nan_elev),
            "reloc_2024": s in set(reloc),
            **{f"nbr_{ref[5:]}": nc.get(ref) for ref in REF_DAYS},
            "min_nbr": min(nc.values()) if nc else None,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "loo_candidates.csv"), index=False, encoding="utf-8-sig"
    )

    # ASOS 배제 목록
    pd.DataFrame({"STN": sorted(asos_like)}).to_csv(
        os.path.join(OUT_DIR, "excluded_asos.csv"), index=False, encoding="utf-8-sig"
    )

    # 엣지케이스 검증셋 요약
    edge = {
        "isolated": sorted(isolated_all),
        "nan_elev": nan_elev[:20],
        "reloc_2024_sample": reloc[:20],
        "dup_zero_pairs": [list(p) for p in sorted(dup_zero)],
    }
    import json
    with open(os.path.join(OUT_DIR, "edge_cases.json"), "w", encoding="utf-8") as f:
        json.dump(edge, f, ensure_ascii=False, indent=2)

    print()
    print("저장:", OUT_DIR)
    print("  n_summary.csv / loo_candidates.csv / excluded_asos.csv / edge_cases.json")
    return with_data, edge


if __name__ == "__main__":
    main()
