"""Stage 1 — 전체 549 LOO raw 생산 (계획서 §2·§4).

fast_idw(extended=True) 로 전체 station × 2024 × (none, fixed_lapse) 를 병렬 생산.
- 병렬 단위 = 월(월당 데이터 1회 로드). 알고리즘 무수정.
- 확장 스키마: 9컬럼 + interpolable + NEIGHBOR_SET_ID + COORD_EPOCH + MONTH + METHOD.
- 재개 원자성: parquet part 를 .tmp -> 원자적 rename + .done sentinel(행수). 재실행 시 skip.
- 병합: 전 part -> SQLite (STN,TM)/(STN,METHOD) 인덱스 + excluded_stations.csv + neighbor_set_map.

출력: KMA_WORK_DIR/loo_stage1/ (OneDrive 밖, 열린결정 H).
사용: python -m scripts.stage1_produce            # 전체 생산 후 병합
      python -m scripts.stage1_produce --merge-only
"""

from project_paths import LOO_DIR

import glob
import json
import os
import sqlite3
import sys
from multiprocessing import Pool

import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN
from weather.load_aws import BASE_FOLDER, prepare_evaluation_data
from weather.fast_idw import evaluate_idw_fast, EXT_COLUMNS

OUT_DIR = str(LOO_DIR)
DB_PATH = os.path.join(OUT_DIR, "loo_raw.db")
METHODS = ["none", "fixed_lapse"]
YEAR = 2024
PART_COLUMNS = EXT_COLUMNS + ["MONTH", "METHOD"]


def discover_months(year=YEAR, base_folder=BASE_FOLDER):
    months = set()
    for path in glob.glob(os.path.join(base_folder, "Station_*", f"AWS_{year}*.csv")):
        months.add(os.path.basename(path).replace("AWS_", "").replace(".csv", ""))
    return sorted(months)


def _paths(ym, method):
    base = os.path.join(OUT_DIR, f"raw_{ym}_{method}")
    return base + ".parquet", base + ".parquet.tmp", base + ".done"


def _is_done(pq, done):
    if not (os.path.exists(pq) and os.path.exists(done)):
        return False
    try:
        expected = int(open(done, encoding="utf-8").read().strip())
    except (ValueError, OSError):
        return False
    try:
        n = pd.read_parquet(pq, columns=["STN"]).shape[0]
    except Exception:
        return False
    return n == expected


def _run_month(ym):
    """워커: 한 달 로드 -> 2 method extended 생산 -> parquet part + nbrmap json."""
    pending = []
    for m in METHODS:
        pq, _, done = _paths(ym, m)
        if not _is_done(pq, done):
            pending.append(m)
    nbrmap_path = os.path.join(OUT_DIR, f"nbrmap_{ym}.json")
    if not pending and os.path.exists(nbrmap_path):
        return ym, "skip"

    meta = load_station_meta()
    y, mo = int(ym[:4]), int(ym[4:])
    start = pd.Timestamp(year=y, month=mo, day=1)
    end = start + pd.offsets.MonthBegin(1)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)

    nbr_map = {}
    status = {}
    for method in METHODS:
        pq, tmp, done = _paths(ym, method)
        if _is_done(pq, done):
            status[method] = "skip"
            # skip 이어도 nbrmap 병합 위해 재계산 필요 -> 아래서 이 method는 map만 다시
        with contextlib.redirect_stdout(buf):
            df, nmap = evaluate_idw_fast(
                data, emeta, target_stations=None,
                elevation_method=method, extended=True,
            )
        nbr_map.update(nmap)
        if not _is_done(pq, done):
            df = df.copy()
            df["MONTH"] = ym
            df["METHOD"] = method
            df = df[PART_COLUMNS]
            df.to_parquet(tmp, index=False)
            os.replace(tmp, pq)
            with open(done, "w", encoding="utf-8") as f:
                f.write(str(len(df)))
            status[method] = f"{len(df)} rows"

    with open(nbrmap_path, "w", encoding="utf-8") as f:
        json.dump(nbr_map, f)
    return ym, status


def produce(months=None, processes=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    months = months or discover_months()
    n_proc = processes or min(len(months), os.cpu_count() or 1)
    print(f"Stage 1 생산: {len(months)}개월 × {len(METHODS)}method, procs={n_proc}, out={OUT_DIR}")
    if n_proc == 1:
        res = [_run_month(m) for m in months]
    else:
        with Pool(n_proc) as pool:
            res = pool.map(_run_month, months)
    for ym, st in sorted(res):
        print(f"  {ym}: {st}")


def merge():
    """전 part -> SQLite + neighbor_set_map + excluded_stations."""
    parts = sorted(glob.glob(os.path.join(OUT_DIR, "raw_*.parquet")))
    if not parts:
        print("part 없음"); return
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    total = 0
    # station별 interpolable 유무 집계(제외집합 판정)
    stn_stats = {}  # STN -> [valid(interp True), total observed rows] (method=none 기준)
    for pq in parts:
        df = pd.read_parquet(pq)
        df["STN"] = df["STN"].astype(str)
        df["TM"] = df["TM"].astype(str)
        df["interpolable"] = df["interpolable"].astype(int)
        df.to_sql("raw", con, if_exists="append", index=False)
        total += len(df)
        # 제외 판정은 method 하나(none)만 세면 충분(선택은 method 무관)
        sub = df[df["METHOD"] == "none"]
        for stn, g in sub.groupby("STN"):
            v, t = stn_stats.get(stn, (0, 0))
            stn_stats[stn] = (v + int(g["interpolable"].sum()), t + len(g))
    con.execute("CREATE INDEX idx_stn_tm ON raw(STN, TM)")
    con.execute("CREATE INDEX idx_stn_method ON raw(STN, METHOD)")
    con.commit()

    # neighbor_set_map 병합
    nbr = {}
    for j in glob.glob(os.path.join(OUT_DIR, "nbrmap_*.json")):
        nbr.update(json.load(open(j, encoding="utf-8")))
    nbr_df = pd.DataFrame(
        [{"NEIGHBOR_SET_ID": k, "neighbor_ids": "-".join(v), "n_neighbors": len(v)}
         for k, v in nbr.items()]
    )
    nbr_df.to_sql("neighbor_set_map", con, if_exists="replace", index=False)

    # excluded_stations: 관측은 있으나 interpolable=True 가 한 번도 없는 지점
    excluded = []
    for stn, (v, t) in sorted(stn_stats.items()):
        if v == 0:
            excluded.append({"STN": stn, "valid_hours": v, "total_hours": t,
                             "valid_ratio": 0.0, "reason": "이웃<3 (상시 고립)"})
    exc_df = pd.DataFrame(excluded)
    exc_df.to_csv(os.path.join(OUT_DIR, "excluded_stations.csv"),
                  index=False, encoding="utf-8-sig")

    # station 요약(유효비율)
    summ = pd.DataFrame(
        [{"STN": s, "valid_hours": v, "total_hours": t,
          "valid_ratio": round(v / t, 4) if t else 0.0}
         for s, (v, t) in sorted(stn_stats.items())]
    )
    summ.to_csv(os.path.join(OUT_DIR, "station_valid_summary.csv"),
                index=False, encoding="utf-8-sig")

    con.close()
    print(f"병합 완료: {DB_PATH}")
    print(f"  raw 총행 {total:,} | 지점 {len(stn_stats)} | 제외 {len(excluded)} "
          f"| 이웃세트 {len(nbr)}")
    print(f"  excluded_stations.csv / station_valid_summary.csv")


if __name__ == "__main__":
    if "--merge-only" in sys.argv:
        merge()
    else:
        produce()
        merge()
