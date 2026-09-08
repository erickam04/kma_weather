"""Stage 4-② neighbor 지도 온디맨드 렌더러 (A안).

stage4_regimes.py가 정한 "이웃 기하 안정 세그먼트"별로, 그 세그먼트 대표 시각의
실제 이웃 지도를 기존 `visualize_idw.save_visualization_outputs`로 렌더한다.
  - 안정 지점: 1장, 이설·churn 지점: regime당 1장(실측 분포 1/2/3+ = 330/148/44).
  - 고립 지점(20개): 대표 시각의 후보 기하를 그대로 렌더(<3 이웃 = 보간불가가 그림에 드러남).
  - 대표는 "실재 정오 시각" — 합성 아님. weight·고도가 서로 정합.

기존 렌더러·알고리즘 모듈 무수정. 로드 비용 절감 위해 **대표일별로 묶어** 하루 1회 로드.
출력: KMA_WORK_DIR/loo_stage1/neighbor_maps/ (OneDrive 밖).

사용:
  neighbor_maps("160")                 # 온디맨드: 한 지점의 세그먼트 지도 전부
  render_report_set(which="tail")      # 보고용 배치: 고립 + 다regime(3+) 꼬리만
"""

from project_paths import LOO_DIR

import os

import pandas as pd

from find_stations import (
    DEFAULT_MAX_DISTANCE_KM, DEFAULT_MAX_NEIGHBORS, load_station_meta,
)
from idw import MIN_NEIGHBORS, POWER, TARGET_COLUMN
from load_aws import prepare_evaluation_data
from run_visualize_targets import pick_valid_time
from visualize_idw import save_visualization_outputs

FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
OUT_DIR = str(LOO_DIR / "neighbor_maps")
DETAIL = os.path.join(FULLNET, "station_regime_detail.csv")
DIAG = os.path.join(FULLNET, "station_neighbor_regimes.csv")


def _regime_times(stn):
    """지점의 (rep_time, cov_pct, first, last) 목록. 고립이면 대표 1개(중년)로."""
    det = pd.read_csv(DETAIL, encoding="utf-8-sig", dtype={"STN": str})
    rows = det[det["STN"] == str(stn)]
    if len(rows):
        return [(pd.Timestamp(r.rep_time), r.cov_pct, r.first_date, r.last_date,
                 int(r.regime_idx)) for r in rows.itertuples()]
    # 고립: 대표시각 6/15 정오 (pick_valid_time이 결측 시 대체)
    return [(pd.Timestamp("2024-06-15 12:00"), None, None, None, 0)]


def _render_day(day, jobs, meta, output_dir):
    """하루 데이터 1회 로드 후 그 날 대표시각인 (stn, sel_time, tag) 전부 렌더."""
    start = pd.Timestamp(day).normalize()
    end = start + pd.Timedelta(days=1)
    data, emeta = prepare_evaluation_data(meta, start, end, TARGET_COLUMN)
    done = []
    for stn, rep_time, tag in jobs:
        sel = pick_valid_time(data, stn, start)
        if sel is None:
            done.append((stn, tag, "no_obs", None)); continue
        paths = save_visualization_outputs(
            data=data, meta=emeta, station_id=stn, selected_time=sel,
            target_column=TARGET_COLUMN, output_dir=output_dir, power=POWER,
            max_distance_km=DEFAULT_MAX_DISTANCE_KM, min_neighbors=MIN_NEIGHBORS,
            max_neighbors=DEFAULT_MAX_NEIGHBORS)
        trace = pd.read_csv(paths["trace"])
        status = "ok" if len(trace) >= MIN_NEIGHBORS else f"isolated({len(trace)}<{MIN_NEIGHBORS})"
        done.append((stn, tag, status, str(paths["map"])))
    return done


def _render_jobs(stn_list, output_dir):
    """(stn, regime) 작업을 대표일별로 묶어 렌더 + 인덱스 반환."""
    os.makedirs(output_dir, exist_ok=True)
    meta = load_station_meta()
    by_day = {}
    metarows = []
    for stn in stn_list:
        for rep_time, cov, first, last, ridx in _regime_times(stn):
            day = pd.Timestamp(rep_time).normalize()
            by_day.setdefault(day, []).append((str(stn), rep_time, ridx))
            metarows.append({"STN": str(stn), "regime_idx": ridx, "rep_time": rep_time,
                             "cov_pct": cov, "first_date": first, "last_date": last})
    index = []
    for day in sorted(by_day):
        res = _render_day(day, by_day[day], meta, output_dir)
        for stn, tag, status, path in res:
            index.append({"STN": stn, "regime_idx": tag, "day": str(day.date()),
                          "status": status, "map": path})
        print(f"  {day.date()}: {len(res)}장 ({sum(1 for r in res if r[2]=='ok')} ok)")
    idx = pd.DataFrame(index).merge(pd.DataFrame(metarows), on=["STN", "regime_idx"], how="left")
    idx.to_csv(os.path.join(output_dir, "neighbor_maps_index.csv"),
               index=False, encoding="utf-8-sig")
    return idx


def neighbor_maps(stn, output_dir=OUT_DIR):
    """온디맨드: 한 지점의 모든 세그먼트 지도를 즉석 렌더."""
    print(f"neighbor_maps({stn}) 렌더 중...")
    return _render_jobs([str(stn)], output_dir)


def render_report_set(which="tail", output_dir=OUT_DIR):
    """보고용 배치. which='tail'=고립+다regime(3+), 'all'=전체 549."""
    diag = pd.read_csv(DIAG, encoding="utf-8-sig", dtype={"STN": str})
    if which == "all":
        stns = diag["STN"].tolist()
    else:  # tail: 해석 가치 높은 소수
        stns = diag[(diag["isolated"]) | (diag["n_regimes"] >= 3)]["STN"].tolist()
    print(f"render_report_set({which}): {len(stns)}지점")
    return _render_jobs(stns, output_dir)


if __name__ == "__main__":
    # 데모: 1장(919)·다regime(160)·고립(530) 세 경로 검증
    idx = _render_jobs(["919", "160", "530"], OUT_DIR)
    print("\n=== 데모 결과 ===")
    print(idx[["STN", "regime_idx", "day", "status", "cov_pct"]].to_string(index=False))
