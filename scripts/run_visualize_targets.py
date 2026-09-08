"""batch target station들의 IDW neighbor 선별 과정을 계절별로 시각화하는 러너.

기존 시각화 로직(`visualize_idw.save_visualization_outputs`)을 그대로 재사용한다.
IDW/고도보정 알고리즘·기존 모듈은 일절 수정하지 않는다 (CLAUDE.md 규칙).

- 대상: `interpolation_batch/target_stations.csv`의 station 전체
- 시각: 4계절 대표일(1/4/7/10월 15일) 정오. 정오 관측값이 결측이면 같은 날 내
  유효 시각으로 자동 대체(빈 trace에서 observed 포매팅 크래시 방지).
- 산출: station별 map(neighbor 선택)·weights·timeseries + trace/predictions CSV
  → `interpolation_visualization/neighbor_detail/{maps,weights,timeseries,trace,predictions}/`
    (카테고리 하위 폴더는 visualize_idw가 pipeline_config.VIZ_SUBDIRS로 라우팅)

실행: python -m scripts.run_visualize_targets
"""

import argparse
from pathlib import Path

import pandas as pd

import pipeline_config as cfg
from weather.find_stations import (
    DEFAULT_MAX_DISTANCE_KM,
    DEFAULT_MAX_NEIGHBORS,
    load_station_meta,
)
from weather.idw import MIN_NEIGHBORS, POWER, TARGET_COLUMN
from weather.load_aws import prepare_evaluation_data
from visualization.visualize_idw import save_visualization_outputs


TARGET_STATIONS_CSV = Path(cfg.OUT_BATCH) / "target_stations.csv"

# 4계절 대표일 (정오 스냅샷) — pipeline_config에서 유도 (batch 기간과 강제 일치).
# 거리는 고정이나 유효 station·결측은 시각마다 다르다.
SEASONS = cfg.get_season_days()
SNAPSHOT_HOUR = "12:00"


def load_target_stations(path=TARGET_STATIONS_CSV):
    """batch에서 선정된 target station id 목록(문자열)을 고도순으로 반환."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["지점"] = df["지점"].astype(str)
    return df


def pick_valid_time(data, station_id, day):
    """지정 날짜(day)에서 target 관측값이 존재하는 시각을 고른다.

    정오를 우선하고, 결측이면 정오에 가장 가까운 유효 시각을 반환한다.
    유효 시각이 전혀 없으면 None.
    """
    if station_id not in data.columns:
        return None

    day = pd.Timestamp(day).normalize()
    preferred = day + pd.Timedelta(SNAPSHOT_HOUR + ":00")

    day_mask = (data.index >= day) & (data.index < day + pd.Timedelta(days=1))
    day_slice = data.loc[day_mask, station_id]
    valid_times = day_slice[day_slice.notna()].index

    if len(valid_times) == 0:
        return None
    if preferred in valid_times:
        return preferred

    deltas = (valid_times - preferred).to_series().abs()
    return valid_times[deltas.values.argmin()]


def run(
    output_dir=Path(cfg.OUT_VIZ),
    target_column=TARGET_COLUMN,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
):
    meta = load_station_meta()
    targets = load_target_stations()
    print(f"대상 target station {len(targets)}개: {list(targets['지점'])}")

    summary = []

    for season_name, day in SEASONS:
        start = pd.to_datetime(day)
        end = start + pd.Timedelta(days=1)
        print(f"\n=== {season_name} ({day}) 데이터 로드 중 ... ===")
        data, evaluation_meta = prepare_evaluation_data(
            meta, start, end, target_column
        )

        for _, trow in targets.iterrows():
            station_id = trow["지점"]
            name = trow.get("지점명", "")

            selected_time = pick_valid_time(data, station_id, day)
            if selected_time is None:
                print(f"  [skip] {station_id} {name}: {day} 관측값 없음")
                summary.append((season_name, station_id, name, "no_obs", None))
                continue

            paths = save_visualization_outputs(
                data=data,
                meta=evaluation_meta,
                station_id=station_id,
                selected_time=selected_time,
                target_column=target_column,
                output_dir=output_dir,
                power=power,
                max_distance_km=max_distance_km,
                min_neighbors=min_neighbors,
                max_neighbors=max_neighbors,
            )

            trace = pd.read_csv(paths["trace"])
            n_sel = len(trace)
            status = "ok" if n_sel >= min_neighbors else f"insufficient({n_sel}<{min_neighbors})"
            print(f"  [{status}] {station_id} {name} @ {selected_time} -> neighbor {n_sel}개")
            summary.append((season_name, station_id, name, status, str(paths["map"])))

    print("\n=== 요약 ===")
    ok = sum(1 for s in summary if s[3] == "ok")
    print(f"총 {len(summary)}건 중 보간 성공 {ok}건")
    for season_name, sid, name, status, _ in summary:
        if status != "ok":
            print(f"  주의: {season_name} {sid} {name} -> {status}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(cfg.OUT_VIZ))
    p.add_argument("--target", default=TARGET_COLUMN)
    p.add_argument("--power", type=float, default=POWER)
    p.add_argument("--max-distance-km", type=float, default=DEFAULT_MAX_DISTANCE_KM)
    p.add_argument("--min-neighbors", type=int, default=MIN_NEIGHBORS)
    p.add_argument("--max-neighbors", type=int, default=DEFAULT_MAX_NEIGHBORS)
    return p.parse_args()


def main():
    args = parse_args()
    run(
        output_dir=args.out,
        target_column=args.target,
        power=args.power,
        max_distance_km=args.max_distance_km,
        min_neighbors=args.min_neighbors,
        max_neighbors=args.max_neighbors,
    )


if __name__ == "__main__":
    main()
