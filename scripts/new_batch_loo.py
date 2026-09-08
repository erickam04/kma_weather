"""전체망 LOO 병렬 배치 러너 (계획서 §2, Stage 0/1).

설계 (열린결정 A=병렬 A1, method=none+fixed 2종으로 락):
  - 알고리즘 무수정: 기존 `evaluate_idw`를 그대로 호출한다. bit-identical.
  - 병렬 단위 = 월(month). 각 워커가 그 달 데이터를 1회 로드(이웃으로 전체 station 필요)
    후 배정된 target × method를 계산. 월은 완전 독립(embarrassingly parallel).
  - 좌표·고도 이설: evaluate_idw 내부의 filter_valid_station_meta가 시각별로 처리하므로
    좌표 epoch가 자동 반영됨(계획서 §2-3).
  - 재개 원자성: part 파일을 .tmp에 쓰고 원자적 rename + .done sentinel(기대 행수 기록).
    .done 있고 행수 일치하면 skip.

출력 스키마 = 현행 golden과 동일(TM,STN,OBSERVED,PREDICTED,ERROR,ABS_ERROR,
NEIGHBOR_COUNT,MONTH,METHOD) — Stage 0 불변식 검증(신러너==evaluate_idw)을 위해.
확장 컬럼(interpolable/NEIGHBOR_SET_ID/COORD_EPOCH)은 Stage 1 별도 레이어.

사용:
  python -m scripts.new_batch_loo                 # 전체 549 target, 12개월, none+fixed
  프로토타입/검증은 run_loo(targets=[...], months=[...], out_dir=...) 호출.
"""

from project_paths import WORK_DIR

import contextlib
import glob
import io
import os
from multiprocessing import Pool

import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN, evaluate_idw
from weather.load_aws import BASE_FOLDER, prepare_evaluation_data

# 열린결정 락: 2 method (none+fixed_lapse). monthly/band는 이번 확장 범위 밖.
DEFAULT_METHODS = ["none", "fixed_lapse"]
# 열린결정 H: OneDrive 밖 로컬 출력(동기화 충돌 방지).
DEFAULT_OUT_DIR = str(WORK_DIR / "loo_raw")
YEAR = 2024
PART_COLUMNS = [
    "TM", "STN", "OBSERVED", "PREDICTED", "ERROR",
    "ABS_ERROR", "NEIGHBOR_COUNT", "MONTH", "METHOD",
]


def discover_months(year=YEAR, base_folder=BASE_FOLDER):
    pattern = os.path.join(base_folder, "Station_*", f"AWS_{year}*.csv")
    months = set()
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        months.add(name.replace("AWS_", "").replace(".csv", ""))
    return sorted(months)


def _part_paths(out_dir, ym, method):
    base = os.path.join(out_dir, f"raw_{ym}_{method}")
    return base + ".csv", base + ".csv.tmp", base + ".done"


def _is_done(csv_path, done_path):
    """.done sentinel 이 있고 그 기대 행수가 실제 csv 행수와 일치하면 완료로 간주."""
    if not (os.path.exists(csv_path) and os.path.exists(done_path)):
        return False
    try:
        with open(done_path, encoding="utf-8") as f:
            expected = int(f.read().strip())
    except (ValueError, OSError):
        return False
    actual = sum(1 for _ in open(csv_path, encoding="utf-8-sig")) - 1  # 헤더 제외
    return actual == expected


def _write_part_atomic(df, csv_path, tmp_path, done_path):
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, csv_path)  # 원자적 rename
    with open(done_path, "w", encoding="utf-8") as f:
        f.write(str(len(df)))


def _run_month(args):
    """워커: 한 달을 로드해 배정된 target×method 를 계산·저장. (프로세스에서 실행)"""
    ym, methods, targets, out_dir, base_folder = args
    year, month = int(ym[:4]), int(ym[4:])
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)

    # 이미 완료된 (월,method) 전부면 로드 없이 skip
    pending = []
    for method in methods:
        csv_path, _, done_path = _part_paths(out_dir, ym, method)
        if not _is_done(csv_path, done_path):
            pending.append(method)
    if not pending:
        return ym, {m: "skip" for m in methods}

    meta = load_station_meta()
    # evaluate_idw 의 진행 출력(대량)은 로그로 흘려보냄 — 알고리즘 무수정
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        data, emeta = prepare_evaluation_data(
            meta, start, end, target_column=TARGET_COLUMN, base_folder=base_folder
        )

    tgt = targets  # None이면 evaluate_idw 가 data.columns 전체(LOO) 사용
    status = {}
    for method in methods:
        csv_path, tmp_path, done_path = _part_paths(out_dir, ym, method)
        if _is_done(csv_path, done_path):
            status[method] = "skip"
            continue
        with contextlib.redirect_stdout(log_buf):
            result = evaluate_idw(
                data, emeta, target_stations=tgt, elevation_method=method
            )
        result = result.copy()
        result["MONTH"] = ym
        result["METHOD"] = method
        result = result[PART_COLUMNS]
        _write_part_atomic(result, csv_path, tmp_path, done_path)
        status[method] = f"{len(result)} rows"
    return ym, status


def run_loo(
    targets=None,
    months=None,
    methods=None,
    out_dir=DEFAULT_OUT_DIR,
    base_folder=BASE_FOLDER,
    processes=None,
    combine_to=None,
):
    """LOO 배치 실행.

    targets  : 지점 리스트. None이면 전체 LOO(data.columns 전체).
    months   : ["202401", ...]. None이면 discover.
    methods  : 기본 none+fixed_lapse.
    out_dir  : part 파일 출력 폴더(기본 OneDrive 밖).
    processes: 병렬 프로세스 수. None이면 min(월수, cpu).
    combine_to: 지정 시 모든 part 를 하나의 CSV로 합쳐 저장(검증·소규모용).
    """
    methods = methods or DEFAULT_METHODS
    months = months or discover_months(base_folder=base_folder)
    if targets is not None:
        targets = [str(t) for t in targets]
    os.makedirs(out_dir, exist_ok=True)

    tasks = [(ym, methods, targets, out_dir, base_folder) for ym in months]
    n_proc = processes or min(len(tasks), os.cpu_count() or 1)

    print(f"LOO 실행: {len(months)}개월 × {len(methods)}method, "
          f"target={'전체' if targets is None else len(targets)}, "
          f"procs={n_proc}, out={out_dir}")

    if n_proc == 1:
        results = [_run_month(t) for t in tasks]
    else:
        with Pool(n_proc) as pool:
            results = pool.map(_run_month, tasks)

    for ym, status in sorted(results):
        print(f"  {ym}: {status}")

    if combine_to:
        parts = []
        for ym in months:
            for method in methods:
                csv_path, _, _ = _part_paths(out_dir, ym, method)
                if os.path.exists(csv_path):
                    parts.append(pd.read_csv(csv_path, encoding="utf-8-sig"))
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined.to_csv(combine_to, index=False, encoding="utf-8-sig")
            print(f"합본 저장: {combine_to} ({len(combined)} rows)")
            return combined
    return None


if __name__ == "__main__":
    run_loo()
