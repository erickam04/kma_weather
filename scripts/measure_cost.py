"""Stage 1 비용 실측 (계획서 §2-2 재추정).

소표본으로 target당 계산비용을 재고 전체(549×12개월×2method) 소요를 외삽한다.
per-target 비용은 idw 이웃 루프가 O(N) 고정이라 소표본으로 정확히 외삽됨.
데이터 로드(월 1회 고정오버헤드)와 순수 계산을 분리 측정.
"""

import time

import pandas as pd

from weather.find_stations import load_station_meta
from weather.idw import TARGET_COLUMN, evaluate_idw
from weather.load_aws import prepare_evaluation_data

N_TARGETS = 549   # 전체 LOO 후보
N_MONTHS = 12
N_SAMPLE = 4      # 측정용 표본 target 수
MONTH = ("2024", "02")


def fmt(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    meta = load_station_meta()
    start = pd.Timestamp(year=int(MONTH[0]), month=int(MONTH[1]), day=1)
    end = start + pd.offsets.MonthBegin(1)

    # 1. 데이터 로드 시간 (월당 고정 오버헤드)
    t0 = time.perf_counter()
    data, emeta = prepare_evaluation_data(meta, start, end, target_column=TARGET_COLUMN)
    load_time = time.perf_counter() - t0
    print(f"[로드] {MONTH[0]}-{MONTH[1]}: {load_time:.1f}s "
          f"(station {len(data.columns)}, row {len(data)})")

    # 2. 순수 계산 시간 (표본 target, 2 method)
    cols = list(data.columns)
    step = max(1, len(cols) // N_SAMPLE)
    sample = cols[::step][:N_SAMPLE]
    print(f"[표본] {N_SAMPLE}개 target: {sample}")

    import contextlib, io
    buf = io.StringIO()
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        for method in ["none", "fixed_lapse"]:
            evaluate_idw(data, emeta, target_stations=sample, elevation_method=method)
    compute_time = time.perf_counter() - t1

    per_tm2 = compute_time / N_SAMPLE  # target·월·2method 당 계산비용
    print(f"[계산] {N_SAMPLE}target × 2method: {compute_time:.1f}s "
          f"-> target·월·2method당 {per_tm2:.2f}s")

    # 3. 외삽
    print("\n" + "=" * 56)
    print("전체(549 × 12개월 × 2method) 소요 추정")
    print("=" * 56)
    single = load_time * N_MONTHS + per_tm2 * N_TARGETS * N_MONTHS
    print(f"단일코어         : {fmt(single)}")
    # 월 병렬(현 러너, 최대 12 워커): 월당 비용이 벽시간
    month_par = load_time + per_tm2 * N_TARGETS
    print(f"월-병렬(12워커)  : {fmt(month_par)}  (현 new_batch_loo 방식, 벽시간=최장월)")
    # (월×target청크) 병렬로 16코어 완전 활용 시 이론 하한
    chunk16 = single / 16
    print(f"16코어 이론하한  : {fmt(chunk16)}  (월×target청크 분할 필요, 러너 보강 시)")
    print(f"\n참고 상수: load {load_time:.1f}s/월, per-target·월·2method {per_tm2:.2f}s")


if __name__ == "__main__":
    main()
