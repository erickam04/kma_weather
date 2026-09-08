import glob
import os

import pandas as pd

import pipeline_config as cfg
from weather.find_stations import load_station_meta, filter_valid_station_meta
from weather.idw import TARGET_COLUMN, ELEVATION_COLUMN, evaluate_idw
from weather.lapse_rate import load_band_table, load_monthly_table
from weather.load_aws import prepare_evaluation_data, BASE_FOLDER


# 설정은 pipeline_config.py 한 곳에서 관리 (아래는 하위호환용 별칭)
YEAR = cfg.YEAR
METHODS = cfg.METHODS
OUTPUT_FOLDER = cfg.OUT_BATCH
N_PER_BAND = cfg.AUTO_N_PER_BAND
ELEVATION_BANDS = cfg.AUTO_ELEVATION_BANDS


def discover_months(year, base_folder=BASE_FOLDER):
    """저장된 AWS 파일에서 해당 연도의 월 목록(YYYYMM)을 찾는다."""
    pattern = os.path.join(base_folder, "Station_*", f"AWS_{year}*.csv")
    months = set()
    for path in glob.glob(pattern):
        name = os.path.basename(path)          # AWS_YYYYMM.csv
        months.add(name.replace("AWS_", "").replace(".csv", ""))
    return sorted(months)


def pick_target_stations(meta, base_folder=BASE_FOLDER, ref_time=None):
    """고도 값 구간(0-100 / 100-500 / 500m+)별로 target station을 선정한다.

    고도 분포가 저지대에 크게 편중돼 있어 순위 3등분은 중/고지대가 비므로,
    실제 고도 값 구간으로 나눈 뒤 각 구간에서 고도순으로 고르게 샘플링한다.
    """
    if ref_time is None:
        ref_time = cfg.get_auto_ref_date()

    valid = filter_valid_station_meta(meta, ref_time)
    valid = valid[valid[ELEVATION_COLUMN].notna()].copy()

    has_folder = valid["지점"].apply(
        lambda s: os.path.isdir(os.path.join(base_folder, f"Station_{s}"))
    )
    valid = valid[has_folder].reset_index(drop=True)

    picks = []
    for lo, hi in ELEVATION_BANDS:
        sub = valid[(valid[ELEVATION_COLUMN] >= lo) & (valid[ELEVATION_COLUMN] < hi)]
        sub = sub.sort_values(ELEVATION_COLUMN).reset_index(drop=True)
        if len(sub) == 0:
            continue

        if len(sub) <= N_PER_BAND:
            idxs = list(range(len(sub)))
        else:
            # 구간 내 고도순 균등 위치 (최저/중간/최고)
            idxs = sorted({
                int(round(i * (len(sub) - 1) / (N_PER_BAND - 1)))
                for i in range(N_PER_BAND)
            })
        picks.append(sub.iloc[idxs])

    picks = pd.concat(picks).drop_duplicates(subset=["지점"])
    return picks[["지점", "지점명", ELEVATION_COLUMN]].reset_index(drop=True)


def build_explicit_targets(meta, target_list, base_folder=BASE_FOLDER, ref_time=None):
    """명시 지정된 지점 리스트를 target 테이블로 변환한다 (explicit 모드).

    metadata에 없는 지점, 데이터 폴더가 없는 지점은 경고 후 제외한다.
    동일 지점의 기간행이 여러 개면 평가 기준일(ref_time)에 유효한 행을 우선 사용
    (auto 모드의 filter_valid_station_meta와 일관) — 없으면 마지막 행으로 fallback.
    반환 형식은 pick_target_stations와 동일: [지점, 지점명, 노장해발고도(m)].
    """
    if ref_time is None:
        ref_time = cfg.get_auto_ref_date()
    valid_at_ref = filter_valid_station_meta(meta, ref_time)

    target_list = [str(s) for s in target_list]
    if not target_list:
        raise ValueError("TARGET_MODE='explicit'이지만 TARGET_LIST가 비어 있음")

    rows = []
    for stn in target_list:
        row = valid_at_ref[valid_at_ref["지점"] == stn]
        if len(row) == 0:
            # 기준일에 유효한 행이 없으면 전체 metadata에서 마지막 행으로 fallback
            row = meta[meta["지점"] == stn]
            if len(row) > 0:
                print(f"  [주의] 지점 {stn}: 기준일({ref_time}) 유효 행 없음 -> 마지막 기간행 사용")
        if len(row) == 0:
            print(f"  [경고] 지점 {stn}: metadata에 없음 -> 제외")
            continue
        if not os.path.isdir(os.path.join(base_folder, f"Station_{stn}")):
            print(f"  [경고] 지점 {stn}: 데이터 폴더 없음 -> 제외")
            continue
        rows.append(row.iloc[[-1]])

    if not rows:
        raise ValueError("TARGET_LIST 중 사용 가능한 지점이 없음")

    picks = pd.concat(rows).drop_duplicates(subset=["지점"])
    return picks[["지점", "지점명", ELEVATION_COLUMN]].reset_index(drop=True)


def select_targets(meta, base_folder=BASE_FOLDER):
    """설정(TARGET_MODE)에 따라 target 테이블을 만든다."""
    if cfg.TARGET_MODE == "explicit":
        return build_explicit_targets(meta, cfg.TARGET_LIST, base_folder)
    if cfg.TARGET_MODE == "auto":
        return pick_target_stations(meta, base_folder)
    raise ValueError(f"알 수 없는 TARGET_MODE: {cfg.TARGET_MODE!r}")


def load_lapse_tables(methods):
    monthly_table = load_monthly_table() if "monthly_lapse" in methods else None
    band_table = load_band_table() if "band_lapse" in methods else None
    return monthly_table, band_table


def main(months=None, output_folder=None):
    """batch 평가 실행. months/output_folder는 회귀 검증용 오버라이드."""
    if output_folder is None:
        output_folder = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)
    meta = load_station_meta()

    if months is None:
        months = discover_months(YEAR)
    print("발견된 월:", months)

    picks = select_targets(meta)
    target_stations = picks["지점"].tolist()
    print(f"선정된 target station ({cfg.TARGET_MODE} 모드):")
    print(picks.to_string(index=False))

    all_results = []
    monthly_table, band_table = load_lapse_tables(METHODS)

    for ym in months:
        year = int(ym[:4])
        month = int(ym[4:])
        start = pd.Timestamp(year=year, month=month, day=1)
        end = (start + pd.offsets.MonthBegin(1))

        print()
        print("=" * 50)
        print("월:", ym, "로드 중...")
        data, emeta = prepare_evaluation_data(
            meta, start, end, target_column=TARGET_COLUMN
        )
        print("사용 가능 station:", len(data.columns), "| row:", len(data))

        for method in METHODS:
            result = evaluate_idw(
                data, emeta,
                target_stations=target_stations,
                elevation_method=method,
                monthly_lapse_table=monthly_table,
                band_lapse_table=band_table,
            )
            if len(result) == 0:
                continue

            result = result.copy()
            result["MONTH"] = ym
            result["METHOD"] = method
            all_results.append(result)

            # 증분 저장 (중단 대비)
            part_path = os.path.join(output_folder, f"raw_{ym}_{method}.csv")
            result.to_csv(part_path, index=False, encoding="utf-8-sig")
            print(f"  {method}: {len(result)} rows -> {part_path}")

    if not all_results:
        print("결과 없음")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(output_folder, "batch_raw_all.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    # target station 고도 정보도 함께 저장 (분석에서 사용)
    picks_path = os.path.join(output_folder, "target_stations.csv")
    picks.to_csv(picks_path, index=False, encoding="utf-8-sig")

    print()
    print("전체 raw 저장:", combined_path)
    print("target 정보 저장:", picks_path)


if __name__ == "__main__":
    main()
