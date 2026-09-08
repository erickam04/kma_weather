"""파이프라인 단일 설정 (single source of truth).

target 집합·평가 기간·보정 method·출력 경로를 이 파일 한 곳에서 관리한다.
`run_pipeline.py`(오케스트레이터)와 각 러너(run_batch_test, analyze_results,
visualize_batch, run_visualize_targets)가 이 값을 읽는다.

계획: .omc/plans/pipeline-config-refactor.md
주의: 알고리즘 모듈(idw, elevation_correction, lapse_rate)은 이 설정을 읽지 않는다
      (로직 무수정 원칙 — 설정은 러너 계층에서만 소비).
"""

import os

# ---------------------------------------------------------------------------
# Target 지정 — 두 모드
# ---------------------------------------------------------------------------
# "auto"     : 고도구간별 자동 선정 (현행 연구용 — 저/중/고지대 균등 샘플링)
# "explicit" : 아래 TARGET_LIST의 지점을 그대로 사용 (eLoran 수신 지점 등 임의 지정)
TARGET_MODE = "auto"

# explicit 모드에서 사용할 지점 번호 리스트 (문자열).
# 예: ["160", "554", "871"]
TARGET_LIST = []

# auto 모드: 고도 값 구간과 구간별 선정 개수 (기존 run_batch_test 상수와 동일)
AUTO_ELEVATION_BANDS = [(0, 100), (100, 500), (500, float("inf"))]
AUTO_N_PER_BAND = 3

# ---------------------------------------------------------------------------
# 평가 기간
# ---------------------------------------------------------------------------
# batch 대상 연도. 저장된 AWS_{YEAR}MM.csv 파일에서 월 목록을 자동 탐색한다.
YEAR = 2024

# auto target 선정 기준일. None이면 f"{YEAR}-06-15" (기존 동작과 동일).
AUTO_REF_DATE = None

# neighbor 시각화용 4계절 대표일. None이면 YEAR의 1/4/7/10월 15일 자동 생성.
# 명시하려면 [("1_winter", "YYYY-MM-DD"), ...] 형태.
SEASON_DAYS = None

# ---------------------------------------------------------------------------
# 보정 method / 출력 경로
# ---------------------------------------------------------------------------
METHODS = ["none", "fixed_lapse", "monthly_lapse", "band_lapse"]

OUT_BATCH = "WeatherData_AWS/interpolation_batch"
OUT_VIZ = "WeatherData_AWS/interpolation_visualization"

# ---------------------------------------------------------------------------
# 시각화 하위 폴더 구조 (카테고리 -> OUT_VIZ 기준 상대경로)
# ---------------------------------------------------------------------------
# 모든 시각화 산출물을 카테고리별 하위 폴더로 분리한다. 생산자 스크립트
# (visualize_idw/visualize_targets_map/visualize_batch/analyze_station_mae/
#  analyze_variable_lapse/build_lapse_tables)가 이 맵을 공유해 경로 일관성 유지.
VIZ_SUBDIRS = {
    "neighbor_map": "neighbor_detail/maps",
    "neighbor_weights": "neighbor_detail/weights",
    "neighbor_timeseries": "neighbor_detail/timeseries",
    "neighbor_trace": "neighbor_detail/trace",
    "neighbor_predictions": "neighbor_detail/predictions",
    "targets_overview": "targets_overview",
    "batch_performance": "batch_performance",
    "mae_analysis": "mae_analysis",
    "variable_lapse": "variable_lapse",
}


def viz_dir(category, base=None):
    """시각화 하위 폴더 경로. base 미지정 시 OUT_VIZ 기준. 존재 보장은 호출측에서."""
    base = OUT_VIZ if base is None else base
    return os.path.join(base, VIZ_SUBDIRS[category])


# ---------------------------------------------------------------------------
# 파생값 헬퍼 (설정에서 유도 — batch와 시각화의 기간을 강제로 일치시킨다)
# ---------------------------------------------------------------------------
def get_auto_ref_date():
    """auto target 선정 기준일 (기존 기본값: 해당 연도 6/15)."""
    return AUTO_REF_DATE if AUTO_REF_DATE is not None else f"{YEAR}-06-15"


def get_season_days():
    """4계절 대표일 [(이름, 날짜)] — SEASON_DAYS 미지정 시 YEAR에서 유도."""
    if SEASON_DAYS is not None:
        return list(SEASON_DAYS)
    return [
        ("1_winter", f"{YEAR}-01-15"),
        ("2_spring", f"{YEAR}-04-15"),
        ("3_summer", f"{YEAR}-07-15"),
        ("4_fall", f"{YEAR}-10-15"),
    ]
