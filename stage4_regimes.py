"""Stage 4-① 이웃 기하 안정성 진단 + 세그먼트(regime) 정의 (A안).

neighbor 지도를 "달력 반기"가 아니라 "이웃 기하가 실제로 안정적인 구간"마다 1장 그리기
위한 사전 계산. SQLite raw의 NEIGHBOR_SET_ID(그 시각 실제 사용 이웃집합 해시)를:
  시간별 → 일-최빈(daily dominant, NaN 노이즈 제거) → RLE → 단기 run 병합
으로 축약해 지점별 "안정 세그먼트"를 정의한다. (근거: 검토 결과 최빈 커버리지 중앙값 95%)

target 이설(COORD_EPOCH 변화)은 이웃집합도 바꾸므로 자동 breakpoint가 되고,
COORD_EPOCH가 못 잡는 '이웃의 개폐·이설'도 이웃집합 변화로 포착된다.

알고리즘 무수정, raw 읽기전용. 출력(소형 CSV, OneDrive):
  fullnet/station_neighbor_regimes.csv   — 지점당 1행(진단: churn·안정성)
  fullnet/station_regime_detail.csv      — regime당 1행(지도 1장분: 대표시각 포함)
"""

from project_paths import LOO_DB, LOO_DIR

import os
import sqlite3

import pandas as pd

from find_stations import filter_valid_station_meta, load_station_meta
from idw import ELEVATION_COLUMN

DB = str(LOO_DB)
FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
VALID_SUMMARY = str(LOO_DIR / "station_valid_summary.csv")
MIN_RUN_DAYS = 14      # 이보다 짧은 안정구간은 세그먼트로 안 침(단기 결측·과분할 억제)
MIN_REGIME_DAYS = 14   # 한 서명이 regime(지도 1장)이 되려면 누적 이 일수 이상 대표해야


def _daily_dominant(df):
    """시간별 (date, sig) → 일별 최빈 서명 Series(index=date)."""
    return df.groupby("day")["sig"].agg(lambda x: x.mode().iloc[0])


def _regimes(daily):
    """일-최빈 서명 시퀀스 → 안정 세그먼트(서명별). 단기 run 병합 후 서명별 집계.

    반환: list of dict(signature, first, last, n_days, longest_run_mid_date).
    규칙: RLE run 중 MIN_RUN_DAYS 미만은 대표성 없음으로 보고, 서명별 '누적 일수'가
      MIN_REGIME_DAYS 이상인 서명만 regime으로 채택. 각 regime 대표일 = 그 서명의
      가장 긴 연속 run의 중앙 날짜(실재 시각 보장).
    """
    # RLE runs
    runs = []
    for date, sig in daily.items():
        if runs and runs[-1]["sig"] == sig:
            runs[-1]["last"] = date
            runs[-1]["days"] += 1
        else:
            runs.append({"sig": sig, "first": date, "last": date, "days": 1})

    # 서명별 누적 일수 + 가장 긴 run
    agg = {}
    for r in runs:
        a = agg.setdefault(r["sig"], {"n_days": 0, "first": r["first"],
                                      "last": r["last"], "longest": r})
        a["n_days"] += r["days"]
        a["first"] = min(a["first"], r["first"])
        a["last"] = max(a["last"], r["last"])
        if r["days"] > a["longest"]["days"]:
            a["longest"] = r

    regimes = []
    for sig, a in agg.items():
        if a["n_days"] < MIN_REGIME_DAYS:
            continue
        lr = a["longest"]
        mid = lr["first"] + (lr["last"] - lr["first"]) / 2
        regimes.append({
            "signature": sig, "first": a["first"], "last": a["last"],
            "n_days": a["n_days"], "rep_date": pd.Timestamp(mid).normalize(),
        })
    # 대표 일수 많은 순
    regimes.sort(key=lambda x: -x["n_days"])
    return regimes


def main():
    os.makedirs(FULLNET, exist_ok=True)
    con = sqlite3.connect(DB)
    meta = load_station_meta()
    vref = filter_valid_station_meta(meta, "2024-06-15")
    elev = {str(r["지점"]): r[ELEVATION_COLUMN] for _, r in vref.iterrows()}
    for stn in meta["지점"].unique():
        elev.setdefault(str(stn), meta[meta["지점"] == str(stn)][ELEVATION_COLUMN].iloc[-1])

    vsum = pd.read_csv(VALID_SUMMARY, encoding="utf-8-sig", dtype={"STN": str})
    vratio = dict(zip(vsum["STN"], vsum["valid_ratio"]))

    # 전체 후보 지점 = raw 에 등장하는 STN (interpolable 무관, 고립도 포함)
    all_stn = pd.read_sql("SELECT DISTINCT STN FROM raw", con)["STN"].astype(str).tolist()

    diag_rows, detail_rows = [], []
    for stn in sorted(all_stn, key=lambda s: int(s) if s.isdigit() else s):
        df = pd.read_sql(
            "SELECT substr(TM,1,10) day, NEIGHBOR_SET_ID sig, COORD_EPOCH "
            "FROM raw WHERE STN=? AND METHOD='none' AND interpolable=1",
            con, params=[stn])
        coord_epochs = pd.read_sql(
            "SELECT COUNT(DISTINCT COORD_EPOCH) c FROM raw WHERE STN=?",
            con, params=[stn]).iloc[0]["c"]

        if len(df) > 0:
            df["day"] = pd.to_datetime(df["day"])

        if len(df) == 0:
            # 상시 고립(보간 성공 시각 0) → regime 없음, 고립 지도 대상
            diag_rows.append({
                "STN": stn, "elev_m": elev.get(stn), "valid_ratio": vratio.get(stn, 0.0),
                "n_days": 0, "distinct_daily_sig": 0, "n_regimes": 0,
                "mode_cov_pct": None, "coord_epochs": coord_epochs, "isolated": True,
            })
            continue

        daily = _daily_dominant(df)
        distinct_daily = daily.nunique()
        # 최빈 커버리지(시간 기준)
        mode_cov = df["sig"].value_counts().iloc[0] / len(df) * 100
        regimes = _regimes(daily)

        diag_rows.append({
            "STN": stn, "elev_m": elev.get(stn), "valid_ratio": vratio.get(stn, 0.0),
            "n_days": len(daily), "distinct_daily_sig": distinct_daily,
            "n_regimes": len(regimes), "mode_cov_pct": round(mode_cov, 1),
            "coord_epochs": coord_epochs, "isolated": False,
        })
        for i, r in enumerate(regimes):
            detail_rows.append({
                "STN": stn, "regime_idx": i, "signature": r["signature"],
                "first_date": r["first"], "last_date": r["last"],
                "n_days_sig": r["n_days"],
                "cov_pct": round(r["n_days"] / len(daily) * 100, 1),
                "rep_time": pd.Timestamp(r["rep_date"]) + pd.Timedelta("12:00:00"),
            })

    diag = pd.DataFrame(diag_rows)
    detail = pd.DataFrame(detail_rows)
    diag.to_csv(os.path.join(FULLNET, "station_neighbor_regimes.csv"),
                index=False, encoding="utf-8-sig")
    detail.to_csv(os.path.join(FULLNET, "station_regime_detail.csv"),
                  index=False, encoding="utf-8-sig")
    con.close()

    # 요약
    print(f"지점 {len(diag)} | 고립 {int(diag.isolated.sum())} | "
          f"보간가능 {int((~diag.isolated).sum())}")
    nr = diag[~diag.isolated]["n_regimes"]
    print(f"지점당 regime(지도 장수) 분포: 1장 {int((nr==1).sum())} / "
          f"2장 {int((nr==2).sum())} / 3장+ {int((nr>=3).sum())}")
    print(f"총 세그먼트 지도 장수: {len(detail)} (+ 고립 진단 {int(diag.isolated.sum())}장)")
    print(f"최빈 커버리지 중앙값: {diag['mode_cov_pct'].median()}%")
    print(f"저장: {FULLNET}/station_neighbor_regimes.csv, station_regime_detail.csv")


if __name__ == "__main__":
    main()
