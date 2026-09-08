import os

import pandas as pd

import pipeline_config as cfg


BATCH_FOLDER = cfg.OUT_BATCH
RAW_PATH = os.path.join(BATCH_FOLDER, "batch_raw_all.csv")
TARGETS_PATH = os.path.join(BATCH_FOLDER, "target_stations.csv")
ELEVATION_COLUMN = "노장해발고도(m)"


def season_of(month):
    if month in (12, 1, 2):
        return "1_winter"
    if month in (3, 4, 5):
        return "2_spring"
    if month in (6, 7, 8):
        return "3_summer"
    return "4_fall"


def elevation_band(elev):
    if pd.isna(elev):
        return "unknown"
    if elev < 100:
        return "0-100m"
    if elev < 500:
        return "100-500m"
    return "500m+"


def metrics(df):
    return pd.Series({
        "samples": len(df),
        "MAE": round(df["ABS_ERROR"].mean(), 4),
        "RMSE": round((df["ERROR"] ** 2).mean() ** 0.5, 4),
        "BIAS": round(df["ERROR"].mean(), 4),
    })


def summarize_by(df, keys):
    """keys + METHOD로 그룹핑 후 MAE를 method 축으로 pivot, ΔMAE 추가."""
    grouped = (
        df.groupby(keys + ["METHOD"], group_keys=False)[["ERROR", "ABS_ERROR"]]
        .apply(metrics)
        .reset_index()
    )
    print(grouped.to_string(index=False))

    if keys:
        pivot = grouped.pivot_table(index=keys, columns="METHOD", values="MAE")
        if "none" in pivot.columns and "fixed_lapse" in pivot.columns:
            pivot["dMAE(none-fixed)"] = (pivot["none"] - pivot["fixed_lapse"]).round(4)
            print()
            print("  [MAE 비교 / dMAE>0 이면 고도보정이 개선]")
            print(pivot.to_string())
    return grouped


def main():
    if not os.path.exists(RAW_PATH):
        print("raw 결과 없음:", RAW_PATH, "-> 먼저 run_batch_test.py 실행")
        return

    df = pd.read_csv(RAW_PATH)
    df["STN"] = df["STN"].astype(str)
    df["TM"] = pd.to_datetime(df["TM"])
    df["month_num"] = df["TM"].dt.month
    df["season"] = df["month_num"].apply(season_of)

    # target 고도 붙이기
    if os.path.exists(TARGETS_PATH):
        targets = pd.read_csv(TARGETS_PATH)
        targets["지점"] = targets["지점"].astype(str)
        elev_map = dict(zip(targets["지점"], targets[ELEVATION_COLUMN]))
        df["target_elev"] = df["STN"].map(elev_map)
        df["elev_band"] = df["target_elev"].apply(elevation_band)
    else:
        df["elev_band"] = "unknown"

    print("=" * 60)
    print("[1] 전체 method별 성능")
    print("=" * 60)
    summarize_by(df, [])  # keys 없음 -> METHOD만

    print()
    print("=" * 60)
    print("[2] 계절별 성능")
    print("=" * 60)
    summarize_by(df, ["season"])

    print()
    print("=" * 60)
    print("[3] 월별 성능")
    print("=" * 60)
    summarize_by(df, ["MONTH"])

    print()
    print("=" * 60)
    print("[4] target 고도구간별 성능")
    print("=" * 60)
    summarize_by(df, ["elev_band"])

    print()
    print("=" * 60)
    print("[5] station별 성능")
    print("=" * 60)
    summarize_by(df, ["STN"])

    # 저장
    out = os.path.join(BATCH_FOLDER, "analysis_summary.csv")
    rows = []
    for keys, label in [([], "overall"), (["season"], "season"),
                        (["MONTH"], "month"), (["elev_band"], "elev_band"),
                        (["STN"], "station")]:
        g = df.groupby(keys + ["METHOD"]).apply(metrics).reset_index()
        g.insert(0, "axis", label)
        rows.append(g)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    print()
    print("분석 요약 저장:", out)


if __name__ == "__main__":
    main()
