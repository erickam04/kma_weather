"""관측소 해발고도에 따른 시간대별 고도보정 효과를 기존 LOO 결과에서 비교한다.

IDW·감률·원자료는 변경하지 않는다. 양의 개선량은 평균 치우침이 0에 가까워진 양이다.
관측소별 짝 비교를 먼저 하고 집단을 요약하며, 일별 절대오차(MAE)를 계산하지 않는다.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bias_data import load_station_pairs, open_readonly_db
from find_stations import filter_station_meta_for_period, filter_valid_station_meta, load_station_meta
from project_paths import META_PATH


ROOT = Path(__file__).resolve().parent
FULLNET = ROOT / "WeatherData_AWS/interpolation_batch/fullnet"
SOURCE = FULLNET / "bias_analysis"
DEFAULT_OUTPUT = FULLNET / "elevation_hour_effect"
DAY_HOURS = tuple(range(9, 18))
NIGHT_HOURS = (21, 22, 23, 0, 1, 2, 3, 4, 5)
SEASONS = ("봄(3~5월)", "여름(6~8월)", "가을(9~11월)", "겨울(1·2·12월)")
BANDS = ("100m 미만", "100~500m 미만", "500m 이상")
METRICS = ("BIAS_BEFORE", "BIAS_AFTER", "ABS_BIAS_BEFORE", "ABS_BIAS_AFTER", "IMPROVEMENT")


def elevation_band(elevation, boundaries=(100.0, 500.0)):
    """해발고도를 중복 없는 표시 구간으로 나눈다. 물리 경계로 해석하지 않는다."""
    z = float(elevation)
    if not np.isfinite(z):
        raise ValueError("elevation must be finite")
    low, high = boundaries
    if low >= high:
        raise ValueError("elevation boundaries must increase")
    if z < low:
        return f"{low:g}m 미만"
    if z < high:
        return f"{low:g}~{high:g}m 미만"
    return f"{high:g}m 이상"


def metadata_elevation_history(meta, station_ids, boundaries=(100.0, 500.0)):
    """2024년 실제 유효 metadata의 고도·구간 변경 여부를 재구성한다."""
    wanted = {str(stn) for stn in station_ids}
    period = filter_station_meta_for_period(meta, "2024-01-01", "2025-01-01")
    period = period.loc[period["지점"].astype(str).isin(wanted)].copy()
    reference_date = pd.Timestamp("2024-06-15")
    events = {pd.Timestamp("2024-01-01"), reference_date}
    for col in ("시작일", "종료일"):
        events.update(period.loc[period[col].between("2024-01-01", "2024-12-31 23:59:59"), col])
    values = {stn: set() for stn in wanted}
    missing = {stn: False for stn in wanted}
    at_reference, last_valid = {}, {}
    for date in sorted(events):
        valid = filter_valid_station_meta(period, date)
        for _, row in valid.iterrows():
            stn = str(row["지점"])
            z = float(row["노장해발고도(m)"])
            if np.isfinite(z):
                values[stn].add(z)
                last_valid[stn] = row
                if date == reference_date:
                    at_reference[stn] = row
            else:
                missing[stn] = True
    records = []
    for stn in sorted(wanted):
        levels = sorted(values[stn])
        if not levels:
            raise ValueError(f"no valid 2024 elevation metadata for {stn}")
        representative = at_reference.get(stn, last_valid[stn])
        records.append({
            "STN": stn, "ELEV_MIN_M": min(levels), "ELEV_MAX_M": max(levels),
            "ELEV_VALUE_COUNT": len(levels), "ELEV_CHANGED": len(levels) > 1,
            "BAND_CHANGED": len({elevation_band(z, boundaries) for z in levels}) > 1,
            "ELEV_METADATA_MISSING": missing[stn],
            "ELEV_REFERENCE_M": float(representative["노장해발고도(m)"]),
            "STATION_NAME_REFERENCE": representative.get("지점명", None),
            "ELEV_REFERENCE_BASIS": "2024-06-15 유효 정보" if stn in at_reference else "2024년 내 마지막 유효 정보",
        })
    return pd.DataFrame(records)


def build_cohort(factors, coverage, meta, boundaries=(100.0, 500.0)):
    """전체 station 보존 + 자료 적격성/고도 구간 변경을 다른 사유로 기록한다."""
    base_cols = ["STN", "지점명", "elev_m"]
    base = factors[base_cols].copy()
    base["STN"] = base["STN"].astype(str)
    cov = coverage[["STN", "COVERAGE_DAYS", "COVERAGE_MONTHS"]].copy()
    cov["STN"] = cov["STN"].astype(str)
    if base.STN.duplicated().any() or cov.STN.duplicated().any():
        raise ValueError("duplicate station in cohort input")
    out = base.merge(cov, on="STN", how="left", validate="one_to_one")
    if out[["COVERAGE_DAYS", "COVERAGE_MONTHS"]].isna().any().any():
        raise ValueError("missing coverage in cohort")
    history = metadata_elevation_history(meta, out.STN, boundaries)
    out = out.merge(history, on="STN", how="left", validate="one_to_one")
    # 기존 요인표의 종료 지점 fallback을 재사용하지 않는다. 원래 값은 감사용으로 보존한다.
    out = out.rename(columns={"elev_m": "FACTOR_ELEV_M", "지점명": "FACTOR_STATION_NAME"})
    out["elev_m"] = out.ELEV_REFERENCE_M
    out["지점명"] = out.STATION_NAME_REFERENCE.fillna(out.FACTOR_STATION_NAME)
    out["FACTOR_METADATA_MISMATCH"] = (
        (out.elev_m - out.FACTOR_ELEV_M).abs().gt(0.051)
        | out["지점명"].ne(out.FACTOR_STATION_NAME)
    )
    out["ELEVATION_BAND"] = out.elev_m.map(lambda z: elevation_band(z, boundaries))
    out["COVERAGE_ELIGIBLE"] = (out.COVERAGE_DAYS >= 180) & (out.COVERAGE_MONTHS >= 9)
    out["COHORT_MAIN"] = out.COVERAGE_ELIGIBLE & ~out.BAND_CHANGED & ~out.ELEV_METADATA_MISSING
    out["MAIN_EXCLUDE_REASON"] = np.select(
        [~out.COVERAGE_ELIGIBLE, out.ELEV_METADATA_MISSING, out.BAND_CHANGED],
        ["연간 자료 부족", "고도 정보 결측", "고도 구간 변경"], default="포함",
    )
    return out.sort_values("STN").reset_index(drop=True)


def prepare_hourly(source, cohort, require_complete=True):
    """입력의 기존 파생 열을 재사용하지 않고 같은 station·시각의 전후만 비교한다."""
    extra = ["SEASON"] if "SEASON" in source else []
    required = ["STN", *extra, "HOUR", "n_hours", "BIAS_none", "BIAS_fixed"]
    out = source[required].copy()
    out["STN"] = out.STN.astype(str)
    keys = ["STN", *extra, "HOUR"]
    if out.duplicated(keys).any():
        raise ValueError("duplicate station/hour record")
    numeric = out[["HOUR", "n_hours", "BIAS_none", "BIAS_fixed"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("hour/count/BIAS must be finite")
    if not ((out.HOUR >= 0) & (out.HOUR <= 23) & (out.HOUR % 1 == 0)).all():
        raise ValueError("HOUR must be an integer in 0..23")
    if not ((out.n_hours > 0) & (out.n_hours % 1 == 0)).all():
        raise ValueError("observation count must be a positive integer")
    if require_complete:
        hour_counts = out.groupby(["STN", *extra]).HOUR.nunique()
        if not hour_counts.eq(24).all():
            raise ValueError("each station/period must have all 24 hours")
    if cohort.STN.duplicated().any() or not set(out.STN).issubset(set(cohort.STN.astype(str))):
        raise ValueError("station missing or duplicate in cohort")
    out = out.rename(columns={"BIAS_none": "BIAS_BEFORE_C", "BIAS_fixed": "BIAS_AFTER_C"})
    out["ABS_BIAS_BEFORE_C"] = out.BIAS_BEFORE_C.abs()
    out["ABS_BIAS_AFTER_C"] = out.BIAS_AFTER_C.abs()
    out["IMPROVEMENT_C"] = out.ABS_BIAS_BEFORE_C - out.ABS_BIAS_AFTER_C
    out = out.merge(cohort, on="STN", how="left", validate="many_to_one")
    return out.sort_values(keys).reset_index(drop=True)


def group_hourly_summary(hourly):
    """시간대별 대표값: 관측소 동일 가중, 전후 짝 비교 후 중앙값·평균 계산."""
    keys = (["SEASON"] if "SEASON" in hourly else []) + ["ELEVATION_BAND", "HOUR"]
    specs = {"N_STATIONS": ("STN", "nunique"), "N_PAIRED_HOURS": ("n_hours", "sum")}
    for metric in METRICS:
        source = f"{metric}_C"
        specs[f"{metric}_MEDIAN_C"] = (source, "median")
        specs[f"{metric}_MEAN_C"] = (source, "mean")
        specs[f"{metric}_Q25_C"] = (source, lambda x: x.quantile(0.25))
        specs[f"{metric}_Q75_C"] = (source, lambda x: x.quantile(0.75))
    return hourly.groupby(keys, sort=True, dropna=False).agg(**specs).reset_index()


def station_daynight(hourly):
    """24시간 효과에서 고정 시각대의 주야 대비를 관측소별로 먼저 계산한다."""
    keys = ["STN"] + (["SEASON"] if "SEASON" in hourly else [])
    records = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key, data in hourly.groupby(grouper, sort=True, dropna=False):
        key = (key,) if len(keys) == 1 else key
        record = dict(zip(keys, key))
        record.update({"ELEVATION_BAND": data.ELEVATION_BAND.iloc[0], "elev_m": data.elev_m.iloc[0]})
        for tag, hours in (("DAY", DAY_HOURS), ("NIGHT", NIGHT_HOURS)):
            part = data.loc[data.HOUR.isin(hours)]
            if len(part) != 9 or part.HOUR.nunique() != 9:
                raise ValueError("day/night contrast requires nine distinct hours per period")
            record[f"{tag}_HOURS"] = len(part)
            record[f"{tag}_IMPROVEMENT_C"] = float(part.IMPROVEMENT_C.mean())
            record[f"{tag}_BEFORE_ABS_C"] = float(part.ABS_BIAS_BEFORE_C.mean())
            record[f"{tag}_AFTER_ABS_C"] = float(part.ABS_BIAS_AFTER_C.mean())
        record["DAY_NIGHT_ADVANTAGE_C"] = record["DAY_IMPROVEMENT_C"] - record["NIGHT_IMPROVEMENT_C"]
        record["ALL_HOUR_MEAN_IMPROVEMENT_C"] = float(data.IMPROVEMENT_C.mean())
        records.append(record)
    return pd.DataFrame(records)


def group_daynight_summary(stations):
    """주야 차이를 관측소별로 구한 후 집단을 요약한다."""
    keys = (["SEASON"] if "SEASON" in stations else []) + ["ELEVATION_BAND"]
    specs = {"N_STATIONS": ("STN", "nunique")}
    metrics = ("DAY_IMPROVEMENT", "NIGHT_IMPROVEMENT", "DAY_NIGHT_ADVANTAGE", "ALL_HOUR_MEAN_IMPROVEMENT")
    for metric in metrics:
        source = f"{metric}_C"
        for tag, op in (("MEDIAN", "median"), ("MEAN", "mean"), ("Q25", lambda x: x.quantile(.25)), ("Q75", lambda x: x.quantile(.75))):
            specs[f"{metric}_{tag}_C"] = (source, op)
    return stations.groupby(keys, dropna=False).agg(**specs).reset_index()


def aggregate_pairs_by_hour(pairs, seasonal=False):
    """오차를 먼저 평균하여 BIAS를 구한다. 계절은 같은 달력연도의 월 묶음이다."""
    data = pairs[["STN", "TM", "error_none", "error_fixed"]].copy()
    data["TM"] = pd.to_datetime(data.TM, errors="raise")
    data["STN"] = data.STN.astype(str)
    if data.duplicated(["STN", "TM"]).any():
        raise ValueError("duplicate paired timestamp")
    if data.empty or data.TM.isna().any() or not np.isfinite(data[["error_none", "error_fixed"]].to_numpy()).all():
        raise ValueError("paired data must be nonempty and finite")
    data["HOUR"] = data.TM.dt.hour
    keys = ["STN"]
    if seasonal:
        month_to_season = {m: SEASONS[0] for m in (3, 4, 5)}
        month_to_season.update({m: SEASONS[1] for m in (6, 7, 8)})
        month_to_season.update({m: SEASONS[2] for m in (9, 10, 11)})
        month_to_season.update({m: SEASONS[3] for m in (1, 2, 12)})
        data["SEASON"] = data.TM.dt.month.map(month_to_season)
        keys.append("SEASON")
    keys.append("HOUR")
    return data.groupby(keys).agg(n_hours=("TM", "size"), BIAS_none=("error_none", "mean"), BIAS_fixed=("error_fixed", "mean")).reset_index()


def verify_coverage(cohort, audit):
    """자료 적격성을 결정한 일수·개월수도 원자료와 대조한다."""
    if set(cohort.STN) != set(audit.STN):
        raise ValueError("coverage mismatch: station set")
    joined = cohort.merge(audit, on="STN", validate="one_to_one")
    if not (joined.COVERAGE_DAYS.eq(joined.PAIRED_DAYS) & joined.COVERAGE_MONTHS.eq(joined.PAIRED_MONTHS)).all():
        raise ValueError("coverage mismatch: days or months")


def common_seasonal_ids(hourly, main_ids):
    """주분석 관측소 중 모든 계절에 24시각 자료가 있는 공통 표본을 고른다."""
    sizes = hourly.groupby(["STN", "SEASON"]).HOUR.nunique().unstack("SEASON")
    return set(sizes.index[sizes.reindex(columns=SEASONS).eq(24).all(axis=1)]) & set(main_ids)


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def plot_outputs(summary, output, seasonal=None):
    """발표용 색상 PNG. 물리 현상을 단정하는 문구·주야 배경 음영은 넣지 않는다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    before_color, after_color = "#2878B5", "#E1812C"
    band_colors = ("#2878B5", "#3C9274", "#9C4F96")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True, layout="constrained")
    for ax, band in zip(axes, BANDS):
        data = summary.loc[summary.ELEVATION_BAND.eq(band)].sort_values("HOUR")
        ax.plot(data.HOUR, data.BIAS_BEFORE_MEDIAN_C, "o-", ms=3, color=before_color, label="보정 전")
        ax.plot(data.HOUR, data.BIAS_AFTER_MEDIAN_C, "s-", ms=3, color=after_color, label="보정 후")
        ax.axhline(0, color="#444444", lw=1)
        ax.set(title=f"{band} (n={int(data.N_STATIONS.iloc[0])})", xlabel="시각 (KST)", xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("관측소별 BIAS의 중앙값 (°C)")
    axes[0].legend(frameon=False)
    fig.suptitle("해발고도별 시간대 BIAS: 고도보정 전·후")
    fig.savefig(plots / "01_bias_before_after.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    for band, color, marker in zip(BANDS, band_colors, ("o", "s", "^")):
        data = summary.loc[summary.ELEVATION_BAND.eq(band)].sort_values("HOUR")
        ax.plot(data.HOUR, data.IMPROVEMENT_MEDIAN_C, color=color, marker=marker, ms=4, label=f"{band} (n={int(data.N_STATIONS.iloc[0])})")
    ax.axhline(0, color="#444444", lw=1)
    ax.set(title="해발고도에 따른 시간대별 고도보정 효과", xlabel="시각 (KST)", ylabel="관측소별 개선량의 중앙값 (°C)\n양수: 개선 / 음수: 악화", xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(plots / "02_hourly_improvement.png", dpi=180)
    plt.close(fig)
    if seasonal is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
        for ax, season in zip(axes.flat, SEASONS):
            for band, color in zip(BANDS, band_colors):
                data = seasonal.loc[seasonal.SEASON.eq(season) & seasonal.ELEVATION_BAND.eq(band)].sort_values("HOUR")
                ax.plot(data.HOUR, data.IMPROVEMENT_MEDIAN_C, color=color, label=f"{band} (n={int(data.N_STATIONS.iloc[0])})")
            ax.axhline(0, color="#444444", lw=1)
            ax.set(title=season, xlabel="시각 (KST)", ylabel="개선량 중앙값 (°C)", xticks=[0, 6, 12, 18, 23])
            ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("계절별 확인: 같은 관측소 집합의 시간대별 보정 효과")
        fig.savefig(plots / "03_seasonal_improvement.png", dpi=180)
        plt.close(fig)


def source_label(path):
    """입력 해시 기록에 기계별 절대경로를 남기지 않는다."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else f"external/{path.name}"


def run(output=DEFAULT_OUTPUT, with_seasons=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    factors_path = SOURCE / "bias_station_factors.csv"
    hourly_path = SOURCE / "bias_diurnal.csv"
    coverage_path = FULLNET / "bias_feedback_analysis/bias_feedback_station_results.csv"
    factors = pd.read_csv(factors_path, encoding="utf-8-sig", dtype={"STN": str})
    coverage = pd.read_csv(coverage_path, encoding="utf-8-sig", dtype={"STN": str})
    raw_hourly = pd.read_csv(hourly_path, encoding="utf-8-sig", dtype={"STN": str}, usecols=["STN", "HOUR", "n_hours", "BIAS_none", "BIAS_fixed"])
    meta_path = META_PATH
    meta = load_station_meta(meta_path)
    cohort = build_cohort(factors, coverage, meta)
    hourly = prepare_hourly(raw_hourly, cohort)
    main = hourly.loc[hourly.COHORT_MAIN].copy()
    station_periods = station_daynight(hourly)
    main_ids = set(cohort.loc[cohort.COHORT_MAIN, "STN"])
    main_periods = station_periods.loc[station_periods.STN.isin(main_ids)].copy()
    hourly_summary = group_hourly_summary(main)
    period_summary = group_daynight_summary(main_periods)
    write_csv(cohort, output / "station_cohort.csv")
    write_csv(hourly, output / "station_hour_bias.csv")
    write_csv(hourly_summary, output / "main_hourly_summary.csv")
    write_csv(station_periods, output / "station_daynight.csv")
    write_csv(period_summary, output / "main_daynight_summary.csv")
    print("MAIN", period_summary[["ELEVATION_BAND", "N_STATIONS", "DAY_IMPROVEMENT_MEDIAN_C", "NIGHT_IMPROVEMENT_MEDIAN_C", "DAY_NIGHT_ADVANTAGE_MEDIAN_C"]].to_json(orient="records", force_ascii=False), flush=True)
    # 승인한 자료 구성과 고도 구간 경계의 민감도만 확인한다.
    masks = {
        "주분석": cohort.COHORT_MAIN,
        "전체 529개 입력": pd.Series(True, index=cohort.index),
        "자료 적격만 적용_구간 변경 포함": cohort.COVERAGE_ELIGIBLE,
        "고도값 변경 모두 제외": cohort.COHORT_MAIN & ~cohort.ELEV_CHANGED,
        "12개월·300일 이상": cohort.COHORT_MAIN & cohort.COVERAGE_MONTHS.eq(12) & cohort.COVERAGE_DAYS.ge(300),
    }
    sensitivity_periods, sensitivity_curves = [], []
    for name, mask in masks.items():
        ids = set(cohort.loc[mask, "STN"])
        sensitivity_periods.append(group_daynight_summary(station_periods.loc[station_periods.STN.isin(ids)]).assign(VARIANT=name))
        sensitivity_curves.append(group_hourly_summary(hourly.loc[hourly.STN.isin(ids)]).assign(VARIANT=name))
    # 첫 경계만 100→200m로 바꾸는 표시 민감도. 새로운 물리 경계가 아니다.
    alt_cohort = build_cohort(factors, coverage, meta, boundaries=(200., 500.))
    alt_hourly = prepare_hourly(raw_hourly, alt_cohort)
    # 경계 변경만 비교하기 위해 관측소 집합은 주 분석과 동일하게 고정한다.
    alt_hourly = alt_hourly.loc[alt_hourly.STN.isin(main_ids)]
    sensitivity_periods.append(group_daynight_summary(station_daynight(alt_hourly)).assign(VARIANT="구간경계200m_대상고정"))
    sensitivity_curves.append(group_hourly_summary(alt_hourly).assign(VARIANT="구간경계200m_대상고정"))
    write_csv(pd.concat(sensitivity_periods, ignore_index=True), output / "sensitivity_daynight.csv")
    write_csv(pd.concat(sensitivity_curves, ignore_index=True), output / "sensitivity_hourly.csv")
    summary = {
        "question": "고도보정 효과의 시간대별 차이가 관측소 해발고도에 따라 달라지는가",
        "n_input_stations": len(cohort), "n_main_stations": len(main_ids),
        "n_input_station_hours": len(hourly), "n_paired_observations": int(hourly.n_hours.sum()),
        "main_counts": cohort.loc[cohort.COHORT_MAIN].ELEVATION_BAND.value_counts().to_dict(),
        "exclusions": cohort.MAIN_EXCLUDE_REASON.value_counts().to_dict(),
        "n_elevation_changed": int(cohort.ELEV_CHANGED.sum()),
        "representative_metadata_corrected_stations": cohort.loc[cohort.FACTOR_METADATA_MISMATCH, "STN"].tolist(),
        "main_hour_count_min": int(main.n_hours.min()), "main_hour_count_max": int(main.n_hours.max()),
        "timezone": "KST", "day_hours": DAY_HOURS, "night_hours": NIGHT_HOURS,
        "inference": "관측된 기술적 비교; 통계적 유의성 검정이나 인과 증명이 아님",
        "known_calendar_gap": "2024-02-29 전 분석 station raw 행 없음(앞선 입력 감사)",
        "sources_sha256": {source_label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (factors_path, hourly_path, coverage_path, meta_path)},
    }
    seasonal_summary = None
    if with_seasons:
        seasonal_chunks, audits = [], []
        reference = raw_hourly.set_index(["STN", "HOUR"])
        with open_readonly_db() as con:
            con.execute("PRAGMA query_only=ON")
            for index, stn in enumerate(cohort.STN, 1):
                pairs = load_station_pairs(con, stn)
                annual = aggregate_pairs_by_hour(pairs).set_index(["STN", "HOUR"]).sort_index()
                expected = reference.loc[[stn]].sort_index()
                max_error = float(np.abs(annual[["BIAS_none", "BIAS_fixed"]].to_numpy() - expected[["BIAS_none", "BIAS_fixed"]].to_numpy()).max())
                counts_match = bool(np.array_equal(annual.n_hours.to_numpy(), expected.n_hours.to_numpy()))
                if max_error > 1e-10 or not counts_match:
                    raise ValueError(f"raw annual aggregation mismatch: {stn} {max_error}")
                seasonal_chunks.append(aggregate_pairs_by_hour(pairs, seasonal=True))
                audits.append({"STN": stn, "N_PAIRS": len(pairs), "MAX_BIAS_RECHECK_ERROR_C": max_error, "COUNTS_MATCH": counts_match, "PAIRED_DAYS": pairs.TM.dt.normalize().nunique(), "PAIRED_MONTHS": pairs.TM.dt.month.nunique()})
                if index % 50 == 0 or index == len(cohort):
                    print(f"SEASONAL {index}/{len(cohort)} stations checked", flush=True)
        seasonal_raw = pd.concat(seasonal_chunks, ignore_index=True)
        verify_coverage(cohort, pd.DataFrame(audits))
        seasonal_all = prepare_hourly(seasonal_raw, cohort, require_complete=False)
        complete_ids = common_seasonal_ids(seasonal_all, main_ids)
        seasonal_main = seasonal_all.loc[seasonal_all.STN.isin(complete_ids)].copy()
        seasonal_summary = group_hourly_summary(seasonal_main)
        seasonal_periods = station_daynight(seasonal_main)
        write_csv(seasonal_all, output / "seasonal_station_hour.csv")
        write_csv(seasonal_summary, output / "seasonal_hourly_summary.csv")
        write_csv(seasonal_periods, output / "seasonal_station_daynight.csv")
        write_csv(group_daynight_summary(seasonal_periods), output / "seasonal_daynight_summary.csv")
        write_csv(pd.DataFrame(audits), output / "source_verification.csv")
        summary["seasonal_common_station_count"] = len(complete_ids)
        summary["seasonal_common_counts"] = cohort.loc[cohort.STN.isin(complete_ids)].ELEVATION_BAND.value_counts().to_dict()
        summary["seasonal_pair_count_min"] = int(seasonal_main.n_hours.min())
        summary["raw_bias_recheck_max_error_c"] = max(row["MAX_BIAS_RECHECK_ERROR_C"] for row in audits)
        summary["all_raw_hour_counts_match"] = all(row["COUNTS_MATCH"] for row in audits)
        summary["all_raw_coverage_days_months_match"] = True
    plot_outputs(hourly_summary, output, seasonal_summary)
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-seasons", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.with_seasons)
