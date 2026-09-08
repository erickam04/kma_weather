"""주513 관측소의 시간대별 BIAS·MAE·RMSE를 고도구간별로 비교한다.

기존 SQLite materialized 집계 ``agg_stn_hour``와 확정 코호트 CSV를 읽고,
``station_hour_bias.csv``의 전 station×hour BIAS·n과 대조한다.
none/fixed_lapse 값을 나란히 보고하며 차이·개선량 지표는 만들지 않는다.
주야 표준 period 지표는 각 관측소 내부 9시간의 n·se·se2·sa를 pooled해
재계산하고, 그 뒤 관측소 간 동일가중 중앙값·IQR로 요약한다.
"""

from project_paths import LOO_DB, WINDOWS_FONT_DIR

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB_PATH = Path(str(LOO_DB))
OUTPUT_DIR = ROOT / "WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect"
COHORT_PATH = OUTPUT_DIR / "station_cohort.csv"
REFERENCE_BIAS_PATH = OUTPUT_DIR / "station_hour_bias.csv"
EXPECTED_METHOD_PAIRED_N = 4_454_204
METHODS = ("none", "fixed_lapse")
METHOD_PREFIX = {"none": "NONE", "fixed_lapse": "FIXED"}
BANDS = ("100m 미만", "100~500m 미만", "500m 이상")
DAY_HOURS = tuple(range(9, 18))
NIGHT_HOURS = (21, 22, 23, 0, 1, 2, 3, 4, 5)
PERIOD_HOURS = {
    "주간(09~17시)": DAY_HOURS,
    "야간(21~05시)": NIGHT_HOURS,
}
METRICS = ("BIAS", "MAE", "RMSE")


def parse_cohort_main(values):
    """COHORT_MAIN의 명시적 bool/true,false/0,1 표현만 bool로 바꾼다."""
    parsed = []
    for index, value in values.items():
        if pd.isna(value):
            raise ValueError(f"COHORT_MAIN contains missing value at index {index}")
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            parsed.append(bool(value))
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                parsed.append(True)
                continue
            if normalized in {"false", "0"}:
                parsed.append(False)
                continue
        raise ValueError(f"COHORT_MAIN contains unsupported value at index {index}: {value!r}")
    return pd.Series(parsed, index=values.index, name=values.name, dtype=bool)


def file_sha256(path):
    """입력 파일을 스트리밍해 SHA-256을 반환한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_agg_stn_hour(db_path=DB_PATH):
    """기존 시간대 materialized 집계를 SQLite 읽기 전용으로 읽는다."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"{path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "agg_stn_hour" not in tables:
            raise ValueError("agg_stn_hour table is missing")
        return pd.read_sql(
            "SELECT STN, bucket, METHOD, n, se, se2, sa FROM agg_stn_hour",
            con,
        )


def prepare_station_hour_metrics(agg, cohort, require_complete=True):
    """저장 합계를 주 코호트의 station×hour×method 표준지표로 바꾼다."""
    agg_required = {"STN", "bucket", "METHOD", "n", "se", "se2", "sa"}
    cohort_required = {"STN", "지점명", "elev_m", "ELEVATION_BAND", "COHORT_MAIN"}
    if missing := agg_required.difference(agg.columns):
        raise ValueError(f"agg missing columns: {sorted(missing)}")
    if missing := cohort_required.difference(cohort.columns):
        raise ValueError(f"cohort missing columns: {sorted(missing)}")

    c = cohort[list(cohort_required)].copy()
    c["STN"] = c["STN"].astype(str)
    if c["STN"].duplicated().any():
        raise ValueError("duplicate station in cohort")
    c["COHORT_MAIN"] = parse_cohort_main(c["COHORT_MAIN"])
    main = c.loc[c["COHORT_MAIN"]].copy()
    if main.empty:
        raise ValueError("main cohort is empty")
    if main[["elev_m", "ELEVATION_BAND"]].isna().any().any():
        raise ValueError("main cohort elevation metadata is missing")

    a = agg[list(agg_required)].copy()
    a["STN"] = a["STN"].astype(str)
    a = a.loc[a["STN"].isin(main["STN"]) & a["METHOD"].isin(METHODS)].copy()
    if a.empty:
        raise ValueError("no aggregate rows for main cohort")
    for column in ("bucket", "n", "se", "se2", "sa"):
        a[column] = pd.to_numeric(a[column], errors="coerce")
    if not np.isfinite(a[["bucket", "n", "se", "se2", "sa"]].to_numpy(float)).all():
        raise ValueError("aggregate values must be finite")
    if not ((a["bucket"] >= 0) & (a["bucket"] <= 23) & (a["bucket"] % 1 == 0)).all():
        raise ValueError("hour bucket must be an integer in 0..23")
    if not ((a["n"] > 0) & (a["n"] % 1 == 0)).all():
        raise ValueError("aggregate count must be a positive integer")
    if (a[["se2", "sa"]] < 0).any().any():
        raise ValueError("squared and absolute error sums must be nonnegative")
    if a.duplicated(["STN", "bucket", "METHOD"]).any():
        raise ValueError("duplicate station/hour/method aggregate")

    method_sets = a.groupby(["STN", "bucket"])["METHOD"].agg(lambda x: frozenset(x))
    if not method_sets.map(lambda x: x == frozenset(METHODS)).all():
        raise ValueError("station/hour methods are not exactly paired")
    counts = a.pivot(index=["STN", "bucket"], columns="METHOD", values="n")
    if not counts["none"].equals(counts["fixed_lapse"]):
        raise ValueError("paired counts differ between methods")

    if require_complete:
        if set(a["STN"]) != set(main["STN"]):
            raise ValueError("main cohort station is missing from aggregate")
        hour_counts = a.groupby(["STN", "METHOD"])["bucket"].nunique()
        if not hour_counts.eq(24).all():
            raise ValueError("each station/method must contain all 24 hours")

    out = a.rename(columns={
        "bucket": "HOUR", "n": "N_PAIRED_HOURS", "se": "ERROR_SUM_C",
        "se2": "ERROR_SQ_SUM_C2", "sa": "ABS_ERROR_SUM_C",
    })
    out["HOUR"] = out["HOUR"].astype(int)
    out["N_PAIRED_HOURS"] = out["N_PAIRED_HOURS"].astype(int)
    out["BIAS_C"] = out["ERROR_SUM_C"] / out["N_PAIRED_HOURS"]
    out["MAE_C"] = out["ABS_ERROR_SUM_C"] / out["N_PAIRED_HOURS"]
    out["RMSE_C"] = np.sqrt(out["ERROR_SQ_SUM_C2"] / out["N_PAIRED_HOURS"])
    out = out.merge(
        main.drop(columns="COHORT_MAIN"), on="STN", how="left", validate="many_to_one"
    )
    columns = [
        "STN", "지점명", "elev_m", "ELEVATION_BAND", "HOUR", "METHOD",
        "N_PAIRED_HOURS", "BIAS_C", "MAE_C", "RMSE_C",
        "ERROR_SUM_C", "ERROR_SQ_SUM_C2", "ABS_ERROR_SUM_C",
    ]
    out = out[columns].sort_values(["STN", "HOUR", "METHOD"]).reset_index(drop=True)
    validate_error_identities(out)
    return out


def _within_tolerance(left, right):
    tolerance = 1e-10 + 1e-12 * np.maximum(np.abs(left), np.abs(right))
    return left <= right + tolerance


def validate_error_identities(station_hour):
    """오차 지표와 sufficient statistics가 만족해야 하는 부등식을 검증한다."""
    if not _within_tolerance(station_hour["BIAS_C"].abs(), station_hour["MAE_C"]).all():
        raise ValueError("identity violated: |BIAS| <= MAE")
    if not _within_tolerance(station_hour["MAE_C"], station_hour["RMSE_C"]).all():
        raise ValueError("identity violated: MAE <= RMSE")
    if not _within_tolerance(station_hour["ERROR_SUM_C"].abs(), station_hour["ABS_ERROR_SUM_C"]).all():
        raise ValueError("identity violated: |se| <= sa")
    if not _within_tolerance(
        station_hour["ERROR_SUM_C"] ** 2,
        station_hour["N_PAIRED_HOURS"] * station_hour["ERROR_SQ_SUM_C2"],
    ).all():
        raise ValueError("identity violated: se^2 <= n*se2")
    return {"checked_rows": int(len(station_hour)), "all_passed": True}


def validate_reference_bias_alignment(
    station_hour, reference, expected_method_total=EXPECTED_METHOD_PAIRED_N,
):
    """주 코호트 전 행의 BIAS/n을 기존 station_hour_bias.csv와 대조한다."""
    required = {
        "STN", "HOUR", "n_hours", "BIAS_BEFORE_C", "BIAS_AFTER_C", "COHORT_MAIN",
    }
    if missing := required.difference(reference.columns):
        raise ValueError(f"reference bias missing columns: {sorted(missing)}")
    ref = reference[list(required)].copy()
    ref["STN"] = ref["STN"].astype(str)
    ref["COHORT_MAIN"] = parse_cohort_main(ref["COHORT_MAIN"])
    ref = ref.loc[ref["COHORT_MAIN"]].copy()
    for column in ("HOUR", "n_hours", "BIAS_BEFORE_C", "BIAS_AFTER_C"):
        ref[column] = pd.to_numeric(ref[column], errors="coerce")
    if not np.isfinite(ref[["HOUR", "n_hours", "BIAS_BEFORE_C", "BIAS_AFTER_C"]].to_numpy(float)).all():
        raise ValueError("reference BIAS/n values must be finite")
    if ref.duplicated(["STN", "HOUR"]).any():
        raise ValueError("duplicate station/hour in reference bias")

    long_parts = []
    for method, bias_column in (("none", "BIAS_BEFORE_C"), ("fixed_lapse", "BIAS_AFTER_C")):
        part = ref[["STN", "HOUR", "n_hours", bias_column]].copy()
        part["METHOD"] = method
        part = part.rename(columns={"n_hours": "REFERENCE_N", bias_column: "REFERENCE_BIAS_C"})
        long_parts.append(part)
    ref_long = pd.concat(long_parts, ignore_index=True)
    actual = station_hour[["STN", "HOUR", "METHOD", "N_PAIRED_HOURS", "BIAS_C"]].copy()
    compared = actual.merge(
        ref_long, on=["STN", "HOUR", "METHOD"], how="outer", indicator=True,
        validate="one_to_one",
    )
    if not compared["_merge"].eq("both").all():
        raise ValueError("reference BIAS/n keys do not match all main station/hour/method rows")

    compared["BIAS_DIFF_C"] = (compared["BIAS_C"] - compared["REFERENCE_BIAS_C"]).abs()
    compared["N_DIFF"] = (compared["N_PAIRED_HOURS"] - compared["REFERENCE_N"]).abs()
    max_bias = compared.groupby("METHOD")["BIAS_DIFF_C"].max().reindex(METHODS)
    max_n = compared.groupby("METHOD")["N_DIFF"].max().reindex(METHODS)
    if (max_bias > 5e-12).any():
        raise ValueError(f"reference BIAS mismatch: {max_bias.to_dict()}")
    if (max_n > 0).any():
        raise ValueError(f"reference n mismatch: {max_n.to_dict()}")

    totals = actual.groupby("METHOD")["N_PAIRED_HOURS"].sum().reindex(METHODS)
    if expected_method_total is not None and not totals.eq(expected_method_total).all():
        raise ValueError(
            f"paired n total must be {expected_method_total} for each method: {totals.to_dict()}"
        )
    return {
        "n_main_station_hours": int(len(ref)),
        "n_method_rows_compared": int(len(compared)),
        "max_abs_bias_diff_c": {method: float(max_bias[method]) for method in METHODS},
        "max_abs_n_diff": {method: int(max_n[method]) for method in METHODS},
        "paired_n_by_method": {method: int(totals[method]) for method in METHODS},
    }


def _summary_records(frame, group_cols):
    records = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, group in frame.groupby(grouper, sort=False, dropna=False):
        keys = (key,) if len(group_cols) == 1 else tuple(key)
        record = dict(zip(group_cols, keys))
        method_station_sets = {
            method: set(group.loc[group["METHOD"].eq(method), "STN"])
            for method in METHODS
        }
        if method_station_sets["none"] != method_station_sets["fixed_lapse"]:
            raise ValueError("station sets differ between methods")
        record["N_STATIONS"] = len(method_station_sets["none"])
        for method in METHODS:
            prefix = METHOD_PREFIX[method]
            part = group.loc[group["METHOD"].eq(method)]
            record[f"{prefix}_N_PAIRED_HOURS"] = int(part["N_PAIRED_HOURS"].sum())
            for metric in METRICS:
                values = part[f"{metric}_C"]
                q25 = float(values.quantile(0.25))
                q75 = float(values.quantile(0.75))
                record[f"{prefix}_{metric}_MEDIAN_C"] = float(values.median())
                record[f"{prefix}_{metric}_Q25_C"] = q25
                record[f"{prefix}_{metric}_Q75_C"] = q75
                record[f"{prefix}_{metric}_IQR_C"] = q75 - q25
        records.append(record)
    return pd.DataFrame(records)


def _sort_bands(frame, secondary):
    out = frame.copy()
    order = {band: index for index, band in enumerate(BANDS)}
    out["_band_order"] = out["ELEVATION_BAND"].map(order)
    return out.sort_values(["_band_order", secondary]).drop(columns="_band_order").reset_index(drop=True)


def summarize_hourly_by_elevation(station_hour):
    """station별 지표를 동일가중으로 고도구간×시각 중앙값·IQR로 요약한다."""
    summary = _summary_records(station_hour, ["ELEVATION_BAND", "HOUR"])
    return _sort_bands(summary, "HOUR")


def build_station_daynight_metrics(station_hour):
    """관측소 내부 주간·야간 9시간의 n·se·se2·sa를 pooled해 표준지표로 환산한다.

    반환된 관측소별 period 지표는 이후 고도구간 안에서 관측소 동일가중
    중앙값·IQR로 요약하며, 시간별 RMSE나 MAE를 단순 평균하지 않는다.
    """
    parts = []
    identity = ["STN", "지점명", "elev_m", "ELEVATION_BAND", "METHOD"]
    for period, hours in PERIOD_HOURS.items():
        selected = station_hour.loc[station_hour["HOUR"].isin(hours)].copy()
        distinct = selected.groupby(identity)["HOUR"].nunique()
        if not distinct.eq(9).all():
            raise ValueError(f"{period} requires nine distinct hours per station/method")
        grouped = selected.groupby(identity, as_index=False).agg(
            N_PAIRED_HOURS=("N_PAIRED_HOURS", "sum"),
            ERROR_SUM_C=("ERROR_SUM_C", "sum"),
            ERROR_SQ_SUM_C2=("ERROR_SQ_SUM_C2", "sum"),
            ABS_ERROR_SUM_C=("ABS_ERROR_SUM_C", "sum"),
        )
        grouped["PERIOD"] = period
        grouped["BIAS_C"] = grouped["ERROR_SUM_C"] / grouped["N_PAIRED_HOURS"]
        grouped["MAE_C"] = grouped["ABS_ERROR_SUM_C"] / grouped["N_PAIRED_HOURS"]
        grouped["RMSE_C"] = np.sqrt(
            grouped["ERROR_SQ_SUM_C2"] / grouped["N_PAIRED_HOURS"]
        )
        parts.append(grouped)
    columns = identity[:-1] + ["PERIOD", "METHOD", "N_PAIRED_HOURS", *[f"{m}_C" for m in METRICS]]
    return pd.concat(parts, ignore_index=True)[columns]


def summarize_daynight_by_elevation(station_period):
    """pooled station별 period 지표를 관측소 동일가중 중앙값·IQR로 요약한다."""
    summary = _summary_records(station_period, ["ELEVATION_BAND", "PERIOD"])
    period_order = {name: index for index, name in enumerate(PERIOD_HOURS)}
    summary["_period_order"] = summary["PERIOD"].map(period_order)
    summary = _sort_bands(summary, "_period_order")
    return summary.drop(columns="_period_order")


def _font(size, bold=False):
    from PIL import ImageFont
    candidates = [
        (WINDOWS_FONT_DIR / ("malgunbd.ttf" if bold else "malgun.ttf")),
        (WINDOWS_FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf")),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _panel_y_limits(summary, metric):
    """BIAS는 공유축, MAE·RMSE는 패널별 독립축 범위를 계산한다."""
    if metric not in METRICS:
        raise ValueError(f"unsupported plot metric: {metric}")

    def padded_limits(frame):
        values = []
        for prefix in METHOD_PREFIX.values():
            values.extend(frame[f"{prefix}_{metric}_MEDIAN_C"].tolist())
        y_min, y_max = min(values), max(values)
        if metric == "BIAS":
            y_min, y_max = min(y_min, 0.0), max(y_max, 0.0)
        pad = max((y_max - y_min) * 0.12, 0.05)
        y_min = y_min - pad if metric == "BIAS" else max(0.0, y_min - pad)
        y_max += pad
        if y_max <= y_min:
            y_max = y_min + 1.0
        return float(y_min), float(y_max)

    if metric == "BIAS":
        common = padded_limits(summary)
        return {band: common for band in BANDS}
    return {
        band: padded_limits(summary.loc[summary["ELEVATION_BAND"].eq(band)])
        for band in BANDS
    }


def _draw_metric_plot(summary, metric, path):
    from PIL import Image, ImageDraw

    width, height = 1800, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, panel_font, label_font, tick_font = (
        _font(30, True), _font(23, True), _font(18), _font(15)
    )
    metric_name = {name: name for name in METRICS}[metric]
    draw.text((width / 2, 25), f"해발고도별 시간대 {metric_name}: 고도보정 전·후 (°C, KST)",
              fill="#222222", font=title_font, anchor="ma")
    colors = {"NONE": "#2878B5", "FIXED": "#D9534F"}
    labels = {"NONE": "고도보정 미적용", "FIXED": "고정감률 적용"}
    y_limits = _panel_y_limits(summary, metric)

    for panel_index, band in enumerate(BANDS):
        data = summary.loc[summary["ELEVATION_BAND"].eq(band)].sort_values("HOUR")
        if len(data) != 24:
            raise ValueError(f"plot requires 24 hourly rows for {band}")
        left = 75 + panel_index * 580
        top, right, bottom = 115, left + 510, 610
        draw.rectangle((left, top, right, bottom), outline="#555555", width=2)
        n = int(data["N_STATIONS"].iloc[0])
        draw.text(((left + right) / 2, 82), f"{band} (n={n})", fill="#222222",
                  font=panel_font, anchor="ma")
        y_min, y_max = y_limits[band]

        def xy(hour, value):
            x = left + 18 + hour / 23 * (right - left - 36)
            y = bottom - 18 - (value - y_min) / (y_max - y_min) * (bottom - top - 36)
            return x, y

        for tick in (0, 6, 12, 18, 23):
            x, _ = xy(tick, y_min)
            draw.line((x, bottom, x, bottom + 6), fill="#555555", width=1)
            draw.text((x, bottom + 10), str(tick), fill="#333333", font=tick_font, anchor="ma")
        for fraction in (0.0, 0.5, 1.0):
            value = y_min + fraction * (y_max - y_min)
            _, y = xy(0, value)
            draw.line((left, y, right, y), fill="#DDDDDD", width=1)
            draw.text((left - 8, y), f"{value:.2f}", fill="#333333", font=tick_font, anchor="rm")
        if metric == "BIAS":
            _, zero_y = xy(0, 0.0)
            draw.line((left, zero_y, right, zero_y), fill="#777777", width=2)

        for prefix, color in colors.items():
            hours = data["HOUR"].astype(int).tolist()
            median = data[f"{prefix}_{metric}_MEDIAN_C"].tolist()
            points = [xy(h, v) for h, v in zip(hours, median)]
            draw.line(points, fill=color, width=4)
            for point in points:
                x, y = point
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text(((left + right) / 2, 655), "시각 (KST)", fill="#222222",
                  font=label_font, anchor="ma")

    legend_y = 675
    for index, prefix in enumerate(("NONE", "FIXED")):
        x = 660 + index * 280
        draw.line((x, legend_y, x + 45, legend_y), fill=colors[prefix], width=5)
        draw.text((x + 55, legend_y), labels[prefix], fill="#222222", font=label_font, anchor="lm")
    image.save(path, format="PNG")

def plot_hourly_standard_metrics(summary, plots_dir):
    """고도구간 3패널 MAE·RMSE·BIAS 전후 PNG를 각각 만든다."""
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    outputs = []
    for metric, name in (
        ("MAE", "04_hourly_mae_standard_metrics.png"),
        ("RMSE", "05_hourly_rmse_standard_metrics.png"),
        ("BIAS", "06_hourly_bias_standard_metrics.png"),
    ):
        path = plots / name
        _draw_metric_plot(summary, metric, path)
        outputs.append(path)
    return outputs


def _write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_run_summary(
    n_main_stations, n_station_hours, band_counts, outputs, validation, input_hashes,
):
    """재현성·집계 정의·도표 축 계약을 포함한 JSON manifest를 만든다."""
    return {
        "n_main_stations": int(n_main_stations),
        "n_station_hours": int(n_station_hours),
        "methods": list(METHODS),
        "bands": band_counts,
        "timezone": "KST",
        "day_hours": list(DAY_HOURS),
        "night_hours": list(NIGHT_HOURS),
        "period_aggregation": (
            "관측소 내부 9시간의 n·se·se2·sa를 pooled해 표준 period 지표를 재계산한 후, "
            "관측소 간 동일가중 중앙값·IQR로 요약"
        ),
        "aggregation": "station별 지표 계산 후 관측소 동일가중 중앙값·IQR",
        "panel_y_scales": "independent",
        "bias_panel_y_scales": "shared",
        "validation": validation,
        "input_hashes": input_hashes,
        "outputs": list(outputs),
    }


def run(
    db_path=DB_PATH, cohort_path=COHORT_PATH, reference_bias_path=REFERENCE_BIAS_PATH,
    output_dir=OUTPUT_DIR,
):
    """실제 주513 입력을 검증하고 새 CSV·PNG만 별도 이름으로 게시한다."""
    output = Path(output_dir)
    cohort = pd.read_csv(cohort_path, encoding="utf-8-sig", dtype={"STN": str})
    cohort_main = parse_cohort_main(cohort["COHORT_MAIN"])
    if int(cohort_main.sum()) != 513:
        raise ValueError("expected 513 COHORT_MAIN stations")
    station_hour = prepare_station_hour_metrics(load_agg_stn_hour(db_path), cohort)
    identity_validation = validate_error_identities(station_hour)
    reference = pd.read_csv(
        reference_bias_path, encoding="utf-8-sig", dtype={"STN": str},
    )
    reference_validation = validate_reference_bias_alignment(station_hour, reference)
    hourly = summarize_hourly_by_elevation(station_hour)
    station_period = build_station_daynight_metrics(station_hour)
    daynight = summarize_daynight_by_elevation(station_period)

    files = {
        "standard_metrics_station_hour.csv": station_hour,
        "standard_metrics_hourly_by_elevation.csv": hourly,
        "standard_metrics_station_daynight.csv": station_period,
        "standard_metrics_daynight_by_elevation.csv": daynight,
    }
    for name, frame in files.items():
        _write_csv(frame, output / name)
    plots = plot_hourly_standard_metrics(hourly, output / "plots")
    input_hashes = {
        "agg_database_sha256": file_sha256(db_path),
        "cohort_csv_sha256": file_sha256(cohort_path),
        "station_hour_bias_csv_sha256": file_sha256(reference_bias_path),
    }
    validation = {
        "error_identities": identity_validation,
        **reference_validation,
    }
    summary = build_run_summary(
        n_main_stations=station_hour["STN"].nunique(),
        n_station_hours=station_hour[["STN", "HOUR"]].drop_duplicates().shape[0],
        band_counts=station_hour.groupby("ELEVATION_BAND")["STN"].nunique().to_dict(),
        outputs=[*files, *(str(path.relative_to(output)) for path in plots)],
        validation=validation,
        input_hashes=input_hashes,
    )
    (output / "standard_metrics_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH)
    parser.add_argument("--reference-bias", type=Path, default=REFERENCE_BIAS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.db, args.cohort, args.reference_bias, args.output_dir)
