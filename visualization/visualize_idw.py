import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from weather.find_stations import (
    DEFAULT_MAX_DISTANCE_KM,
    DEFAULT_MAX_NEIGHBORS,
    filter_valid_station_meta,
    get_distance_km,
    load_station_meta,
)
from weather.idw import ELEVATION_COLUMN, MIN_NEIGHBORS, POWER, TARGET_COLUMN, idw_interpolate_with_trace
from weather.load_aws import prepare_evaluation_data
import pipeline_config as cfg


DEFAULT_OUTPUT_DIR = Path("WeatherData_AWS/interpolation_visualization")
ELEV_CMAP = "terrain"  # neighbor 선정 지도의 고도 heatmap 색상


def build_output_paths(output_dir, station_id, target_column, selected_time):
    """산출물 5종의 경로. 각 종류는 output_dir 아래 카테고리 하위 폴더로 분리
    (cfg.VIZ_SUBDIRS). output_dir는 OUT_VIZ 또는 --out으로 받은 임의 경로."""
    output_dir = Path(output_dir)
    time_key = pd.Timestamp(selected_time).strftime("%Y%m%d_%H%M")
    day_key = pd.Timestamp(selected_time).strftime("%Y%m%d")
    prefix = f"station_{station_id}_{target_column}"

    def sub(category):
        return output_dir / cfg.VIZ_SUBDIRS[category]

    return {
        "map": sub("neighbor_map") / f"{prefix}_{time_key}_map.png",
        "weights": sub("neighbor_weights") / f"{prefix}_{time_key}_weights.png",
        "timeseries": sub("neighbor_timeseries") / f"{prefix}_{day_key}_timeseries.png",
        "trace": sub("neighbor_trace") / f"{prefix}_{time_key}_trace.csv",
        "predictions": sub("neighbor_predictions") / f"{prefix}_{day_key}_predictions.csv",
    }


def build_snapshot(
    data,
    meta,
    target_station_id,
    selected_time,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
):
    selected_time = pd.Timestamp(selected_time)
    target_station_id = str(target_station_id)

    if selected_time not in data.index:
        raise ValueError(f"selected_time is not in data index: {selected_time}")

    values_at_time = data.loc[selected_time]
    observed = values_at_time[target_station_id]
    valid_meta = filter_valid_station_meta(meta, selected_time)
    target_rows = valid_meta[valid_meta["지점"] == target_station_id]

    if len(target_rows) == 0:
        raise ValueError(f"target station has no valid metadata: {target_station_id}")

    predicted, trace = idw_interpolate_with_trace(
        target_station_id=target_station_id,
        values_at_time=values_at_time,
        meta=valid_meta,
        power=power,
        max_distance_km=max_distance_km,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
    )

    error = None if predicted is None or pd.isna(observed) else predicted - observed

    return {
        "time": selected_time,
        "target_station_id": target_station_id,
        "target": target_rows.iloc[0].to_dict(),
        "observed": observed,
        "predicted": predicted,
        "error": error,
        "trace": trace,
        "valid_meta": valid_meta,
        "values_at_time": values_at_time,
    }


def build_candidate_stations(snapshot, max_distance_km=DEFAULT_MAX_DISTANCE_KM):
    target = snapshot["target"]
    valid_meta = snapshot["valid_meta"]
    values_at_time = snapshot["values_at_time"]
    target_station_id = snapshot["target_station_id"]
    selected_ids = set(snapshot["trace"]["station_id"].astype(str))
    rows = []

    for _, row in valid_meta.iterrows():
        station_id = str(row["지점"])

        if station_id == target_station_id:
            continue

        if station_id not in values_at_time.index:
            continue

        value = values_at_time[station_id]

        if pd.isna(value):
            continue

        distance = get_distance_km(
            target["위도"],
            target["경도"],
            row["위도"],
            row["경도"],
        )

        if max_distance_km is not None and distance > max_distance_km:
            continue

        rows.append(
            {
                "station_id": station_id,
                "station_name": row.get("지점명", ""),
                "latitude": row["위도"],
                "longitude": row["경도"],
                "value": value,
                "elevation": row.get(ELEVATION_COLUMN),
                "distance_km": distance,
                "selected": station_id in selected_ids,
            }
        )

    candidates = pd.DataFrame(rows)

    if len(candidates) == 0:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "latitude",
                "longitude",
                "value",
                "elevation",
                "distance_km",
                "selected",
            ]
        )

    return candidates.sort_values("distance_km").reset_index(drop=True)


def build_timeseries(
    data,
    meta,
    target_station_id,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
):
    rows = []
    meta_cache = {}
    target_station_id = str(target_station_id)

    for tm, values_at_time in data.iterrows():
        observed = values_at_time[target_station_id]

        if pd.isna(observed):
            continue

        date_key = pd.Timestamp(tm).normalize()

        if date_key not in meta_cache:
            meta_cache[date_key] = filter_valid_station_meta(meta, date_key)

        predicted, trace = idw_interpolate_with_trace(
            target_station_id=target_station_id,
            values_at_time=values_at_time,
            meta=meta_cache[date_key],
            power=power,
            max_distance_km=max_distance_km,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
        )

        if predicted is None:
            continue

        error = predicted - observed
        rows.append(
            {
                "TM": tm,
                "STN": target_station_id,
                "OBSERVED": observed,
                "PREDICTED": predicted,
                "ERROR": error,
                "ABS_ERROR": abs(error),
                "NEIGHBOR_COUNT": len(trace),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "TM",
            "STN",
            "OBSERVED",
            "PREDICTED",
            "ERROR",
            "ABS_ERROR",
            "NEIGHBOR_COUNT",
        ],
    )


def plot_station_map(snapshot, candidates, output_path, max_distance_km):
    """neighbor 선정 지도. 관측소 마커를 해발고도(elevation) 색으로 칠한다(heatmap).

    색=고도(공유 colorbar), 역할(target/selected/candidate) 구분은 마커 크기·테두리로.
    고도 결측 station은 회색 사각 마커로 별도 표시. 거리 weight 선은 유지.
    실제 DEM이 아니라 관측소 지점 고도만 사용 (합성 배경면 없음).
    """
    target = snapshot["target"]
    trace = snapshot["trace"]
    selected = candidates[candidates["selected"]]
    unselected = candidates[~candidates["selected"]]
    target_elev = target.get(ELEVATION_COLUMN)

    # 공유 고도 색 스케일 (candidate + selected + target)
    elevs = list(candidates["elevation"].dropna()) if "elevation" in candidates else []
    if pd.notna(target_elev):
        elevs.append(float(target_elev))
    if elevs:
        vmin, vmax = min(elevs), max(elevs)
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.6, 7))

    def scatter_group(df, size, edge, lw):
        """고도 있는 것은 colormap, 없는 것은 회색 사각으로. 마지막 scatter 핸들 반환."""
        handle = None
        if len(df) == 0:
            return None
        has = df[df["elevation"].notna()]
        nan = df[df["elevation"].isna()]
        if len(has):
            handle = ax.scatter(has["longitude"], has["latitude"], s=size,
                                c=has["elevation"], cmap=ELEV_CMAP, norm=norm,
                                edgecolors=edge, linewidths=lw, zorder=3)
        if len(nan):
            ax.scatter(nan["longitude"], nan["latitude"], s=size, c="#dddddd",
                       marker="s", edgecolors=edge, linewidths=lw, zorder=3)
        return handle

    # 거리 weight 선 (아래층)
    for _, row in selected.iterrows():
        ax.plot([target["경도"], row["longitude"]], [target["위도"], row["latitude"]],
                c="#5c87d6", linewidth=1, alpha=0.6, zorder=1)

    sc = scatter_group(unselected, 45, "#999999", 0.5)
    sc_sel = scatter_group(selected, 120, "#111111", 1.2)
    sc = sc_sel if sc_sel is not None else sc

    # target 별
    if pd.notna(target_elev):
        sc_t = ax.scatter([target["경도"]], [target["위도"]], s=360,
                          c=[float(target_elev)], cmap=ELEV_CMAP, norm=norm,
                          marker="*", edgecolors="#d62728", linewidths=1.8, zorder=5)
        sc = sc if sc is not None else sc_t
    else:
        ax.scatter([target["경도"]], [target["위도"]], s=360, c="#dddddd",
                   marker="*", edgecolors="#d62728", linewidths=1.8, zorder=5)

    # 선택 이웃 라벨: 지점 / 거리 / 고도
    for _, row in selected.iterrows():
        elev_txt = f" {row['elevation']:.0f}m" if pd.notna(row["elevation"]) else ""
        ax.annotate(
            f"{row['station_id']}\n{row['distance_km']:.1f}km{elev_txt}",
            (row["longitude"], row["latitude"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7.5,
        )

    # 위경도 축 비율 보정: 경도 1도는 위도 1도의 cos(위도)배 (거리 왜곡 방지)
    ax.set_aspect(1.0 / math.cos(math.radians(float(target["위도"]))))

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("Elevation (m)")

    # 색=고도이므로, 역할 구분은 마커 모양 legend로 별도 표기
    legend_handles = [
        Line2D([], [], marker="*", linestyle="none", markerfacecolor="#cfcfcf",
               markeredgecolor="#d62728", markeredgewidth=1.6, markersize=17,
               label=f"Target {snapshot['target_station_id']}"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#cfcfcf",
               markeredgecolor="#111111", markeredgewidth=1.2, markersize=11,
               label="Selected neighbor"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#cfcfcf",
               markeredgecolor="#999999", markersize=8, label="Candidate"),
    ]

    radius_text = "unlimited" if max_distance_km is None else f"{max_distance_km} km"
    title = (
        f"IDW Neighbor Selection: station {snapshot['target_station_id']}\n"
        f"{snapshot['time']} / radius {radius_text} / selected {len(trace)} "
        f"(marker color = elevation)"
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_weights(snapshot, output_path):
    trace = snapshot["trace"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if len(trace) == 0:
        axes[0].text(0.5, 0.5, "No selected neighbors", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No interpolation result", ha="center", va="center")
    else:
        labels = trace["station_id"].astype(str)
        axes[0].bar(labels, trace["weight_normalized"], color="#2d6cdf")
        axes[0].set_ylabel("Normalized weight")
        axes[0].set_ylim(0, max(1.0, trace["weight_normalized"].max() * 1.15))

        axes[1].bar(labels, trace["contribution"], color="#4aa564")
        axes[1].set_ylabel("Contribution")
        axes[1].set_xlabel("Neighbor station")

        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)

    observed = snapshot["observed"]
    predicted = snapshot["predicted"]
    error = snapshot["error"]
    predicted_text = "None" if predicted is None else f"{predicted:.3f}"
    error_text = "None" if error is None else f"{error:.3f}"
    fig.suptitle(
        f"IDW Weights: station {snapshot['target_station_id']} at {snapshot['time']}\n"
        f"observed={observed:.3f}, predicted={predicted_text}, error={error_text}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_timeseries(timeseries, output_path, station_id):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    if len(timeseries) == 0:
        axes[0].text(0.5, 0.5, "No predictions", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No errors", ha="center", va="center")
    else:
        axes[0].plot(timeseries["TM"], timeseries["OBSERVED"], label="Observed", color="#222222")
        axes[0].plot(timeseries["TM"], timeseries["PREDICTED"], label="Predicted", color="#d62728")
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].axhline(0, color="#666666", linewidth=1)
        axes[1].plot(timeseries["TM"], timeseries["ERROR"], label="Error", color="#2d6cdf")
        axes[1].set_ylabel("Predicted - observed")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"Observed vs IDW Predicted: station {station_id}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_visualization_outputs(
    data,
    meta,
    station_id,
    selected_time,
    target_column=TARGET_COLUMN,
    output_dir=DEFAULT_OUTPUT_DIR,
    power=POWER,
    max_distance_km=DEFAULT_MAX_DISTANCE_KM,
    min_neighbors=MIN_NEIGHBORS,
    max_neighbors=DEFAULT_MAX_NEIGHBORS,
):
    output_dir = Path(output_dir)
    paths = build_output_paths(output_dir, station_id, target_column, selected_time)
    for p in paths.values():
        os.makedirs(Path(p).parent, exist_ok=True)
    snapshot = build_snapshot(
        data=data,
        meta=meta,
        target_station_id=station_id,
        selected_time=selected_time,
        power=power,
        max_distance_km=max_distance_km,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
    )
    candidates = build_candidate_stations(snapshot, max_distance_km=max_distance_km)
    timeseries = build_timeseries(
        data=data,
        meta=meta,
        target_station_id=station_id,
        power=power,
        max_distance_km=max_distance_km,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
    )

    plot_station_map(snapshot, candidates, paths["map"], max_distance_km)
    plot_weights(snapshot, paths["weights"])
    plot_timeseries(timeseries, paths["timeseries"], station_id)
    snapshot["trace"].to_csv(paths["trace"], index=False, encoding="utf-8-sig")
    timeseries.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")

    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize IDW neighbor selection and interpolation process."
    )
    parser.add_argument("--station", required=True, help="Target station id, e.g. 116")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2024-02-01")
    parser.add_argument("--end", required=True, help="End date, inclusive, e.g. 2024-02-01")
    parser.add_argument(
        "--time",
        required=True,
        help='Selected timestamp for map and weights, e.g. "2024-02-01 12:00"',
    )
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column, default TA")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory, default {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--power", type=float, default=POWER, help="IDW power")
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=DEFAULT_MAX_DISTANCE_KM,
        help="Maximum neighbor distance in km",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=MIN_NEIGHBORS,
        help="Minimum neighbors required for interpolation",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS,
        help="Maximum selected neighbors",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) + pd.Timedelta(days=1)
    selected_time = pd.to_datetime(args.time)

    meta = load_station_meta()
    data, evaluation_meta = prepare_evaluation_data(
        meta,
        start,
        end,
        target_column=args.target,
    )

    paths = save_visualization_outputs(
        data=data,
        meta=evaluation_meta,
        station_id=args.station,
        selected_time=selected_time,
        target_column=args.target,
        output_dir=args.out,
        power=args.power,
        max_distance_km=args.max_distance_km,
        min_neighbors=args.min_neighbors,
        max_neighbors=args.max_neighbors,
    )

    print("Saved visualization outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
