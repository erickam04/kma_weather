"""Metadata 기반 보간 geometry 지표를 독립적으로 재구성한다."""

from dataclasses import asdict, dataclass
from math import cos, radians, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from find_stations import filter_valid_station_meta, get_distance_km
from idw import POWER


ELEV_COL = "노장해발고도(m)"


@dataclass(frozen=True)
class GeometryMetrics:
    dz_geom_km: float
    neighbor_count_geom: int
    nearest_distance_km: float
    weighted_distance_km: float
    centroid_offset_km: float
    neighbor_elev_sd_m: float


def _elevation_delta_m(target_elevation, neighbor_elevation):
    if pd.isna(target_elevation) or pd.isna(neighbor_elevation):
        return 0.0
    return float(target_elevation) - float(neighbor_elevation)


def compute_geometry_metrics(
    target_id: str,
    neighbor_ids: Sequence[str],
    valid_meta: pd.DataFrame,
    power: int = POWER,
) -> GeometryMetrics:
    """한 시각에 유효한 metadata로 target-neighbor geometry를 계산한다."""
    lut = valid_meta.set_index("지점", drop=False)
    target_id = str(target_id)
    if target_id not in lut.index:
        raise ValueError(f"target missing from valid metadata: {target_id}")
    target = lut.loc[target_id]
    if isinstance(target, pd.DataFrame):
        raise ValueError(f"duplicate target metadata after validity filtering: {target_id}")

    entries = []
    for order, sid in enumerate(neighbor_ids):
        sid = str(sid)
        if sid not in lut.index:
            raise ValueError(f"neighbor missing from valid metadata: {sid}")
        row = lut.loc[sid]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"duplicate neighbor metadata after validity filtering: {sid}")
        distance = get_distance_km(
            target["위도"], target["경도"], row["위도"], row["경도"]
        )
        entries.append((distance, order, sid, row))
    if not entries:
        raise ValueError("neighbor_ids must not be empty")
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    if entries[0][0] == 0.0:
        row = entries[0][3]
        return GeometryMetrics(
            dz_geom_km=_elevation_delta_m(target[ELEV_COL], row[ELEV_COL]) / 1000.0,
            neighbor_count_geom=len(entries),
            nearest_distance_km=0.0,
            weighted_distance_km=0.0,
            centroid_offset_km=0.0,
            neighbor_elev_sd_m=0.0,
        )

    distances = np.array([entry[0] for entry in entries], dtype=float)
    weights = distances ** (-power)
    weights /= weights.sum()
    per_neighbor_delta_m = np.array([
        _elevation_delta_m(target[ELEV_COL], entry[3][ELEV_COL])
        for entry in entries
    ])
    dz_geom_km = float(np.dot(weights, per_neighbor_delta_m)) / 1000.0

    lat0 = float(target["위도"])
    lon0 = float(target["경도"])
    dx = np.array([
        (float(entry[3]["경도"]) - lon0) * 111.320 * cos(radians(lat0))
        for entry in entries
    ])
    dy = np.array([
        (float(entry[3]["위도"]) - lat0) * 110.574 for entry in entries
    ])
    centroid_offset_km = sqrt(
        float(np.dot(weights, dx)) ** 2 + float(np.dot(weights, dy)) ** 2
    )

    finite = np.array([not pd.isna(entry[3][ELEV_COL]) for entry in entries])
    if finite.sum() < 2:
        neighbor_elev_sd_m = np.nan
    else:
        finite_weights = weights[finite] / weights[finite].sum()
        finite_elevations = np.array([
            float(entries[index][3][ELEV_COL])
            for index in range(len(entries))
            if finite[index]
        ])
        mean_elevation = float(np.dot(finite_weights, finite_elevations))
        neighbor_elev_sd_m = sqrt(
            float(np.dot(finite_weights, (finite_elevations - mean_elevation) ** 2))
        )

    return GeometryMetrics(
        dz_geom_km=dz_geom_km,
        neighbor_count_geom=len(entries),
        nearest_distance_km=float(distances.min()),
        weighted_distance_km=float(np.dot(weights, distances)),
        centroid_offset_km=centroid_offset_km,
        neighbor_elev_sd_m=neighbor_elev_sd_m,
    )


def reconstruct_geometry_for_pairs(
    pairs: pd.DataFrame,
    neighbor_map: dict[str, Sequence[str]],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """pair별 metadata epoch와 neighbor signature를 사용해 geometry를 재구성한다."""
    required = {"TM", "STN", "NEIGHBOR_SET_ID", "COORD_EPOCH"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing columns: {sorted(missing)}")

    out = pairs.copy()
    out["TM"] = pd.to_datetime(out["TM"])
    out["DATE"] = out["TM"].dt.normalize()
    key_cols = ["DATE", "STN", "NEIGHBOR_SET_ID", "COORD_EPOCH"]
    keys = out[key_cols].drop_duplicates().reset_index(drop=True)
    records = []
    valid_cache = {}

    for row in keys.itertuples(index=False):
        date = pd.Timestamp(row.DATE)
        if date not in valid_cache:
            valid_cache[date] = filter_valid_station_meta(meta, date)
        valid = valid_cache[date]
        target_row = valid[valid["지점"] == str(row.STN)]
        if len(target_row) != 1:
            raise ValueError(f"target metadata count != 1: {row.STN} {date.date()}")
        epoch = pd.Timestamp(target_row.iloc[0]["시작일"]).normalize()
        coord_epoch = pd.Timestamp(row.COORD_EPOCH).normalize()
        if coord_epoch != epoch:
            raise ValueError(f"COORD_EPOCH mismatch: {row.STN} {date.date()}")
        signature = str(row.NEIGHBOR_SET_ID)
        if signature not in neighbor_map:
            raise ValueError(f"unknown neighbor signature: {signature}")
        metrics = compute_geometry_metrics(str(row.STN), neighbor_map[signature], valid)
        records.append({**dict(zip(key_cols, row)), **asdict(metrics)})

    geometry = pd.DataFrame(records)
    merged = out.merge(geometry, on=key_cols, how="left", validate="many_to_one")
    metric_cols = list(GeometryMetrics.__dataclass_fields__)
    if merged[metric_cols].isna().all(axis=1).any():
        raise ValueError("geometry merge produced an entirely missing metric row")
    return merged.drop(columns=["DATE"])
