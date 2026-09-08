"""BIAS 변화의 수학적 분해와 보정 mechanism 판정 순수 함수."""

import numpy as np
import pandas as pd


BIAS_TOL = 1e-10


def classify_bias_mechanism(bias_none, correction_shift, tol=BIAS_TOL):
    """기준 BIAS와 보정 이동량으로 BIAS 변화 mechanism을 판정한다."""
    b0 = float(bias_none)
    c = float(correction_shift)
    if not np.isfinite(b0) or not np.isfinite(c):
        raise ValueError("bias_none and correction_shift must be finite")
    if abs(c) <= tol:
        return "no_correction"
    if abs(b0) <= tol:
        return "zero_baseline_worsened"

    boundary = c * (2.0 * b0 + c)
    if abs(boundary) <= tol:
        return "equal_magnitude"
    if boundary < 0.0:
        if b0 * (b0 + c) < 0.0:
            return "improved_with_flip"
        return "improved_without_flip"
    if b0 * c > 0.0:
        return "same_direction_worsened"
    return "overshoot_worsened"


def compute_bias_metrics(error_none, error_fixed, tol=BIAS_TOL):
    """짝지어진 시간별 오차 배열에서 station BIAS 지표를 계산한다."""
    en = np.asarray(error_none, dtype=float)
    ef = np.asarray(error_fixed, dtype=float)
    if en.ndim != 1 or ef.ndim != 1 or len(en) != len(ef) or len(en) == 0:
        raise ValueError(
            "paired error arrays must be non-empty 1-D arrays of equal length"
        )
    if not np.isfinite(en).all() or not np.isfinite(ef).all():
        raise ValueError("paired error arrays must contain only finite values")

    b0 = float(en.mean())
    b1 = float(ef.mean())
    c = b1 - b0
    y = abs(b1) - abs(b0)
    return {
        "n_hours": int(len(en)),
        "BIAS_none": b0,
        "BIAS_fixed": b1,
        "CORRECTION_SHIFT": c,
        "DELTA_ABS_BIAS": y,
        "MECHANISM": classify_bias_mechanism(b0, c, tol=tol),
    }


def aggregate_bias_groups(pairs, group_cols):
    """짝지어진 시간별 오차를 기간·관측소별 편향 지표로 집계한다."""
    required = set(group_cols) | {"error_none", "error_fixed", "dz_geom_km"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing aggregation columns: {sorted(missing)}")
    dz_geom_km = pd.to_numeric(pairs["dz_geom_km"], errors="coerce")
    if not np.isfinite(dz_geom_km.to_numpy(dtype=float)).all():
        raise ValueError("dz_geom_km must be finite")
    geometry_fields = [
        ("neighbor_count_geom", "mean_neighbor_count"),
        ("nearest_distance_km", "mean_nearest_distance_km"),
        ("weighted_distance_km", "mean_weighted_distance_km"),
        ("centroid_offset_km", "mean_centroid_offset_km"),
        ("neighbor_elev_sd_m", "mean_neighbor_elev_sd_m"),
    ]
    normalized_pairs = pairs.copy()
    normalized_pairs["dz_geom_km"] = dz_geom_km
    for source, _ in geometry_fields:
        if source not in normalized_pairs:
            continue
        values = pd.to_numeric(normalized_pairs[source], errors="coerce")
        invalid = normalized_pairs[source].notna() & ~np.isfinite(
            values.to_numpy(dtype=float)
        )
        if invalid.any():
            raise ValueError(f"{source} must be finite when non-null")
        normalized_pairs[source] = values
    rows = []
    if group_cols:
        grouper = group_cols[0] if len(group_cols) == 1 else group_cols
        groups = normalized_pairs.groupby(grouper, sort=True, dropna=False)
    else:
        groups = [((), normalized_pairs)]
    for key, group in groups:
        keys = (key,) if len(group_cols) == 1 else tuple(key)
        record = dict(zip(group_cols, keys))
        record.update(compute_bias_metrics(group["error_none"], group["error_fixed"]))
        record.update({
            "mean_dz_geom_km": float(group["dz_geom_km"].mean()),
            "mean_absdz_geom_km": float(group["dz_geom_km"].abs().mean()),
            "sd_dz_geom_km": float(group["dz_geom_km"].std(ddof=0)),
            "n_valid_dz_geom_km": int(group["dz_geom_km"].notna().sum()),
        })
        for source, target in geometry_fields:
            if source in group:
                record[target] = float(group[source].mean())
                record[f"n_valid_{source}"] = int(group[source].notna().sum())
            else:
                record[target] = np.nan
                record[f"n_valid_{source}"] = 0
        rows.append(record)
    return pd.DataFrame(rows)
