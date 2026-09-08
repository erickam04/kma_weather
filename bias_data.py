"""SQLite LOO 결과에서 편향 분석용 paired row를 읽는다."""

from project_paths import LOO_DB

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from bias_analysis_core import aggregate_bias_groups


DB_PATH = str(LOO_DB)


def open_readonly_db(path=DB_PATH):
    """SQLite DB를 읽기 전용 URI로 연다."""
    uri = f"{Path(path).expanduser().resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_analysis_station_ids(con):
    """두 보정 방법 모두 보간 가능한 분석 대상 관측소를 반환한다."""
    sql = """
    SELECT DISTINCT STN
    FROM raw
    WHERE interpolable=1 AND METHOD IN ('none', 'fixed_lapse')
    ORDER BY CAST(STN AS INTEGER), STN
    """
    station_ids = pd.read_sql(sql, con)["STN"].astype(str).tolist()
    for stn in station_ids:
        try:
            load_station_pairs(con, stn)
        except ValueError as exc:
            raise ValueError(f"invalid analysis station {stn}: {exc}") from exc
    return station_ids


def load_neighbor_map(con):
    """이웃 집합 서명별 관측소 ID 목록을 반환한다."""
    df = pd.read_sql(
        "SELECT NEIGHBOR_SET_ID, neighbor_ids FROM neighbor_set_map", con
    )
    return {
        str(row.NEIGHBOR_SET_ID): str(row.neighbor_ids).split("-")
        for row in df.itertuples(index=False)
    }


def load_station_pairs(con, stn):
    candidate_sql = """
    SELECT TM, STN, OBSERVED, ERROR, NEIGHBOR_SET_ID, COORD_EPOCH, METHOD
    FROM raw
    WHERE STN=? AND interpolable=1 AND METHOD IN ('none', 'fixed_lapse')
    """
    candidates = pd.read_sql(candidate_sql, con, params=[str(stn)])
    if candidates.empty:
        raise ValueError(f"no paired rows for station {stn}")
    try:
        candidates["TM_KEY"] = pd.to_datetime(
            candidates["TM"], errors="raise", format="mixed"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid analysis timestamp: {stn}") from exc
    if candidates["TM_KEY"].isna().any():
        raise ValueError(f"invalid analysis timestamp: {stn}")

    counts = candidates.groupby(["TM_KEY", "METHOD"], dropna=False).size()
    if (counts != 1).any():
        raise ValueError(f"duplicate analysis rows for station {stn}")
    none_tms = set(candidates.loc[candidates["METHOD"] == "none", "TM_KEY"])
    fixed_tms = set(candidates.loc[candidates["METHOD"] == "fixed_lapse", "TM_KEY"])
    if not none_tms or not fixed_tms or none_tms != fixed_tms:
        raise ValueError(f"timestamp mismatch between methods: {stn}")

    """동일 STN/TM의 none·fixed_lapse 쌍만 읽고 정합성을 검증한다."""
    none_rows = candidates.loc[candidates["METHOD"] == "none"].rename(
        columns={
            "STN": "stn_none",
            "OBSERVED": "observed_none",
            "ERROR": "error_none",
            "NEIGHBOR_SET_ID": "sig_none",
            "COORD_EPOCH": "epoch_none",
        }
    )
    fixed_rows = candidates.loc[candidates["METHOD"] == "fixed_lapse"].rename(
        columns={
            "STN": "stn_fixed",
            "OBSERVED": "observed_fixed",
            "ERROR": "error_fixed",
            "NEIGHBOR_SET_ID": "sig_fixed",
            "COORD_EPOCH": "epoch_fixed",
        }
    )
    df = none_rows.merge(
        fixed_rows,
        on="TM_KEY",
        how="inner",
        validate="one_to_one",
        suffixes=("_none", "_fixed"),
    ).sort_values("TM_KEY").reset_index(drop=True)
    expected_count = len(none_tms)
    if len(df) != expected_count or df["TM_KEY"].nunique(dropna=False) != expected_count:
        raise ValueError(f"canonical merge did not produce exact 1:1 pairs: {stn}")
    if df[["sig_none", "sig_fixed"]].isna().any().any():
        raise ValueError(f"neighbor signature missing between methods: {stn}")
    if df[["epoch_none", "epoch_fixed"]].isna().any().any():
        raise ValueError(f"coordinate epoch missing between methods: {stn}")
    if not np.array_equal(
        df["observed_none"].to_numpy(), df["observed_fixed"].to_numpy()
    ):
        raise ValueError(f"OBSERVED mismatch between methods: {stn}")
    if not df["sig_none"].equals(df["sig_fixed"]):
        raise ValueError(f"neighbor signature mismatch between methods: {stn}")
    if not df["epoch_none"].equals(df["epoch_fixed"]):
        raise ValueError(f"coordinate epoch mismatch between methods: {stn}")

    out = pd.DataFrame({
        "TM": df["TM_KEY"],
        "STN": df["stn_none"].astype(str),
        "OBSERVED": df["observed_none"].astype(float),
        "error_none": df["error_none"].astype(float),
        "error_fixed": df["error_fixed"].astype(float),
        "NEIGHBOR_SET_ID": df["sig_none"].astype(str),
        "COORD_EPOCH": df["epoch_none"].astype(str),
    })
    out["MONTH"] = out["TM"].dt.strftime("%Y%m")
    out["HOUR"] = out["TM"].dt.hour
    return out
