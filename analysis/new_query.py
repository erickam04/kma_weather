"""Stage 3 — 탐색 인터페이스 (계획서 §6).

Stage 1 SQLite(loo_raw.db)와 Stage 2 집계만 조회해 원하는 지점·기간·집계단위로
표/그래프를 즉석에서 뽑는다. **IDW 재계산 없음** — 저장소 조회만.

핵심 API:
  query(stn|group_by, start, end, agg, view, method) -> DataFrame
  plot_query(...) -> PNG 저장
  top_stations(n, by)         -> 난이도 랭킹 (station_factors)
  station_report(stn)         -> 한 지점 종합 요약

agg  ∈ {timeseries, daily, monthly, diurnal, seasonal, overall}
view ∈ {error(MAE/BIAS/RMSE), compare(none vs fixed), gamma(γ_eff)}

CLI 데모: python -m analysis.new_query
"""

from project_paths import LOO_DB

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.new_aggregate import elev_band, season_of  # 재사용 (모듈 import 안전)

DB = str(LOO_DB)
FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
OUTDIR = os.path.join(FULLNET, "queries")
GAMMA = -6.5
DZ_MIN_KM = 0.10
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def _con():
    return sqlite3.connect(DB)


def _fetch(stn, start, end, methods, con):
    """지점(들)·기간·method 의 raw 행(interpolable=1) 조회 (행 단위 — timeseries/gamma용)."""
    stns = [str(stn)] if isinstance(stn, (str, int)) else [str(s) for s in stn]
    q = ("SELECT TM,STN,OBSERVED,PREDICTED,ERROR,ABS_ERROR,METHOD "
         "FROM raw WHERE interpolable=1 AND STN IN (%s) AND METHOD IN (%s)"
         % (",".join("?" * len(stns)), ",".join("?" * len(methods))))
    params = stns + list(methods)
    if start:
        q += " AND TM>=?"; params.append(str(start))
    if end:
        q += " AND TM<?"; params.append(str(end))
    df = pd.read_sql(q, con, params=params)
    df["TM"] = pd.to_datetime(df["TM"])
    df["STN"] = df["STN"].astype(str)
    return df


def _bucket_sql(agg):
    """agg → SQL 버킷 표현식 (행을 끌어오지 않고 DB에서 집계)."""
    return {
        "monthly": "MONTH",
        "daily": "substr(TM,1,10)",
        "diurnal": "CAST(substr(TM,12,2) AS INTEGER)",
        "seasonal": ("CASE substr(MONTH,5,2) WHEN '12' THEN 'winter' "
                     "WHEN '01' THEN 'winter' WHEN '02' THEN 'winter' "
                     "WHEN '03' THEN 'spring' WHEN '04' THEN 'spring' WHEN '05' THEN 'spring' "
                     "WHEN '06' THEN 'summer' WHEN '07' THEN 'summer' WHEN '08' THEN 'summer' "
                     "ELSE 'fall' END"),
        "overall": "'overall'",
    }.get(agg)


def _ensure_agg_tables(con):
    """소형 materialized 집계표(STN×월×method, STN×시각×method)를 1회 생성.

    9M raw를 매번 GROUP BY하면 그룹쿼리가 ~40s. 이 13k/26k행 표로 롤업하면 즉시.
    build_agg_tables()로 명시 재생성도 가능(raw 갱신 시).
    """
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "agg_stn_month" in have and "agg_stn_hour" in have:
        return
    con.execute(
        "CREATE TABLE IF NOT EXISTS agg_stn_month AS "
        "SELECT STN, MONTH AS bucket, METHOD, COUNT(*) n, SUM(ERROR) se, "
        "SUM(ERROR*ERROR) se2, SUM(ABS_ERROR) sa FROM raw WHERE interpolable=1 "
        "GROUP BY STN, MONTH, METHOD")
    con.execute(
        "CREATE TABLE IF NOT EXISTS agg_stn_hour AS "
        "SELECT STN, CAST(substr(TM,12,2) AS INTEGER) AS bucket, METHOD, COUNT(*) n, "
        "SUM(ERROR) se, SUM(ERROR*ERROR) se2, SUM(ABS_ERROR) sa FROM raw "
        "WHERE interpolable=1 GROUP BY STN, bucket, METHOD")
    con.commit()


def build_agg_tables():
    """materialized 집계표 강제 재생성 (raw 갱신 후 호출)."""
    con = _con()
    for t in ("agg_stn_month", "agg_stn_hour"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.commit()
    _ensure_agg_tables(con)
    con.close()
    print("agg_stn_month / agg_stn_hour 재생성 완료")


def _agg_sums(stns, start, end, methods, agg, con):
    """(STN, bucket, METHOD)별 합계. 전기간(날짜필터 없음)은 materialized 표에서 즉시,
    날짜 범위 지정 시에만 raw 스캔."""
    stns = [str(s) for s in stns]
    # 전기간 + 표준 agg → materialized (즉시)
    if start is None and end is None and agg in ("monthly", "overall", "seasonal", "diurnal"):
        _ensure_agg_tables(con)
        src = "agg_stn_hour" if agg == "diurnal" else "agg_stn_month"
        df = pd.read_sql(f"SELECT * FROM {src} WHERE STN IN "
                         f"({','.join('?'*len(stns))}) AND METHOD IN "
                         f"({','.join('?'*len(methods))})",
                         con, params=stns + list(methods))
        df["STN"] = df["STN"].astype(str)
        if agg == "overall":
            df["bucket"] = "overall"
        elif agg == "seasonal":
            df["bucket"] = df["bucket"].astype(str).str[4:6].astype(int).map(season_of)
        # monthly/diurnal 은 bucket 그대로
        return df

    # 날짜 범위 지정 → raw 스캔 (덜 흔한 경로)
    bexpr = _bucket_sql(agg)
    if bexpr is None:
        raise ValueError(f"SQL 집계 미지원 agg: {agg}")
    q = (f"SELECT STN, {bexpr} AS bucket, METHOD, COUNT(*) n, SUM(ERROR) se, "
         f"SUM(ERROR*ERROR) se2, SUM(ABS_ERROR) sa FROM raw "
         f"WHERE interpolable=1 AND STN IN ({','.join('?'*len(stns))}) "
         f"AND METHOD IN ({','.join('?'*len(methods))})")
    params = stns + list(methods)
    if start:
        q += " AND TM>=?"; params.append(str(start))
    if end:
        q += " AND TM<?"; params.append(str(end))
    q += " GROUP BY STN, bucket, METHOD"
    df = pd.read_sql(q, con, params=params)
    df["STN"] = df["STN"].astype(str)
    return df


def _finalize(sums, group_keys):
    """합계 → MAE/RMSE/BIAS (임의 상위 그룹으로 재합산 가능)."""
    g = sums.groupby(group_keys, as_index=False).agg(
        n=("n", "sum"), se=("se", "sum"), se2=("se2", "sum"), sa=("sa", "sum"))
    g["MAE"] = g["sa"] / g["n"]
    g["BIAS"] = g["se"] / g["n"]
    g["RMSE"] = np.sqrt(g["se2"] / g["n"])
    return g.drop(columns=["se", "se2", "sa"])


def _bucket(df, agg):
    """TM 에 집계 버킷 컬럼 부여."""
    if agg == "timeseries":
        df["bucket"] = df["TM"]
    elif agg == "daily":
        df["bucket"] = df["TM"].dt.date
    elif agg == "monthly":
        df["bucket"] = df["TM"].dt.strftime("%Y-%m")
    elif agg == "diurnal":
        df["bucket"] = df["TM"].dt.hour
    elif agg == "seasonal":
        df["bucket"] = df["TM"].dt.month.map(season_of)
    elif agg == "overall":
        df["bucket"] = "overall"
    else:
        raise ValueError(f"알 수 없는 agg: {agg}")
    return df


def _scores(g):
    return pd.Series({
        "n": len(g),
        "MAE": g["ABS_ERROR"].mean(),
        "BIAS": g["ERROR"].mean(),
        "RMSE": np.sqrt((g["ERROR"] ** 2).mean()),
    })


def query(stn=None, group_by=None, start=None, end=None,
          agg="monthly", view="error", method="fixed_lapse"):
    """탐색 조회. stn(단일/리스트) 또는 group_by='elev_band' 중 하나 지정.

    view='error'  : method 의 MAE/BIAS/RMSE (agg 버킷별)
    view='compare': none vs fixed MAE 나란히 (고도보정 효과)
    view='gamma'  : γ_eff (none·fixed 필요, 게이트 |ΔZ|≥0.10km)
    """
    con = _con()
    try:
        methods = ["none", "fixed_lapse"] if view in ("compare", "gamma") else [method]

        # gamma / timeseries 는 행 단위 필요 (단일·소수 지점 전제)
        if view == "gamma" or agg == "timeseries":
            if stn is None:
                raise ValueError("gamma/timeseries view는 stn 지정 필요")
            df = _bucket(_fetch(stn, start, end, methods, con), agg)
            if len(df) == 0:
                return pd.DataFrame()
            if view == "gamma":
                return _gamma_view(df, agg)
            keys = ["bucket", "METHOD"] if len(methods) > 1 else ["bucket"]
            out = df.groupby(keys).apply(_scores, include_groups=False).reset_index()
            return _shape_view(out, view)

        # error/compare + 집계 agg → DB SQL 합계 (빠름)
        if group_by == "elev_band":
            from weather.find_stations import load_station_meta, filter_valid_station_meta
            from weather.idw import ELEVATION_COLUMN
            meta = load_station_meta()
            vref = filter_valid_station_meta(meta, "2024-06-15")
            elev = {str(r["지점"]): r[ELEVATION_COLUMN] for _, r in vref.iterrows()}
            stns = pd.read_sql("SELECT DISTINCT STN FROM raw", con)["STN"].astype(str).tolist()
            sums = _agg_sums(stns, start, end, methods, agg, con)
            sums["band"] = sums["STN"].map(lambda s: elev_band(elev.get(s)))
            keys = ["band", "bucket", "METHOD"] if len(methods) > 1 else ["band", "bucket"]
            out = _finalize(sums, keys)
            return _shape_view(out, view)

        if stn is None:
            raise ValueError("stn 또는 group_by 중 하나는 지정해야 함")
        stns = [stn] if isinstance(stn, (str, int)) else list(stn)
        sums = _agg_sums(stns, start, end, methods, agg, con)
        if len(sums) == 0:
            return pd.DataFrame()
        keys = ["bucket", "METHOD"] if len(methods) > 1 else ["bucket"]
        out = _finalize(sums, keys)
        return _shape_view(out, view)
    finally:
        con.close()


def _shape_view(out, view, group_col=None):
    if view == "compare":
        idx = [c for c in ["band", "bucket"] if c in out.columns]
        piv = out.pivot_table(index=idx, columns="METHOD", values="MAE").reset_index()
        return piv
    return out


def _gamma_view(df, agg):
    """γ_eff = −ERR_none/ΔZ, ΔZ=(ERR_fixed−ERR_none)/(−6.5), 게이트 후 버킷별 중앙값."""
    n = df[df.METHOD == "none"][["TM", "STN", "ERROR", "bucket"]].rename(columns={"ERROR": "e_none"})
    f = df[df.METHOD == "fixed_lapse"][["TM", "STN", "ERROR"]].rename(columns={"ERROR": "e_fix"})
    m = n.merge(f, on=["TM", "STN"])
    dz = (m["e_fix"] - m["e_none"]) / GAMMA
    gated = dz.abs() >= DZ_MIN_KM
    m = m[gated].copy()
    m["gamma_eff"] = -m["e_none"] / ((m["e_fix"] - m["e_none"]) / GAMMA)
    return m.groupby("bucket").agg(
        n=("gamma_eff", "size"),
        gamma_eff_median=("gamma_eff", "median"),
        gamma_eff_q25=("gamma_eff", lambda x: x.quantile(0.25)),
        gamma_eff_q75=("gamma_eff", lambda x: x.quantile(0.75)),
    ).reset_index()


def top_stations(n=15, by="MAE", ascending=False):
    """난이도 랭킹 (station_factors.csv 조회)."""
    sf = pd.read_csv(os.path.join(FULLNET, "station_factors.csv"),
                     encoding="utf-8-sig", dtype={"STN": str})
    cols = ["STN", "elev_m", "MAE", "MAE_none", "RI", "mean_absdz_km",
            "gamma_eff_median", "night_inversion_ratio", "n_hours"]
    return sf.sort_values(by, ascending=ascending)[cols].head(n).reset_index(drop=True)


def station_report(stn):
    """한 지점 종합: station_factors 1행 + 월별·시간대별 MAE."""
    sf = pd.read_csv(os.path.join(FULLNET, "station_factors.csv"),
                     encoding="utf-8-sig", dtype={"STN": str})
    row = sf[sf["STN"] == str(stn)]
    return {
        "factors": row.to_dict("records")[0] if len(row) else None,
        "monthly": query(stn=stn, agg="monthly", view="error"),
        "diurnal": query(stn=stn, agg="diurnal", view="error"),
    }


def plot_query(stn=None, group_by=None, start=None, end=None,
               agg="monthly", view="error", method="fixed_lapse",
               title=None, save=None):
    """query 결과를 그림으로. save 미지정 시 OUTDIR 에 자동 명명 저장."""
    os.makedirs(OUTDIR, exist_ok=True)
    d = query(stn=stn, group_by=group_by, start=start, end=end,
              agg=agg, view=view, method=method)
    if d is None or len(d) == 0:
        print("데이터 없음"); return None
    fig, ax = plt.subplots(figsize=(10, 5))
    x = d["bucket"] if "bucket" in d.columns else d.iloc[:, 0]

    if view == "compare":
        ax.plot(x, d["none"], "o-", label="보정 없음(none)")
        ax.plot(x, d["fixed_lapse"], "s-", label="고도보정(fixed)")
        ax.set_ylabel("MAE (°C)"); ax.legend()
    elif view == "gamma":
        ax.plot(x, d["gamma_eff_median"], "o-", color="tab:red")
        ax.fill_between(x, d["gamma_eff_q25"], d["gamma_eff_q75"], alpha=0.2, color="tab:red")
        ax.axhline(-6.5, ls="--", color="gray", label="고정 −6.5")
        ax.set_ylabel("γ_eff (°C/km)"); ax.legend()
    else:
        ax.plot(x, d["MAE"], "o-", label="MAE")
        ax.plot(x, d["BIAS"], "s--", label="BIAS", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.5); ax.set_ylabel("°C"); ax.legend()

    who = f"STN {stn}" if stn else f"group={group_by}"
    ax.set_title(title or f"{who} | {agg} | {view}")
    ax.set_xlabel(agg)
    if agg in ("monthly", "seasonal") or (agg == "daily"):
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = save or os.path.join(OUTDIR, f"query_{stn or group_by}_{agg}_{view}.png")
    plt.savefig(path, dpi=120); plt.close()
    print("저장:", path)
    return path


if __name__ == "__main__":
    # 데모: 대과제 결론을 인터페이스로 재현
    pd.set_option("display.width", 120)
    print("=== 난이도 최악 10지점 ===")
    print(top_stations(10).to_string(index=False))
    print("\n=== 160 부산레: 월별 MAE/BIAS ===")
    print(query(stn="160", agg="monthly", view="error").to_string(index=False))
    print("\n=== 고도구간별 월별 MAE (none vs fixed) — overall ===")
    print(query(group_by="elev_band", agg="overall", view="compare").to_string(index=False))
    # 그림 3종
    plot_query(stn="160", agg="monthly", view="compare", title="160 부산레 — 고도보정 효과(월별)")
    plot_query(stn="160", agg="diurnal", view="gamma", title="160 부산레 — 시간대별 실제감률 γ_eff")
    plot_query(group_by="elev_band", agg="monthly", view="compare",
               title="고도구간별 월별 MAE — none vs fixed")
