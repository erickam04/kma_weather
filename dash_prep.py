"""대시보드용 데이터 준비 — SQLite/CSV/지도를 조회해 자체완결 JSON 1개로 뽑는다.

대시보드(아티팩트)는 외부 요청이 막혀 있어 필요한 집계를 전부 미리 계산해 심어 넣는다.
[MAE / BIAS / 최적 lapse γ] × [월별 / 일별 / 시간대별] 조합 + ΔZ–MAE 산점을 지원.

산출 (fullnet/dashboard_data.json):
  provinces  : 시도 경계 폴리곤(단순화)
  stations   : 지점 메타(좌표·고도·지역·지형·전체 MAE/BIAS/ΔZ 등) — 산점·표·색에 사용
  monthly    : 지점×월 합계 {n:[n,se,sa], f:[...]} (none/fixed, 가중집계용) → MAE·BIAS
  diurnal_mae/diurnal_bias: 지점×시(24) fixed MAE·BIAS (int×100)
  optlapse_m : 지점×월 [최적γ, mean|ΔZ|]
  optlapse_h : 지점×시 [최적γ, mean|ΔZ|]
  stations[].gopt : 지점 전체기간 최적γ (지도 heatmap 전환용)
"""

from project_paths import LOO_DB

import json
import os
import sqlite3

import numpy as np
import pandas as pd
from shapely.geometry import shape, Point

from find_stations import filter_valid_station_meta, load_station_meta
from idw import ELEVATION_COLUMN

DB = str(LOO_DB)
FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
PROV_JSON = "지도데이터/korea_boundary_geo.json"
OUT = os.path.join(FULLNET, "dashboard_data.json")
GAMMA = -6.5
# 최적 γ 탐색 격자. ⚠️ 범위를 좁게 잡으면 경계에서 잘려 "평평한 구간"이라는 가짜 신호가 생긴다.
# 실측: 160의 오후 최적은 −10.05, 885(역전)는 **양수**(+1.45)까지 감 → 양수 포함 넓은 범위 필수.
# (역전 시각에는 "고도가 오를수록 기온 상승"이므로 γ>0이 물리적으로 정상)
GRID = np.round(np.arange(-20.0, 10.0 + 1e-9, 0.05), 3)
GRID_LO, GRID_HI = float(GRID[0]), float(GRID[-1])
DZ_GATE = 0.10
SCALE = 100


def load_provinces():
    gj = json.load(open(PROV_JSON, encoding="utf-8"))
    prov_geom, prov_draw = {}, {}
    for ft in gj["features"]:
        name = ft["properties"]["name"]
        g = shape(ft["geometry"])
        prov_geom[name] = g
        s = g.simplify(0.008, preserve_topology=True)
        polys = list(s.geoms) if s.geom_type == "MultiPolygon" else [s]
        polys = [p for p in polys if p.area > 0.002]
        prov_draw[name] = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords]
                           for p in polys]
    return prov_geom, prov_draw


def assign_region(lat, lon, prov_geom):
    p = Point(lon, lat)
    for name, g in prov_geom.items():
        if g.covers(p):
            return name
    return min(prov_geom, key=lambda n: prov_geom[n].distance(p))


def short_region(name):
    return (name.replace("특별자치도", "").replace("특별자치시", "")
            .replace("특별시", "").replace("광역시", "").replace("도", "")) or name


def opt_gamma(en, dz):
    """유효 게이트 통과 시 ERROR(γ)=en+γ·dz 의 MAE 최소 γ. 부족하면 None.

    반환 [γ, mean|ΔZ|]. γ가 격자 경계에 붙으면(=진짜 최적이 범위 밖) 경고 카운트.
    """
    gated = np.abs(dz) >= DZ_GATE
    if gated.sum() < 12:
        return None
    # ★ 최적화는 '정보가 있는' 표본(|ΔZ|≥게이트)으로만 한다.
    #   ΔZ≈0 표본까지 넣으면 γ에 거의 반응하지 않는 항들이 곡선을 뭉개 argmin이 경계로 튄다.
    en_g, dz_g = en[gated], dz[gated]
    errs = en_g[:, None] + GRID[None, :] * dz_g[:, None]
    g = float(GRID[int(np.abs(errs).mean(axis=0).argmin())])
    if g <= GRID_LO + 1e-9 or g >= GRID_HI - 1e-9:
        opt_gamma.saturated += 1   # 그래도 경계면 식별 불가 → 표시하지 않음
        return None
    return [g, round(float(np.abs(dz_g).mean()), 3)]


opt_gamma.saturated = 0


def ic(v):
    """MAE/BIAS → int×100 or None."""
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v) * SCALE)


def main():
    con = sqlite3.connect(DB)
    meta = load_station_meta()
    vref = filter_valid_station_meta(meta, "2024-06-15")
    metarow = {str(r["지점"]): r for _, r in vref.iterrows()}
    for s in meta["지점"].unique():
        metarow.setdefault(str(s), meta[meta["지점"] == str(s)].iloc[-1])

    sf = pd.read_csv(os.path.join(FULLNET, "station_factors.csv"),
                     encoding="utf-8-sig", dtype={"STN": str}).set_index("STN")
    terr = pd.read_csv(os.path.join(FULLNET, "station_terrain.csv"),
                       encoding="utf-8-sig", dtype={"STN": str}).set_index("STN")

    print("시도 폴리곤 로드·단순화 ...")
    prov_geom, prov_draw = load_provinces()
    stn_list = pd.read_sql("SELECT DISTINCT STN FROM raw", con)["STN"].astype(str).tolist()

    # 월별 합계(none/fixed) + 시간대별 합계(fixed) — materialized 표에서
    am = pd.read_sql("SELECT STN,bucket,METHOD,n,se,sa FROM agg_stn_month", con)
    am["STN"] = am["STN"].astype(str); am["mm"] = am["bucket"].astype(str).str[4:6].astype(int)
    ah = pd.read_sql("SELECT STN,bucket AS h,n,se,sa FROM agg_stn_hour WHERE METHOD='fixed_lapse'", con)
    ah["STN"] = ah["STN"].astype(str)

    stations = []
    monthly = {}
    diurnal_mae, diurnal_bias = {}, {}
    optlapse_m, optlapse_h = {}, {}

    print(f"지점 {len(stn_list)} — 집계·최적γ(월·시간) 계산 중 ...")
    for i, stn in enumerate(sorted(stn_list, key=lambda s: int(s) if s.isdigit() else 1e9)):
        r = metarow.get(stn)
        if r is None:
            continue
        lat, lon = float(r["위도"]), float(r["경도"])
        t = terr.loc[stn] if stn in terr.index else None
        f = sf.loc[stn] if stn in sf.index else None
        gv = lambda k: (None if f is None or pd.isna(f[k]) else round(float(f[k]), 3))
        stations.append({
            "stn": stn, "name": str(r.get("지점명", "")),
            "lat": round(lat, 4), "lon": round(lon, 4),
            "elev": None if pd.isna(r[ELEVATION_COLUMN]) else round(float(r[ELEVATION_COLUMN]), 0),
            "region": short_region(assign_region(lat, lon, prov_geom)),
            "mtn": bool(t["is_mountain"]) if t is not None else False,
            "coast": bool(t["is_coastal"]) if t is not None else False,
            "island": bool(t["is_island"]) if t is not None else False,
            "mae": gv("MAE"), "mae_none": gv("MAE_none"), "bias": gv("BIAS"),
            "dz": gv("mean_absdz_km"), "ri": gv("RI"), "night_inv": gv("night_inversion_ratio"),
        })

        # 월별 합계
        sub = am[am["STN"] == stn]
        mrec = {}
        for mm in range(1, 13):
            row = {}
            for method, key in [("none", "n"), ("fixed_lapse", "f")]:
                x = sub[(sub["mm"] == mm) & (sub["METHOD"] == method)]
                if len(x):
                    row[key] = [int(x["n"].iloc[0]), round(float(x["se"].iloc[0]), 1),
                               round(float(x["sa"].iloc[0]), 1)]
            if row:
                mrec[str(mm)] = row
        monthly[stn] = mrec

        # 시간대별 (fixed) → 24
        hsub = ah[ah["STN"] == stn]
        hm = [None] * 24; hb = [None] * 24
        for h, n, se, sa in zip(hsub["h"], hsub["n"], hsub["se"], hsub["sa"]):
            if 0 <= h < 24 and n:
                hm[int(h)] = ic(sa / n); hb[int(h)] = ic(se / n)
        diurnal_mae[stn] = hm; diurnal_bias[stn] = hb

        # 최적 γ (월·시간) — per-hour raw 필요
        nd = pd.read_sql("SELECT TM, ERROR en FROM raw WHERE STN=? AND METHOD='none' AND interpolable=1",
                         con, params=[stn])
        fd = pd.read_sql("SELECT TM, ERROR ef FROM raw WHERE STN=? AND METHOD='fixed_lapse' AND interpolable=1",
                         con, params=[stn])
        om, oh = {}, {}
        if len(nd) and len(fd):
            m = nd.merge(fd, on="TM")
            en = m["en"].to_numpy(); dz = (m["ef"].to_numpy() - en) / GAMMA
            # 지점 전체기간 최적γ (지도 heatmap 전환용)
            gall = opt_gamma(en, dz)
            if gall:
                stations[-1]["gopt"] = round(gall[0], 2)
            tm = pd.to_datetime(m["TM"])
            mo = tm.dt.month.to_numpy(); hr = tm.dt.hour.to_numpy()
            for mm in range(1, 13):
                sel = mo == mm
                if sel.sum() >= 24:
                    g = opt_gamma(en[sel], dz[sel])
                    if g: om[str(mm)] = [round(g[0], 2), g[1]]
            for hh in range(24):
                sel = hr == hh
                if sel.sum() >= 24:
                    g = opt_gamma(en[sel], dz[sel])
                    if g: oh[str(hh)] = [round(g[0], 2), g[1]]
        optlapse_m[stn] = om; optlapse_h[stn] = oh
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(stn_list)}")

    data = {
        "meta": {"n_stations": len(stations), "year": 2024, "gamma_fixed": GAMMA, "scale": SCALE},
        "provinces": prov_draw, "stations": stations, "monthly": monthly,
        "diurnal_mae": diurnal_mae, "diurnal_bias": diurnal_bias,
        "optlapse_m": optlapse_m, "optlapse_h": optlapse_h,
    }
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, separators=(",", ":"))
    con.close()
    print(f"\n저장: {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB, 지점 {len(stations)})")
    print(f"최적γ 격자 {GRID_LO}~{GRID_HI} | 식별불가(경계 포화)로 제외 {opt_gamma.saturated}건"
          " — ΔZ가 너무 작아 γ가 결정되지 않는 구간(표시 안 함)")


if __name__ == "__main__":
    main()
