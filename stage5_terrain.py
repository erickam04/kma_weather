"""Stage 5 — 지형 라벨링 (산악=기복 기반 / 해안 / 섬).

지점마다 지형 라벨을 붙인다. MAE와 결합하지 않음(순환성 차단, §1) — 라벨만.
근거·기준(선행연구):
  - 산악: **기복(起伏, 국지 고도 최대−최소)** 기반. 국토교통부 '산' = 기복량 ≥100m.
    절대고도 아님(우리 ΔZ 발견과 정합). 기복은 DEM(Copernicus GLO-30, 30m)에서 1km 창으로 계산.
  - 해안: 진짜 해안선까지 ≤10km (민숙주·최영은 2019, 웹·미검증 → 잠정, 민감도 병기).
    ⚠️ 행정경계 dissolve 필수(안 하면 내륙 대구가 8.4km로 오염).
  - 섬: dissolve 후 육지 최대 폴리곤(본토)이 아닌 폴리곤 안이면 섬 (제주·울릉 등).

기존 모듈·알고리즘 무수정. 신규 의존: shapely, rasterio, pyproj (설치 완료).
출력: WeatherData_AWS/interpolation_batch/fullnet/station_terrain.csv (라벨) + 분포 요약.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import shape, Point, MultiPolygon, Polygon
from shapely.ops import unary_union, transform as shp_transform, nearest_points
from pyproj import Transformer, Geod

from find_stations import filter_valid_station_meta, load_station_meta
from idw import ELEVATION_COLUMN

FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
BOUNDARY = "지도데이터/korea_boundary_geo.json"   # full-res (dissolve용)
REF_DATE = "2024-06-15"
RELIEF_RADIUS_M = 1000        # 국지 기복 창 반경(±1km)
MOUNTAIN_RELIEF_M = 100       # 국토교통부 '산' 기복 임계 (민감도 200/300 병기)
COAST_KM = 10.0               # 해안 임계 (민감도 5/15/20 병기)
S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
_GEOD = Geod(ellps="WGS84")


# ---------------- 기복 (DEM /vsicurl 원격 윈도우) ----------------
def _tile_url(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    ns = "N" if la >= 0 else "S"; ew = "E" if lo >= 0 else "W"
    t = f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
    return f"/vsicurl/{S3}/{t}/{t}.tif"


def compute_relief(meta_pts):
    """지점별 1km 창 기복. 타일별로 묶어 원격 COG를 타일당 1회 열고 윈도우만 읽음."""
    d = RELIEF_RADIUS_M / 111320.0
    by_tile = {}
    for stn, lat, lon in meta_pts:
        by_tile.setdefault((math.floor(lat), math.floor(lon)), []).append((stn, lat, lon))

    out = {}
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env:
        for i, (tile, pts) in enumerate(sorted(by_tile.items()), 1):
            url = _tile_url(pts[0][1], pts[0][2])
            try:
                src = rasterio.open(url)
            except Exception as e:
                print(f"  [타일 {tile} 열기 실패: {type(e).__name__}] {len(pts)}지점 relief=NaN")
                for stn, la, lo in pts:
                    out[stn] = (np.nan, True)
                continue
            with src:
                for stn, la, lo in pts:
                    try:
                        win = from_bounds(lo - d, la - d, lo + d, la + d, src.transform)
                        arr = src.read(1, window=win).astype(float)
                        arr = arr[arr > -1000]
                        # 기대 픽셀수 대비 부족하면 타일경계 clip으로 표시
                        exp = (2 * d / src.res[0]) ** 2
                        clipped = arr.size < 0.8 * exp
                        out[stn] = (float(arr.max() - arr.min()) if arr.size else np.nan, clipped)
                    except Exception:
                        out[stn] = (np.nan, True)
            print(f"  [{i}/{len(by_tile)}] 타일 {tile}: {len(pts)}지점")
    return out


# ---------------- 해안거리 + 섬 (shapely, dissolve) ----------------
def load_coastline():
    """full-res 경계 → dissolve(본토+섬 폴리곤) + 외곽선(진짜 해안선) 반환."""
    with open(BOUNDARY, encoding="utf-8") as f:
        gj = json.load(f)
    geoms = []
    feats = gj.get("features", gj if isinstance(gj, list) else [])
    for ft in feats:
        g = ft.get("geometry") if isinstance(ft, dict) else None
        if g:
            try:
                geoms.append(shape(g))
            except Exception:
                pass
    land = unary_union(geoms)                 # 행정경계 내부선 제거 → 진짜 육지 덩어리
    polys = list(land.geoms) if isinstance(land, MultiPolygon) else [land]
    polys.sort(key=lambda p: p.area, reverse=True)
    mainland = polys[0]                        # 최대 = 본토
    coastline = unary_union([p.boundary for p in polys])  # 모든 육지 외곽 = 해안선
    return land, mainland, polys, coastline


def coast_and_island(pts, coastline, mainland, polys):
    """지점별 해안거리(km, 지오데식) + 섬여부(본토 아닌 폴리곤 안)."""
    res = {}
    for stn, lat, lon in pts:
        p = Point(lon, lat)
        # 최근접 해안점(도 단위) → 지오데식 거리(km)
        np_pt = nearest_points(p, coastline)[1]
        _, _, dist_m = _GEOD.inv(lon, lat, np_pt.x, np_pt.y)
        # 섬: 본토에 안 들고 다른 육지 폴리곤에 들면 섬 (경계 근처 tolerance)
        on_main = mainland.covers(p) or mainland.distance(p) < 1e-4
        in_land = any(poly.covers(p) or poly.distance(p) < 1e-4 for poly in polys)
        is_island = (not on_main) and in_land
        res[stn] = (dist_m / 1000.0, is_island)
    return res


def terrain_group(r):
    """복합 지형 라벨 — 한 지점이 섬·산악·해안 여럿에 해당하면 모두 표기(·구분).
    (예: 제주 한라산 고봉 = '섬·산악'). 아무것도 없으면 '내륙평지'."""
    labels = []
    if r["is_island"]:
        labels.append("섬")
    if r["is_mountain"]:
        labels.append("산악")
    if r["is_coastal"]:
        labels.append("해안")
    return "·".join(labels) if labels else "내륙평지"


def elev_class(e):
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return "unknown"
    for lo, hi, name in [(0, 100, "저지대"), (100, 500, "중지대"),
                         (500, 1000, "고지대"), (1000, 1e9, "고산")]:
        if lo <= e < hi:
            return name
    return "unknown"


def main():
    os.makedirs(FULLNET, exist_ok=True)
    meta = load_station_meta()
    vref = filter_valid_station_meta(meta, REF_DATE)
    # 기준일 유효 없으면 마지막 행 fallback
    seen = {}
    for _, r in vref.iterrows():
        seen[str(r["지점"])] = r
    for stn in meta["지점"].unique():
        s = str(stn)
        if s not in seen:
            seen[s] = meta[meta["지점"] == s].iloc[-1]

    pts = [(s, float(r["위도"]), float(r["경도"])) for s, r in seen.items()
           if pd.notna(r["위도"]) and pd.notna(r["경도"])]
    print(f"대상 지점 {len(pts)}개")

    print("[1/2] 기복(DEM) 계산 ...")
    relief = compute_relief(pts)

    print("[2/2] 해안선 dissolve + 거리·섬 판정 ...")
    land, mainland, polys, coastline = load_coastline()
    print(f"  육지 폴리곤 {len(polys)}개 (본토 면적 {mainland.area:.3f})")
    ci = coast_and_island(pts, coastline, mainland, polys)

    rows = []
    for s, r in seen.items():
        if s not in relief:
            continue
        rel, clipped = relief[s]
        cd, isl = ci.get(s, (np.nan, False))
        e = r[ELEVATION_COLUMN]
        rows.append({
            "STN": s, "지점명": r.get("지점명", ""),
            "위도": round(float(r["위도"]), 5), "경도": round(float(r["경도"]), 5),
            "elev_m": None if pd.isna(e) else round(float(e), 1),
            "relief_1km_m": None if rel is None or np.isnan(rel) else round(rel, 1),
            "relief_clipped": bool(clipped),
            "is_mountain": (rel is not None and not np.isnan(rel) and rel >= MOUNTAIN_RELIEF_M),
            "coast_dist_km": None if np.isnan(cd) else round(cd, 2),
            "is_coastal": (not np.isnan(cd)) and cd <= COAST_KM,
            "is_island": bool(isl),
            "elev_class": elev_class(e),
        })
    df = pd.DataFrame(rows)

    df["terrain_group"] = df.apply(terrain_group, axis=1)

    df.to_csv(os.path.join(FULLNET, "station_terrain.csv"),
              index=False, encoding="utf-8-sig")

    print("\n=== 지형 분류 결과 ===")
    print("terrain_group 분포:")
    print(df["terrain_group"].value_counts().to_string())
    print(f"\nis_mountain(기복≥{MOUNTAIN_RELIEF_M}m): {int(df.is_mountain.sum())}  "
          f"| is_coastal(≤{COAST_KM}km): {int(df.is_coastal.sum())}  "
          f"| is_island: {int(df.is_island.sum())}")
    print("\n기복 임계 민감도:", {f'≥{t}m': int((df.relief_1km_m >= t).sum())
                              for t in (100, 200, 300, 500)})
    print("해안 임계 민감도:", {f'≤{t}km': int((df.coast_dist_km <= t).sum())
                            for t in (5, 10, 15, 20)})
    print(f"relief_clipped(타일경계): {int(df.relief_clipped.sum())}지점")
    print(f"\n저장: {FULLNET}/station_terrain.csv ({len(df)}지점)")


if __name__ == "__main__":
    main()
