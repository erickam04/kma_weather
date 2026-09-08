"""Target 분포 지도 + 이웃 고도 시각화 (TODO 1~4, 파이프라인 2단계).

기존 산출물만 읽어서 그린다 — IDW/고도보정 재계산 없음 (로직 무수정 원칙):
- `{OUT_BATCH}/target_stations.csv`                       : target 목록 (run_batch_test 산출)
- `{OUT_VIZ}/neighbor_detail/trace/station_..._trace.csv` : 계절별 neighbor trace (run_visualize_targets 산출)
- metadata                                                 : 전체 station 위경도·고도 (배경 지도용)

산출 (→ {OUT_VIZ}/targets_overview/):
1. `targets_overview_map.png`   — target 전체를 실제 한국 지도 위에 표시 (TODO 1)
2. `targets_neighbor_elevation_dist.png` — target별 이웃 고도분포 diagram:
   이웃 고도 산점 + target 고도 + weight 가중평균 이웃 고도 (TODO 4)

이웃 고도 자체(TODO 2)는 neighbor 선정 지도(`neighbor_detail/maps/`)에 마커 고도색
heatmap으로 통합됨 (visualize_idw.plot_station_map). 지도는 위도 보정 aspect(1/cos φ) 적용 (TODO 3).

실행: python -m visualization.visualize_targets_map  (또는 run_pipeline.py의 viz_maps 단계)
계획: .omc/plans/pipeline-config-refactor.md §3.4
"""

import glob
import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

import pipeline_config as cfg
from weather.find_stations import (
    filter_valid_station_meta,
    load_station_meta,
)
from weather.idw import ELEVATION_COLUMN, TARGET_COLUMN


TARGETS_PATH = os.path.join(cfg.OUT_BATCH, "target_stations.csv")
SUMMARY_PATH = os.path.join(cfg.OUT_BATCH, "analysis_summary.csv")
ELEV_CMAP = "terrain"

# 한국 시도 경계 (통계청 2013, southkorea-maps에서 단순화 — 지도데이터/simplify_boundary.py)
KOREA_BOUNDARY_PATH = os.path.join("지도데이터", "korea_boundary_simplified.json")
LAND_COLOR = "#f5f2e8"
SEA_COLOR = "#dbe9f4"
BORDER_COLOR = "#9a9484"


def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def lat_aspect(lat_deg):
    """위경도 축 비율 (TODO 3): 경도 1도가 위도 1도보다 cos(φ)배 짧음을 보정."""
    return 1.0 / math.cos(math.radians(lat_deg))


def load_targets_with_coords(meta):
    """target CSV에 기준일 유효 metadata의 위경도를 붙여 반환 (고도순 정렬)."""
    targets = pd.read_csv(TARGETS_PATH, encoding="utf-8-sig")
    targets["지점"] = targets["지점"].astype(str)

    valid = filter_valid_station_meta(meta, cfg.get_auto_ref_date())
    coord = valid.drop_duplicates(subset=["지점"]).set_index("지점")[["위도", "경도"]]
    targets = targets.join(coord, on="지점")

    missing = targets[targets["위도"].isna()]["지점"].tolist()
    if missing:
        print(f"  [주의] 기준일 유효 metadata에 위경도 없음: {missing} (지도에서 제외)")
    return targets.dropna(subset=["위도", "경도"]).sort_values(ELEVATION_COLUMN).reset_index(drop=True)


def find_season_traces(station_id, out_viz=None):
    """target 하나의 계절별 trace CSV 경로 {계절이름: path}를 찾는다.

    run_visualize_targets는 정오 결측 시 같은 날 다른 시각으로 대체하므로
    시각 부분은 glob(*)으로 매칭한다. 파일이 없는 계절(관측 결측)은 건너뜀.
    """
    if out_viz is None:
        out_viz = cfg.OUT_VIZ
    trace_dir = cfg.viz_dir("neighbor_trace", base=out_viz)
    found = {}
    for season_name, day in cfg.get_season_days():
        day_key = pd.Timestamp(day).strftime("%Y%m%d")
        pattern = os.path.join(
            trace_dir, f"station_{station_id}_{TARGET_COLUMN}_{day_key}_*_trace.csv"
        )
        paths = sorted(glob.glob(pattern))
        if paths:
            found[season_name] = paths[0]
    return found


def draw_korea_basemap(ax):
    """실제 한국 경계 폴리곤을 지도 배경으로 그린다 (위경도 좌표 그대로 사용).

    경계 파일이 없으면 경고 후 흰 배경으로 fallback (그림은 계속 생성).
    """
    if not os.path.exists(KOREA_BOUNDARY_PATH):
        print(f"  [주의] 경계 파일 없음: {KOREA_BOUNDARY_PATH} -> 흰 배경으로 대체")
        return False
    import json
    with open(KOREA_BOUNDARY_PATH, encoding="utf-8") as f:
        rings = json.load(f)["rings"]
    ax.set_facecolor(SEA_COLOR)
    for ring in rings:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        ax.fill(xs, ys, fc=LAND_COLOR, ec=BORDER_COLOR, lw=0.5, zorder=0)
    return True


def interpolated_station_ids():
    """batch 결과가 실제로 존재하는 station 집합 (보간 불가 지점 제외용).

    analysis_summary.csv의 station axis에서 읽는다. 파일이 없으면 None (제외 안 함).
    """
    if not os.path.exists(SUMMARY_PATH):
        return None
    s = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    s = s[s["axis"] == "station"]
    if len(s) == 0:
        return None
    return set(s["STN"].astype(float).astype(int).astype(str))


# ---------------------------------------------------------------------------
# 1. Target 분포 지도 (TODO 1)
# ---------------------------------------------------------------------------
def plot_targets_overview(meta, targets, output_path):
    valid = filter_valid_station_meta(meta, cfg.get_auto_ref_date())
    valid = valid.dropna(subset=["위도", "경도"])

    # 보간 불가(batch 결과 없음) target은 지도에서 제외 — 예: 530 태하(이웃<3)
    done = interpolated_station_ids()
    excluded = []
    if done is not None:
        excluded = [f"{r['지점명']}({r['지점']})" for _, r in targets.iterrows()
                    if r["지점"] not in done]
        targets = targets[targets["지점"].isin(done)].reset_index(drop=True)
        if excluded:
            print(f"  분포 지도에서 보간 불가 지점 제외: {', '.join(excluded)}")

    fig, ax = plt.subplots(figsize=(9, 11))
    draw_korea_basemap(ax)

    ax.scatter(valid["경도"], valid["위도"], s=5, c="#a8a8a8", alpha=0.45,
               label=f"AWS 관측소 전체 ({len(valid)}개)", zorder=1)

    elev = targets[ELEVATION_COLUMN]
    norm = plt.Normalize(vmin=0, vmax=max(float(elev.max()), 1.0))
    sc = ax.scatter(targets["경도"], targets["위도"], s=260, c=elev,
                    cmap=ELEV_CMAP, norm=norm, marker="*",
                    edgecolors="#d62728", linewidths=1.4,
                    label=f"Target ({len(targets)}개)", zorder=3)

    # 라벨: 가까운 target이 이미 있으면 반대쪽(좌하)으로 밀어 겹침 방지 (예: 사상904·부산레160)
    placed = []
    for _, row in targets.iterrows():
        lon, lat = float(row["경도"]), float(row["위도"])
        near = sum(1 for plon, plat in placed
                   if abs(plon - lon) < 0.5 and abs(plat - lat) < 0.35)
        if near % 2 == 0:
            xytext, ha, va = (8, 5), "left", "bottom"
        else:
            xytext, ha, va = (-8, -7), "right", "top"
        placed.append((lon, lat))
        ax.annotate(
            f"{row['지점명']}({row['지점']})\n{row[ELEVATION_COLUMN]:.0f}m",
            (lon, lat),
            xytext=xytext, textcoords="offset points", ha=ha, va=va,
            fontsize=9, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.8),
            zorder=4,
        )

    ax.set_aspect(lat_aspect(float(valid["위도"].mean())))
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Target 해발고도 (m)")
    ax.set_xlabel("경도 (°E)")
    ax.set_ylabel("위도 (°N)")
    ax.set_title(f"Target 관측소 분포 ({cfg.YEAR}년 batch 평가 대상)\n"
                 f"회색 점: 기준일({cfg.get_auto_ref_date()}) 유효 AWS 관측소",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. 이웃 고도분포 diagram (TODO 4)
# ---------------------------------------------------------------------------
# (target별 4계절 neighbor 지도는 제거됨 — 이웃 고도는 neighbor 선정 지도
#  `neighbor_detail/maps/`에 마커 고도색 heatmap으로 통합, visualize_idw.plot_station_map)
def plot_neighbor_elevation_dist(targets, all_season_traces, output_path):
    """target(고도순)별로 이웃 고도 분포(4계절 합산)를 산점으로 직접 보여준다.

    - 회색 점: 이웃 고도 (계절 중복 포함, 좌우 jitter 대신 계절별 미세 오프셋)
    - 빨간 별: target 고도
    - 파란 대시: weight 가중평균 이웃 고도 (MAE 요인 분석의 delta_elev와 동일 개념)
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    season_order = [s for s, _ in cfg.get_season_days()]
    offsets = np.linspace(-0.18, 0.18, max(len(season_order), 1))

    xticklabels = []
    for i, (_, trow) in enumerate(targets.iterrows()):
        sid = trow["지점"]
        traces = all_season_traces.get(sid, {})
        xticklabels.append(f"{trow['지점명']}\n({sid})")

        wsum, wesum = 0.0, 0.0
        for j, season in enumerate(season_order):
            if season not in traces:
                continue
            tr = pd.read_csv(traces[season], encoding="utf-8-sig")
            ax.scatter(np.full(len(tr), i + offsets[j]), tr["elevation_m"],
                       s=26, c="#8f8f8f", alpha=0.65, zorder=2)
            # NaN weight(보간 미수행 케이스)는 가중평균에서 제외
            valid = tr["weight_normalized"].notna() & tr["elevation_m"].notna()
            w = tr.loc[valid, "weight_normalized"].astype(float)
            wsum += float(w.sum())
            wesum += float((w * tr.loc[valid, "elevation_m"].astype(float)).sum())

        if wsum > 0:
            wmean = wesum / wsum
            ax.hlines(wmean, i - 0.28, i + 0.28, colors="#2d6cdf", lw=2.4,
                      linestyles="--", zorder=3)
        ax.scatter([i], [trow[ELEVATION_COLUMN]], s=210, c="#d62728", marker="*",
                   edgecolors="#7a1010", linewidths=0.8, zorder=4)

    handles = [
        Line2D([], [], marker="*", color="none", markerfacecolor="#d62728",
               markeredgecolor="#7a1010", markersize=15, label="Target 고도"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#8f8f8f",
               markersize=7, alpha=0.7, label="이웃 고도 (4계절 합산)"),
        Line2D([], [], color="#2d6cdf", lw=2.4, ls="--",
               label="이웃 고도 weight 가중평균"),
    ]
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("해발고도 (m)")
    ax.set_title(f"Target별 이웃 고도분포 ({cfg.YEAR}년 4계절 대표시각)\n"
                 "빨간 별과 파란 대시의 차이가 고도보정이 메워야 하는 Δ고도",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    setup_korean_font()
    out_dir = cfg.viz_dir("targets_overview")
    os.makedirs(out_dir, exist_ok=True)

    meta = load_station_meta()
    targets = load_targets_with_coords(meta)
    print(f"target {len(targets)}개 (고도순): {list(targets['지점'])}")

    saved = []

    # 1. 분포 지도
    path = os.path.join(out_dir, "targets_overview_map.png")
    plot_targets_overview(meta, targets, path)
    saved.append(path)

    # 2. 이웃 고도분포 diagram용 trace 수집 (계절별 지도는 제거 — 고도는 neighbor
    #    선정 지도에 heatmap으로 통합됨)
    all_season_traces = {}
    for _, trow in targets.iterrows():
        sid = trow["지점"]
        season_traces = find_season_traces(sid)
        all_season_traces[sid] = season_traces
        if not season_traces:
            print(f"  [주의] {sid}: trace 없음 (run_visualize_targets 미실행?)")

    # 3. 이웃 고도분포 diagram
    path = os.path.join(out_dir, "targets_neighbor_elevation_dist.png")
    plot_neighbor_elevation_dist(targets, all_season_traces, path)
    saved.append(path)

    print("저장 완료:")
    for p in saved:
        print("  -", p)


if __name__ == "__main__":
    main()
