"""고도보정(fixed_lapse, γ=-6.5°C/km) 후에도 남는 station별 MAE 차이의 원인 규명 —
   선행연구 기반 정식 재현 스크립트 (TODO #1 Phase 4).

리포트 `WeatherData_AWS/interpolation_batch/station_mae_prior_analysis_20260720.md`
(ground truth)의 수치를 그대로 재현한다. 리포트 문서는 수정하지 않는다 — 이 스크립트만 관리.

핵심 항등식 (§3):
    station별 MAE = mean|γ_eff − (−6.5)| × |ΔZ|
      · ΔZ      = target 고도 − Σ(weight_normalized × 이웃 고도)   (외삽 거리, 4계절 trace 평균)
      · γ_eff   = −ERROR_none / ΔZ_km                             (실측 국지 감률, 시각별)

재현 계산:
  1. ΔZ (station별, 4계절 trace 평균)
  2. 오차 분해 항등식 MAE = mean|γ_eff+6.5| × |ΔZ|  (|ΔZ|>0.10km 5개)
  3. 계절·주야 BIAS (봄=4·5월, 겨울=12·1월, 주간=12–15시, 야간=00–05시)
     + 겨울야간−여름야간 MAE (기상학적 계절: 겨울 DJF=12·1·2, 여름 JJA=6·7·8)
  4. 야간 역전 비율 (야간 γ_eff>0 비율)
  5. 국지 γ 통계 (주간/야간/일변화 진폭/연평균±std, 160 월×시간대 γ표, 전국 γ 비교)
  6. 공변량 회귀 MAE~설명변수 R² (n=8 편의표본 — R²는 참고용, 인과 아님)

알고리즘 모듈(idw/elevation_correction/lapse_rate) 무수정 — 기존 batch 산출물·trace·메타만 읽음.
metadata 조회는 반드시 find_stations.filter_valid_station_meta 사용
  (단순 dict(zip)은 유효기간 중복행에서 904 고도가 0.73m 대신 30.0m로 오염됨).

산출물:
  WeatherData_AWS/interpolation_batch/station_mae_prior_factors.csv  (utf-8-sig)
  <viz_dir("mae_analysis")>/prior_mae_decomposition.png
  <viz_dir("mae_analysis")>/prior_local_gamma_160_vs_871.png
  <viz_dir("mae_analysis")>/prior_bias_signreversal_160.png
  <viz_dir("mae_analysis")>/prior_night_inversion.png
  <viz_dir("mae_analysis")>/prior_mae_vs_covariates.png

실행: python analyze_station_mae_prior.py
"""

import glob
import math
import sys
from pathlib import Path

# Windows 콘솔(cp949)에서 em-dash 등 유니코드 출력이 깨지지 않도록 stdout을 utf-8로.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from find_stations import filter_valid_station_meta, load_station_meta
import pipeline_config as cfg

# ---------------------------------------------------------------------------
# 경로·상수
# ---------------------------------------------------------------------------
BATCH_DIR = Path("WeatherData_AWS/interpolation_batch")
TRACE_DIR = Path("WeatherData_AWS/interpolation_visualization/neighbor_detail/trace")
VIZ_DIR = Path(cfg.viz_dir("mae_analysis"))

TARGETS_CSV = BATCH_DIR / "target_stations.csv"
SUMMARY_CSV = BATCH_DIR / "analysis_summary.csv"
HOURLY_GAMMA_CSV = BATCH_DIR / "hourly_gamma_2024.csv"
FACTORS_CSV = BATCH_DIR / "station_mae_prior_factors.csv"

# 530 태하: 반경30km 내 이웃 1개 → 보간 불가. target 8개.
EXCLUDE_STATIONS = {"530"}

GAMMA_APPLIED = -6.5  # °C/km (fixed_lapse 적용값)
DZ_MIN_KM = 0.10      # γ_eff 해석에 쓰는 |ΔZ| 하한 (작으면 나눗셈 불안정)

# 4계절 대표 trace (정오). 554는 봄(0415) 관측 중단 → 3계절만 존재.
SEASON_TRACES = {
    "1_winter": "20240115",
    "2_spring": "20240415",
    "3_summer": "20240715",
    "4_fall": "20241015",
}

# 시간대·계절 정의
DAY_HOURS = (12, 15)      # 주간
NIGHT_HOURS = (0, 5)      # 야간
SPRING_MONTHS = [4, 5]    # 봄 BIAS
WINTER_MONTHS = [12, 1]   # 겨울 BIAS
# 겨울야간−여름야간 MAE 열은 기상학적 계절(DJF/JJA) 사용 (§2-3 재현)
DJF = [12, 1, 2]
JJA = [6, 7, 8]

# ADW(Angular Distance Weighting) 원식 파라미터 (New/Caesar/CRU TS v4)
ADW_CDD = 30.0  # 상관감쇠거리(km) = 탐색반경
ADW_M = 4       # 거리가중 지수

ELEV_COL = "노장해발고도(m)"

# 리포트(ground truth) 검증 목표값
REPORT_DZ = {  # §7 (km)
    "160": 0.4167, "554": 0.3636, "871": 0.2714, "904": -0.1306,
    "768": -0.1143, "537": 0.0126, "648": 0.0067, "919": 0.0045,
}
REPORT_DECOMP = {  # §3-2 (|Δγ|, |ΔZ|km, 실제MAE)
    "160": (2.74, 0.417, 1.142), "554": (3.05, 0.364, 1.086),
    "871": (2.85, 0.271, 0.774), "904": (5.53, 0.131, 0.722),
    "768": (4.98, 0.114, 0.569),
}
REPORT_NIGHT_INV = {  # §4-6 (야간 역전 비율 %)
    "904": 30.5, "768": 19.5, "160": 8.3, "554": 5.5, "871": 2.4,
}
REPORT_LOCAL_GAMMA = {  # §4 (주간, 야간)
    "160": (-9.56, -4.87), "871": (-4.84, -7.33), "554": (-8.23, -6.23),
    "904": (-5.79, -1.97), "768": (-6.79, -4.28),
}
REPORT_ANNUAL_GAMMA = {"160": -6.94, "554": -7.05, "871": -6.35}  # §4-5
REPORT_NATIONAL_GAMMA = (-6.15, -5.28, 0.87)  # §4-1 (주간, 야간, 진폭)
REPORT_R2 = {  # §2-6 / §5-3 / §6 (MAE ~ 설명변수)
    "delta_z": 0.698, "gap_elev": 0.766, "abs_elev": 0.147, "adw_alpha": 0.054,
    "octant_occ": 0.022, "quadrant_occ": 0.026, "max_angular_gap": 0.055,
    "nearest_dist": 0.222, "neighbor_elev_std": 0.023,
}
REPORT_BIAS_REG = {  # §6 (slope, R²)
    "bias_none": (6.60, 0.976), "bias_fixed": (0.28, 0.056),
}


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def setup_korean_font():
    for name in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def bearing_deg(lat1, lon1, lat2, lon2):
    """target(1)에서 이웃(2)을 바라보는 방위각(0~360, 북=0, 시계방향)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def max_angular_gap(bearings):
    """이웃 방위각들 사이의 가장 큰 빈 각도(도). 클수록 한쪽 쏠림."""
    b = sorted(bearings)
    if len(b) == 0:
        return float("nan")
    if len(b) == 1:
        return 360.0
    gaps = [b[i + 1] - b[i] for i in range(len(b) - 1)]
    gaps.append(360.0 - b[-1] + b[0])
    return max(gaps)


def adw_alpha_weighted(dist, bearings):
    """ADW 방위가중 α의 거리가중 평균 (station별 단일 지표, 0~2).

    원식(New 2000 / Caesar 2006 / CRU TS v4):
        f_k = (exp(−d_k / CDD))^m
        α_k = Σ_{l≠k} f_l·(1 − cos(θ_l − θ_k)) / Σ_{l≠k} f_l
    α_k(방위적 고립도)를 f_k로 가중평균해 이웃 구성 전체의 방위 분산도를 나타냄.
    """
    d = np.asarray(dist, dtype=float)
    th = np.radians(np.asarray(bearings, dtype=float))
    fk = (np.exp(-d / ADW_CDD)) ** ADW_M
    n = len(d)
    alpha = np.full(n, np.nan)
    for k in range(n):
        num = 0.0
        den = 0.0
        for l in range(n):
            if l == k:
                continue
            num += fk[l] * (1.0 - math.cos(th[l] - th[k]))
            den += fk[l]
        alpha[k] = num / den if den > 0 else np.nan
    fsum = np.nansum(fk)
    return float(np.nansum(fk * alpha) / fsum) if fsum > 0 else float("nan")


def r2_of(x, y):
    """단순선형회귀 결정계수 R² = Pearson r². (n=8 편의표본 — 참고용)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)


def slope_of(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    return float(np.polyfit(x[m], y[m], 1)[0])


def time_band(hour):
    if hour <= 5:
        return "야간"
    if hour <= 11:
        return "오전"
    if hour <= 15:
        return "주간"
    return "저녁"


# ---------------------------------------------------------------------------
# 로드
# ---------------------------------------------------------------------------
def load_targets():
    df = pd.read_csv(TARGETS_CSV, encoding="utf-8-sig")
    df["지점"] = df["지점"].astype(str)
    return df[~df["지점"].isin(EXCLUDE_STATIONS)].reset_index(drop=True)


def load_raw(method):
    """월별 raw_2024MM_<method>.csv 12개월을 이어붙여 로드."""
    files = sorted(glob.glob(str(BATCH_DIR / f"raw_2024??_{method}.csv")))
    if not files:
        raise FileNotFoundError(f"raw 파일 없음: {method}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df["STN"] = df["STN"].astype(str)
    df["TM"] = pd.to_datetime(df["TM"])
    df["hour"] = df["TM"].dt.hour
    df["month"] = df["TM"].dt.month
    return df


def load_summary_station():
    """analysis_summary.csv에서 station축 MAE·BIAS (method별)."""
    s = pd.read_csv(SUMMARY_CSV)
    st = s[s["axis"] == "station"].copy()
    st["STN"] = st["STN"].astype("Int64").astype(str)
    out = {}
    for method in ["fixed_lapse", "none"]:
        sub = st[st["METHOD"] == method]
        out[method] = {
            "MAE": dict(zip(sub["STN"], sub["MAE"])),
            "BIAS": dict(zip(sub["STN"], sub["BIAS"])),
        }
    return out


# ---------------------------------------------------------------------------
# 1. ΔZ + 이웃 기하 (4계절 trace 평균, meta는 filter_valid_station_meta 사용)
# ---------------------------------------------------------------------------
def compute_delta_z_and_geometry(meta, targets):
    """station별 ΔZ(외삽 거리, km) 및 공변량 기하 지표를 4계절 trace 평균으로 산출.

    ΔZ = target 고도 − Σ(weight_normalized × 이웃 elevation_m).
    이웃 고도 결측은 정규화에서 제외(고도 결측 fallback). target 좌표·고도는
    각 trace 시각에 filter_valid_station_meta로 조회(유효기간 중복행 오염 방지).
    """
    rows = []
    for sid in targets["지점"]:
        per_season = []
        for date_str in SEASON_TRACES.values():
            f = TRACE_DIR / f"station_{sid}_TA_{date_str}_1200_trace.csv"
            if not f.exists():
                continue
            tr = pd.read_csv(f, encoding="utf-8-sig")
            if len(tr) == 0:
                continue
            ts = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 12:00")
            valid = filter_valid_station_meta(meta, ts)
            trow = valid[valid["지점"] == sid]
            if len(trow) == 0:
                continue
            trow = trow.iloc[0]
            t_elev = pd.to_numeric(trow.get(ELEV_COL), errors="coerce")
            t_lat = float(trow["위도"])
            t_lon = float(trow["경도"])

            w = tr["weight_normalized"].to_numpy(dtype=float)
            elev = pd.to_numeric(tr["elevation_m"], errors="coerce").to_numpy()
            dist = tr["distance_km"].to_numpy(dtype=float)
            emask = ~np.isnan(elev)
            if not emask.any() or pd.isna(t_elev):
                continue
            w_elev = float(np.sum(w[emask] * elev[emask]) / np.sum(w[emask]))

            bearings = [
                bearing_deg(t_lat, t_lon, r["latitude"], r["longitude"])
                for _, r in tr.iterrows()
            ]
            nearest_idx = int(np.argmin(dist))
            per_season.append(
                {
                    "delta_z": (float(t_elev) - w_elev) / 1000.0,  # km
                    # gap_elev = target 고도 − 최근접 이웃 고도 (비표준 보조지표, §6)
                    "gap_elev": float(t_elev) - float(elev[nearest_idx]),
                    "nearest_dist": float(dist[nearest_idx]),
                    "neighbor_elev_std": float(np.nanstd(elev)),
                    "adw_alpha": adw_alpha_weighted(dist, bearings),
                    "octant_occ": len({int(b // 45) for b in bearings}),
                    "quadrant_occ": len({int(b // 90) for b in bearings}),
                    "max_angular_gap": max_angular_gap(bearings),
                    "abs_elev": float(t_elev),
                    "n_neighbors": len(tr),
                }
            )
        if not per_season:
            continue
        df = pd.DataFrame(per_season)
        agg = {"STN": sid, "n_season_traces": len(df)}
        for c in df.columns:
            agg[c] = float(np.nanmean(df[c]))
        # |gap_elev| 은 R²가 ΔZ를 상회(§6). 크기(간극)로 쓰는 편이 부호형보다 설명력 큼.
        agg["abs_gap_elev"] = abs(agg["gap_elev"])
        rows.append(agg)
    return pd.DataFrame(rows).set_index("STN")


# ---------------------------------------------------------------------------
# 2. γ_eff + 오차 분해
# ---------------------------------------------------------------------------
def compute_gamma_eff(raw_none, delta_z):
    """시각별 γ_eff = −ERROR_none / ΔZ_km (station 고정 ΔZ 사용)."""
    df = raw_none.copy()
    df["dz"] = df["STN"].map(delta_z)
    df = df[df["dz"].notna() & (df["dz"] != 0)].copy()
    df["gamma_eff"] = -df["ERROR"] / df["dz"]  # °C/km
    return df


def decomposition_table(gamma_df, delta_z, mae_fixed):
    """MAE = mean|γ_eff+6.5| × |ΔZ| (|ΔZ|>0.10km). 항등식 → 실제 MAE와 대조."""
    rows = []
    for sid, dz in delta_z.items():
        if abs(dz) <= DZ_MIN_KM:
            continue
        g = gamma_df[gamma_df["STN"] == sid]
        if len(g) == 0:
            continue
        d_gamma = float(np.mean(np.abs(g["gamma_eff"] - GAMMA_APPLIED)))
        calc_mae = d_gamma * abs(dz)
        rows.append(
            {
                "STN": sid,
                "delta_gamma": d_gamma,   # mean|γ_eff+6.5|
                "abs_dz_km": abs(dz),
                "calc_MAE": calc_mae,
                "actual_MAE": mae_fixed.get(sid, float("nan")),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_MAE", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 계절·주야 BIAS + 겨울야간−여름야간 MAE
# ---------------------------------------------------------------------------
def bias_table(raw_fixed):
    rows = []
    for sid, g in raw_fixed.groupby("STN"):
        err = g["ERROR"]
        day = g[g["hour"].between(*DAY_HOURS)]["ERROR"]
        night = g[g["hour"].between(*NIGHT_HOURS)]["ERROR"]
        w_night = g[(g["month"].isin(DJF)) & (g["hour"].between(*NIGHT_HOURS))]["ABS_ERROR"]
        s_night = g[(g["month"].isin(JJA)) & (g["hour"].between(*NIGHT_HOURS))]["ABS_ERROR"]
        rows.append(
            {
                "STN": sid,
                "BIAS_year": float(err.mean()),
                "BIAS_spring": float(g[g["month"].isin(SPRING_MONTHS)]["ERROR"].mean()),
                "BIAS_winter": float(g[g["month"].isin(WINTER_MONTHS)]["ERROR"].mean()),
                "BIAS_day": float(day.mean()),
                "BIAS_night": float(night.mean()),
                "MAE_wnight": float(w_night.mean()),
                "MAE_snight": float(s_night.mean()),
                "MAE_wnight_minus_snight": float(w_night.mean() - s_night.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("STN")


# ---------------------------------------------------------------------------
# 4. 야간 역전 비율
# ---------------------------------------------------------------------------
def night_inversion(gamma_df, delta_z):
    """야간(00–05) γ_eff>0(실제 역전) 비율. |ΔZ|>0.10km만 의미 있음."""
    rows = []
    for sid, dz in delta_z.items():
        if abs(dz) <= DZ_MIN_KM:
            continue
        g = gamma_df[(gamma_df["STN"] == sid) & (gamma_df["hour"].between(*NIGHT_HOURS))]
        if len(g) == 0:
            continue
        geff = g["gamma_eff"].to_numpy()
        rows.append(
            {
                "STN": sid,
                "night_inv_pct": float(100.0 * np.mean(geff > 0)),
                "night_gamma_median": float(np.median(geff)),
                "abs_elev": float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("STN")


# ---------------------------------------------------------------------------
# 5. 국지 γ 통계
# ---------------------------------------------------------------------------
def local_gamma_stats(gamma_df, delta_z):
    rows = []
    for sid, dz in delta_z.items():
        if abs(dz) <= DZ_MIN_KM:
            continue
        g = gamma_df[gamma_df["STN"] == sid]
        geff = g["gamma_eff"]
        day = geff[g["hour"].between(*DAY_HOURS)].mean()
        night = geff[g["hour"].between(*NIGHT_HOURS)].mean()
        rows.append(
            {
                "STN": sid,
                "gamma_day": float(day),
                "gamma_night": float(night),
                # 일변화 진폭 = 야간 − 주간 (§4: 주간이 더 가파르면 +, 부호가 위상 지시)
                "gamma_amp": float(night - day),
                "gamma_annual": float(geff.mean()),
                "gamma_std": float(geff.std()),
            }
        )
    return pd.DataFrame(rows).set_index("STN")


def national_gamma():
    """전국 γ (hourly_gamma_2024.csv, Phase B 산출물). gamma는 °C/m → ×1000."""
    hg = pd.read_csv(HOURLY_GAMMA_CSV, encoding="utf-8-sig")
    hg["TM"] = pd.to_datetime(hg["TM"])
    hg["hour"] = hg["TM"].dt.hour
    day = hg[hg["hour"].between(*DAY_HOURS)]["gamma"].mean() * 1000.0
    night = hg[hg["hour"].between(*NIGHT_HOURS)]["gamma"].mean() * 1000.0
    return float(day), float(night), float(night - day)  # 진폭 = 야간 − 주간


def month_band_gamma_160(gamma_df):
    """160의 월×시간대 국지 γ 중앙값 (§4-4)."""
    g = gamma_df[gamma_df["STN"] == "160"].copy()
    g["band"] = g["hour"].apply(time_band)
    piv = g.groupby(["month", "band"])["gamma_eff"].median().unstack("band")
    return piv.reindex(columns=["야간", "오전", "주간", "저녁"])


# ---------------------------------------------------------------------------
# 6. 공변량 회귀
# ---------------------------------------------------------------------------
def covariate_r2(factors, mae_fixed):
    """MAE ~ 각 설명변수 R² (참고용 — n=8 편의표본, 인과 아님)."""
    y = [mae_fixed[s] for s in factors.index]
    out = {}
    cov_cols = [
        "delta_z", "abs_gap_elev", "abs_elev", "adw_alpha", "octant_occ",
        "quadrant_occ", "max_angular_gap", "nearest_dist", "neighbor_elev_std",
    ]
    for c in cov_cols:
        out[c] = r2_of(factors[c].to_numpy(), y)
    # gap_elev(부호형)은 정의상 대조용으로 함께 기록
    out["gap_elev_signed"] = r2_of(factors["gap_elev"].to_numpy(), y)
    return out


# ---------------------------------------------------------------------------
# 재현성 검증 (재계산 vs 리포트 문서)
# ---------------------------------------------------------------------------
def verify(delta_z, decomp, night_inv, local_g, nat_gamma, cov_r2,
           bias_reg, mae_fixed_recomp, mae_fixed_summary):
    def chk(name, got, exp, tol):
        flag = "OK" if (not np.isnan(got) and abs(got - exp) <= tol) else "MISMATCH"
        print(f"  {name:34s} 재계산={got:8.3f}  문서={exp:8.3f}  |Δ|={abs(got-exp):6.3f} [{flag}]")
        return flag == "OK"

    ok = True
    print("\n" + "=" * 74)
    print("재현성 검증 — 재계산 vs 리포트(ground truth) station_mae_prior_analysis_20260720.md")
    print("=" * 74)

    print("\n[0] 재계산 MAE(raw) vs analysis_summary (fixed_lapse)")
    for sid in sorted(mae_fixed_recomp):
        ok &= chk(f"MAE {sid}", mae_fixed_recomp[sid], mae_fixed_summary[sid], 0.001)

    print("\n[1] ΔZ (km, §7)")
    for sid, exp in REPORT_DZ.items():
        ok &= chk(f"ΔZ {sid}", delta_z.get(sid, float("nan")), exp, 0.01)

    print("\n[2] 오차 분해 계산MAE = mean|γ_eff+6.5|×|ΔZ| (§3-2, |ΔZ|>0.10)")
    dmap = decomp.set_index("STN")
    for sid, (dg, dz, mae) in REPORT_DECOMP.items():
        got = dmap.loc[sid, "calc_MAE"] if sid in dmap.index else float("nan")
        ok &= chk(f"calc_MAE {sid}", got, mae, 0.05)

    print("\n[4] 야간 역전 비율 % (§4-6)")
    for sid, exp in REPORT_NIGHT_INV.items():
        got = night_inv.loc[sid, "night_inv_pct"] if sid in night_inv.index else float("nan")
        ok &= chk(f"야간역전 {sid}", got, exp, 0.5)

    print("\n[5a] 국지 γ 주간/야간 (§4)")
    for sid, (d, n) in REPORT_LOCAL_GAMMA.items():
        ok &= chk(f"γ주간 {sid}", local_g.loc[sid, "gamma_day"], d, 0.1)
        ok &= chk(f"γ야간 {sid}", local_g.loc[sid, "gamma_night"], n, 0.1)
    print("\n[5b] 연평균 국지 γ (§4-5)")
    for sid, exp in REPORT_ANNUAL_GAMMA.items():
        ok &= chk(f"γ연평균 {sid}", local_g.loc[sid, "gamma_annual"], exp, 0.1)
    print("\n[5c] 전국 γ 주간/야간/진폭 (§4-1)")
    ok &= chk("전국γ 주간", nat_gamma[0], REPORT_NATIONAL_GAMMA[0], 0.05)
    ok &= chk("전국γ 야간", nat_gamma[1], REPORT_NATIONAL_GAMMA[1], 0.05)
    ok &= chk("전국γ 진폭", nat_gamma[2], REPORT_NATIONAL_GAMMA[2], 0.05)

    print("\n[6a] 공변량 MAE~설명변수 R² (§2-6/§5-3/§6, 참고용)")
    name_map = {
        "delta_z": "MAE~ΔZ", "abs_gap_elev": "MAE~|gap_elev|", "abs_elev": "MAE~절대고도",
        "adw_alpha": "MAE~ADW α", "octant_occ": "MAE~팔분면", "quadrant_occ": "MAE~사분면",
        "max_angular_gap": "MAE~최대방위공백", "nearest_dist": "MAE~최근접거리",
        "neighbor_elev_std": "MAE~이웃고도std",
    }
    r2_key = {
        "delta_z": "delta_z", "abs_gap_elev": "gap_elev", "abs_elev": "abs_elev",
        "adw_alpha": "adw_alpha", "octant_occ": "octant_occ", "quadrant_occ": "quadrant_occ",
        "max_angular_gap": "max_angular_gap", "nearest_dist": "nearest_dist",
        "neighbor_elev_std": "neighbor_elev_std",
    }
    for c, label in name_map.items():
        ok &= chk(label, cov_r2[c], REPORT_R2[r2_key[c]], 0.1)
    print(f"  {'(참고) MAE~gap_elev 부호형':34s} 재계산={cov_r2['gap_elev_signed']:8.3f}  "
          f"— |gap|=0.766 재현엔 크기형 사용, 부호형은 대조용")

    print("\n[6b] BIAS ~ ΔZ 회귀 (§6)")
    ok &= chk("BIAS_none~ΔZ slope", bias_reg["bias_none"][0], REPORT_BIAS_REG["bias_none"][0], 0.05)
    ok &= chk("BIAS_none~ΔZ R²", bias_reg["bias_none"][1], REPORT_BIAS_REG["bias_none"][1], 0.02)
    ok &= chk("BIAS_fixed~ΔZ slope", bias_reg["bias_fixed"][0], REPORT_BIAS_REG["bias_fixed"][0], 0.05)
    ok &= chk("BIAS_fixed~ΔZ R²", bias_reg["bias_fixed"][1], REPORT_BIAS_REG["bias_fixed"][1], 0.02)

    print("\n" + "=" * 74)
    print("  => 전부 일치 (허용오차 내)" if ok else "  => 불일치 존재 (아래 MISMATCH 확인)")
    print("=" * 74)
    return ok


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def fig_decomposition(decomp, path):
    setup_korean_font()
    names = {"160": "부산(레)", "554": "미시령", "871": "윗세오름", "904": "사상", "768": "곡성"}
    d = decomp.sort_values("actual_MAE", ascending=False)
    labels = [f"{r.STN}\n{names.get(r.STN, '')}" for r in d.itertuples()]
    x = np.arange(len(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 좌: 두 인자
    ax1.bar(x - 0.2, d["delta_gamma"], width=0.4, color="#d1913c", label="① |Δγ| = mean|γ_eff+6.5| (°C/km)")
    ax1b = ax1.twinx()
    ax1b.bar(x + 0.2, d["abs_dz_km"], width=0.4, color="#4aa564", label="② |ΔZ| 외삽거리 (km)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("① 감률오차 |Δγ| (°C/km)", color="#a5701f")
    ax1b.set_ylabel("② 외삽거리 |ΔZ| (km)", color="#2f6b42")
    ax1.set_title("두 인자: 물리(감률오차) × 기하(외삽거리)", fontsize=12, fontweight="bold")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.2)

    # 우: 곱 vs 실제 MAE
    ax2.bar(x, d["calc_MAE"], width=0.55, color="#8aa8d6",
            label="계산 MAE = |Δγ| × |ΔZ| (항등식)")
    ax2.plot(x, d["actual_MAE"], "o-", color="#d62728", markersize=9,
             label="실제 MAE (analysis_summary)", zorder=3)
    for xi, r in zip(x, d.itertuples()):
        ax2.annotate(f"{r.delta_gamma:.2f}×{r.abs_dz_km:.3f}", (xi, r.calc_MAE),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 3),
                     textcoords="offset points")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("MAE (°C)")
    ax2.set_title("MAE = |Δγ| × |ΔZ| : 계산 vs 실제", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.2)

    fig.suptitle("보정 후 MAE의 물리·기하 분해 (§3 항등식, |ΔZ|>0.10km 5개)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_local_gamma_160_871(gamma_df, path):
    setup_korean_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 좌: 시간대별 γ_eff 평균 (부호 반대 위상)
    for sid, name, color in [("160", "부산(레) 512m", "#d62728"), ("871", "윗세오름 1676m", "#2d6cdf")]:
        g = gamma_df[gamma_df["STN"] == sid]
        hourly = g.groupby("hour")["gamma_eff"].mean()
        ax1.plot(hourly.index, hourly.values, "o-", color=color, label=f"{sid} {name}")
    ax1.axhline(GAMMA_APPLIED, color="gray", ls="--", lw=1, label="적용값 -6.5")
    ax1.axvspan(0, 5, color="#3b4a66", alpha=0.08)
    ax1.axvspan(12, 15, color="#e8a13b", alpha=0.10)
    ax1.set_xlabel("시각 (시)")
    ax1.set_ylabel("국지 γ_eff (°C/km)")
    ax1.set_title("시간대별 국지 감률 — 160·871 일변화 위상 정반대", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # 우: 주간/야간 막대 대비
    rows = []
    for sid in ["160", "871"]:
        g = gamma_df[gamma_df["STN"] == sid]
        day = g[g["hour"].between(*DAY_HOURS)]["gamma_eff"].mean()
        night = g[g["hour"].between(*NIGHT_HOURS)]["gamma_eff"].mean()
        rows.append((sid, day, night))
    xb = np.arange(2)
    ax2.bar(xb - 0.2, [r[1] for r in rows], width=0.4, color="#e8a13b", label="주간(12–15시)")
    ax2.bar(xb + 0.2, [r[2] for r in rows], width=0.4, color="#3b4a66", label="야간(00–05시)")
    for i, (sid, day, night) in enumerate(rows):
        ax2.annotate(f"{day:.2f}", (i - 0.2, day), ha="center", va="top", fontsize=9)
        ax2.annotate(f"{night:.2f}", (i + 0.2, night), ha="center", va="top", fontsize=9)
    ax2.axhline(GAMMA_APPLIED, color="gray", ls="--", lw=1, label="적용값 -6.5")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["160 부산(레)\n주간 급, 야간 완", "871 윗세오름\n주간 완, 야간 급"], fontsize=10)
    ax2.set_ylabel("국지 γ_eff (°C/km)")
    ax2.set_title("주야 국지 γ 부호 반전 (전국 단일 γ 불가의 근거)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("국지 감률의 지점 간 위상 대립 — 160 vs 871 (§4-2)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_bias_signreversal_160(raw_fixed, path):
    setup_korean_font()
    g = raw_fixed[raw_fixed["STN"] == "160"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 좌: 계절·주야 BIAS 막대 (부호별 색)
    cats = ["연평균", "봄(4·5)", "겨울(12·1)", "주간(12–15)", "야간(00–05)"]
    vals = [
        g["ERROR"].mean(),
        g[g["month"].isin(SPRING_MONTHS)]["ERROR"].mean(),
        g[g["month"].isin(WINTER_MONTHS)]["ERROR"].mean(),
        g[g["hour"].between(*DAY_HOURS)]["ERROR"].mean(),
        g[g["hour"].between(*NIGHT_HOURS)]["ERROR"].mean(),
    ]
    colors = ["#d62728" if v >= 0 else "#2d6cdf" for v in vals]
    ax1.bar(cats, vals, color=colors)
    for i, v in enumerate(vals):
        ax1.annotate(f"{v:+.2f}", (i, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("BIAS = mean(예측-관측) (°C)")
    ax1.set_title("160: 계절·주야로 BIAS 부호 반전\n(연평균 +0.18 작지만 MAE 1.14 최대)",
                  fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(True, axis="y", alpha=0.25)

    # 우: 월별 BIAS (연중 부호 뒤집힘)
    monthly = g.groupby("month")["ERROR"].mean()
    mcolors = ["#d62728" if v >= 0 else "#2d6cdf" for v in monthly.values]
    ax2.bar(monthly.index, monthly.values, color=mcolors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("월")
    ax2.set_ylabel("월별 BIAS (°C)")
    ax2.set_xticks(range(1, 13))
    ax2.set_title("160: 월별 BIAS — 연중 부호가 계속 뒤집힘\n(상쇄되어 연평균↓, MAE엔 누적)",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("감률오차의 부호 반전 서명 — 160 부산(레) (§2-2)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_night_inversion(night_inv, factors, path):
    setup_korean_font()
    names = {"904": "사상", "768": "곡성", "160": "부산(레)", "554": "미시령", "871": "윗세오름"}
    d = night_inv.copy()
    d["elev"] = [factors.loc[s, "abs_elev"] for s in d.index]
    d = d.sort_values("elev")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(d["elev"], d["night_inv_pct"], s=140, c="#2d6cdf", zorder=3)
    ax.plot(d["elev"], d["night_inv_pct"], "-", color="#2d6cdf", alpha=0.4, zorder=2)
    for s, r in d.iterrows():
        ax.annotate(f"{s} {names.get(s, '')}\n{r.night_inv_pct:.1f}%",
                    (r.elev, r.night_inv_pct), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("관측소 해발고도 (m, 로그축)")
    ax.set_ylabel("야간(00–05) 역전 비율 — γ_eff > 0 시각 비율 (%)")
    ax.set_title("저지대일수록 야간 역전 빈발 (§4-6)\n선형 감률이 표현 못 하는 성분 → 고정값 조정으로 개선 불가",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mae_vs_covariates(factors, mae_fixed, cov_r2, path):
    setup_korean_font()
    names = {"904": "사상", "919": "창녕", "648": "장동", "768": "곡성",
             "537": "임계", "160": "부산(레)", "554": "미시령", "871": "윗세오름"}
    specs = [
        ("delta_z", "ΔZ 외삽거리 (km)", cov_r2["delta_z"], "강함"),
        ("abs_gap_elev", "|gap_elev| = |target-최근접이웃 고도| (m)", cov_r2["abs_gap_elev"], "강함(비표준 보조)"),
        ("abs_elev", "관측소 절대고도 (m)", cov_r2["abs_elev"], "약함(871 반례)"),
        ("adw_alpha", "ADW 방위가중 α (0~2)", cov_r2["adw_alpha"], "없음(904 반례)"),
    ]
    y = {s: mae_fixed[s] for s in factors.index}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (col, xlabel, r2v, verdict) in zip(axes.ravel(), specs):
        xs = factors[col]
        ax.scatter(xs, [y[s] for s in factors.index], s=90, c="#2d6cdf", zorder=3)
        for s in factors.index:
            ax.annotate(f"{s} {names.get(s, '')}", (factors.loc[s, col], y[s]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        # 회귀선
        xv = factors[col].to_numpy()
        yv = np.array([y[s] for s in factors.index])
        m = ~np.isnan(xv)
        if m.sum() >= 2:
            a, b = np.polyfit(xv[m], yv[m], 1)
            xr = np.linspace(xv[m].min(), xv[m].max(), 50)
            ax.plot(xr, a * xr + b, "--", color="#d62728", alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("보정 후 MAE (°C)")
        ax.set_title(f"MAE ~ {xlabel.split(' ')[0]}   R²={r2v:.3f}  [{verdict}]",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
    fig.suptitle("보정 후 MAE vs 설명변수 — 연직(ΔZ·gap) 강함 / 수평방위(α) 없음\n"
                 "(n=8 편의표본, R²는 참고용 — 인과 단정 아님)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_station_meta()
    targets = load_targets()
    print(f"분석 대상 {len(targets)} station (530 태하 제외): {list(targets['지점'])}")

    print("\n[1/6] ΔZ·이웃 기하 (4계절 trace 평균, filter_valid_station_meta) ...")
    geom = compute_delta_z_and_geometry(meta, targets)
    delta_z = geom["delta_z"].to_dict()

    print("[2/6] raw 로드 (fixed_lapse / none, 12개월) ...")
    raw_fixed = load_raw("fixed_lapse")
    raw_none = load_raw("none")
    summ = load_summary_station()
    mae_fixed_summary = summ["fixed_lapse"]["MAE"]
    bias_none_summary = summ["none"]["BIAS"]
    bias_fixed_summary = summ["fixed_lapse"]["BIAS"]
    # 재계산 MAE (raw 기준) — summary와 교차검증
    mae_fixed_recomp = {
        sid: float(g["ABS_ERROR"].mean()) for sid, g in raw_fixed.groupby("STN")
    }

    print("[3/6] γ_eff·오차분해·야간역전 ...")
    gamma_df = compute_gamma_eff(raw_none, delta_z)
    decomp = decomposition_table(gamma_df, delta_z, mae_fixed_summary)
    night_inv = night_inversion(gamma_df, delta_z)

    print("[4/6] BIAS·국지 γ 통계 ...")
    bias = bias_table(raw_fixed)
    local_g = local_gamma_stats(gamma_df, delta_z)
    nat_gamma = national_gamma()
    mb160 = month_band_gamma_160(gamma_df)

    print("[5/6] 공변량 회귀 ...")
    cov_r2 = covariate_r2(geom, mae_fixed_summary)
    stns = list(geom.index)
    bias_reg = {
        "bias_none": (
            slope_of([delta_z[s] for s in stns], [bias_none_summary[s] for s in stns]),
            r2_of([delta_z[s] for s in stns], [bias_none_summary[s] for s in stns]),
        ),
        "bias_fixed": (
            slope_of([delta_z[s] for s in stns], [bias_fixed_summary[s] for s in stns]),
            r2_of([delta_z[s] for s in stns], [bias_fixed_summary[s] for s in stns]),
        ),
    }

    # ---- station별 요인표 CSV ----
    factors = (
        targets.rename(columns={"지점": "STN"}).set_index("STN")
        .join(geom.drop(columns=["abs_elev"]))
        .join(bias)
        .join(local_g)
        .join(night_inv[["night_inv_pct", "night_gamma_median"]])
    )
    factors["MAE_fixed"] = factors.index.map(mae_fixed_summary)
    factors["BIAS_none"] = factors.index.map(bias_none_summary)
    factors = factors.reset_index()
    factors.to_csv(FACTORS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n요인표 저장: {FACTORS_CSV}  ({len(factors)} station)")

    # ---- 재현성 검증 ----
    ok = verify(delta_z, decomp, night_inv, local_g, nat_gamma, cov_r2,
                bias_reg, mae_fixed_recomp, mae_fixed_summary)

    # ---- 160 월×시간대 γ표 ----
    print("\n[160 월×시간대 국지 γ 중앙값 (°C/km, §4-4)]")
    print(mb160.round(1).to_string())

    # ---- 겨울야간−여름야간 MAE 표 ----
    print("\n[겨울야간(DJF)−여름야간(JJA) MAE (§2-3)]")
    show = bias[["MAE_wnight", "MAE_snight", "MAE_wnight_minus_snight", "BIAS_night"]].round(2)
    print(show.sort_values("MAE_wnight_minus_snight", ascending=False).to_string())

    print("\n[6/6] 그림 생성 ...")
    fig_decomposition(decomp, VIZ_DIR / "prior_mae_decomposition.png")
    fig_local_gamma_160_871(gamma_df, VIZ_DIR / "prior_local_gamma_160_vs_871.png")
    fig_bias_signreversal_160(raw_fixed, VIZ_DIR / "prior_bias_signreversal_160.png")
    fig_night_inversion(night_inv, geom, VIZ_DIR / "prior_night_inversion.png")
    fig_mae_vs_covariates(geom, mae_fixed_summary, cov_r2, VIZ_DIR / "prior_mae_vs_covariates.png")
    for p in ["prior_mae_decomposition", "prior_local_gamma_160_vs_871",
              "prior_bias_signreversal_160", "prior_night_inversion", "prior_mae_vs_covariates"]:
        print(f"  저장: {VIZ_DIR / (p + '.png')}")

    print("\n완료." + ("  [재현성 전부 일치]" if ok else "  [불일치 존재 — 위 검증 확인]"))


if __name__ == "__main__":
    main()
