"""유동적 lapse rate 추정·적용 모듈 (Phase B).

두 가지 유동적 γ를 제공한다 (계획: .omc/plans/phase-b-variable-lapse-rate.md):

- monthly_lapse: 시각별 전국 (고도, 기온) 단순선형회귀 γ(t)를 월별 중앙값으로 집계한
  12개 테이블. 적용 시 해당 월 γ를 기존 fixed_lapse 경로(0m 기준면 경유)에 주입.
  근거: Rolland(2003) 월별 회귀, Stahl(2006) SLR 계열, 자체 2024 실측(3~4월 -3, 12월 -7.7).

- band_lapse: 고도구간별로 γ가 다른 계단형 프로파일. 구간 내 회귀는 불안정하므로
  (자체 탐색: 0-100m 구간 σ±13.6) 구간별 평균기온-평균고도(binned centroid)의
  인접 구간 간 기울기로 추정. 보정은 0m→z 경로적분(연속 C(z)).
  근거: Gao(2012) Method IV 2층 분리, PRISM 수직층, 최광용(2011) 고도 비균일.

γ 추정 회귀에서 batch target station들은 제외한다(in-sample 완화).
γ는 [-0.012, +0.003] °C/m 로 클램프(역전 허용, 비물리값 차단).

불변식: 모든 구간 γ가 동일하면 band 경로적분은 fixed_lapse와 수학적으로 완전 동일.
기존 elevation_correction.py 는 수정하지 않는다.
"""

import numpy as np
import pandas as pd

from weather.elevation_correction import FIXED_LAPSE_RATE
from weather.find_stations import filter_valid_station_meta

ELEVATION_COLUMN = "노장해발고도(m)"

# γ 클램프 (°C/m): 강한 역전(+3°C/km)까지 허용, 비물리적 극단값 차단
GAMMA_MIN = -0.012
GAMMA_MAX = 0.003

# 시각별 전국 회귀 최소 표본 (Lute 2021: 소표본 회귀 위험)
MIN_REGRESSION_STATIONS = 100

# 고도구간 (binned centroid 추정용). 마지막 구간은 상한 없음.
BAND_EDGES = [0.0, 100.0, 300.0, 600.0, float("inf")]
MIN_BIN_COUNT = 10  # 구간별 최소 관측소 수 (미달 구간은 그 시각에서 제외)

MONTHLY_TABLE_PATH = "WeatherData_AWS/interpolation_batch/monthly_lapse_table.csv"
BAND_TABLE_PATH = "WeatherData_AWS/interpolation_batch/band_lapse_table.csv"
HOURLY_GAMMA_PATH = "WeatherData_AWS/interpolation_batch/hourly_gamma_2024.csv"


def clamp_gamma(gamma):
    if pd.isna(gamma):
        return np.nan
    return min(max(gamma, GAMMA_MIN), GAMMA_MAX)


# ---------------------------------------------------------------------------
# 1) 시각별 γ 추정 (전국 회귀 + 구간 binned 기울기)
# ---------------------------------------------------------------------------

def _station_elevations(meta, when):
    """해당 시각에 유효한 station의 고도 Series(지점 -> m)를 반환."""
    valid = filter_valid_station_meta(meta, when)
    elev = pd.to_numeric(valid.set_index("지점")[ELEVATION_COLUMN], errors="coerce")
    return elev[~elev.index.duplicated(keep="first")]


def estimate_gamma_at_time(values_at_time, elevations, exclude_stations=()):
    """한 시각의 전국 회귀 γ(°C/m)와 구간 기울기들을 추정한다.

    반환: (reg_row dict | None, band_rows list[dict])
      reg_row  = {"n", "gamma", "corr"}          — 표본 부족 시 None
      band_rows= [{"pair", "z_lo", "z_hi", "gamma"}] — centroid 쌍별 기울기
    """
    df = pd.DataFrame({
        "ta": pd.to_numeric(values_at_time, errors="coerce"),
        "z": elevations.reindex(values_at_time.index.astype(str)).values,
    })
    df = df.dropna()
    df = df[df["z"] >= 0]
    if exclude_stations:
        df = df[~df.index.astype(str).isin({str(s) for s in exclude_stations})]

    reg_row = None
    if len(df) >= MIN_REGRESSION_STATIONS and df["z"].std() > 0:
        slope = np.polyfit(df["z"], df["ta"], 1)[0]
        corr = float(np.corrcoef(df["z"], df["ta"])[0, 1])
        reg_row = {"n": len(df), "gamma": clamp_gamma(slope), "corr": corr}

    # 구간별 centroid (평균고도, 평균기온) → 인접 구간 간 기울기
    centroids = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        sub = df[(df["z"] >= lo) & (df["z"] < hi)]
        if len(sub) >= MIN_BIN_COUNT:
            centroids.append((float(sub["z"].mean()), float(sub["ta"].mean())))

    band_rows = []
    for i in range(len(centroids) - 1):
        (z0, t0), (z1, t1) = centroids[i], centroids[i + 1]
        if z1 - z0 <= 0:
            continue
        band_rows.append({
            "pair": i,
            "z_lo": z0,
            "z_hi": z1,
            "gamma": clamp_gamma((t1 - t0) / (z1 - z0)),
        })
    return reg_row, band_rows


def estimate_gamma_series(data, meta, exclude_stations=()):
    """wide-format 데이터 전체 시각에 대해 γ를 추정한다.

    반환: (gamma_df, band_df)
      gamma_df: TM, n, gamma, corr   (시각별 전국 회귀)
      band_df : TM, pair, z_lo, z_hi, gamma (시각별 구간 기울기)
    """
    gamma_rows, band_rows = [], []
    elev_cache = {}

    for tm, values_at_time in data.iterrows():
        date_key = pd.Timestamp(tm).normalize()
        if date_key not in elev_cache:
            elev_cache[date_key] = _station_elevations(meta, date_key)

        reg, bands = estimate_gamma_at_time(
            values_at_time, elev_cache[date_key], exclude_stations
        )
        if reg is not None:
            gamma_rows.append({"TM": tm, **reg})
        for b in bands:
            band_rows.append({"TM": tm, **b})

    return pd.DataFrame(gamma_rows), pd.DataFrame(band_rows)


# ---------------------------------------------------------------------------
# 2) 테이블 집계 (중앙값 — 이상 시각에 견고)
# ---------------------------------------------------------------------------

def build_monthly_table(gamma_df):
    """시각별 γ를 월별 중앙값으로 집계한 DataFrame(month, gamma, n_hours)을 만든다."""
    df = gamma_df.copy()
    df["month"] = pd.to_datetime(df["TM"]).dt.month
    agg = df.groupby("month")["gamma"].agg(["median", "count"]).reset_index()
    agg.columns = ["month", "gamma", "n_hours"]

    # 누락 월은 고정 γ로 fallback (정상 데이터에서는 발생하지 않음)
    missing = set(range(1, 13)) - set(agg["month"])
    if missing:
        agg = pd.concat([
            agg,
            pd.DataFrame([
                {"month": m, "gamma": FIXED_LAPSE_RATE, "n_hours": 0} for m in missing
            ]),
        ]).sort_values("month").reset_index(drop=True)
    return agg


def build_band_table(band_df):
    """시각별 구간 기울기를 pair별 중앙값으로 집계해 계단형 γ 프로파일을 만든다.

    프로파일 = binned 프로파일의 조각별 선형보간과 동치:
      구간 경계는 centroid 중앙값, 최하단(0~c1)은 첫 기울기,
      최상단(c_last~inf)은 마지막 기울기로 연장.
    반환 DataFrame: z_lo, z_hi, gamma  (0부터 inf까지 연속 커버)
    """
    agg = band_df.groupby("pair").agg(
        z_lo=("z_lo", "median"),
        z_hi=("z_hi", "median"),
        gamma=("gamma", "median"),
        n_hours=("gamma", "count"),
    ).sort_index()

    if len(agg) == 0:
        # fallback: 단일 고정 γ 구간
        return pd.DataFrame([{"z_lo": 0.0, "z_hi": float("inf"),
                              "gamma": FIXED_LAPSE_RATE, "n_hours": 0}])

    rows = []
    boundaries = list(agg["z_hi"][:-1])  # 내부 경계 = centroid 중앙값들
    lows = [0.0] + boundaries
    highs = boundaries + [float("inf")]
    for (lo, hi), (_, r) in zip(zip(lows, highs), agg.iterrows()):
        rows.append({"z_lo": lo, "z_hi": hi, "gamma": r["gamma"],
                     "n_hours": int(r["n_hours"])})
    return pd.DataFrame(rows)


def save_table(table, path):
    out = table.copy()
    out["gamma_C_per_km"] = out["gamma"] * 1000
    out.to_csv(path, index=False, encoding="utf-8-sig")


def load_monthly_table(path=MONTHLY_TABLE_PATH):
    """CSV -> {month(int): gamma(°C/m)} dict."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {int(r["month"]): float(r["gamma"]) for _, r in df.iterrows()}


def load_band_table(path=BAND_TABLE_PATH):
    """CSV -> DataFrame(z_lo, z_hi, gamma). z_hi 결측/문자 'inf' 허용."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["z_hi"] = df["z_hi"].apply(
        lambda v: float("inf") if pd.isna(v) or str(v).strip() in ("inf", "") else float(v)
    )
    return df[["z_lo", "z_hi", "gamma"]].copy()


def resolve_monthly_gamma(tm, monthly_table):
    """시각 tm의 월 γ. 테이블 누락 시 고정 γ fallback."""
    month = pd.Timestamp(tm).month
    if monthly_table is None:
        return FIXED_LAPSE_RATE
    return monthly_table.get(month, FIXED_LAPSE_RATE)


# ---------------------------------------------------------------------------
# 3) band 경로적분 보정
# ---------------------------------------------------------------------------

def cumulative_offset(elevation_m, band_table):
    """C(z) = ∫_0^z γ(z')dz'  (°C). z<0이면 첫 구간 γ로 선형 연장.

    band_table이 [0, inf)를 연속 커버한다고 가정한다 (build_band_table 보장).
    """
    z = elevation_m
    if pd.isna(z):
        return np.nan

    if z < 0:
        return float(band_table.iloc[0]["gamma"]) * z

    total = 0.0
    for _, r in band_table.iterrows():
        lo, hi, g = float(r["z_lo"]), float(r["z_hi"]), float(r["gamma"])
        if z <= lo:
            break
        seg_hi = min(z, hi)
        total += g * (seg_hi - lo)
        if z <= hi:
            break
    return total


def band_to_reference(value, source_elevation_m, band_table):
    """이웃 기온을 0m 기준면 값으로 환산 (경로적분). 결측 시 원본 반환.

    T_ref = T_source - C(z_source)   (모든 γ 동일 시 fixed_lapse와 동일)
    """
    if pd.isna(value) or pd.isna(source_elevation_m) or band_table is None:
        return value
    return value - cumulative_offset(source_elevation_m, band_table)


def band_from_reference(value, target_elevation_m, band_table):
    """0m 기준면 보간값을 target 고도로 되돌림. 결측 시 원본 반환."""
    if pd.isna(value) or pd.isna(target_elevation_m) or band_table is None:
        return value
    return value + cumulative_offset(target_elevation_m, band_table)
