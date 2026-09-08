"""Stage 2 — 파생·집계·진단 (계획서 §4-3/§4-6/§4-7/§1).

Stage 1 SQLite raw(9.3M행)에서:
  ① 파생: ΔZ(t)=(ERR_fixed−ERR_none)/(−6.5), γ_eff(t)=−ERR_none/ΔZ (게이트 |ΔZ|≥0.10km).
  ② 집계표 analysis_summary.csv: axis(overall/season/month/elev_band/station)×method의 n·MAE·RMSE·BIAS.
  ③ 지점 진단 station_factors.csv: MAE/RMSE/BIAS, Willmott PSE·RMSE_s/u·d_r, γ_eff(median·IQR·
     주야), 야간역전%, RI(개선율), mean ΔZ, 고도, valid_ratio.
  ④ 가설 재검정(n=8→549): H1 MAE~|ΔZ|, 절대고도 무관 확인.

알고리즘 무수정, raw 읽기 전용. 메모리 안전(지점별 인덱스 조회 루프).
출력: WeatherData_AWS/interpolation_batch/fullnet/ (소형 CSV, OneDrive 유지).
"""

from project_paths import LOO_DB, LOO_DIR

import os
import sqlite3

import numpy as np
import pandas as pd

from find_stations import filter_valid_station_meta, load_station_meta
from idw import ELEVATION_COLUMN

DB = str(LOO_DB)
OUT = "WeatherData_AWS/interpolation_batch/fullnet"
GAMMA = -6.5           # °C/km (fixed_lapse)
DZ_MIN_KM = 0.10       # γ_eff 게이트 (작은 ΔZ 발산 방지, 기존 관행)
ELEV_BANDS = [(0, 100), (100, 500), (500, float("inf"))]
REF_DATE = "2024-06-15"


def season_of(month):
    m = int(month)
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall"}[m]


def elev_band(e):
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return "unknown"
    for lo, hi in ELEV_BANDS:
        if lo <= e < hi:
            return f"{lo}-{hi if hi != float('inf') else 'inf'}"
    return "unknown"


def _scores_from_sums(n, s_err, s_err2, s_abs):
    mae = s_abs / n
    rmse = np.sqrt(s_err2 / n)
    bias = s_err / n
    return mae, rmse, bias


def willmott(observed, predicted):
    """Willmott 분해: 예측을 관측에 OLS 회귀(P̂=a+bO). PSE=MSE_s/MSE, d_r(2012)."""
    O = np.asarray(observed, float); P = np.asarray(predicted, float)
    n = len(O)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    b, a = np.polyfit(O, P, 1)          # P ≈ a + b·O
    Phat = a + b * O
    mse = np.mean((P - O) ** 2)
    mse_s = np.mean((Phat - O) ** 2)
    mse_u = np.mean((P - Phat) ** 2)
    pse = mse_s / mse if mse > 0 else np.nan
    # d_r (refined index of agreement, Willmott 2012), c=2
    num = np.sum(np.abs(P - O))
    den = 2 * np.sum(np.abs(O - O.mean()))
    if den == 0:
        dr = np.nan
    elif num <= den:
        dr = 1 - num / den
    else:
        dr = den / num - 1
    return pse, np.sqrt(mse_s), np.sqrt(mse_u), dr


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(DB)

    # 고도 lookup (기준일 유효행)
    meta = load_station_meta()
    vref = filter_valid_station_meta(meta, REF_DATE)
    elev = {str(r["지점"]): r[ELEVATION_COLUMN] for _, r in vref.iterrows()}
    # 기준일 미유효 지점은 아무 행 고도로 fallback
    for stn in meta["지점"].unique():
        elev.setdefault(str(stn), meta[meta["지점"] == str(stn)][ELEVATION_COLUMN].iloc[-1])

    # --- ② 집계: (STN,MONTH,METHOD) 합계 SQL → 롤업 --------------------
    base = pd.read_sql(
        "SELECT STN, MONTH, METHOD, COUNT(*) n, SUM(ERROR) s_err, "
        "SUM(ERROR*ERROR) s_err2, SUM(ABS_ERROR) s_abs "
        "FROM raw WHERE interpolable=1 GROUP BY STN, MONTH, METHOD", con)
    base["STN"] = base["STN"].astype(str)
    base["season"] = base["MONTH"].str[4:].astype(int).apply(season_of)
    base["elev_m"] = base["STN"].map(elev)
    base["band"] = base["elev_m"].apply(elev_band)

    def rollup(keys, label_cols):
        g = base.groupby(keys, as_index=False).agg(
            n=("n", "sum"), s_err=("s_err", "sum"),
            s_err2=("s_err2", "sum"), s_abs=("s_abs", "sum"))
        g["MAE"], g["RMSE"], g["BIAS"] = _scores_from_sums(
            g["n"], g["s_err"], g["s_err2"], g["s_abs"])
        g["axis"] = label_cols
        return g

    summ = pd.concat([
        rollup(["METHOD"], "overall"),
        rollup(["season", "METHOD"], "season"),
        rollup(["MONTH", "METHOD"], "month"),
        rollup(["band", "METHOD"], "elev_band"),
        rollup(["STN", "METHOD"], "station"),
    ], ignore_index=True)
    for c in ["MAE", "RMSE", "BIAS"]:
        summ[c] = summ[c].round(5)
    summ.to_csv(os.path.join(OUT, "analysis_summary.csv"),
                index=False, encoding="utf-8-sig")

    # overall 검증용
    ov = summ[summ["axis"] == "overall"].set_index("METHOD")
    print("=== overall MAE/RMSE/BIAS ===")
    print(ov[["n", "MAE", "RMSE", "BIAS"]].to_string())

    # --- ③ 지점 진단 (지점별 인덱스 조회 루프) ------------------------
    stations = sorted(base["STN"].unique())
    vsum = pd.read_csv(str(LOO_DIR / "station_valid_summary.csv"),
                       encoding="utf-8-sig", dtype={"STN": str})
    vratio = dict(zip(vsum["STN"], vsum["valid_ratio"]))

    rows = []
    for stn in stations:
        n_df = pd.read_sql(
            "SELECT TM, OBSERVED, PREDICTED, ERROR FROM raw "
            "WHERE STN=? AND METHOD='none' AND interpolable=1", con, params=[stn])
        f_df = pd.read_sql(
            "SELECT TM, OBSERVED, PREDICTED, ERROR FROM raw "
            "WHERE STN=? AND METHOD='fixed_lapse' AND interpolable=1", con, params=[stn])
        if len(f_df) < 3:
            continue
        m = n_df.merge(f_df, on="TM", suffixes=("_none", "_fix"))
        err_none = m["ERROR_none"].to_numpy()
        err_fix = m["ERROR_fix"].to_numpy()
        obs = m["OBSERVED_fix"].to_numpy()
        pred = m["PREDICTED_fix"].to_numpy()

        mae_f = np.mean(np.abs(err_fix)); rmse_f = np.sqrt(np.mean(err_fix**2))
        bias_f = np.mean(err_fix)
        mae_n = np.mean(np.abs(err_none))
        ri = (mae_n - mae_f) / mae_n if mae_n > 0 else np.nan
        pse, rmse_s, rmse_u, dr = willmott(obs, pred)

        # 파생 ΔZ, γ_eff (게이트)
        dz = (err_fix - err_none) / GAMMA          # km
        gated = np.abs(dz) >= DZ_MIN_KM
        gamma_eff = np.where(gated, -err_none / dz, np.nan)
        ge = gamma_eff[~np.isnan(gamma_eff)]
        mean_absdz = np.mean(np.abs(dz))
        mean_dz = np.mean(dz)

        # 주야 분리 + 야간역전
        hours = pd.to_datetime(m["TM"]).dt.hour.to_numpy()
        night = (hours >= 21) | (hours <= 5)
        day = (hours >= 9) & (hours <= 17)
        ge_night = gamma_eff[night & gated]; ge_night = ge_night[~np.isnan(ge_night)]
        ge_day = gamma_eff[day & gated]; ge_day = ge_day[~np.isnan(ge_day)]
        # 역전 = γ_eff>0 (실제 기온이 고도와 함께 상승)
        night_gated = gamma_eff[night & gated]; night_gated = night_gated[~np.isnan(night_gated)]
        inv_ratio = np.mean(night_gated > 0) if len(night_gated) else np.nan

        rows.append({
            "STN": stn, "elev_m": elev.get(stn), "n_hours": len(m),
            "valid_ratio": vratio.get(stn),
            "MAE": round(mae_f, 5), "RMSE": round(rmse_f, 5), "BIAS": round(bias_f, 5),
            "MAE_none": round(mae_n, 5), "RI": round(ri, 4) if not np.isnan(ri) else np.nan,
            "PSE": round(pse, 4) if not np.isnan(pse) else np.nan,
            "RMSE_s": round(rmse_s, 4) if not np.isnan(rmse_s) else np.nan,
            "RMSE_u": round(rmse_u, 4) if not np.isnan(rmse_u) else np.nan,
            "d_r": round(dr, 4) if not np.isnan(dr) else np.nan,
            "mean_dz_km": round(mean_dz, 4), "mean_absdz_km": round(mean_absdz, 4),
            "gamma_eff_median": round(np.median(ge), 3) if len(ge) else np.nan,
            "gamma_eff_iqr": round(np.subtract(*np.percentile(ge, [75, 25])), 3) if len(ge) else np.nan,
            "gamma_eff_day": round(np.median(ge_day), 3) if len(ge_day) else np.nan,
            "gamma_eff_night": round(np.median(ge_night), 3) if len(ge_night) else np.nan,
            "night_inversion_ratio": round(inv_ratio, 4) if not np.isnan(inv_ratio) else np.nan,
            "n_gated": int(gated.sum()),
        })
    sf = pd.DataFrame(rows)
    sf.to_csv(os.path.join(OUT, "station_factors.csv"), index=False, encoding="utf-8-sig")
    print(f"\nstation_factors: {len(sf)} 지점")

    # --- ④ 가설 재검정 (n=8 → 전체) ----------------------------------
    def r2(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
        if len(x) < 3:
            return np.nan, np.nan, 0
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        return (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, b, len(x)

    # ΔZ 의미있는 지점만 (mean_absdz 충분)
    valid = sf[sf["mean_absdz_km"] >= DZ_MIN_KM]
    r2_dz, slope_dz, n_dz = r2(valid["mean_absdz_km"], valid["MAE"])
    r2_elev, slope_elev, n_elev = r2(sf["elev_m"].abs(), sf["MAE"])
    print("\n" + "=" * 56)
    print("가설 재검정 (n=8 편의표본 → 전체 549)")
    print("=" * 56)
    print(f"H1  MAE ~ mean|ΔZ| : R²={r2_dz:.3f}  기울기={slope_dz:.3f}°C/km  (n={n_dz})")
    print(f"    (n=8 참고값 R²=0.698)")
    print(f"대조 MAE ~ 절대고도 : R²={r2_elev:.3f}  (n={n_elev})  '절대고도 무관' 명제 확인")

    lines = [
        "# Stage 2 가설 재검정 요약 (전체 549 station)",
        "",
        f"- **H1: MAE ~ mean|ΔZ|** — R²={r2_dz:.3f}, 기울기={slope_dz:.3f}°C/km (n={n_dz} 지점, |ΔZ|≥{DZ_MIN_KM}km).",
        f"  - n=8 편의표본에서는 R²=0.698. 전체망에서 {'재현됨' if r2_dz >= 0.3 else '약화됨'}.",
        f"- **대조: MAE ~ 절대고도** — R²={r2_elev:.3f} (n={n_elev}). '절대고도는 예측변수 아님' 명제 {'유지' if r2_elev < 0.2 else '재검토'}.",
        "- H2(이웃 방위분포 ADW α 무관)는 이웃세트 방위각 계산 필요 → 후속 서브단계.",
        "",
        "상세 수치: `station_factors.csv`, `analysis_summary.csv`.",
    ]
    with open(os.path.join(OUT, "stage2_hypothesis_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    con.close()
    print(f"\n저장: {OUT}/ (analysis_summary.csv, station_factors.csv, stage2_hypothesis_tests.md)")


if __name__ == "__main__":
    main()
