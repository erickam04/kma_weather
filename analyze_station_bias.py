"""전체망 station별 BIAS 증감 원인 분석 runner와 산출물 생성."""

from project_paths import LOO_DIR

import json
import os
import shutil
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from bias_analysis_core import aggregate_bias_groups
from bias_data import (
    load_analysis_station_ids,
    load_neighbor_map,
    load_station_pairs,
    open_readonly_db,
)
from bias_geometry import reconstruct_geometry_for_pairs
from bias_statistics import build_effect_table, fit_standardized_ols
from find_stations import META_FILE, load_station_meta


GAMMA_C_PER_KM = -6.5
DZ_GATE_KM = 0.10
FULLNET = "WeatherData_AWS/interpolation_batch/fullnet"
OUT_DIR = os.path.join(FULLNET, "bias_analysis")
EXCLUDED_PATH = str(LOO_DIR / "excluded_stations.csv")
CANDIDATE_PATH = "WeatherData_AWS/interpolation_batch/stage0/loo_candidates.csv"
BOUNDARY_PATH = "지도데이터/korea_boundary_simplified.json"

INDEPENDENT_PREDICTORS = [
    "mean_dz_geom_km",
    "mean_absdz_geom_km",
    "sd_dz_geom_km",
    "mean_neighbor_count",
    "mean_nearest_distance_km",
    "mean_weighted_distance_km",
    "mean_centroid_offset_km",
    "mean_neighbor_elev_sd_m",
    "n_regimes",
    "mode_cov_pct",
    "elev_m",
    "relief_1km_m",
    "coast_dist_km",
    "is_mountain",
    "is_island",
]
DIAGNOSTIC_PREDICTORS = [
    "gamma_eff_median",
    "gamma_eff_iqr",
    "gamma_eff_day",
    "gamma_eff_night",
    "night_inversion_ratio",
]
OLS_PREDICTORS = [
    "mean_dz_geom_km",
    "sd_dz_geom_km",
    "mean_weighted_distance_km",
    "mean_centroid_offset_km",
    "mean_neighbor_elev_sd_m",
    "n_regimes",
    "elev_m",
    "relief_1km_m",
    "coast_dist_km",
    "is_mountain",
    "is_island",
]
EFFECT_COLUMNS = [
    "HYPOTHESIS",
    "EVIDENCE_LEVEL",
    "ANALYSIS",
    "OUTCOME",
    "PREDICTOR",
    "N",
    "EFFECT_NAME",
    "EFFECT_VALUE",
    "R2",
    "R2_TYPE",
    "VIF",
    "NOTES",
]
SMOKE_ARTIFACTS = [
    "bias_station_factors.csv",
    "bias_monthly.csv",
    "bias_diurnal.csv",
    "bias_run_summary.json",
]
FULL_CSV_ARTIFACTS = [
    "bias_station_factors.csv",
    "bias_monthly.csv",
    "bias_diurnal.csv",
    "bias_excluded_stations.csv",
    "bias_hypothesis_tests.csv",
]
FULL_FIGURE_ARTIFACTS = [
    "bias_phase.png",
    "bias_delta_map.png",
    "bias_delta_distribution.png",
    "bias_temporal.png",
    "bias_physical_factors.png",
    "bias_case_studies.png",
]
REPORT_ARTIFACT = "bias_station_analysis_report.md"
SUMMARY_ARTIFACT = "bias_run_summary.json"
FULL_ARTIFACTS = (
    FULL_CSV_ARTIFACTS
    + FULL_FIGURE_ARTIFACTS
    + [REPORT_ARTIFACT, SUMMARY_ARTIFACT]
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _add_gamma_diagnostics(pairs):
    out = pairs.copy()
    gated = out["dz_geom_km"].abs() >= DZ_GATE_KM
    out["gamma_eff"] = np.where(
        gated, -out["error_none"] / out["dz_geom_km"], np.nan
    )
    hour = out["TM"].dt.hour
    out["is_night"] = (hour >= 21) | (hour <= 5)
    out["is_day"] = (hour >= 9) & (hour <= 17)
    return out


def _annual_diagnostics(pairs):
    ge = pairs["gamma_eff"]
    day = pairs.loc[pairs["is_day"], "gamma_eff"]
    night = pairs.loc[pairs["is_night"], "gamma_eff"]
    return {
        "gamma_eff_median": float(ge.median()),
        "gamma_eff_iqr": float(ge.quantile(0.75) - ge.quantile(0.25)),
        "gamma_eff_day": float(day.median()),
        "gamma_eff_night": float(night.median()),
        "night_inversion_ratio": float((night.dropna() > 0).mean()),
        "n_gamma_gated": int(ge.notna().sum()),
    }


def _write_csvs(frames, out_dir):
    for name, frame in frames.items():
        frame.to_csv(
            os.path.join(out_dir, name), index=False, encoding="utf-8-sig"
        )


def _write_summary(summary, out_dir):
    path = os.path.join(out_dir, SUMMARY_ARTIFACT)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return path


def _base_summary(station_df, excluded_count, max_c, max_b1):
    return {
        "analyzed_stations": int(len(station_df)),
        "excluded_stations": int(excluded_count),
        "paired_hours": int(station_df["n_hours"].sum()),
        "improved_stations": int((station_df["DELTA_ABS_BIAS"] < 0).sum()),
        "worsened_stations": int((station_df["DELTA_ABS_BIAS"] > 0).sum()),
        "unchanged_stations": int((station_df["DELTA_ABS_BIAS"] == 0).sum()),
        "max_identity_residual_c": max_c,
        "max_identity_residual_b1": max_b1,
    }


def _complete_summary(summary, artifacts):
    result = dict(summary)
    result["status"] = "complete"
    result["artifacts"] = list(artifacts)
    return result


def _validate_explicit_stations(stations):
    selected = [str(stn) for stn in stations]
    if not selected:
        raise ValueError("explicit analysis requires at least one station")
    if len(selected) != len(set(selected)):
        raise ValueError("explicit analysis station list contains duplicates")
    return selected


def _collect_analysis_frames(con, selected):
    """선택 station을 한 번에 하나씩 읽어 세 집계표를 만든다."""
    neighbor_map = load_neighbor_map(con)
    meta = load_station_meta(META_FILE)
    annual_parts = []
    monthly_parts = []
    diurnal_parts = []
    for stn in selected:
        pairs = load_station_pairs(con, stn)
        pairs = reconstruct_geometry_for_pairs(pairs, neighbor_map, meta)
        pairs = _add_gamma_diagnostics(pairs)
        annual = aggregate_bias_groups(pairs, ["STN"])
        annual = annual.assign(**_annual_diagnostics(pairs))
        annual_parts.append(annual)
        monthly_parts.append(aggregate_bias_groups(pairs, ["STN", "MONTH"]))
        diurnal_parts.append(aggregate_bias_groups(pairs, ["STN", "HOUR"]))
    return (
        pd.concat(annual_parts, ignore_index=True),
        pd.concat(monthly_parts, ignore_index=True),
        pd.concat(diurnal_parts, ignore_index=True),
    )


def _validate_population_sets(analyzed, excluded, candidates):
    """전국 분석·제외·후보 STN 모집단 계약을 검증한다."""
    frames = [
        ("analyzed", analyzed),
        ("excluded", excluded),
        ("candidate", candidates),
    ]
    station_sets = {}
    for label, frame in frames:
        if "STN" not in frame.columns:
            raise AssertionError(f"{label} STN column missing")
        ids = frame["STN"].astype(str)
        if ids.duplicated().any():
            raise AssertionError(f"{label} STN values must be unique")
        station_sets[label] = set(ids)
    if len(candidates) != 549:
        raise AssertionError(f"expected 549 candidates, got {len(candidates)}")
    if len(excluded) != 20:
        raise AssertionError(f"expected 20 excluded stations, got {len(excluded)}")
    if len(analyzed) != 529:
        raise AssertionError(f"expected 529 analyzed stations, got {len(analyzed)}")
    if station_sets["analyzed"] & station_sets["excluded"]:
        raise AssertionError("analyzed and excluded station sets overlap")
    if station_sets["analyzed"] | station_sets["excluded"] != station_sets["candidate"]:
        raise AssertionError(
            "analyzed + excluded station sets do not match Stage 0 candidates"
        )


def _new_sibling_directory(out_dir, role):
    absolute = os.path.abspath(out_dir)
    parent = os.path.dirname(absolute)
    os.makedirs(parent, exist_ok=True)
    name = os.path.basename(absolute.rstrip(os.sep))
    path = os.path.join(parent, f".{name}.{role}-{uuid.uuid4().hex}")
    os.makedirs(path)
    return path


def _verify_artifacts(directory, artifacts):
    expected = set(artifacts)
    actual = set(os.listdir(directory))
    if actual != expected:
        raise AssertionError(
            f"artifact set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    empty = [
        name
        for name in artifacts
        if not os.path.isfile(os.path.join(directory, name))
        or os.path.getsize(os.path.join(directory, name)) == 0
    ]
    if empty:
        raise AssertionError(f"empty or non-file artifacts: {empty}")


def _publish_staging(staging_dir, out_dir):
    """완성 staging 디렉터리를 기존 출력과 교체하고 실패 시 복구한다."""
    absolute_out = os.path.abspath(out_dir)
    backup = None
    if os.path.exists(absolute_out):
        backup = _new_sibling_directory(absolute_out, "backup")
        os.rmdir(backup)
        os.replace(absolute_out, backup)
    try:
        os.replace(staging_dir, absolute_out)
    except BaseException:
        if backup is not None and not os.path.exists(absolute_out):
            os.replace(backup, absolute_out)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _merge_full_population_features(station_df):
    terrain = pd.read_csv(
        os.path.join(FULLNET, "station_terrain.csv"),
        encoding="utf-8-sig",
        dtype={"STN": str},
    )
    regimes = pd.read_csv(
        os.path.join(FULLNET, "station_neighbor_regimes.csv"),
        encoding="utf-8-sig",
        dtype={"STN": str},
    )
    terrain_cols = [
        "STN",
        "지점명",
        "위도",
        "경도",
        "elev_m",
        "relief_1km_m",
        "coast_dist_km",
        "is_mountain",
        "is_island",
        "terrain_group",
    ]
    regime_cols = ["STN", "n_regimes", "mode_cov_pct", "isolated"]
    result = station_df.merge(
        terrain[terrain_cols], on="STN", how="left", validate="one_to_one"
    ).merge(
        regimes[regime_cols], on="STN", how="left", validate="one_to_one"
    )
    for col in ["is_mountain", "is_island", "isolated"]:
        result[col] = result[col].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        )
        if result[col].isna().any():
            raise ValueError(f"unparseable boolean feature: {col}")
        result[col] = result[col].astype(int)
    missing_independent = result[INDEPENDENT_PREDICTORS].isna().sum()
    missing_independent = missing_independent[missing_independent > 0]
    if len(missing_independent):
        raise ValueError(
            f"missing independent features: {missing_independent.to_dict()}"
        )
    return result


def _build_evidence(station_df, max_c, max_b1):
    associations = build_effect_table(
        station_df, INDEPENDENT_PREDICTORS, DIAGNOSTIC_PREDICTORS
    )
    exact = pd.DataFrame(
        [
            {
                "HYPOTHESIS": "H1",
                "EVIDENCE_LEVEL": "exact_invariant",
                "ANALYSIS": "max_abs_residual",
                "OUTCOME": "CORRECTION_SHIFT",
                "PREDICTOR": "GAMMA_C_PER_KM * mean_dz_geom_km",
                "N": len(station_df),
                "EFFECT_NAME": "max_abs_residual_c",
                "EFFECT_VALUE": max_c,
                "R2": 1.0,
                "R2_TYPE": "exact_identity",
                "VIF": np.nan,
                "NOTES": "geometry reconstructed independently from error",
            },
            {
                "HYPOTHESIS": "H2",
                "EVIDENCE_LEVEL": "exact_invariant",
                "ANALYSIS": "max_abs_residual",
                "OUTCOME": "BIAS_fixed",
                "PREDICTOR": "BIAS_none + CORRECTION_SHIFT",
                "N": len(station_df),
                "EFFECT_NAME": "max_abs_residual_b1",
                "EFFECT_VALUE": max_b1,
                "R2": 1.0,
                "R2_TYPE": "exact_identity",
                "VIF": np.nan,
                "NOTES": "algebraic decomposition",
            },
        ],
        columns=EFFECT_COLUMNS,
    )
    coef, ols_summary = fit_standardized_ols(
        station_df, "BIAS_none", OLS_PREDICTORS
    )
    ols_rows = coef.assign(
        HYPOTHESIS="H5",
        EVIDENCE_LEVEL="independent_association",
        ANALYSIS="standardized_ols",
        OUTCOME="BIAS_none",
        PREDICTOR=coef["predictor"],
        N=ols_summary["n"],
        EFFECT_NAME="standardized_beta",
        EFFECT_VALUE=coef["std_beta"],
        R2=ols_summary["adjusted_r2"],
        R2_TYPE="adjusted_r_squared",
        NOTES="association only",
    )[EFFECT_COLUMNS]
    return pd.concat(
        [exact, associations[EFFECT_COLUMNS], ols_rows], ignore_index=True
    )


def run_analysis(
    stations=None,
    out_dir=OUT_DIR,
    enforce_population=True,
    make_all_figures=True,
):
    """station별로 데이터를 읽어 BIAS 증감 분석 산출물을 생성한다."""
    explicit_smoke = stations is not None and not enforce_population
    selected = None if stations is None else _validate_explicit_stations(stations)

    con = open_readonly_db()
    try:
        if not explicit_smoke:
            population = load_analysis_station_ids(con)
            if selected is None:
                selected = population
            unknown = sorted(set(selected).difference(population))
            if unknown:
                raise ValueError(f"stations outside paired population: {unknown}")
            if enforce_population and len(population) != 529:
                raise AssertionError(
                    f"expected 529 paired stations, got {len(population)}"
                )
        try:
            station_df, monthly_df, diurnal_df = _collect_analysis_frames(
                con, selected
            )
        except ValueError as error:
            if explicit_smoke:
                station = selected[0] if len(selected) == 1 else selected
                raise ValueError(
                    f"invalid explicit smoke station {station}: {error}"
                ) from error
            raise
    finally:
        con.close()

    station_df["IDENTITY_RESIDUAL_C"] = (
        station_df["CORRECTION_SHIFT"]
        - GAMMA_C_PER_KM * station_df["mean_dz_geom_km"]
    )
    station_df["IDENTITY_RESIDUAL_B1"] = (
        station_df["BIAS_fixed"]
        - station_df["BIAS_none"]
        - station_df["CORRECTION_SHIFT"]
    )
    max_c = float(station_df["IDENTITY_RESIDUAL_C"].abs().max())
    max_b1 = float(station_df["IDENTITY_RESIDUAL_B1"].abs().max())
    if max_c > 1e-10 or max_b1 > 1e-10:
        raise AssertionError(f"identity failure: C={max_c}, B1={max_b1}")

    if stations is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_csvs(
            {
                "bias_station_factors.csv": station_df,
                "bias_monthly.csv": monthly_df,
                "bias_diurnal.csv": diurnal_df,
            },
            out_dir,
        )
        summary = _complete_summary(
            _base_summary(station_df, 0, max_c, max_b1), SMOKE_ARTIFACTS
        )
        _write_summary(summary, out_dir)
        return summary

    station_df = _merge_full_population_features(station_df)
    evidence = _build_evidence(station_df, max_c, max_b1)

    excluded = pd.read_csv(
        EXCLUDED_PATH, encoding="utf-8-sig", dtype={"STN": str}
    )
    candidates = pd.read_csv(
        CANDIDATE_PATH, encoding="utf-8-sig", dtype={"STN": str}
    )
    if enforce_population:
        _validate_population_sets(station_df, excluded, candidates)

    summary = _base_summary(station_df, len(excluded), max_c, max_b1)
    summary.update(
        {
            "overall_bias_none_station_mean": float(
                station_df["BIAS_none"].mean()
            ),
            "overall_bias_fixed_station_mean": float(
                station_df["BIAS_fixed"].mean()
            ),
            "mechanism_counts": {
                str(key): int(value)
                for key, value in station_df["MECHANISM"].value_counts().items()
            },
        }
    )
    artifacts = (
        FULL_ARTIFACTS
        if make_all_figures
        else FULL_CSV_ARTIFACTS + [REPORT_ARTIFACT, SUMMARY_ARTIFACT]
    )
    summary = _complete_summary(summary, artifacts)

    staging_dir = _new_sibling_directory(out_dir, "staging")
    try:
        _write_csvs(
            {
                "bias_station_factors.csv": station_df,
                "bias_monthly.csv": monthly_df,
                "bias_diurnal.csv": diurnal_df,
                "bias_excluded_stations.csv": excluded,
                "bias_hypothesis_tests.csv": evidence,
            },
            staging_dir,
        )
        if make_all_figures:
            plot_bias_phase(station_df, staging_dir)
            plot_bias_map(station_df, BOUNDARY_PATH, staging_dir)
            plot_bias_distribution(station_df, staging_dir)
            plot_bias_temporal(monthly_df, diurnal_df, staging_dir)
            plot_physical_factors(station_df, evidence, staging_dir)
            plot_case_studies(station_df, monthly_df, staging_dir)
        write_report(
            station_df,
            monthly_df,
            diurnal_df,
            evidence,
            summary,
            staging_dir,
        )
        _verify_artifacts(staging_dir, artifacts[:-1])
        _write_summary(summary, staging_dir)
        _verify_artifacts(staging_dir, artifacts)
        _publish_staging(staging_dir, out_dir)
        staging_dir = None
    finally:
        if staging_dir is not None and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
    return summary


def _save_figure(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _symmetric_norm(values):
    vmax = float(np.nanquantile(np.abs(np.asarray(values, dtype=float)), 0.95))
    vmax = max(vmax, 1e-6)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_bias_phase(station_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    x = station_df["BIAS_none"].to_numpy(float)
    correction = station_df["CORRECTION_SHIFT"].to_numpy(float)
    delta = station_df["DELTA_ABS_BIAS"].to_numpy(float)
    scatter = ax.scatter(
        x,
        correction,
        c=delta,
        cmap="coolwarm",
        norm=_symmetric_norm(delta),
        s=24,
        alpha=0.8,
        edgecolors="none",
    )
    grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 300)
    ax.axhline(0.0, color="black", lw=1.0, label="C=0")
    ax.plot(grid, -grid, color="green", lw=1.2, label=r"$C=-B_0$ (BIAS=0)")
    ax.plot(
        grid,
        -2.0 * grid,
        color="purple",
        lw=1.2,
        label=r"$C=-2B_0$ (|BIAS| 동일)",
    )
    ax.set_xlabel(r"BIAS_none $B_0$ (°C)")
    ax.set_ylabel("보정 이동량 C (°C)")
    ax.set_title(r"529개 station의 $B_0$-$C$ 위상도")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(scatter, ax=ax, label="Δ|BIAS| (°C, 음수=개선)")
    return _save_figure(fig, out_dir, "bias_phase.png")


def plot_bias_map(station_df, boundary_path, out_dir):
    with open(boundary_path, encoding="utf-8") as file:
        rings = json.load(file)["rings"]
    fig, ax = plt.subplots(figsize=(8, 9))
    for ring in rings:
        xy = np.asarray(ring, dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], color="#b8b8b8", lw=0.35, zorder=1)
    delta = station_df["DELTA_ABS_BIAS"].to_numpy(float)
    scatter = ax.scatter(
        station_df["경도"],
        station_df["위도"],
        c=delta,
        cmap="coolwarm",
        norm=_symmetric_norm(delta),
        s=18,
        edgecolors="black",
        linewidths=0.15,
        zorder=2,
    )
    mean_lat = float(station_df["위도"].mean())
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_title("Station별 고도보정 후 Δ|BIAS|")
    fig.colorbar(scatter, ax=ax, label="Δ|BIAS| (°C, 음수=개선)")
    return _save_figure(fig, out_dir, "bias_delta_map.png")


def plot_bias_distribution(station_df, out_dir):
    values = station_df["DELTA_ABS_BIAS"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=35, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="black", lw=1.2)
    ax.set_xlabel("Δ|BIAS| (°C, 음수=개선)")
    ax.set_ylabel("Station 수")
    ax.set_title(f"전체 분포 (중앙값={np.median(values):.3f}°C)")
    return _save_figure(fig, out_dir, "bias_delta_distribution.png")


def _period_quantiles(frame, period):
    grouped = frame.groupby(period)["DELTA_ABS_BIAS"]
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        }
    ).reset_index()


def plot_bias_temporal(monthly_df, diurnal_df, out_dir):
    month = _period_quantiles(monthly_df, "MONTH")
    hour = _period_quantiles(diurnal_df, "HOUR")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    month_x = np.arange(len(month))
    axes[0].plot(month_x, month["median"].to_numpy(float), color="#4c78a8")
    axes[0].fill_between(
        month_x,
        month["q25"].to_numpy(float),
        month["q75"].to_numpy(float),
        color="#4c78a8",
        alpha=0.25,
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xticks(month_x, month["MONTH"].astype(str).str[-2:])
    axes[0].set_xlabel("월")
    axes[0].set_ylabel("Δ|BIAS| (°C)")
    axes[0].set_title("월별 station 중앙값과 IQR")
    hour_x = hour["HOUR"].to_numpy(float)
    axes[1].plot(hour_x, hour["median"].to_numpy(float), color="#f58518")
    axes[1].fill_between(
        hour_x,
        hour["q25"].to_numpy(float),
        hour["q75"].to_numpy(float),
        color="#f58518",
        alpha=0.25,
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("시각")
    axes[1].set_ylabel("Δ|BIAS| (°C)")
    axes[1].set_title("시간대별 station 중앙값과 IQR")
    return _save_figure(fig, out_dir, "bias_temporal.png")


def plot_physical_factors(station_df, evidence, out_dir):
    del station_df
    rows = evidence[
        (evidence["EVIDENCE_LEVEL"] == "independent_association")
        & (evidence["ANALYSIS"] == "spearman")
        & evidence["PREDICTOR"].isin(INDEPENDENT_PREDICTORS)
    ]
    pivot = rows.pivot(
        index="PREDICTOR", columns="OUTCOME", values="EFFECT_VALUE"
    ).reindex(INDEPENDENT_PREDICTORS)
    display_names = {
        "mean_dz_geom_km": "평균 ΔZ (km)",
        "mean_absdz_geom_km": "평균 절대 ΔZ (km)",
        "sd_dz_geom_km": "ΔZ 변동폭 (km)",
        "mean_neighbor_count": "이웃 station 수",
        "mean_nearest_distance_km": "가장 가까운 이웃 거리 (km)",
        "mean_weighted_distance_km": "가중평균 이웃 거리 (km)",
        "mean_centroid_offset_km": "이웃 중심과의 거리 (km)",
        "mean_neighbor_elev_sd_m": "이웃 고도 산포 (m)",
        "n_regimes": "이웃 구성 변화 횟수",
        "elev_m": "target station 고도 (m)",
        "relief_1km_m": "주변 1km 기복 (m)",
        "coast_dist_km": "해안까지 거리 (km)",
        "is_mountain": "산악 여부",
        "is_island": "섬 여부",
        "mode_cov_pct": "최빈 이웃 구성 비율 (%)",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    ypos = np.arange(len(pivot))
    width = 0.38
    ax.barh(
        ypos - width / 2,
        pivot["BIAS_none"],
        height=width,
        label="보정 전 BIAS",
    )
    ax.barh(
        ypos + width / 2,
        pivot["DELTA_ABS_BIAS"],
        height=width,
        label="BIAS 변화량 (Δ|BIAS|)",
    )
    ax.set_yticks(ypos, [display_names.get(name, name) for name in pivot.index])
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ (순위 상관계수)")
    ax.set_title("독립 요인과 BIAS의 관계")
    ax.legend()
    return _save_figure(fig, out_dir, "bias_physical_factors_readable.png")


def plot_case_studies(station_df, monthly_df, out_dir):
    chosen = (
        station_df.assign(_abs_y=station_df["DELTA_ABS_BIAS"].abs())
        .sort_values(["MECHANISM", "_abs_y"], ascending=[True, False])
        .groupby("MECHANISM", as_index=False)
        .head(1)
    )
    count = len(chosen)
    fig, axes = plt.subplots(
        count, 1, figsize=(9, max(3.2 * count, 4)), squeeze=False
    )
    for ax, row in zip(axes[:, 0], chosen.itertuples(index=False)):
        data = monthly_df[monthly_df["STN"] == str(row.STN)].sort_values("MONTH")
        x = np.arange(len(data))
        ax.plot(x, data["BIAS_none"], marker="o", label="none")
        ax.plot(x, data["BIAS_fixed"], marker="s", label="fixed_lapse")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x, data["MONTH"].astype(str).str[-2:])
        ax.set_ylabel("BIAS (°C)")
        case_label = f"STN {row.STN} — {row.MECHANISM}"
        ax.set_title(case_label)
        ax.legend(fontsize=8, title=case_label, title_fontsize=8)
    axes[-1, 0].set_xlabel("월")
    return _save_figure(fig, out_dir, "bias_case_studies.png")


def _format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5f}"
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "<br>").replace("|", "\\|")


def _markdown_table(df):
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(station_df, monthly_df, diurnal_df, evidence, summary, out_dir):
    mechanism = (
        station_df["MECHANISM"]
        .value_counts()
        .rename_axis("MECHANISM")
        .reset_index(name="stations")
    )
    direct = evidence[evidence["EVIDENCE_LEVEL"] == "exact_invariant"][
        [
            "HYPOTHESIS",
            "OUTCOME",
            "PREDICTOR",
            "N",
            "EFFECT_NAME",
            "EFFECT_VALUE",
        ]
    ]
    independent = evidence[
        (evidence["EVIDENCE_LEVEL"] == "independent_association")
        & (evidence["ANALYSIS"] == "spearman")
    ].copy()
    independent["abs_effect"] = independent["EFFECT_VALUE"].abs()
    independent = independent.sort_values(
        "abs_effect", ascending=False
    ).head(12)[
        [
            "OUTCOME",
            "PREDICTOR",
            "N",
            "EFFECT_VALUE",
            "R2",
            "R2_TYPE",
            "NOTES",
        ]
    ]
    cases = (
        station_df.assign(abs_y=station_df["DELTA_ABS_BIAS"].abs())
        .sort_values(["MECHANISM", "abs_y"], ascending=[True, False])
        .groupby("MECHANISM", as_index=False)
        .head(1)[
            [
                "STN",
                "지점명",
                "MECHANISM",
                "BIAS_none",
                "BIAS_fixed",
                "CORRECTION_SHIFT",
                "DELTA_ABS_BIAS",
                "mean_dz_geom_km",
            ]
        ]
    )
    month_summary = (
        monthly_df.groupby("MONTH")["DELTA_ABS_BIAS"]
        .agg(station_median="median", station_mean="mean", n="count")
        .reset_index()
    )
    hour_summary = (
        diurnal_df.groupby("HOUR")["DELTA_ABS_BIAS"]
        .agg(station_median="median", station_mean="mean", n="count")
        .reset_index()
    )

    report = f"""# Station별 BIAS 증감 원인 분석

## 1. 연구 질문과 BIAS 정의

[증거 수준: `algebraic identity`]

오차는 `PREDICTED−OBSERVED`다. 연간 `B0=BIAS_none`,
`C=BIAS_fixed−BIAS_none`, `B1=B0+C`, 주 결과는
`Δ|BIAS|=|B1|−|B0|`로 정의했다. 음수는 개선, 양수는 악화다.

## 2. 모집단 529개와 비보간 20개

[증거 수준: `outcome-derived diagnostic`]

연간 분석은 극단값 선별 없이 paired 결과가 있는 {summary['analyzed_stations']}개 station을
전부 포함했다. BIAS를 계산할 수 없는 {summary['excluded_stations']}개는 별도 제외표에 남겼다.
총 paired 시각 수는 {summary['paired_hours']:,}개다.

## 3. 직접 항등식: B0, C, B1

[증거 수준: `algebraic identity`]

`ΔZ_geom`은 오차 차이에서 역산하지 않고 metadata와 neighbor geometry에서 독립 재구성했다.
`C=γ_fixed×mean(ΔZ_geom)`와 `B1=B0+C`의 검산 결과는 다음과 같다.

{_markdown_table(direct)}

## 4. 전체 station의 Δ|BIAS| 분포

[증거 수준: `algebraic identity`]

개선 {summary['improved_stations']}개, 악화 {summary['worsened_stations']}개,
정확히 동일 {summary['unchanged_stations']}개다. mechanism별 분포는 다음과 같다.

{_markdown_table(mechanism)}

## 5. 감률·계절·주야 진단

[증거 수준: `outcome-derived diagnostic`]

`γ_eff`와 역전 비율은 interpolation error에서 파생한 진단값이며 독립적인 인과 증거가 아니다.
월별 및 시간대별 station 분포는 연간 차이가 특정 시기에 누적 또는 상쇄되는지 확인하는 데만 사용했다.

### 월별 요약

{_markdown_table(month_summary)}

### 시간대별 요약

{_markdown_table(hour_summary)}

## 6. 독립 이웃 기하·지형 요인

[증거 수준: `independent physical association`]

아래 표는 사전 지정 독립 변수의 Spearman 연관성만 제시하며 인과효과로 해석하지 않는다.
표준화 OLS 계수·adjusted R²·VIF는 `bias_hypothesis_tests.csv` hypothesis CSV를 참조한다.

{_markdown_table(independent)}

## 7. 대표 사례

[증거 수준: `outcome-derived diagnostic`]

전체 population 결과를 확정한 후 각 mechanism 내부에서 `|Δ|BIAS||`가 가장 큰 station을
설명용 사례로 선정했다. 이 사례들은 전체 결론을 만드는 표본으로 사용하지 않았다.

{_markdown_table(cases)}

## 8. 증거 수준과 인과해석 한계

본 분석은 (1) algebraic identity, (2) outcome-derived diagnostic,
(3) independent physical association을 구분한다. 2024년 한 해의 LOO 결과이므로
다른 연도에 대한 일반화와 지형 변수의 인과효과는 별도 검증이 필요하다.

## 9. 재현 방법

```powershell
python analyze_station_bias.py
```
"""
    path = os.path.join(out_dir, "bias_station_analysis_report.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(report)
    return path


def main():
    summary = run_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
