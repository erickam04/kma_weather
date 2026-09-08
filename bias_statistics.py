"""관측소 편향 요인 분석을 위한 효과크기와 보조 회귀 도구."""

import numpy as np
import pandas as pd


def _paired_finite(x, y):
    """두 1차원 입력을 숫자로 강제변환하고 유한한 관측쌍만 반환한다."""
    a_raw = np.asarray(x, dtype=object)
    b_raw = np.asarray(y, dtype=object)
    if a_raw.ndim != 1 or b_raw.ndim != 1:
        raise ValueError("paired effects require one-dimensional inputs")
    if len(a_raw) != len(b_raw):
        raise ValueError("paired effects require equal-length inputs")
    a = pd.to_numeric(pd.Series(a_raw), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b_raw), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def spearman_effect(x, y):
    """Spearman 순위상관계수와 유효 관측쌍 수를 반환한다."""
    a, b = _paired_finite(x, y)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, int(len(a))
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1]), int(len(a))


def simple_r2(x, y):
    """두 변수의 Pearson 상관계수 제곱(R²)을 반환한다."""
    a, b = _paired_finite(x, y)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def variance_inflation_factors(x_standardized, predictor_names):
    """표준화된 예측변수 행렬의 predictor별 VIF를 계산한다."""
    try:
        x = np.asarray(x_standardized, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("VIF matrix must be numeric") from error
    if x.ndim != 2:
        raise ValueError("VIF matrix must be two-dimensional")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("VIF matrix must have rows and predictors")
    names = list(predictor_names)
    if len(names) != x.shape[1]:
        raise ValueError("predictor_names length must match VIF matrix columns")
    if len(set(names)) != len(names):
        raise ValueError("predictor_names must be unique")
    if not np.isfinite(x).all():
        raise ValueError("VIF matrix must contain only finite values")
    if np.any(x.std(axis=0, ddof=0) == 0):
        raise ValueError("VIF matrix must not contain constant predictors")
    if x.shape[1] == 1:
        return pd.DataFrame({"predictor": names, "VIF": [1.0]})
    rows = []
    for j, name in enumerate(names):
        y = x[:, j]
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(len(y)), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        sst = float(np.sum((y - y.mean()) ** 2))
        sse = float(np.sum((y - fitted) ** 2))
        r2 = 1.0 - sse / sst
        vif = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"predictor": name, "VIF": vif})
    return pd.DataFrame(rows)


def fit_standardized_ols(df, outcome, predictors):
    """완전 관측 행으로 표준화 OLS와 VIF를 계산한다."""
    predictors = list(predictors)
    if outcome not in df.columns:
        raise ValueError(f"outcome column not found: {outcome}")
    if not predictors:
        raise ValueError("standardized OLS requires at least one predictor")
    if len(set(predictors)) != len(predictors):
        raise ValueError("standardized OLS predictors must be unique")
    missing_predictors = [name for name in predictors if name not in df.columns]
    if missing_predictors:
        raise ValueError(f"predictor columns not found: {missing_predictors}")
    clean = df[[outcome] + predictors].apply(pd.to_numeric, errors="coerce")
    clean = clean.loc[np.isfinite(clean.to_numpy(dtype=float)).all(axis=1)]
    if len(clean) <= len(predictors) + 1:
        raise ValueError("not enough complete rows for standardized OLS")
    y = clean[outcome].to_numpy(float)
    x = clean[predictors].to_numpy(float)
    x_sd = x.std(axis=0, ddof=0)
    if np.any(x_sd == 0):
        raise ValueError("constant predictor in standardized OLS")
    y_sd = y.std(ddof=0)
    if y_sd == 0:
        raise ValueError("constant outcome in standardized OLS")
    xz = (x - x.mean(axis=0)) / x_sd
    yz = (y - y.mean()) / y_sd
    if np.linalg.matrix_rank(xz) < len(predictors):
        raise ValueError("rank-deficient predictors in standardized OLS")
    design = np.column_stack([np.ones(len(yz)), xz])
    beta = np.linalg.lstsq(design, yz, rcond=None)[0]
    fitted = design @ beta
    sse = float(np.sum((yz - fitted) ** 2))
    sst = float(np.sum((yz - yz.mean()) ** 2))
    r2 = 1.0 - sse / sst
    n = len(yz)
    p = len(predictors)
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    vif = variance_inflation_factors(xz, predictors)
    coef = pd.DataFrame({"predictor": predictors, "std_beta": beta[1:]})
    coef = coef.merge(vif, on="predictor", validate="one_to_one")
    return coef, {"n": n, "r2": r2, "adjusted_r2": adjusted_r2}


def build_effect_table(station_df, independent_predictors, diagnostic_predictors):
    """독립 예측변수와 오차 유래 진단변수의 증거수준을 분리한 효과표를 만든다."""
    independent_predictors = list(independent_predictors)
    diagnostic_predictors = list(diagnostic_predictors)
    required = ["BIAS_none", "DELTA_ABS_BIAS"] + independent_predictors + diagnostic_predictors
    missing_columns = [name for name in dict.fromkeys(required) if name not in station_df.columns]
    if missing_columns:
        raise ValueError(f"effect table columns not found: {missing_columns}")
    rows = []
    for evidence_level, predictors in [
        ("independent_association", independent_predictors),
        ("outcome_derived_diagnostic", diagnostic_predictors),
    ]:
        for outcome in ["BIAS_none", "DELTA_ABS_BIAS"]:
            for predictor in predictors:
                rho, n = spearman_effect(station_df[predictor], station_df[outcome])
                rows.append(
                    {
                        "HYPOTHESIS": "H5"
                        if evidence_level == "independent_association"
                        else "H3_H4",
                        "EVIDENCE_LEVEL": evidence_level,
                        "ANALYSIS": "spearman",
                        "OUTCOME": outcome,
                        "PREDICTOR": predictor,
                        "N": n,
                        "EFFECT_NAME": "spearman_rho",
                        "EFFECT_VALUE": rho,
                        "R2": simple_r2(station_df[predictor], station_df[outcome]),
                        "R2_TYPE": "pearson_r_squared",
                        "VIF": np.nan,
                        "NOTES": "Pearson R²; association only"
                        if evidence_level == "independent_association"
                        else "Pearson R²; derived from interpolation error; diagnostic only",
                    }
                )
    return pd.DataFrame(rows)
