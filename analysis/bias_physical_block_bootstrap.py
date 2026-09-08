"""물리요인 후속분석용 월층화 달력 연속 날짜 블록 재표집.

이 모듈은 파일 I/O를 하지 않는다. runner가 한 station의 시간별 오차와
공간구조 상태를 전달하면, 월 경계를 넘지 않는 동일 날짜 weight로
두 상태의 Δ|BIAS| 차이를 재계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


REQUIRED_STATES = ("inversion_like", "normal_negative")
ALLOWED_STRUCTURE_STATES = frozenset(
    {"inversion_like", "normal_negative", "uncertain", "ineligible"}
)
ALLOWED_SOLAR_STATES = frozenset({"bright", "transition", "dark"})
STRATA_WEIGHTING = "equal_common_month_solar_strata"
ESTIMAND = "standardized_signed_bias_then_delta_abs_bias_contrast"
SAMPLING_SCHEME = "monthly_calendar_contiguous_fixed_length_blocks"
REQUIRED_COLUMNS = (
    "STN",
    "TM",
    "SOLAR_STATE",
    "STRUCTURE_STATE",
    "error_none",
    "error_fixed",
)


@dataclass(frozen=True)
class MonthlyCircularBlockSample:
    """월층화 원형 블록 표본.

    `sampled_indices[r]`는 replicate r에서 선택된 `dates`의 위치이고,
    `weights[r, d]`는 날짜 d가 replicate r에 뽑힌 횟수다.
    """

    dates: pd.DatetimeIndex
    sampled_indices: np.ndarray
    weights: np.ndarray
    block_length_days: int
    n_replicates: int
    seed: Optional[int]
    sampling_scheme: str
    month_diagnostics: tuple[dict[str, object], ...]
    unique_weight_patterns: int
    supported_date_mask: np.ndarray
    unsupported_dates: tuple[str, ...]
    supported_date_count: int
    unsupported_date_count: int
    date_support_fraction: float


@dataclass(frozen=True)
class SharedStateContrastResult:
    """같은 날짜 weight로 계산한 두 공간상태의 Δ|BIAS| contrast."""

    station_id: str
    coverage_eligible: bool
    reason: str
    common_strata_count: int
    inversion_unique_days: int
    normal_unique_days: int
    inversion_months: int
    normal_months: int
    inversion_delta_abs_bias: float
    normal_delta_abs_bias: float
    point_contrast: float
    ci_low: float
    ci_high: float
    requested_replicates: int
    valid_replicates: int
    valid_replicate_fraction: float
    replicate_contrasts: np.ndarray
    sample: Optional[MonthlyCircularBlockSample]
    seed: Optional[int]
    block_length_days: int
    min_unique_days_per_state: int
    min_months_per_state: int
    min_valid_replicate_fraction: float
    strata_keys: tuple[tuple[str, str], ...]
    strata_weights: np.ndarray
    strata_weighting: str
    estimand: str
    sampling_scheme: str
    calendar_month_diagnostics: tuple[dict[str, object], ...]
    analysis_support_checked: bool
    analysis_date_count: int
    supported_analysis_date_count: int
    uncovered_analysis_date_count: int
    analysis_date_support_fraction: float
    uncovered_analysis_dates: tuple[str, ...]
    analysis_support_diagnostics: tuple[dict[str, object], ...]


class CalendarBlockUnavailableError(ValueError):
    """한 달 이상에서 달력상 연속 block을 만들 수 없을 때 발생한다."""

    def __init__(self, month_diagnostics: tuple[dict[str, object], ...]):
        self.month_diagnostics = month_diagnostics
        invalid = [
            str(item["month"])
            for item in month_diagnostics
            if int(item["available_contiguous_blocks"]) == 0
        ]
        super().__init__(
            "no complete calendar-contiguous block for month(s): " + ", ".join(invalid)
        )


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _probability(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be in (0, 1]")
    number = float(value)
    if not np.isfinite(number) or not 0 < number <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return number


def _normalized_unique_dates(dates: Sequence) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("dates must be valid timestamps") from exc
    if len(index) == 0:
        raise ValueError("dates must not be empty")
    if index.hasnans:
        raise ValueError("dates must not contain NaT")
    return pd.DatetimeIndex(index.normalize().unique()).sort_values()


def generate_monthly_circular_block_weights(
    dates: Sequence,
    *,
    n_replicates: int,
    block_length_days: int = 7,
    seed: Optional[int] = 0,
) -> MonthlyCircularBlockSample:
    """월별 달력상 연속인 고정길이 날짜 block의 index와 weight를 만든다.

    public 함수명은 초기 API 호환을 위해 유지한다. 실제 표본 block은 결측일을
    건너뛰거나 월말에서 월초로 wrap하지 않는다. 월 안에 완전한 연속 block이
    하나도 없으면 :class:`CalendarBlockUnavailableError`를 발생시킨다.
    """

    replicate_count = _positive_integer("n_replicates", n_replicates)
    block_length = _positive_integer("block_length_days", block_length_days)
    unique_dates = _normalized_unique_dates(dates)
    month_labels = unique_dates.strftime("%Y-%m").to_numpy()
    months = pd.unique(month_labels)
    rng = np.random.default_rng(seed)

    blocks_by_month: list[tuple[np.ndarray, np.ndarray]] = []
    diagnostics: list[dict[str, object]] = []
    supported_date_mask = np.zeros(len(unique_dates), dtype=bool)
    for month in months:
        positions = np.flatnonzero(month_labels == month)
        month_dates = unique_dates[positions]
        position_by_date = {date: int(position) for date, position in zip(month_dates, positions)}
        valid_blocks = []
        for start_date in month_dates:
            block_dates = [
                start_date + pd.Timedelta(days=offset)
                for offset in range(block_length)
            ]
            if all(date in position_by_date for date in block_dates):
                valid_blocks.append([position_by_date[date] for date in block_dates])
        span_days = int((month_dates[-1] - month_dates[0]).days + 1)
        day_differences = [
            int((month_dates[index] - month_dates[index - 1]).days)
            for index in range(1, len(month_dates))
        ]
        supported_positions = (
            np.unique(np.asarray(valid_blocks, dtype=np.int64).reshape(-1))
            if valid_blocks
            else np.empty(0, dtype=np.int64)
        )
        supported_date_mask[supported_positions] = True
        unsupported_month_dates = tuple(
            date.strftime("%Y-%m-%d")
            for date, position in zip(month_dates, positions)
            if position not in set(supported_positions.tolist())
        )
        diagnostics.append(
            {
                "month": str(month),
                "unique_dates": int(len(month_dates)),
                "first_date": month_dates[0].isoformat(),
                "last_date": month_dates[-1].isoformat(),
                "span_calendar_days": span_days,
                "missing_days_within_span": int(span_days - len(month_dates)),
                "max_gap_days": max(day_differences) if day_differences else 0,
                "available_contiguous_blocks": int(len(valid_blocks)),
                "supported_unique_dates": int(len(supported_positions)),
                "unsupported_unique_dates": int(len(unsupported_month_dates)),
                "unsupported_dates": unsupported_month_dates,
            }
        )
        block_array = (
            np.asarray(valid_blocks, dtype=np.int64)
            if valid_blocks
            else np.empty((0, block_length), dtype=np.int64)
        )
        blocks_by_month.append((positions, block_array))

    diagnostic_tuple = tuple(diagnostics)
    if any(len(blocks) == 0 for _, blocks in blocks_by_month):
        raise CalendarBlockUnavailableError(diagnostic_tuple)

    sampled_indices = np.empty(
        (replicate_count, len(unique_dates)), dtype=np.int64
    )
    for replicate in range(replicate_count):
        cursor = 0
        for positions, valid_blocks in blocks_by_month:
            month_size = len(positions)
            block_count = (month_size + block_length - 1) // block_length
            choices = rng.integers(0, len(valid_blocks), size=block_count)
            selected = valid_blocks[choices].reshape(-1)[:month_size]
            sampled_indices[replicate, cursor : cursor + month_size] = selected
            cursor += month_size

    weights = np.zeros_like(sampled_indices)
    row_indices = np.repeat(np.arange(replicate_count), len(unique_dates))
    np.add.at(weights, (row_indices, sampled_indices.reshape(-1)), 1)
    unique_weight_patterns = int(len(np.unique(weights, axis=0)))
    unsupported_dates = tuple(
        date.strftime("%Y-%m-%d")
        for date, supported in zip(unique_dates, supported_date_mask)
        if not supported
    )
    supported_date_count = int(supported_date_mask.sum())
    return MonthlyCircularBlockSample(
        dates=unique_dates,
        sampled_indices=sampled_indices,
        weights=weights,
        block_length_days=block_length,
        n_replicates=replicate_count,
        seed=seed,
        sampling_scheme=SAMPLING_SCHEME,
        month_diagnostics=diagnostic_tuple,
        unique_weight_patterns=unique_weight_patterns,
        supported_date_mask=supported_date_mask,
        unsupported_dates=unsupported_dates,
        supported_date_count=supported_date_count,
        unsupported_date_count=len(unsupported_dates),
        date_support_fraction=supported_date_count / len(unique_dates),
    )


def _empty_result(
    *,
    station_id: str,
    reason: str,
    common_strata_count: int,
    inversion_unique_days: int = 0,
    normal_unique_days: int = 0,
    inversion_months: int = 0,
    normal_months: int = 0,
    inversion_delta_abs_bias: float = np.nan,
    normal_delta_abs_bias: float = np.nan,
    point_contrast: float = np.nan,
    requested_replicates: int,
    seed: Optional[int],
    block_length_days: int,
    min_unique_days_per_state: int,
    min_months_per_state: int,
    min_valid_replicate_fraction: float,
    strata_keys: tuple[tuple[str, str], ...] = (),
    strata_weights: Optional[np.ndarray] = None,
    sample: Optional[MonthlyCircularBlockSample] = None,
    calendar_month_diagnostics: tuple[dict[str, object], ...] = (),
    valid_replicates: int = 0,
    valid_replicate_fraction: float = 0.0,
    replicate_contrasts: Optional[np.ndarray] = None,
    analysis_support_checked: bool = False,
    analysis_date_count: int = 0,
    supported_analysis_date_count: int = 0,
    uncovered_analysis_date_count: int = 0,
    analysis_date_support_fraction: float = np.nan,
    uncovered_analysis_dates: tuple[str, ...] = (),
    analysis_support_diagnostics: tuple[dict[str, object], ...] = (),
) -> SharedStateContrastResult:
    if strata_weights is None:
        strata_weights = np.empty(0, dtype=float)
    if replicate_contrasts is None:
        replicate_contrasts = np.empty(0, dtype=float)
    return SharedStateContrastResult(
        station_id=station_id,
        coverage_eligible=False,
        reason=reason,
        common_strata_count=common_strata_count,
        inversion_unique_days=inversion_unique_days,
        normal_unique_days=normal_unique_days,
        inversion_months=inversion_months,
        normal_months=normal_months,
        inversion_delta_abs_bias=float(inversion_delta_abs_bias),
        normal_delta_abs_bias=float(normal_delta_abs_bias),
        point_contrast=float(point_contrast),
        ci_low=np.nan,
        ci_high=np.nan,
        requested_replicates=requested_replicates,
        valid_replicates=int(valid_replicates),
        valid_replicate_fraction=float(valid_replicate_fraction),
        replicate_contrasts=np.asarray(replicate_contrasts, dtype=float),
        sample=sample,
        seed=seed,
        block_length_days=block_length_days,
        min_unique_days_per_state=min_unique_days_per_state,
        min_months_per_state=min_months_per_state,
        min_valid_replicate_fraction=min_valid_replicate_fraction,
        strata_keys=strata_keys,
        strata_weights=np.asarray(strata_weights, dtype=float),
        strata_weighting=STRATA_WEIGHTING,
        estimand=ESTIMAND,
        sampling_scheme=SAMPLING_SCHEME,
        calendar_month_diagnostics=calendar_month_diagnostics,
        analysis_support_checked=analysis_support_checked,
        analysis_date_count=analysis_date_count,
        supported_analysis_date_count=supported_analysis_date_count,
        uncovered_analysis_date_count=uncovered_analysis_date_count,
        analysis_date_support_fraction=float(analysis_date_support_fraction),
        uncovered_analysis_dates=uncovered_analysis_dates,
        analysis_support_diagnostics=analysis_support_diagnostics,
    )


def _standardized_state_delta_abs_bias(
    frame: pd.DataFrame,
    state: str,
    strata_keys: tuple[tuple[str, str], ...],
    strata_weights: np.ndarray,
) -> float:
    standardized = {}
    for method in ("error_none", "error_fixed"):
        stratum_means = []
        for month, solar in strata_keys:
            selected = frame.loc[
                frame["STRUCTURE_STATE"].eq(state)
                & frame["MONTH"].eq(month)
                & frame["SOLAR_STATE"].eq(solar),
                method,
            ]
            stratum_means.append(float(selected.mean()))
        standardized[method] = float(np.dot(strata_weights, stratum_means))
    return abs(standardized["error_fixed"]) - abs(standardized["error_none"])


def _analysis_support_metadata(
    common: pd.DataFrame,
    strata_keys: tuple[tuple[str, str], ...],
    sample: MonthlyCircularBlockSample,
) -> dict[str, object]:
    supported_dates = set(sample.dates[sample.supported_date_mask])
    analysis_dates = set(pd.DatetimeIndex(common["DATE"].unique()))
    uncovered = sorted(analysis_dates - supported_dates)
    diagnostics = []
    for month, solar in strata_keys:
        for state in REQUIRED_STATES:
            selected = common.loc[
                common["MONTH"].eq(month)
                & common["SOLAR_STATE"].eq(solar)
                & common["STRUCTURE_STATE"].eq(state),
                "DATE",
            ]
            state_dates = set(pd.DatetimeIndex(selected.unique()))
            state_uncovered = sorted(state_dates - supported_dates)
            supported_count = len(state_dates) - len(state_uncovered)
            diagnostics.append(
                {
                    "month": month,
                    "solar_state": solar,
                    "structure_state": state,
                    "analysis_unique_dates": len(state_dates),
                    "supported_unique_dates": supported_count,
                    "uncovered_unique_dates": len(state_uncovered),
                    "support_fraction": (
                        supported_count / len(state_dates) if state_dates else np.nan
                    ),
                    "uncovered_dates": tuple(
                        date.strftime("%Y-%m-%d") for date in state_uncovered
                    ),
                }
            )
    supported_analysis_count = len(analysis_dates) - len(uncovered)
    return {
        "analysis_support_checked": True,
        "analysis_date_count": len(analysis_dates),
        "supported_analysis_date_count": supported_analysis_count,
        "uncovered_analysis_date_count": len(uncovered),
        "analysis_date_support_fraction": (
            supported_analysis_count / len(analysis_dates) if analysis_dates else np.nan
        ),
        "uncovered_analysis_dates": tuple(
            date.strftime("%Y-%m-%d") for date in uncovered
        ),
        "analysis_support_diagnostics": tuple(diagnostics),
    }


def _validated_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("frame must not be empty")

    work = frame.loc[:, REQUIRED_COLUMNS].copy()
    work["STN"] = work["STN"].astype("string").str.strip()
    if work["STN"].isna().any() or work["STN"].eq("").any():
        raise ValueError("STN must be a non-empty identifier")
    stations = work["STN"].unique().tolist()
    if len(stations) != 1:
        raise ValueError("frame must contain exactly one station")
    station_id = str(stations[0])

    try:
        work["TM"] = pd.to_datetime(work["TM"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("TM must contain valid timestamps") from exc
    if work["TM"].isna().any():
        raise ValueError("TM must not contain NaT")
    if work.duplicated(["STN", "TM"]).any():
        raise ValueError("STN/TM must be unique")

    domains = {
        "SOLAR_STATE": ALLOWED_SOLAR_STATES,
        "STRUCTURE_STATE": ALLOWED_STRUCTURE_STATES,
    }
    for column, allowed in domains.items():
        if work[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")
        values = work[column].astype(str)
        invalid = sorted(set(values) - set(allowed))
        if invalid:
            raise ValueError(f"{column} contains invalid values: {invalid}")
        work[column] = values
    for column in ("error_none", "error_fixed"):
        try:
            work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not np.isfinite(work[column].to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")

    work["DATE"] = work["TM"].dt.normalize()
    work["MONTH"] = work["TM"].dt.strftime("%Y-%m")
    return work, station_id


def bootstrap_shared_state_delta_abs_bias_contrast(
    frame: pd.DataFrame,
    *,
    n_replicates: int = 2000,
    block_length_days: int = 7,
    seed: Optional[int] = 0,
    min_unique_days_per_state: int = 30,
    min_months_per_state: int = 6,
    min_valid_replicate_fraction: float = 0.95,
) -> SharedStateContrastResult:
    """공통 월×태양상태 층에서 두 상태의 Δ|BIAS| 차이와 95% CI를 구한다.

    contrast의 부호는
    `inversion_like Δ|BIAS| - normal_negative Δ|BIAS|`이다.
    같은 replicate의 날짜 weight를 none/fixed와 두 상태 모두에 적용한다.
    기본 추론 coverage는 각 상태 30개 고유 날짜와 6개월이다.
    """

    replicate_count = _positive_integer("n_replicates", n_replicates)
    block_length = _positive_integer("block_length_days", block_length_days)
    minimum_days = _positive_integer(
        "min_unique_days_per_state", min_unique_days_per_state
    )
    minimum_months = _positive_integer("min_months_per_state", min_months_per_state)
    minimum_valid_fraction = _probability(
        "min_valid_replicate_fraction", min_valid_replicate_fraction
    )
    work, station_id = _validated_frame(frame)

    method_metadata = dict(
        seed=seed,
        block_length_days=block_length,
        min_unique_days_per_state=minimum_days,
        min_months_per_state=minimum_months,
        min_valid_replicate_fraction=minimum_valid_fraction,
    )

    required = work.loc[work["STRUCTURE_STATE"].isin(REQUIRED_STATES)].copy()
    present_states = set(required["STRUCTURE_STATE"])
    if not set(REQUIRED_STATES).issubset(present_states):
        return _empty_result(
            station_id=station_id,
            reason="missing_required_state",
            common_strata_count=0,
            requested_replicates=replicate_count,
            **method_metadata,
        )

    common_keys = []
    for key, group in required.groupby(["MONTH", "SOLAR_STATE"], sort=True):
        if set(REQUIRED_STATES).issubset(set(group["STRUCTURE_STATE"])):
            common_keys.append(key)
    if not common_keys:
        return _empty_result(
            station_id=station_id,
            reason="no_common_month_solar_strata",
            common_strata_count=0,
            requested_replicates=replicate_count,
            **method_metadata,
        )

    strata_keys = tuple((str(month), str(solar)) for month, solar in sorted(common_keys))
    strata_weights = np.full(len(strata_keys), 1.0 / len(strata_keys), dtype=float)
    key_set = set(strata_keys)
    common_mask = [
        (month, solar) in key_set
        for month, solar in zip(required["MONTH"], required["SOLAR_STATE"])
    ]
    common = required.loc[common_mask].copy()

    counts = {}
    for state in REQUIRED_STATES:
        state_rows = common.loc[common["STRUCTURE_STATE"].eq(state)]
        counts[state] = (
            int(state_rows["DATE"].nunique()),
            int(state_rows["MONTH"].nunique()),
        )
    inversion_days, inversion_months = counts["inversion_like"]
    normal_days, normal_months = counts["normal_negative"]
    inversion_effect = _standardized_state_delta_abs_bias(
        common, "inversion_like", strata_keys, strata_weights
    )
    normal_effect = _standardized_state_delta_abs_bias(
        common, "normal_negative", strata_keys, strata_weights
    )
    point_contrast = inversion_effect - normal_effect

    empty_kwargs = dict(
        station_id=station_id,
        common_strata_count=len(common_keys),
        inversion_unique_days=inversion_days,
        normal_unique_days=normal_days,
        inversion_months=inversion_months,
        normal_months=normal_months,
        inversion_delta_abs_bias=inversion_effect,
        normal_delta_abs_bias=normal_effect,
        point_contrast=point_contrast,
        requested_replicates=replicate_count,
        strata_keys=strata_keys,
        strata_weights=strata_weights,
        **method_metadata,
    )
    if min(inversion_days, normal_days) < minimum_days:
        return _empty_result(reason="insufficient_unique_days", **empty_kwargs)
    if min(inversion_months, normal_months) < minimum_months:
        return _empty_result(reason="insufficient_months", **empty_kwargs)

    common_months = {month for month, _ in strata_keys}
    sampling_dates = work.loc[work["MONTH"].isin(common_months), "DATE"]
    try:
        sample = generate_monthly_circular_block_weights(
            sampling_dates,
            n_replicates=replicate_count,
            block_length_days=block_length,
            seed=seed,
        )
    except CalendarBlockUnavailableError as exc:
        return _empty_result(
            reason="insufficient_calendar_blocks",
            calendar_month_diagnostics=exc.month_diagnostics,
            **empty_kwargs,
        )
    support_metadata = _analysis_support_metadata(common, strata_keys, sample)
    supported_empty_kwargs = {**empty_kwargs, **support_metadata}
    if int(support_metadata["uncovered_analysis_date_count"]) > 0:
        return _empty_result(
            reason="uncovered_analysis_dates",
            sample=sample,
            calendar_month_diagnostics=sample.month_diagnostics,
            **supported_empty_kwargs,
        )
    if sample.unique_weight_patterns < 2:
        return _empty_result(
            reason="resampling_degenerate",
            sample=sample,
            calendar_month_diagnostics=sample.month_diagnostics,
            **supported_empty_kwargs,
        )
    date_position = {date: index for index, date in enumerate(sample.dates)}
    row_positions = common["DATE"].map(date_position).to_numpy(dtype=np.int64)
    row_weights = sample.weights[:, row_positions].astype(float)
    states = common["STRUCTURE_STATE"].to_numpy()
    none = common["error_none"].to_numpy(float)
    fixed = common["error_fixed"].to_numpy(float)

    replicate_effects: dict[str, np.ndarray] = {}
    valid = np.ones(replicate_count, dtype=bool)
    for state in REQUIRED_STATES:
        standardized_bias = {
            "error_none": np.zeros(replicate_count, dtype=float),
            "error_fixed": np.zeros(replicate_count, dtype=float),
        }
        for stratum_index, (month, solar) in enumerate(strata_keys):
            stratum_state_mask = (
                (states == state)
                & common["MONTH"].to_numpy().astype(str).__eq__(month)
                & common["SOLAR_STATE"].to_numpy().astype(str).__eq__(solar)
            )
            stratum_row_weights = row_weights[:, stratum_state_mask]
            denominator = stratum_row_weights.sum(axis=1)
            valid &= denominator > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                standardized_bias["error_none"] += strata_weights[stratum_index] * (
                    (stratum_row_weights @ none[stratum_state_mask]) / denominator
                )
                standardized_bias["error_fixed"] += strata_weights[stratum_index] * (
                    (stratum_row_weights @ fixed[stratum_state_mask]) / denominator
                )
        replicate_effects[state] = (
            np.abs(standardized_bias["error_fixed"])
            - np.abs(standardized_bias["error_none"])
        )

    contrasts = (
        replicate_effects["inversion_like"]
        - replicate_effects["normal_negative"]
    )
    valid &= np.isfinite(contrasts)
    valid_contrasts = contrasts[valid]
    valid_count = int(len(valid_contrasts))
    valid_fraction = valid_count / replicate_count
    if len(valid_contrasts) == 0:
        return _empty_result(
            reason="no_valid_replicates",
            sample=sample,
            calendar_month_diagnostics=sample.month_diagnostics,
            **supported_empty_kwargs,
        )
    if valid_fraction < minimum_valid_fraction:
        return _empty_result(
            reason="insufficient_valid_replicates",
            sample=sample,
            calendar_month_diagnostics=sample.month_diagnostics,
            valid_replicates=valid_count,
            valid_replicate_fraction=valid_fraction,
            replicate_contrasts=valid_contrasts,
            **supported_empty_kwargs,
        )
    ci_low, ci_high = np.quantile(valid_contrasts, [0.025, 0.975])

    return SharedStateContrastResult(
        station_id=station_id,
        coverage_eligible=True,
        reason="eligible",
        common_strata_count=len(common_keys),
        inversion_unique_days=inversion_days,
        normal_unique_days=normal_days,
        inversion_months=inversion_months,
        normal_months=normal_months,
        inversion_delta_abs_bias=float(inversion_effect),
        normal_delta_abs_bias=float(normal_effect),
        point_contrast=float(point_contrast),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        requested_replicates=replicate_count,
        valid_replicates=valid_count,
        valid_replicate_fraction=valid_fraction,
        replicate_contrasts=valid_contrasts,
        sample=sample,
        seed=seed,
        block_length_days=block_length,
        min_unique_days_per_state=minimum_days,
        min_months_per_state=minimum_months,
        min_valid_replicate_fraction=minimum_valid_fraction,
        strata_keys=strata_keys,
        strata_weights=strata_weights,
        strata_weighting=STRATA_WEIGHTING,
        estimand=ESTIMAND,
        sampling_scheme=SAMPLING_SCHEME,
        calendar_month_diagnostics=sample.month_diagnostics,
        **support_metadata,
    )


__all__ = [
    "CalendarBlockUnavailableError",
    "MonthlyCircularBlockSample",
    "SharedStateContrastResult",
    "bootstrap_shared_state_delta_abs_bias_contrast",
    "generate_monthly_circular_block_weights",
]
