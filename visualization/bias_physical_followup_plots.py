"""BIAS 물리요인 후속보고용 독립 시각화 함수.

각 함수는 분석이 끝난 DataFrame을 주입받아 한 개의 PNG만 저장한다.
분석·분류·bootstrap 계산은 수행하지 않으며, 사용자에게 보이는 표기는
관측소·ΔZ·BIAS 용어로 통일한다.
"""

from __future__ import annotations

from project_paths import WINDOWS_FONT_DIR

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


BEFORE_COLOR = "#2C7FB8"
AFTER_COLOR = "#F28E2B"
IMPROVEMENT_COLOR = "#377EB8"
WORSENING_COLOR = "#D95F5F"
EQUIVALENT_COLOR = "#7F7F7F"
UNCERTAIN_COLOR = "#D99A00"
H1_COLOR = "#7A5195"
H2_COLOR = "#168F8C"
GRID_COLOR = "#E3E3E3"
TEXT_COLOR = "#222222"


def _korean_font_family() -> str:
    """설치된 한글 sans 글꼴을 골라 matplotlib에 등록한다."""

    preferred_files = (
        (WINDOWS_FONT_DIR / "malgun.ttf"),
        (WINDOWS_FONT_DIR / "malgunbd.ttf"),
    )
    for path in preferred_files:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                return font_manager.FontProperties(fname=str(path)).get_name()
            except (OSError, RuntimeError):
                continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic"):
        if family in available:
            return family
    return "DejaVu Sans"


_FONT_FAMILY = _korean_font_family()
_RC_PARAMS = {
    "font.family": _FONT_FAMILY,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#555555",
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
}


def _validate_frame(frame: pd.DataFrame, required: Iterable[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("입력 자료는 pandas DataFrame이어야 합니다.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")


def _validate_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("출력 경로는 .png 확장자의 PNG 파일이어야 합니다.")
    return path


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")


def _empty_figure(title: str, message: str = "표시할 유효 자료가 없습니다.") -> Figure:
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    axis.axis("off")
    axis.text(
        0.5,
        0.52,
        message,
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
        transform=axis.transAxes,
    )
    figure.suptitle(title, x=0.02, y=0.96, ha="left", fontsize=14, fontweight="bold")
    return figure


def _save_plot(
    output_path: str | Path,
    *,
    title: str,
    description: str,
    builder: Callable[[], Figure],
) -> Path:
    """PNG 저장 성공·실패와 관계없이 figure를 닫는다."""

    path = _validate_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure: Figure | None = None
    try:
        with plt.rc_context(_RC_PARAMS):
            figure = builder()
            figure.patch.set_facecolor("white")
            figure.savefig(
                path,
                format="png",
                dpi=200,
                facecolor="white",
                bbox_inches="tight",
                metadata={
                    "Title": title,
                    "Description": description,
                },
            )
        return path
    finally:
        if figure is not None:
            plt.close(figure)


def _add_caption(
    figure: Figure,
    caption: str,
    *,
    bottom: float = 0.12,
    top: float = 0.93,
) -> None:
    figure.text(0.02, 0.018, caption, ha="left", va="bottom", fontsize=8.4, color="#444444")
    figure.tight_layout(rect=(0.0, bottom, 1.0, top))


def equivalent_before_after(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """‘차이 거의 없음’ 관측소의 보정 전·후 절대 BIAS를 그린다."""

    required = (
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "ABS_BIAS_NONE",
        "ABS_BIAS_FIXED",
        "BIAS_FIXED_LEVEL",
    )
    _validate_frame(frame, required)
    path = _validate_output_path(output_path)
    data = _numeric(frame, ("ABS_BIAS_NONE", "ABS_BIAS_FIXED"))
    data = data.dropna(subset=["ABS_BIAS_NONE", "ABS_BIAS_FIXED"])
    title = "‘차이 거의 없음’ 관측소의 보정 전·후 절대 BIAS"

    if data.empty:
        description = (
            "관측소 n=0, 유효 자료 없음. BIAS = 예측기온 - 실제 관측기온 (°C); "
            "이 그림은 절대 BIAS를 비교한다."
        )
        return _save_plot(path, title=title, description=description, builder=lambda: _empty_figure(title))

    description = (
        f"관측소 n={len(data)}. BIAS = 예측기온 - 실제 관측기온 (°C); "
        "점은 절대 BIAS이며 대각선 아래는 보정 후 절대 BIAS 감소를 뜻한다."
    )

    def build() -> Figure:
        figure, axis = plt.subplots(figsize=(8.2, 5.0))
        level_style = {
            "low": ("낮음 (≤0.1°C)", IMPROVEMENT_COLOR, "o"),
            "middle": ("보통 (0.1-0.5°C)", UNCERTAIN_COLOR, "s"),
            "high": ("높음 (>0.5°C)", WORSENING_COLOR, "^"),
            "낮음": ("낮음 (≤0.1°C)", IMPROVEMENT_COLOR, "o"),
            "보통": ("보통 (0.1-0.5°C)", UNCERTAIN_COLOR, "s"),
            "높음": ("높음 (>0.5°C)", WORSENING_COLOR, "^"),
        }
        plotted = pd.Series(False, index=data.index)
        for key, (label, color, marker) in level_style.items():
            mask = data["BIAS_FIXED_LEVEL"].astype("string").str.lower().eq(str(key).lower()) & ~plotted
            if not mask.any():
                continue
            axis.scatter(
                data.loc[mask, "ABS_BIAS_NONE"],
                data.loc[mask, "ABS_BIAS_FIXED"],
                s=32,
                alpha=0.72,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.4,
                label=label,
                zorder=3,
            )
            plotted |= mask
        if (~plotted).any():
            axis.scatter(
                data.loc[~plotted, "ABS_BIAS_NONE"],
                data.loc[~plotted, "ABS_BIAS_FIXED"],
                s=30,
                alpha=0.7,
                color=EQUIVALENT_COLOR,
                label="기타",
                zorder=3,
            )

        limit = float(np.nanmax(data[["ABS_BIAS_NONE", "ABS_BIAS_FIXED"]].to_numpy()))
        limit = max(0.6, limit * 1.08)
        axis.plot([0, limit], [0, limit], color="#333333", linewidth=1.0, label="변화 없음")
        for threshold in (0.1, 0.5):
            axis.axvline(threshold, color="#BDBDBD", linewidth=0.8, linestyle="--")
            axis.axhline(threshold, color="#BDBDBD", linewidth=0.8, linestyle="--")
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("보정 전 절대 BIAS (°C)")
        axis.set_ylabel("보정 후 절대 BIAS (°C)")
        axis.set_title(title, loc="left", fontweight="bold")
        _style_axis(axis, grid_axis="both")
        axis.legend(loc="upper left", frameon=False, ncol=2)
        _add_caption(figure, description)
        return figure

    return _save_plot(path, title=title, description=description, builder=build)


def worsening_by_dz(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """유의미한 악화 관측소의 평균 절대 ΔZ와 BIAS 증가량을 그린다."""

    required = ("STN", "MECHANISM", "DELTA_ABS_BIAS", "ABS_DZ_M")
    _validate_frame(frame, required)
    path = _validate_output_path(output_path)
    data = _numeric(frame, ("DELTA_ABS_BIAS", "ABS_DZ_M"))
    data = data.dropna(subset=["DELTA_ABS_BIAS", "ABS_DZ_M"])
    title = "평균 절대 ΔZ와 BIAS 악화 크기"

    if data.empty:
        description = (
            "관측소 n=0, 유효 자료 없음. 평균 절대 ΔZ 단위는 m이고, "
            "BIAS = 예측기온 - 실제 관측기온 (°C)이다."
        )
        return _save_plot(path, title=title, description=description, builder=lambda: _empty_figure(title))

    description = (
        f"관측소 n={len(data)}, 평균 절대 ΔZ 단위는 m. "
        "BIAS = 예측기온 - 실제 관측기온 (°C); BIAS 변화는 |보정 후|-|보정 전|이며 양수는 악화를 뜻한다."
    )

    def build() -> Figure:
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        mechanism_style = {
            "same_direction_worsened": ("같은 방향 악화", WORSENING_COLOR, "o"),
            "overshoot": ("과보정", UNCERTAIN_COLOR, "^"),
            "같은 방향 악화": ("같은 방향 악화", WORSENING_COLOR, "o"),
            "과보정": ("과보정", UNCERTAIN_COLOR, "^"),
        }
        plotted = pd.Series(False, index=data.index)
        mechanism_text = data["MECHANISM"].astype("string")
        for key, (label, color, marker) in mechanism_style.items():
            mask = mechanism_text.str.lower().eq(str(key).lower()) & ~plotted
            if not mask.any():
                continue
            axis.scatter(
                data.loc[mask, "ABS_DZ_M"],
                data.loc[mask, "DELTA_ABS_BIAS"],
                s=38,
                alpha=0.72,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.4,
                label=label,
                zorder=3,
            )
            plotted |= mask
        if (~plotted).any():
            axis.scatter(
                data.loc[~plotted, "ABS_DZ_M"],
                data.loc[~plotted, "DELTA_ABS_BIAS"],
                s=34,
                color=EQUIVALENT_COLOR,
                alpha=0.7,
                label="기타",
                zorder=3,
            )

        axis.axhline(0, color="#333333", linewidth=0.9)
        for sid in ("310", "594", "885"):
            mask = data["STN"].astype("string").str.replace(r"\.0$", "", regex=True).eq(sid)
            for _, row in data.loc[mask].iterrows():
                axis.annotate(
                    f"관측소 {sid}",
                    (row["ABS_DZ_M"], row["DELTA_ABS_BIAS"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_xlabel("평균 절대 ΔZ (m)")
        axis.set_ylabel("BIAS 변화 (°C, 양수=악화)")
        axis.set_title(title, loc="left", fontweight="bold")
        _style_axis(axis, grid_axis="both")
        axis.legend(loc="upper left", frameon=False)
        _add_caption(figure, description)
        return figure

    return _save_plot(path, title=title, description=description, builder=build)


def _group_dotplot(axis, groups: list[np.ndarray], labels: list[str], *, color: str) -> None:
    for position, values in enumerate(groups):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        offsets = np.linspace(-0.06, 0.06, finite.size) if finite.size > 1 else np.array([0.0])
        axis.scatter(
            np.full(finite.size, position, dtype=float) + offsets,
            finite,
            s=30,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        median = float(np.median(finite))
        axis.plot([position - 0.16, position + 0.16], [median, median], color="#222222", linewidth=1.6)
    axis.set_xticks(range(len(labels)), labels)


def high_residual_factors(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """보정 후 큰 BIAS 집단의 전·후 BIAS와 두 간접 물리요인을 그린다."""

    required = (
        "STN",
        "BIAS_none",
        "BIAS_fixed",
        "IS_PRIMARY_HIGH_RESIDUAL",
        "MEDIAN_DELTA_COAST_KM",
        "MEDIAN_RELIEF_MISMATCH_M",
    )
    _validate_frame(frame, required)
    path = _validate_output_path(output_path)
    data = _numeric(
        frame,
        ("BIAS_none", "BIAS_fixed", "MEDIAN_DELTA_COAST_KM", "MEDIAN_RELIEF_MISMATCH_M"),
    )
    data = data.assign(_HIGH=_truthy(data["IS_PRIMARY_HIGH_RESIDUAL"]))
    valid_bias = data["_HIGH"] & data[["BIAS_none", "BIAS_fixed"]].notna().all(axis=1)
    valid_factor = data[["MEDIAN_DELTA_COAST_KM", "MEDIAN_RELIEF_MISMATCH_M"]].notna().any(axis=1)
    title = "보정 후 큰 BIAS 관측소와 간접 물리요인"

    if not valid_bias.any() and not valid_factor.any():
        description = (
            "관측소 n=0, 유효 자료 없음. BIAS = 예측기온 - 실제 관측기온 (°C); "
            "해안거리·지형기복 지표는 간접 진단이다."
        )
        return _save_plot(path, title=title, description=description, builder=lambda: _empty_figure(title))

    high_n = int(data.loc[valid_bias, "STN"].astype("string").nunique())
    control_n = int(data.loc[~data["_HIGH"], "STN"].astype("string").nunique())
    description = (
        f"큰 잔여 BIAS 관측소 n={high_n}, 대조 관측소 n={control_n}. "
        "BIAS = 예측기온 - 실제 관측기온 (°C); 해안거리 차이와 지형기복 불일치는 원인의 직접 측정이 아닌 간접 진단이다."
    )

    def build() -> Figure:
        figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.4))

        paired = data.loc[valid_bias]
        for _, row in paired.iterrows():
            axes[0].plot(
                [0, 1],
                [row["BIAS_none"], row["BIAS_fixed"]],
                color="#B8B8B8",
                linewidth=0.9,
                alpha=0.8,
                zorder=1,
            )
        if not paired.empty:
            axes[0].scatter(np.zeros(len(paired)), paired["BIAS_none"], color=BEFORE_COLOR, s=32, label="보정 전", zorder=3)
            axes[0].scatter(np.ones(len(paired)), paired["BIAS_fixed"], color=AFTER_COLOR, s=32, label="보정 후", zorder=3)
        axes[0].axhline(0, color="#333333", linewidth=0.9)
        axes[0].set_xticks([0, 1], ["보정 전", "보정 후"])
        axes[0].set_ylabel("부호 있는 BIAS (°C)")
        axes[0].set_title("큰 잔여 BIAS 집단", loc="left")
        _style_axis(axes[0])

        high = data["_HIGH"]
        _group_dotplot(
            axes[1],
            [
                data.loc[high, "MEDIAN_DELTA_COAST_KM"].to_numpy(),
                data.loc[~high, "MEDIAN_DELTA_COAST_KM"].to_numpy(),
            ],
            ["큰 잔여 BIAS", "대조"],
            color=H2_COLOR,
        )
        axes[1].axhline(0, color="#555555", linewidth=0.8)
        axes[1].set_ylabel("대상 관측소-이웃 해안거리 차이 (km)")
        axes[1].set_title("해안-내륙 배치", loc="left")
        _style_axis(axes[1])

        _group_dotplot(
            axes[2],
            [
                data.loc[high, "MEDIAN_RELIEF_MISMATCH_M"].to_numpy(),
                data.loc[~high, "MEDIAN_RELIEF_MISMATCH_M"].to_numpy(),
            ],
            ["큰 잔여 BIAS", "대조"],
            color=H1_COLOR,
        )
        axes[2].set_ylabel("지형기복 불일치 (m)")
        axes[2].set_title("국지지형 불일치", loc="left")
        _style_axis(axes[2])

        figure.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold")
        _add_caption(figure, description, bottom=0.15)
        return figure

    return _save_plot(path, title=title, description=description, builder=build)


def structure_condition_bias(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """공간구조 상태별 부호 있는 BIAS와 절대 BIAS 변화를 그린다."""

    required = (
        "STN",
        "CONFIG",
        "STRUCTURE_STATE",
        "CONDITION_VARIABLE",
        "N_HOURS",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
    )
    _validate_frame(frame, required)
    path = _validate_output_path(output_path)
    data = _numeric(frame, ("N_HOURS", "BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"))
    data = data.loc[
        data["CONFIG"].astype("string").str.lower().eq("primary")
        & data["CONDITION_VARIABLE"].astype("string").str.upper().eq("ALL")
    ].copy()
    data = data.dropna(subset=["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"], how="all")
    title = "공간구조 상태별 BIAS와 보정 효과"

    if data.empty:
        description = (
            "관측소 n=0, 유효 자료 없음. 부호 있는 BIAS = 예측기온 - 실제 관측기온 (°C); "
            "공간구조는 실제 연직 역전의 직접 관측이 아니다."
        )
        return _save_plot(path, title=title, description=description, builder=lambda: _empty_figure(title))

    states = ["inversion_like", "normal_negative", "uncertain", "ineligible"]
    labels = {
        "inversion_like": "역전형 공간구조",
        "normal_negative": "정상 음의 구조",
        "uncertain": "불확실",
        "ineligible": "판정 불가",
    }
    present = [state for state in states if data["STRUCTURE_STATE"].astype("string").eq(state).any()]
    unknown = [
        state
        for state in data["STRUCTURE_STATE"].dropna().astype("string").unique().tolist()
        if state not in present
    ]
    present.extend(unknown)
    station_n = int(data["STN"].astype("string").nunique())
    hours = int(data["N_HOURS"].fillna(0).sum())
    description = (
        f"관측소 n={station_n}, 합계 {hours}시간. 부호 있는 BIAS = 예측기온 - 실제 관측기온 (°C)을 먼저 비교한다. "
        "Δ|BIAS| 양수는 악화이며, 이웃 AWS 공간구조는 실제 연직 역전의 직접 관측이 아니다."
    )

    summary = (
        data.groupby("STRUCTURE_STATE", dropna=False)[["BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]]
        .median(numeric_only=True)
    )

    def build() -> Figure:
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
        positions = np.arange(len(present), dtype=float)
        width = 0.34
        before_values = [summary.loc[state, "BIAS_none"] if state in summary.index else np.nan for state in present]
        after_values = [summary.loc[state, "BIAS_fixed"] if state in summary.index else np.nan for state in present]
        delta_values = [summary.loc[state, "DELTA_ABS_BIAS"] if state in summary.index else np.nan for state in present]

        axes[0].bar(positions - width / 2, before_values, width, color=BEFORE_COLOR, label="보정 전", zorder=3)
        axes[0].bar(positions + width / 2, after_values, width, color=AFTER_COLOR, label="보정 후", zorder=3)
        axes[0].axhline(0, color="#333333", linewidth=0.9)
        axes[0].set_xticks(positions, [labels.get(state, "기타") for state in present], rotation=12, ha="right")
        axes[0].set_ylabel("관측소별 BIAS 중앙값 (°C)")
        axes[0].set_title("부호 있는 BIAS", loc="left")
        axes[0].legend(frameon=False, loc="upper right")
        _style_axis(axes[0])

        colors = [H1_COLOR if state == "inversion_like" else EQUIVALENT_COLOR for state in present]
        axes[1].bar(positions, delta_values, width=0.58, color=colors, zorder=3)
        axes[1].axhline(0, color="#333333", linewidth=0.9)
        axes[1].set_xticks(positions, [labels.get(state, "기타") for state in present], rotation=12, ha="right")
        axes[1].set_ylabel("Δ|BIAS| 중앙값 (°C)")
        axes[1].set_title("절대 BIAS 변화 (양수=악화)", loc="left")
        _style_axis(axes[1])

        figure.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold")
        _add_caption(figure, description, bottom=0.16)
        return figure

    return _save_plot(path, title=title, description=description, builder=build)


def station_885_case(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """관측소 885의 월·태양상태별 BIAS와 공간구조 진단을 그린다."""

    required = (
        "STN",
        "MONTH",
        "SOLAR_STATE",
        "STRUCTURE_STATE",
        "N_HOURS",
        "BIAS_none",
        "BIAS_fixed",
        "DELTA_ABS_BIAS",
    )
    _validate_frame(frame, required)
    path = _validate_output_path(output_path)
    data = _numeric(frame, ("MONTH", "N_HOURS", "BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"))
    station_number = pd.to_numeric(data["STN"], errors="coerce")
    data = data.loc[station_number.eq(885)].copy()
    data = data.dropna(subset=["MONTH", "BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"], how="any")
    title = "관측소 885 사례: 월·태양상태별 BIAS"

    if data.empty:
        description = (
            "관측소 885 사례, 유효 자료 없음. BIAS = 예측기온 - 실제 관측기온 (°C); "
            "단일 사례는 전체 관측소의 대표 원인이 아니다."
        )
        return _save_plot(path, title=title, description=description, builder=lambda: _empty_figure(title))

    solar_order = {"dark": 0, "transition": 1, "bright": 2}
    data = data.assign(
        _SOLAR_ORDER=data["SOLAR_STATE"].astype("string").str.lower().map(solar_order).fillna(9)
    ).sort_values(["MONTH", "_SOLAR_ORDER"])
    state_labels = {
        "inversion_like": "역전형 공간구조",
        "normal_negative": "정상 음의 구조",
        "uncertain": "불확실",
        "ineligible": "판정 불가",
    }
    state_colors = {
        "inversion_like": H1_COLOR,
        "normal_negative": H2_COLOR,
        "uncertain": UNCERTAIN_COLOR,
        "ineligible": EQUIVALENT_COLOR,
    }
    solar_labels = {"dark": "어두움", "transition": "전환", "bright": "밝음"}
    xlabels = [
        f"{int(row.MONTH)}월\n{solar_labels.get(str(row.SOLAR_STATE).lower(), '미분류')}"
        for row in data.itertuples()
    ]
    hours = int(data["N_HOURS"].fillna(0).sum())
    description = (
        f"관측소 885 사례, 합계 {hours}시간. BIAS = 예측기온 - 실제 관측기온 (°C); "
        "Δ|BIAS| 양수는 악화이며 이 사례를 전체 관측소의 원인으로 일반화하지 않는다."
    )

    def build() -> Figure:
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
        positions = np.arange(len(data), dtype=float)
        axes[0].plot(positions, data["BIAS_none"], color=BEFORE_COLOR, marker="o", linewidth=1.6, label="보정 전")
        axes[0].plot(positions, data["BIAS_fixed"], color=AFTER_COLOR, marker="o", linewidth=1.6, label="보정 후")
        axes[0].axhline(0, color="#333333", linewidth=0.9)
        axes[0].set_xticks(positions, xlabels, rotation=0)
        axes[0].set_ylabel("부호 있는 BIAS (°C)")
        axes[0].set_title("보정 전·후 BIAS", loc="left")
        axes[0].legend(frameon=False, loc="best")
        _style_axis(axes[0])

        states = data["STRUCTURE_STATE"].astype("string").fillna("ineligible")
        bar_colors = [state_colors.get(str(state), EQUIVALENT_COLOR) for state in states]
        axes[1].bar(positions, data["DELTA_ABS_BIAS"], color=bar_colors, width=0.62, zorder=3)
        axes[1].axhline(0, color="#333333", linewidth=0.9)
        axes[1].set_xticks(positions, xlabels, rotation=0)
        axes[1].set_ylabel("Δ|BIAS| (°C, 양수=악화)")
        axes[1].set_title("공간구조별 보정 효과", loc="left")
        for state in states.unique():
            axes[1].scatter([], [], color=state_colors.get(str(state), EQUIVALENT_COLOR), label=state_labels.get(str(state), "기타"))
        handles, legend_labels = axes[1].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.965),
            ncol=3,
        )
        _style_axis(axes[1])

        figure.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold")
        _add_caption(figure, description, bottom=0.16, top=0.86)
        return figure

    return _save_plot(path, title=title, description=description, builder=build)


__all__ = [
    "equivalent_before_after",
    "worsening_by_dz",
    "high_residual_factors",
    "structure_condition_bias",
    "station_885_case",
]
