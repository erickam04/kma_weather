"""BIAS 물리요인 후속보고 시각화 모듈의 주입형 계약 테스트."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


try:
    plots = importlib.import_module("bias_physical_followup_plots")
    MODULE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    plots = None
    MODULE_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parent


def _equivalent_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STN": ["100", "200", "300", "400"],
            "BIAS_none": [0.05, 0.30, -0.80, 0.55],
            "BIAS_fixed": [0.06, 0.25, -0.75, 0.48],
            "ABS_BIAS_NONE": [0.05, 0.30, 0.80, 0.55],
            "ABS_BIAS_FIXED": [0.06, 0.25, 0.75, 0.48],
            "CORRECTION_SHIFT": [0.01, -0.05, 0.05, -0.07],
            "ABS_CORRECTION_SHIFT": [0.01, 0.05, 0.05, 0.07],
            "DELTA_ABS_BIAS": [0.01, -0.05, -0.05, -0.07],
            "ABS_DZ_M": [3.0, 8.0, 12.0, 20.0],
            "BIAS_NONE_LEVEL": ["low", "middle", "high", "high"],
            "BIAS_FIXED_LEVEL": ["low", "middle", "high", "middle"],
        }
    )


def _worsening_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STN": ["310", "594", "885", "100"],
            "BIAS_none": [-0.463, -0.674, 0.272, 0.10],
            "BIAS_fixed": [0.938, 0.989, 2.481, 0.20],
            "MECHANISM": [
                "overshoot",
                "overshoot",
                "same_direction_worsened",
                "same_direction_worsened",
            ],
            "CORRECTION_SHIFT": [1.401, 1.663, 2.209, 0.10],
            "DELTA_ABS_BIAS": [0.475, 0.315, 2.209, 0.10],
            "ABS_DZ_M": [215.6, 255.9, 339.9, 12.0],
            "DZ_BIN": ["200-300m", "200-300m", ">300m", "<=50m"],
        }
    )


def _high_residual_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STN": ["100", "200", "300", "400"],
            "BIAS_none": [0.70, -0.80, 0.20, -0.15],
            "BIAS_fixed": [0.72, -0.75, 0.25, -0.10],
            "BIAS_FIXED_SIGN": ["positive", "negative", "positive", "negative"],
            "BIAS_NONE_LEVEL": ["high", "high", "middle", "middle"],
            "BIAS_FIXED_LEVEL": ["high", "high", "middle", "low"],
            "HIGH_BEFORE": [True, True, False, False],
            "HIGH_AFTER": [True, True, False, False],
            "HIGH_COMMON": [True, True, False, False],
            "IS_PRIMARY_HIGH_RESIDUAL": [True, True, False, False],
            "PRIMARY_STRUCTURE_ELIGIBLE": [True, False, True, True],
            "MEDIAN_DELTA_COAST_KM": [-18.0, 22.0, -2.0, 3.0],
            "MEDIAN_RELIEF_MISMATCH_M": [130.0, 95.0, 20.0, 35.0],
            "TPI_STATUS": ["NOT_MEASURED"] * 4,
            "ANALYSIS_POPULATION": ["all_529_stations"] * 4,
        }
    )


def _structure_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STN": ["100", "100", "100", "100"],
            "CONFIG": ["primary"] * 4,
            "STRUCTURE_STATE": [
                "inversion_like",
                "normal_negative",
                "uncertain",
                "ineligible",
            ],
            "CONDITION_VARIABLE": ["ALL"] * 4,
            "CONDITION_LEVEL": ["ALL"] * 4,
            "N_HOURS": [120, 180, 60, 0],
            "N_DAYS": [30, 35, 20, 0],
            "N_MONTHS": [8, 9, 6, 0],
            "N_STRATA": [10, 12, 8, 0],
            "BIAS_none": [0.40, -0.15, 0.05, np.nan],
            "BIAS_fixed": [0.70, -0.10, 0.06, np.nan],
            "DELTA_ABS_BIAS": [0.30, -0.05, 0.01, np.nan],
            "MAE_none": [0.9, 0.7, 0.8, np.nan],
            "MAE_fixed": [1.0, 0.65, 0.8, np.nan],
            "DELTA_MAE": [0.1, -0.05, 0.0, np.nan],
            "BLOCK7_CI_LOW": [0.20, -0.10, -0.03, np.nan],
            "BLOCK7_CI_HIGH": [0.40, 0.00, 0.05, np.nan],
            "BLOCK7_REPLICATES": [2000, 2000, 2000, 0],
            "COVERAGE_ELIGIBLE": [True, True, True, False],
            "DROP_REASON": ["", "", "", "insufficient_coverage"],
            "EVIDENCE_GRADE": ["proxy", "proxy", "proxy", "not_assessed"],
            "LIMITATION": ["regional spatial proxy"] * 4,
            "ANALYSIS_POPULATION": ["all_529_stations"] * 4,
            "SUBSET_MEANINGFUL_WORSENING": [False] * 4,
            "SUBSET_HIGH_AFTER": [False] * 4,
        }
    )


def _station_885_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STN": ["885"] * 4,
            "MONTH": [1, 1, 7, 7],
            "SOLAR_STATE": ["dark", "bright", "transition", "bright"],
            "STRUCTURE_STATE": [
                "inversion_like",
                "normal_negative",
                "inversion_like",
                "uncertain",
            ],
            "N_HOURS": [50, 70, 40, 60],
            "N_DAYS": [20, 25, 18, 22],
            "N_MONTHS": [1, 1, 1, 1],
            "BIAS_none": [0.20, 0.30, 0.40, 0.25],
            "BIAS_fixed": [2.20, 2.00, 2.80, 2.30],
            "DELTA_ABS_BIAS": [2.00, 1.70, 2.40, 2.05],
            "MAE_none": [0.8, 0.9, 1.0, 0.85],
            "MAE_fixed": [2.3, 2.2, 3.0, 2.5],
            "DELTA_MAE": [1.5, 1.3, 2.0, 1.65],
            "DELTA_COAST_KM": [8.0, 8.0, 7.5, 7.5],
            "H2_EXPECTED_BIAS_SIGN": ["negative", "not_prespecified", "not_prespecified", "positive"],
            "CASE_SCOPE": ["station_885_case"] * 4,
            "EVIDENCE_GRADE": ["case_diagnostic"] * 4,
            "LIMITATION": ["not a vertical inversion measurement"] * 4,
        }
    )


PLOT_CASES = {
    "equivalent_before_after": _equivalent_fixture,
    "worsening_by_dz": _worsening_fixture,
    "high_residual_factors": _high_residual_fixture,
    "structure_condition_bias": _structure_fixture,
    "station_885_case": _station_885_fixture,
}


class PlotContractTests(unittest.TestCase):
    def function(self, name):
        self.assertIsNotNone(
            plots,
            msg=f"bias_physical_followup_plots module is missing: {MODULE_IMPORT_ERROR}",
        )
        function = getattr(plots, name, None)
        self.assertTrue(callable(function), msg=f"required plot function is missing: {name}")
        return function

    def assert_decodable_png(self, path: Path) -> dict:
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1_000)
        image = mpimg.imread(path)
        self.assertGreater(image.shape[0], 100)
        self.assertGreater(image.shape[1], 100)
        with Image.open(path) as opened:
            opened.load()
            self.assertEqual(opened.format, "PNG")
            return dict(opened.info)

    def run_plot_case(self, name: str, frame: pd.DataFrame) -> tuple[Path, dict]:
        function = self.function(name)
        temp_dir = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp_dir.cleanup)
        output = Path(temp_dir.name) / "nested" / f"{name}.png"
        before = set(plt.get_fignums())

        returned = function(frame, output)

        self.assertEqual(Path(returned), output)
        self.assertTrue(output.parent.is_dir())
        self.assertEqual(set(plt.get_fignums()), before)
        return output, self.assert_decodable_png(output)

    def test_equivalent_before_after_writes_png_and_closes_figure(self):
        """산점도 저장·상위폴더 생성·figure 정리 계약이 사라지는 회귀를 잡는다."""
        _, info = self.run_plot_case("equivalent_before_after", _equivalent_fixture())
        self.assertIn("관측소", info["Description"])
        self.assertIn("BIAS = 예측기온 - 실제 관측기온", info["Description"])

    def test_worsening_by_dz_writes_png_with_delta_z_caption(self):
        """악화 그림이 ΔZ 단위와 양의 BIAS 변화 의미를 잃는 회귀를 잡는다."""
        _, info = self.run_plot_case("worsening_by_dz", _worsening_fixture())
        self.assertIn("ΔZ", info["Description"])
        self.assertIn("양수", info["Description"])

    def test_high_residual_factors_writes_png_and_declares_proxy_limit(self):
        """큰 잔여 BIAS 그림이 직접 원인증명처럼 보이는 회귀를 잡는다."""
        _, info = self.run_plot_case("high_residual_factors", _high_residual_fixture())
        self.assertIn("간접", info["Description"])
        self.assertIn("관측소", info["Description"])

    def test_structure_condition_bias_writes_png_with_signed_bias_first(self):
        """구조별 그림이 signed BIAS 대신 시간별 절대오차만 그리는 회귀를 잡는다."""
        _, info = self.run_plot_case("structure_condition_bias", _structure_fixture())
        self.assertIn("부호 있는 BIAS", info["Description"])
        self.assertIn("실제 연직 역전", info["Description"])

    def test_station_885_case_writes_png_with_case_scope(self):
        """한남 사례가 전체 관측소의 대표 원인처럼 일반화되는 회귀를 잡는다."""
        _, info = self.run_plot_case("station_885_case", _station_885_fixture())
        self.assertIn("관측소 885", info["Description"])
        self.assertIn("사례", info["Description"])

    def test_empty_and_all_na_frames_render_explicit_no_data_png(self):
        """자료 없음이 예외·빈 파일·오해 가능한 빈 축으로 바뀌는 회귀를 잡는다."""
        for name, fixture in PLOT_CASES.items():
            with self.subTest(name=name, kind="empty"):
                empty = fixture().iloc[0:0]
                _, info = self.run_plot_case(name, empty)
                self.assertIn("유효 자료 없음", info["Description"])

            with self.subTest(name=name, kind="all_na"):
                all_na = fixture().copy()
                for column in all_na.select_dtypes(include=[np.number]).columns:
                    all_na[column] = np.nan
                _, info = self.run_plot_case(name, all_na)
                self.assertIn("유효 자료 없음", info["Description"])

    def test_missing_columns_and_non_png_path_fail_explicitly(self):
        """schema 누락이나 잘못된 확장자가 조용히 왜곡된 그림을 만드는 회귀를 잡는다."""
        for name, fixture in PLOT_CASES.items():
            function = self.function(name)
            with self.subTest(name=name, kind="schema"):
                with self.assertRaisesRegex(ValueError, "필수 열"):
                    function(pd.DataFrame({"wrong": [1]}), Path("unused.png"))
            with self.subTest(name=name, kind="extension"):
                with self.assertRaisesRegex(ValueError, "PNG"):
                    function(fixture(), Path("unused.jpg"))

    def test_save_failure_still_closes_figure(self):
        """PNG 저장 실패가 열린 matplotlib figure를 남기는 회귀를 잡는다."""
        function = self.function("equivalent_before_after")
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            output = Path(temp_dir) / "nested" / "figure.png"
            with mock.patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=RuntimeError("injected save failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected save failure"):
                    function(_equivalent_fixture(), output)
        self.assertEqual(set(plt.get_fignums()), before)

    def test_accessibility_metadata_never_exposes_forbidden_display_terms(self):
        """사용자-facing 문구에 내부 영문·기하 변수명이 되살아나는 회귀를 잡는다."""
        for name, fixture in PLOT_CASES.items():
            with self.subTest(name=name):
                _, info = self.run_plot_case(name, fixture())
                description = info["Description"].lower()
                self.assertNotIn("station", description)
                self.assertNotIn("dz_geom", description)

    def test_visible_labels_use_korean_observation_terms(self):
        """실제 축·범례에 영문 target/station 또는 내부 ΔZ명이 노출되는 회귀를 잡는다."""
        function = self.function("high_residual_factors")
        visible_text = []

        def inspect_figure(figure, *args, **kwargs):
            visible_text.extend(text.get_text() for text in figure.texts)
            for axis in figure.axes:
                visible_text.extend(
                    [axis.get_title(), axis.get_xlabel(), axis.get_ylabel()]
                )
                visible_text.extend(text.get_text() for text in axis.get_xticklabels())
                visible_text.extend(text.get_text() for text in axis.get_yticklabels())
                legend = axis.get_legend()
                if legend is not None:
                    visible_text.extend(text.get_text() for text in legend.get_texts())

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            output = Path(temp_dir) / "visible-labels.png"
            with mock.patch("matplotlib.figure.Figure.savefig", new=inspect_figure):
                function(_high_residual_fixture(), output)

        joined = " ".join(visible_text).lower()
        self.assertNotIn("target", joined)
        self.assertNotIn("station", joined)
        self.assertNotIn("dz_geom", joined)

    def test_station_885_legend_does_not_cover_panel_title(self):
        """공간구조 범례가 오른쪽 panel 제목을 가리는 시각 회귀를 잡는다."""
        function = self.function("station_885_case")
        overlap_results = []

        def inspect_figure(figure, *args, **kwargs):
            figure.canvas.draw()
            axis = figure.axes[1]
            legends = [legend for legend in [axis.get_legend(), *figure.legends] if legend is not None]
            self.assertEqual(len(legends), 1)
            renderer = figure.canvas.get_renderer()
            legend_box = legends[0].get_window_extent(renderer)
            title_box = axis.title.get_window_extent(renderer)
            overlap_results.append(legend_box.overlaps(title_box))

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            output = Path(temp_dir) / "legend-layout.png"
            with mock.patch("matplotlib.figure.Figure.savefig", new=inspect_figure):
                function(_station_885_fixture(), output)

        self.assertEqual(overlap_results, [False])

    def test_station_885_uses_classifier_solar_domain_without_day_night_overclaim(self):
        """transition을 기타로 버리거나 태양고도 상태를 주간·야간으로 과장하는 회귀를 잡는다."""
        function = self.function("station_885_case")
        frame = _station_885_fixture().iloc[:3].copy()
        frame.loc[:, "MONTH"] = [1, 1, 1]
        frame.loc[:, "SOLAR_STATE"] = ["dark", "transition", "bright"]
        captured_labels = []

        def inspect_figure(figure, *args, **kwargs):
            figure.canvas.draw()
            for axis in figure.axes:
                captured_labels.append([label.get_text() for label in axis.get_xticklabels()])

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            output = Path(temp_dir) / "solar-state-labels.png"
            with mock.patch("matplotlib.figure.Figure.savefig", new=inspect_figure):
                function(frame, output)

        expected = ["1월\n어두움", "1월\n전환", "1월\n밝음"]
        self.assertEqual(captured_labels, [expected, expected])
        joined = " ".join(label for labels in captured_labels for label in labels)
        for forbidden in ("주간", "야간", "박명", "기타"):
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
