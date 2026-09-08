
from project_paths import LOO_DB
from contextlib import contextmanager
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid
import warnings

import numpy as np
import pandas as pd

import analyze_station_bias as bias_analysis
from analyze_station_bias import run_analysis


DB = str(LOO_DB)
SMOKE_OUTPUTS = {
    "bias_station_factors.csv",
    "bias_monthly.csv",
    "bias_diurnal.csv",
    "bias_run_summary.json",
}
FULL_ARTIFACTS = [
    "bias_station_factors.csv",
    "bias_monthly.csv",
    "bias_diurnal.csv",
    "bias_excluded_stations.csv",
    "bias_hypothesis_tests.csv",
    "bias_phase.png",
    "bias_delta_map.png",
    "bias_delta_distribution.png",
    "bias_temporal.png",
    "bias_physical_factors.png",
    "bias_case_studies.png",
    "bias_station_analysis_report.md",
    "bias_run_summary.json",
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


@contextmanager
def temporary_output_dir():
    temp_root = os.environ.get("BIAS_TEST_TMPDIR")
    if not temp_root:
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp
        return

    os.makedirs(temp_root, exist_ok=True)
    tmp = os.path.join(temp_root, f"bias-test-{uuid.uuid4().hex}")
    os.makedirs(tmp)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp)


def synthetic_frames(station_ids):
    count = len(station_ids)
    index = np.arange(count, dtype=float)
    dz = ((index % 19) - 9.0) / 100.0
    bias_none = 0.4 + index / 10000.0
    correction = -6.5 * dz
    bias_fixed = bias_none + correction
    station = pd.DataFrame(
        {
            "STN": station_ids,
            "n_hours": np.full(count, 2),
            "BIAS_none": bias_none,
            "BIAS_fixed": bias_fixed,
            "CORRECTION_SHIFT": correction,
            "DELTA_ABS_BIAS": np.abs(bias_fixed) - np.abs(bias_none),
            "MECHANISM": np.where(index % 2 == 0, "improved", "worsened"),
            "mean_dz_geom_km": dz,
            "mean_absdz_geom_km": np.abs(dz),
            "sd_dz_geom_km": 0.01 + index / 100000.0,
            "mean_neighbor_count": 3.0 + index % 8,
            "mean_nearest_distance_km": 1.0 + index / 1000.0,
            "mean_weighted_distance_km": 2.0 + index / 900.0,
            "mean_centroid_offset_km": 0.2 + index / 8000.0,
            "mean_neighbor_elev_sd_m": 10.0 + index / 10.0,
            "n_regimes": 1.0 + index % 3,
            "mode_cov_pct": 90.0 + index % 10,
            "elev_m": 20.0 + index,
            "relief_1km_m": 30.0 + index / 2.0,
            "coast_dist_km": 5.0 + index / 3.0,
            "is_mountain": (index.astype(int) % 3 == 0).astype(int),
            "is_island": (index.astype(int) % 11 == 0).astype(int),
            "isolated": np.zeros(count, dtype=int),
            "gamma_eff_median": -5.0 + index / 1000.0,
            "gamma_eff_iqr": 1.0 + index / 10000.0,
            "gamma_eff_day": -5.5 + index / 1100.0,
            "gamma_eff_night": -4.5 + index / 900.0,
            "night_inversion_ratio": (index % 10) / 10.0,
            "n_gamma_gated": np.full(count, 2),
            "지점명": [f"지점 {stn}" for stn in station_ids],
            "위도": 33.0 + index / 1000.0,
            "경도": 126.0 + index / 1000.0,
            "terrain_group": np.where(index % 2 == 0, "내륙", "해안"),
        }
    )
    monthly = station[
        ["STN", "n_hours", "BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
    ].copy()
    monthly.insert(1, "MONTH", "202401")
    diurnal = station[
        ["STN", "n_hours", "BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS"]
    ].copy()
    diurnal.insert(1, "HOUR", 0)
    return station, monthly, diurnal


def synthetic_evidence(count):
    rows = [
        [
            "H1",
            "exact_invariant",
            "max_abs_residual",
            "CORRECTION_SHIFT",
            "GAMMA_C_PER_KM * mean_dz_geom_km",
            count,
            "max_abs_residual_c",
            0.0,
            1.0,
            "exact_identity",
            np.nan,
            "geometry reconstructed independently from error",
        ],
        [
            "H5",
            "independent_association",
            "spearman",
            "BIAS_none",
            "mean_dz_geom_km",
            count,
            "spearman_rho",
            0.25,
            0.10,
            "pearson_r_squared",
            np.nan,
            "line one\nline two",
        ],
        [
            "H5",
            "independent_association",
            "standardized_ols",
            "BIAS_none",
            "mean_dz_geom_km",
            count,
            "standardized_beta",
            0.20,
            0.15,
            "adjusted_r_squared",
            1.2,
            "association only",
        ],
    ]
    return pd.DataFrame(rows, columns=EFFECT_COLUMNS)


def write_population_inputs(root, analyzed_ids):
    excluded_ids = [str(8000 + index) for index in range(20)]
    candidates = analyzed_ids + excluded_ids
    excluded_path = os.path.join(root, "excluded.csv")
    candidate_path = os.path.join(root, "candidates.csv")
    pd.DataFrame({"STN": excluded_ids}).to_csv(
        excluded_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"STN": candidates}).to_csv(
        candidate_path, index=False, encoding="utf-8-sig"
    )
    return excluded_path, candidate_path


def write_fake_figure(filename, *args):
    out_dir = args[-1]
    Path(out_dir, filename).write_bytes(b"synthetic png")
    return os.path.join(out_dir, filename)


@contextmanager
def mocked_full_inputs(root, fail_figure=False):
    analyzed_ids = [str(1000 + index) for index in range(529)]
    station, monthly, diurnal = synthetic_frames(analyzed_ids)
    evidence = synthetic_evidence(len(station))
    excluded_path, candidate_path = write_population_inputs(root, analyzed_ids)
    connection = Mock()
    with ExitStack() as stack:
        stack.enter_context(
            patch("analyze_station_bias.open_readonly_db", return_value=connection)
        )
        stack.enter_context(
            patch(
                "analyze_station_bias.load_analysis_station_ids",
                return_value=analyzed_ids,
            )
        )
        stack.enter_context(
            patch(
                "analyze_station_bias._collect_analysis_frames",
                return_value=(station.copy(), monthly.copy(), diurnal.copy()),
            )
        )
        stack.enter_context(
            patch(
                "analyze_station_bias._merge_full_population_features",
                side_effect=lambda frame: frame,
            )
        )
        stack.enter_context(
            patch("analyze_station_bias._build_evidence", return_value=evidence)
        )
        stack.enter_context(
            patch("analyze_station_bias.EXCLUDED_PATH", excluded_path)
        )
        stack.enter_context(
            patch("analyze_station_bias.CANDIDATE_PATH", candidate_path)
        )
        figure_names = FULL_ARTIFACTS[5:11]
        figure_functions = [
            "plot_bias_phase",
            "plot_bias_map",
            "plot_bias_distribution",
            "plot_bias_temporal",
            "plot_physical_factors",
            "plot_case_studies",
        ]
        for function, filename in zip(figure_functions, figure_names):
            if fail_figure and function == "plot_bias_phase":
                side_effect = RuntimeError("synthetic figure failure")
            else:
                side_effect = lambda *args, name=filename: write_fake_figure(
                    name, *args
                )
            stack.enter_context(
                patch(f"analyze_station_bias.{function}", side_effect=side_effect)
            )
        yield analyzed_ids, evidence, connection


@unittest.skipUnless(os.path.exists(DB), "full-network SQLite not available")
class BiasAnalysisIntegrationTests(unittest.TestCase):
    def test_phase_plot_emits_no_missing_glyph_warning(self):
        station, _, _ = synthetic_frames(["101", "102"])
        with temporary_output_dir() as out_dir:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bias_analysis.plot_bias_phase(station, out_dir)

        missing_glyph = [
            str(item.message)
            for item in caught
            if "Glyph" in str(item.message)
            and "missing from current font" in str(item.message)
        ]
        self.assertEqual(missing_glyph, [])

    def test_case_study_legend_identifies_station_and_mechanism(self):
        station, monthly, _ = synthetic_frames(["101"])
        captured = {}

        def capture_figure(fig, out_dir, filename):
            del out_dir, filename
            captured["figure"] = fig
            return "captured"

        with patch.object(bias_analysis, "_save_figure", side_effect=capture_figure):
            bias_analysis.plot_case_studies(station, monthly, "unused")

        fig = captured["figure"]
        try:
            legend_title = fig.axes[0].get_legend().get_title().get_text()
            self.assertIn("STN 101", legend_title)
            self.assertIn("improved", legend_title)
        finally:
            bias_analysis.plt.close(fig)

    def test_single_station_smoke_run(self):
        with temporary_output_dir() as tmp:
            with patch(
                "analyze_station_bias.load_analysis_station_ids",
                side_effect=AssertionError(
                    "explicit smoke must not scan the full population"
                ),
            ):
                summary = run_analysis(
                    stations=["116"],
                    out_dir=tmp,
                    enforce_population=False,
                    make_all_figures=False,
                )

            self.assertEqual(summary["analyzed_stations"], 1)
            self.assertEqual(set(os.listdir(tmp)), SMOKE_OUTPUTS)
            station = pd.read_csv(
                os.path.join(tmp, "bias_station_factors.csv"),
                encoding="utf-8-sig",
                dtype={"STN": str},
            )
            self.assertEqual(station["STN"].tolist(), ["116"])
            self.assertLess(summary["max_identity_residual_c"], 1e-10)
            self.assertLess(summary["max_identity_residual_b1"], 1e-10)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(set(summary["artifacts"]), SMOKE_OUTPUTS)
            for filename in SMOKE_OUTPUTS:
                if filename.endswith(".csv"):
                    with open(os.path.join(tmp, filename), "rb") as file:
                        self.assertEqual(file.read(3), b"\xef\xbb\xbf")

    def test_explicit_calls_reject_empty_and_duplicate_station_lists(self):
        for enforce_population in [False, True]:
            for stations, message in [
                ([], "at least one"),
                (["116", "116"], "duplicate"),
            ]:
                with self.subTest(
                    enforce_population=enforce_population, stations=stations
                ):
                    with patch(
                        "analyze_station_bias.open_readonly_db",
                        side_effect=AssertionError("validation must precede DB open"),
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            run_analysis(
                                stations=stations,
                                enforce_population=enforce_population,
                                make_all_figures=False,
                            )

    def test_unknown_station_errors_in_both_population_modes(self):
        connection = Mock()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "analyze_station_bias.open_readonly_db",
                    return_value=connection,
                )
            )
            full_loader = stack.enter_context(
                patch(
                    "analyze_station_bias.load_analysis_station_ids",
                    side_effect=AssertionError("smoke must skip full loader"),
                )
            )
            stack.enter_context(
                patch("analyze_station_bias.load_neighbor_map", return_value={})
            )
            stack.enter_context(
                patch("analyze_station_bias.load_station_meta", return_value=pd.DataFrame())
            )
            stack.enter_context(
                patch(
                    "analyze_station_bias.load_station_pairs",
                    side_effect=ValueError("no paired rows for station 999"),
                )
            )
            with self.assertRaisesRegex(ValueError, "invalid explicit smoke station 999"):
                run_analysis(
                    stations=["999"],
                    enforce_population=False,
                    make_all_figures=False,
                )
            full_loader.assert_not_called()

        connection = Mock()
        with patch(
            "analyze_station_bias.open_readonly_db", return_value=connection
        ), patch(
            "analyze_station_bias.load_analysis_station_ids", return_value=["116"]
        ):
            with self.assertRaisesRegex(ValueError, "outside paired population.*999"):
                run_analysis(
                    stations=["999"],
                    enforce_population=True,
                    make_all_figures=False,
                )

    def test_enforced_partial_call_uses_population_check_then_smoke_outputs(self):
        station, monthly, diurnal = synthetic_frames(["116"])
        population = ["116"] + [str(2000 + index) for index in range(528)]
        connection = Mock()
        with temporary_output_dir() as tmp, patch(
            "analyze_station_bias.open_readonly_db", return_value=connection
        ), patch(
            "analyze_station_bias.load_analysis_station_ids",
            return_value=population,
        ) as full_loader, patch(
            "analyze_station_bias._collect_analysis_frames",
            return_value=(station, monthly, diurnal),
        ):
            summary = run_analysis(
                stations=["116"],
                out_dir=tmp,
                enforce_population=True,
                make_all_figures=True,
            )
        full_loader.assert_called_once_with(connection)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(set(summary["artifacts"]), SMOKE_OUTPUTS)

    def test_population_contract_rejects_duplicates_size_overlap_and_union(self):
        analyzed = pd.DataFrame({"STN": [str(1000 + i) for i in range(529)]})
        excluded = pd.DataFrame({"STN": [str(8000 + i) for i in range(20)]})
        candidates = pd.concat([analyzed, excluded], ignore_index=True)
        cases = []
        duplicate_analyzed = analyzed.copy()
        duplicate_analyzed.loc[1, "STN"] = duplicate_analyzed.loc[0, "STN"]
        cases.append((duplicate_analyzed, excluded, candidates, "analyzed.*unique"))
        duplicate_excluded = excluded.copy()
        duplicate_excluded.loc[1, "STN"] = duplicate_excluded.loc[0, "STN"]
        cases.append((analyzed, duplicate_excluded, candidates, "excluded.*unique"))
        duplicate_candidates = candidates.copy()
        duplicate_candidates.loc[1, "STN"] = duplicate_candidates.loc[0, "STN"]
        cases.append((analyzed, excluded, duplicate_candidates, "candidate.*unique"))
        cases.append((analyzed, excluded, candidates.iloc[:-1], "549 candidates"))
        overlapping = excluded.copy()
        overlapping.loc[0, "STN"] = analyzed.loc[0, "STN"]
        cases.append((analyzed, overlapping, candidates, "overlap"))
        wrong_union = candidates.copy()
        wrong_union.loc[len(wrong_union) - 1, "STN"] = "99999"
        cases.append((analyzed, excluded, wrong_union, "do not match"))
        for analyzed_case, excluded_case, candidate_case, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AssertionError, message):
                    bias_analysis._validate_population_sets(
                        analyzed_case, excluded_case, candidate_case
                    )

    def test_full_branch_stages_complete_artifacts_and_manifest(self):
        with temporary_output_dir() as root:
            out_dir = os.path.join(root, "published")
            os.makedirs(out_dir)
            Path(out_dir, "old_marker.txt").write_text("old", encoding="utf-8")
            with mocked_full_inputs(root) as (_, _, connection):
                summary = run_analysis(out_dir=out_dir, make_all_figures=True)

            self.assertEqual(set(os.listdir(out_dir)), set(FULL_ARTIFACTS))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["artifacts"], FULL_ARTIFACTS)
            manifest = json.loads(
                Path(out_dir, "bias_run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, summary)
            evidence = pd.read_csv(
                os.path.join(out_dir, "bias_hypothesis_tests.csv"),
                encoding="utf-8-sig",
            )
            self.assertEqual(evidence.columns.tolist(), EFFECT_COLUMNS)
            self.assertFalse(
                any("pvalue" in column.lower().replace("_", "") for column in evidence)
            )
            manifest_time = os.stat(
                os.path.join(out_dir, "bias_run_summary.json")
            ).st_mtime_ns
            self.assertGreaterEqual(
                manifest_time,
                max(
                    os.stat(os.path.join(out_dir, name)).st_mtime_ns
                    for name in FULL_ARTIFACTS[:-1]
                ),
            )
            report = Path(out_dir, "bias_station_analysis_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("R2_TYPE", report)
            self.assertIn("Spearman", report)
            self.assertIn("표준화 OLS", report)
            self.assertIn("hypothesis CSV", report)
            self.assertIn("line one<br>line two", report)
            expected_evidence_labels = {
                "## 1. 연구 질문과 BIAS 정의": "algebraic identity",
                "## 2. 모집단 529개와 비보간 20개": "outcome-derived diagnostic",
                "## 3. 직접 항등식: B0, C, B1": "algebraic identity",
                "## 4. 전체 station의 Δ|BIAS| 분포": "algebraic identity",
                "## 5. 감률·계절·주야 진단": "outcome-derived diagnostic",
                "## 6. 독립 이웃 기하·지형 요인": "independent physical association",
                "## 7. 대표 사례": "outcome-derived diagnostic",
            }
            for heading, evidence_level in expected_evidence_labels.items():
                with self.subTest(heading=heading):
                    self.assertIn(
                        f"{heading}\n\n[증거 수준: `{evidence_level}`]",
                        report,
                    )
            connection.close.assert_called_once()

    def test_full_branch_failure_preserves_previous_output_and_cleans_staging(self):
        with temporary_output_dir() as root:
            out_dir = os.path.join(root, "published")
            os.makedirs(out_dir)
            marker = Path(out_dir, "old_marker.txt")
            marker.write_text("previous complete output", encoding="utf-8")
            with mocked_full_inputs(root, fail_figure=True):
                with self.assertRaisesRegex(RuntimeError, "synthetic figure failure"):
                    run_analysis(out_dir=out_dir, make_all_figures=True)
            self.assertEqual(marker.read_text(encoding="utf-8"), "previous complete output")
            self.assertEqual(os.listdir(out_dir), ["old_marker.txt"])
            self.assertFalse(
                any(".published.staging-" in name for name in os.listdir(root))
            )

    def test_markdown_cells_escape_line_breaks(self):
        self.assertEqual(
            bias_analysis._format_cell("line one\r\nline two\nline three"),
            "line one<br>line two<br>line three",
        )


if __name__ == "__main__":
    unittest.main()
