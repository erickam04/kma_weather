"""월층화 달력 연속 7일 블록 재표집의 순수 계약 테스트."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bias_physical_block_bootstrap import (
    bootstrap_shared_state_delta_abs_bias_contrast,
    generate_monthly_circular_block_weights,
)


def _reference_monthly_calendar_blocks(dates, *, n_replicates, block_length, seed):
    """달력상 연속 block만 허용하는 독립 명세용 reference resampler."""
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    month_codes = normalized.to_period("M")
    months = month_codes.unique().sort_values()
    rng = np.random.default_rng(seed)
    selected = np.empty((n_replicates, len(normalized)), dtype=np.int64)

    for replicate in range(n_replicates):
        output = []
        for month in months:
            positions = np.flatnonzero(month_codes == month)
            position_by_date = {normalized[position]: position for position in positions}
            valid_blocks = []
            for position in positions:
                start = normalized[position]
                block_dates = [start + pd.Timedelta(days=offset) for offset in range(block_length)]
                if all(date in position_by_date for date in block_dates):
                    valid_blocks.append([position_by_date[date] for date in block_dates])
            if not valid_blocks:
                raise ValueError("month has no complete calendar-contiguous block")
            month_output = []
            while len(month_output) < len(positions):
                block = valid_blocks[int(rng.integers(0, len(valid_blocks)))]
                month_output.extend(block)
            output.extend(month_output[: len(positions)])
        selected[replicate] = output

    weights = np.zeros_like(selected)
    for replicate, indices in enumerate(selected):
        weights[replicate] = np.bincount(indices, minlength=len(normalized))
    return normalized, selected, weights


def _paired_state_frame(months=(1, 2), days_per_month=9):
    rows = []
    for month in months:
        for day in range(1, days_per_month + 1):
            date = pd.Timestamp(2024, month, day)
            day_signal = ((day * 3 + month) % 7) - 3
            rows.extend(
                [
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=2),
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": "inversion_like",
                        "error_none": -1.0 + 0.2 * day_signal,
                        "error_fixed": -0.2 + 0.1 * day_signal,
                    },
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=3),
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": "normal_negative",
                        "error_none": 0.4 - 0.15 * day_signal,
                        "error_fixed": 0.7 + 0.05 * day_signal,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _reference_shared_contrast(frame, weights, dates):
    work = frame.copy()
    work["DATE"] = pd.to_datetime(work["TM"]).dt.normalize()
    date_position = {date: index for index, date in enumerate(dates)}
    row_positions = work["DATE"].map(date_position).to_numpy(dtype=np.int64)
    states = work["STRUCTURE_STATE"].to_numpy()
    none = work["error_none"].to_numpy(float)
    fixed = work["error_fixed"].to_numpy(float)
    output = []
    for replicate_weights in weights:
        row_weights = replicate_weights[row_positions].astype(float)
        state_effects = {}
        for state in ("inversion_like", "normal_negative"):
            mask = states == state
            denominator = row_weights[mask].sum()
            bias_none = np.dot(row_weights[mask], none[mask]) / denominator
            bias_fixed = np.dot(row_weights[mask], fixed[mask]) / denominator
            state_effects[state] = abs(bias_fixed) - abs(bias_none)
        output.append(state_effects["inversion_like"] - state_effects["normal_negative"])
    return np.asarray(output)


class MonthlyCircularWeightTests(unittest.TestCase):
    def test_exact_indices_and_weights_match_independent_reference(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D").append(
            pd.date_range("2024-02-01", periods=8, freq="D")
        )
        expected_dates, expected_indices, expected_weights = (
            _reference_monthly_calendar_blocks(
                dates, n_replicates=8, block_length=7, seed=9381
            )
        )

        actual = generate_monthly_circular_block_weights(
            dates, n_replicates=8, block_length_days=7, seed=9381
        )

        pd.testing.assert_index_equal(actual.dates, expected_dates)
        np.testing.assert_array_equal(actual.sampled_indices, expected_indices)
        np.testing.assert_array_equal(actual.weights, expected_weights)
        self.assertEqual(actual.block_length_days, 7)

    def test_same_seed_repeats_and_different_seed_changes_sample(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        first = generate_monthly_circular_block_weights(dates, n_replicates=10, seed=1)
        repeated = generate_monthly_circular_block_weights(dates, n_replicates=10, seed=1)
        different = generate_monthly_circular_block_weights(dates, n_replicates=10, seed=2)

        np.testing.assert_array_equal(first.sampled_indices, repeated.sampled_indices)
        self.assertFalse(np.array_equal(first.sampled_indices, different.sampled_indices))

    def test_calendar_blocks_never_wrap_or_cross_into_next_month(self):
        dates = pd.date_range("2024-01-25", periods=7, freq="D").append(
            pd.date_range("2024-02-01", periods=7, freq="D")
        )
        sample = generate_monthly_circular_block_weights(
            dates, n_replicates=12, block_length_days=7, seed=3
        )

        for indices in sample.sampled_indices:
            january, february = indices[:7], indices[7:]
            np.testing.assert_array_equal(january, np.arange(7))
            np.testing.assert_array_equal(february, np.arange(7, 14))
            self.assertTrue(
                np.all(np.diff(sample.dates[january].to_numpy()) == np.timedelta64(1, "D"))
            )
            self.assertTrue(
                np.all(np.diff(sample.dates[february].to_numpy()) == np.timedelta64(1, "D"))
            )

    def test_six_days_are_insufficient_but_seven_contiguous_leap_days_pass(self):
        six_days = pd.date_range("2024-01-01", periods=6, freq="D")
        seven_leap_days = pd.date_range("2024-02-23", periods=7, freq="D")
        with self.assertRaises(ValueError):
            generate_monthly_circular_block_weights(
                six_days, n_replicates=5, block_length_days=7, seed=11
            )
        sample = generate_monthly_circular_block_weights(
            seven_leap_days, n_replicates=5, block_length_days=7, seed=11
        )

        self.assertIn(pd.Timestamp("2024-02-29"), sample.dates)
        np.testing.assert_array_equal(sample.weights.sum(axis=1), np.full(5, 7))
        for indices in sample.sampled_indices:
            np.testing.assert_array_equal(indices, np.arange(7))

    def test_sparse_calendar_gap_reports_unsupported_dates_and_literal_eighteen_day_gap(self):
        dates = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        ).append(pd.date_range("2024-01-20", periods=7, freq="D"))
        sample = generate_monthly_circular_block_weights(
            dates, n_replicates=10, block_length_days=7, seed=19
        )
        for indices in sample.sampled_indices:
            for start in range(0, 7, 7):
                chosen = sample.dates[indices[start : start + 7]]
                self.assertLessEqual((chosen.max() - chosen.min()).days, 6)
            self.assertFalse(np.isin(indices, [0, 1]).any())
        self.assertEqual(sample.month_diagnostics[0]["max_gap_days"], 18)
        self.assertEqual(
            sample.unsupported_dates,
            ("2024-01-01", "2024-01-02"),
        )
        np.testing.assert_array_equal(sample.weights[:, :2], 0)

    def test_duplicate_hours_do_not_change_date_sampling_population(self):
        dates = pd.date_range("2024-03-01", periods=12, freq="D")
        duplicated = dates.repeat(24)
        unique_sample = generate_monthly_circular_block_weights(
            dates, n_replicates=4, seed=15
        )
        duplicate_sample = generate_monthly_circular_block_weights(
            duplicated, n_replicates=4, seed=15
        )
        pd.testing.assert_index_equal(unique_sample.dates, duplicate_sample.dates)
        np.testing.assert_array_equal(unique_sample.weights, duplicate_sample.weights)


class SharedStateContrastTests(unittest.TestCase):
    def test_shared_date_weights_match_independent_replicate_contrast_and_ci(self):
        frame = _paired_state_frame()
        _, _, reference_weights = _reference_monthly_calendar_blocks(
            pd.to_datetime(frame["TM"]).dt.normalize(),
            n_replicates=200,
            block_length=7,
            seed=72,
        )
        dates = pd.DatetimeIndex(pd.to_datetime(frame["TM"]).dt.normalize()).unique().sort_values()
        expected_replicates = _reference_shared_contrast(frame, reference_weights, dates)
        expected_ci = np.quantile(expected_replicates, [0.025, 0.975])

        result = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=200,
            seed=72,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        self.assertTrue(result.coverage_eligible)
        self.assertEqual(result.reason, "eligible")
        self.assertEqual(result.valid_replicates, 200)
        np.testing.assert_allclose(result.replicate_contrasts, expected_replicates, rtol=0, atol=1e-12)
        np.testing.assert_allclose([result.ci_low, result.ci_high], expected_ci, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result.sample.weights, reference_weights)

    def test_same_seed_repeats_and_different_seed_changes_contrast_replicates(self):
        frame = _paired_state_frame(days_per_month=12)
        kwargs = dict(min_unique_days_per_state=1, min_months_per_state=1, n_replicates=80)
        first = bootstrap_shared_state_delta_abs_bias_contrast(frame, seed=7, **kwargs)
        repeated = bootstrap_shared_state_delta_abs_bias_contrast(frame, seed=7, **kwargs)
        different = bootstrap_shared_state_delta_abs_bias_contrast(frame, seed=8, **kwargs)

        np.testing.assert_array_equal(first.replicate_contrasts, repeated.replicate_contrasts)
        self.assertFalse(np.array_equal(first.replicate_contrasts, different.replicate_contrasts))

    def test_default_coverage_requires_each_state_thirty_days_and_six_months(self):
        sufficient = _paired_state_frame(months=range(1, 7), days_per_month=5)
        calendar_support = []
        for month in range(1, 7):
            for day in range(6, 11):
                calendar_support.append(
                    {
                        "STN": "100",
                        "TM": pd.Timestamp(2024, month, day, 12),
                        "SOLAR_STATE": "transition",
                        "STRUCTURE_STATE": "uncertain",
                        "error_none": 0.0,
                        "error_fixed": 0.0,
                    }
                )
        sufficient = pd.concat(
            [sufficient, pd.DataFrame(calendar_support)], ignore_index=True
        )
        passed = bootstrap_shared_state_delta_abs_bias_contrast(
            sufficient, n_replicates=20, seed=2
        )
        self.assertTrue(passed.coverage_eligible)
        self.assertEqual((passed.inversion_unique_days, passed.normal_unique_days), (30, 30))
        self.assertEqual((passed.inversion_months, passed.normal_months), (6, 6))
        self.assertEqual(passed.valid_replicates, 20)

        insufficient_days = sufficient.drop(
            sufficient.index[
                (sufficient["STRUCTURE_STATE"] == "normal_negative")
                & (sufficient["TM"] == pd.Timestamp("2024-06-05 03:00"))
            ]
        )
        failed_days = bootstrap_shared_state_delta_abs_bias_contrast(
            insufficient_days, n_replicates=20, seed=2
        )
        self.assertFalse(failed_days.coverage_eligible)
        self.assertEqual(failed_days.reason, "insufficient_unique_days")
        self.assertEqual(failed_days.valid_replicates, 0)
        self.assertEqual(len(failed_days.replicate_contrasts), 0)
        self.assertTrue(np.isnan(failed_days.ci_low))
        self.assertTrue(np.isnan(failed_days.ci_high))

        five_months = sufficient.loc[pd.to_datetime(sufficient["TM"]).dt.month <= 5]
        failed_months = bootstrap_shared_state_delta_abs_bias_contrast(
            five_months,
            n_replicates=20,
            seed=2,
            min_unique_days_per_state=20,
        )
        self.assertFalse(failed_months.coverage_eligible)
        self.assertEqual(failed_months.reason, "insufficient_months")
        self.assertEqual(failed_months.valid_replicates, 0)

    def test_missing_state_and_no_common_month_solar_stratum_are_explicit(self):
        only_inversion = _paired_state_frame().loc[
            lambda value: value["STRUCTURE_STATE"].eq("inversion_like")
        ]
        missing = bootstrap_shared_state_delta_abs_bias_contrast(
            only_inversion,
            n_replicates=10,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )
        self.assertFalse(missing.coverage_eligible)
        self.assertEqual(missing.reason, "missing_required_state")
        self.assertEqual(missing.valid_replicates, 0)

        separated = _paired_state_frame(months=(1,), days_per_month=9)
        separated.loc[
            separated["STRUCTURE_STATE"].eq("normal_negative"), "SOLAR_STATE"
        ] = "bright"
        no_common = bootstrap_shared_state_delta_abs_bias_contrast(
            separated,
            n_replicates=10,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )
        self.assertFalse(no_common.coverage_eligible)
        self.assertEqual(no_common.reason, "no_common_month_solar_strata")

    def test_common_strata_filter_excludes_unmatched_rows_from_point_and_replicates(self):
        frame = _paired_state_frame(months=(1,), days_per_month=10)
        unmatched = pd.DataFrame(
            [
                {
                    "STN": "100",
                    "TM": pd.Timestamp("2024-02-01 12:00"),
                    "SOLAR_STATE": "bright",
                    "STRUCTURE_STATE": "inversion_like",
                    "error_none": 1000.0,
                    "error_fixed": -1000.0,
                }
            ]
        )
        base = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=30,
            seed=4,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )
        augmented = bootstrap_shared_state_delta_abs_bias_contrast(
            pd.concat([frame, unmatched], ignore_index=True),
            n_replicates=30,
            seed=4,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )
        self.assertAlmostEqual(base.point_contrast, augmented.point_contrast)
        np.testing.assert_array_equal(base.replicate_contrasts, augmented.replicate_contrasts)

    def test_equal_stratum_signed_bias_standardization_blocks_simpson_reversal(self):
        rows = []
        for month, fixed_error, inversion_hours, normal_hours in (
            (1, 10.0, range(0, 10), range(10, 11)),
            (2, 0.0, range(0, 1), range(1, 11)),
        ):
            for day in range(1, 11):
                date = pd.Timestamp(2024, month, day)
                for hour in inversion_hours:
                    rows.append(
                        {
                            "STN": "100",
                            "TM": date + pd.Timedelta(hours=hour),
                            "SOLAR_STATE": "dark",
                            "STRUCTURE_STATE": "inversion_like",
                            "error_none": 0.0,
                            "error_fixed": fixed_error,
                        }
                    )
                for hour in normal_hours:
                    rows.append(
                        {
                            "STN": "100",
                            "TM": date + pd.Timedelta(hours=hour),
                            "SOLAR_STATE": "dark",
                            "STRUCTURE_STATE": "normal_negative",
                            "error_none": 0.0,
                            "error_fixed": fixed_error,
                        }
                    )
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            pd.DataFrame(rows),
            n_replicates=100,
            seed=27,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        self.assertAlmostEqual(result.point_contrast, 0.0, places=12)
        np.testing.assert_allclose(result.replicate_contrasts, 0.0, rtol=0, atol=1e-12)
        self.assertEqual(result.strata_weighting, "equal_common_month_solar_strata")
        self.assertEqual(
            result.estimand,
            "standardized_signed_bias_then_delta_abs_bias_contrast",
        )
        self.assertEqual(result.strata_keys, (("2024-01", "dark"), ("2024-02", "dark")))
        np.testing.assert_allclose(result.strata_weights, [0.5, 0.5], rtol=0, atol=0)

    def test_uncovered_analysis_date_is_ineligible_without_changing_full_point_estimand(self):
        rows = []
        for day in range(1, 9):
            date = pd.Timestamp(2024, 1, day)
            for state, hour in (("inversion_like", 2), ("normal_negative", 3)):
                rows.append(
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=hour),
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": state,
                        "error_none": 0.0,
                        "error_fixed": 0.0,
                    }
                )
        rows.append(
            {
                "STN": "100",
                "TM": pd.Timestamp("2024-01-20 02:00"),
                "SOLAR_STATE": "dark",
                "STRUCTURE_STATE": "inversion_like",
                "error_none": 0.0,
                "error_fixed": 100.0,
            }
        )
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            pd.DataFrame(rows),
            n_replicates=100,
            seed=12,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        self.assertFalse(result.coverage_eligible)
        self.assertEqual(result.reason, "uncovered_analysis_dates")
        self.assertAlmostEqual(result.point_contrast, 100 / 9)
        self.assertEqual(result.valid_replicates, 0)
        self.assertEqual(len(result.replicate_contrasts), 0)
        self.assertTrue(np.isnan(result.ci_low))
        self.assertTrue(np.isnan(result.ci_high))
        self.assertTrue(result.analysis_support_checked)
        self.assertEqual(result.analysis_date_count, 9)
        self.assertEqual(result.supported_analysis_date_count, 8)
        self.assertEqual(result.uncovered_analysis_date_count, 1)
        self.assertAlmostEqual(result.analysis_date_support_fraction, 8 / 9)
        self.assertEqual(result.uncovered_analysis_dates, ("2024-01-20",))
        jan20_position = result.sample.dates.get_loc(pd.Timestamp("2024-01-20"))
        self.assertEqual(int(result.sample.weights[:, jan20_position].max()), 0)

    def test_analysis_support_diagnostics_cover_both_states_and_every_common_stratum(self):
        rows = []
        for month in (1, 2):
            for day in range(1, 9):
                date = pd.Timestamp(2024, month, day)
                for state, hour in (("inversion_like", 2), ("normal_negative", 3)):
                    rows.append(
                        {
                            "STN": "100",
                            "TM": date + pd.Timedelta(hours=hour),
                            "SOLAR_STATE": "dark",
                            "STRUCTURE_STATE": state,
                            "error_none": 0.0,
                            "error_fixed": 0.0,
                        }
                    )
        rows.extend(
            [
                {
                    "STN": "100",
                    "TM": pd.Timestamp("2024-01-20 02:00"),
                    "SOLAR_STATE": "dark",
                    "STRUCTURE_STATE": "inversion_like",
                    "error_none": 0.0,
                    "error_fixed": 1.0,
                },
                {
                    "STN": "100",
                    "TM": pd.Timestamp("2024-02-20 03:00"),
                    "SOLAR_STATE": "dark",
                    "STRUCTURE_STATE": "normal_negative",
                    "error_none": 0.0,
                    "error_fixed": 1.0,
                },
            ]
        )
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            pd.DataFrame(rows),
            n_replicates=50,
            seed=13,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        self.assertEqual(result.reason, "uncovered_analysis_dates")
        diagnostics = {
            (item["month"], item["solar_state"], item["structure_state"]): item
            for item in result.analysis_support_diagnostics
        }
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(
            diagnostics[("2024-01", "dark", "inversion_like")]["uncovered_dates"],
            ("2024-01-20",),
        )
        self.assertEqual(
            diagnostics[("2024-02", "dark", "normal_negative")]["uncovered_dates"],
            ("2024-02-20",),
        )
        self.assertEqual(
            diagnostics[("2024-01", "dark", "normal_negative")]["uncovered_unique_dates"],
            0,
        )
        self.assertEqual(
            diagnostics[("2024-02", "dark", "inversion_like")]["uncovered_unique_dates"],
            0,
        )

    def test_resampling_degenerate_weights_are_not_an_inferential_ci(self):
        frame = _paired_state_frame(months=range(1, 7), days_per_month=7)
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            frame, n_replicates=100, seed=4
        )

        self.assertFalse(result.coverage_eligible)
        self.assertEqual(result.reason, "resampling_degenerate")
        self.assertEqual(result.valid_replicates, 0)
        self.assertTrue(np.isnan(result.ci_low))
        self.assertTrue(np.isnan(result.ci_high))
        self.assertEqual(result.sample.unique_weight_patterns, 1)

    def test_valid_replicate_fraction_gate_has_inclusive_prespecified_edge(self):
        rows = []
        for day in range(1, 15):
            date = pd.Timestamp(2024, 1, day)
            if day == 1:
                rows.append(
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=2),
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": "inversion_like",
                        "error_none": -1.0,
                        "error_fixed": -0.5,
                    }
                )
            elif day == 14:
                rows.append(
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=3),
                        "SOLAR_STATE": "dark",
                        "STRUCTURE_STATE": "normal_negative",
                        "error_none": 1.0,
                        "error_fixed": 0.5,
                    }
                )
            else:
                rows.append(
                    {
                        "STN": "100",
                        "TM": date + pd.Timedelta(hours=12),
                        "SOLAR_STATE": "transition",
                        "STRUCTURE_STATE": "uncertain",
                        "error_none": 0.0,
                        "error_fixed": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        failed = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=1000,
            seed=33,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )
        self.assertFalse(failed.coverage_eligible)
        self.assertEqual(failed.reason, "insufficient_valid_replicates")
        self.assertLess(failed.valid_replicate_fraction, 0.95)
        self.assertGreater(failed.valid_replicates, 0)
        self.assertTrue(np.isnan(failed.ci_low))

        edge = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=1000,
            seed=33,
            min_unique_days_per_state=1,
            min_months_per_state=1,
            min_valid_replicate_fraction=failed.valid_replicate_fraction,
        )
        self.assertTrue(edge.coverage_eligible)
        self.assertEqual(edge.reason, "eligible")
        self.assertAlmostEqual(edge.valid_replicate_fraction, failed.valid_replicate_fraction)

        above_edge = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=1000,
            seed=33,
            min_unique_days_per_state=1,
            min_months_per_state=1,
            min_valid_replicate_fraction=np.nextafter(
                failed.valid_replicate_fraction, 1.0
            ),
        )
        self.assertFalse(above_edge.coverage_eligible)
        self.assertEqual(above_edge.reason, "insufficient_valid_replicates")

    def test_no_valid_replicates_is_explicit_with_complete_calendar_support(self):
        """달력 support는 완전하지만 두 상태가 한 replicate에 함께 안 뽑히는 분기를 고정한다."""
        rows = []
        for day in range(1, 15):
            date = pd.Timestamp(2024, 1, day)
            if day == 1:
                state, error_none, error_fixed = "inversion_like", -1.0, -0.5
            elif day == 14:
                state, error_none, error_fixed = "normal_negative", 1.0, 0.5
            else:
                state, error_none, error_fixed = "uncertain", 0.0, 0.0
            rows.append(
                {
                    "STN": "100",
                    "TM": date + pd.Timedelta(hours=2),
                    "SOLAR_STATE": "dark",
                    "STRUCTURE_STATE": state,
                    "error_none": error_none,
                    "error_fixed": error_fixed,
                }
            )

        result = bootstrap_shared_state_delta_abs_bias_contrast(
            pd.DataFrame(rows),
            n_replicates=2,
            seed=0,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        self.assertEqual(result.analysis_date_support_fraction, 1.0)
        self.assertFalse(result.coverage_eligible)
        self.assertEqual(result.reason, "no_valid_replicates")
        self.assertEqual(result.valid_replicates, 0)
        self.assertTrue(np.isnan(result.ci_low))
        self.assertTrue(np.isnan(result.ci_high))

    def test_state_and_solar_domains_reject_typos_case_and_empty_values(self):
        frame = _paired_state_frame(months=(1,), days_per_month=9)
        mutations = [
            ("STRUCTURE_STATE", "inversion_like "),
            ("STRUCTURE_STATE", "INVERSION_LIKE"),
            ("STRUCTURE_STATE", "unknown"),
            ("SOLAR_STATE", "dark "),
            ("SOLAR_STATE", "DARK"),
            ("SOLAR_STATE", ""),
        ]
        for column, value in mutations:
            with self.subTest(column=column, value=value):
                invalid = frame.copy()
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    bootstrap_shared_state_delta_abs_bias_contrast(
                        invalid,
                        n_replicates=10,
                        min_unique_days_per_state=1,
                        min_months_per_state=1,
                    )

    def test_result_always_records_estimand_sampling_and_gate_metadata(self):
        frame = _paired_state_frame(months=(1,), days_per_month=9)
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=10,
            block_length_days=7,
            seed=81,
            min_unique_days_per_state=20,
            min_months_per_state=3,
            min_valid_replicate_fraction=0.9,
        )
        self.assertFalse(result.coverage_eligible)
        self.assertEqual(result.seed, 81)
        self.assertEqual(result.block_length_days, 7)
        self.assertEqual(result.min_unique_days_per_state, 20)
        self.assertEqual(result.min_months_per_state, 3)
        self.assertEqual(result.min_valid_replicate_fraction, 0.9)
        self.assertEqual(result.strata_weighting, "equal_common_month_solar_strata")
        self.assertEqual(
            result.estimand,
            "standardized_signed_bias_then_delta_abs_bias_contrast",
        )

    def test_bias_contrast_is_not_timewise_mae_contrast(self):
        frame = pd.DataFrame(
            {
                "STN": ["100"] * 14,
                "TM": [
                    pd.Timestamp(2024, 1, day, hour)
                    for day in range(1, 8)
                    for hour in (2, 3)
                ],
                "SOLAR_STATE": ["dark"] * 14,
                "STRUCTURE_STATE": ["inversion_like", "normal_negative"] * 7,
                "error_none": [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 8,
                "error_fixed": [0.0, 1.0, 2.0, 1.0, 2.0, 1.0] + [1.0] * 8,
            }
        )
        result = bootstrap_shared_state_delta_abs_bias_contrast(
            frame,
            n_replicates=20,
            seed=9,
            min_unique_days_per_state=1,
            min_months_per_state=1,
        )

        state_mae_changes = {}
        for state, group in frame.groupby("STRUCTURE_STATE"):
            state_mae_changes[state] = (
                group["error_fixed"].abs().mean() - group["error_none"].abs().mean()
            )
        wrong_mae_contrast = (
            state_mae_changes["inversion_like"] - state_mae_changes["normal_negative"]
        )
        self.assertAlmostEqual(result.point_contrast, 3 / 7)
        self.assertNotAlmostEqual(result.point_contrast, wrong_mae_contrast)

    def test_rejects_multiple_stations_nonfinite_errors_and_invalid_parameters(self):
        frame = _paired_state_frame(months=(1,), days_per_month=9)
        multiple = pd.concat([frame, frame.assign(STN="200")], ignore_index=True)
        with self.assertRaises(ValueError):
            bootstrap_shared_state_delta_abs_bias_contrast(multiple)

        nonfinite = frame.copy()
        nonfinite.loc[0, "error_none"] = np.inf
        with self.assertRaises(ValueError):
            bootstrap_shared_state_delta_abs_bias_contrast(nonfinite)

        with self.assertRaises(ValueError):
            generate_monthly_circular_block_weights(frame["TM"], n_replicates=0)
        with self.assertRaises(ValueError):
            generate_monthly_circular_block_weights(
                frame["TM"], n_replicates=2, block_length_days=0
            )
        with self.assertRaises(ValueError):
            bootstrap_shared_state_delta_abs_bias_contrast(
                frame,
                n_replicates=2,
                min_unique_days_per_state=1,
                min_months_per_state=1,
                min_valid_replicate_fraction=0.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
