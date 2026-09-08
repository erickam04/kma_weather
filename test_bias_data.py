import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bias_data import (
    aggregate_bias_groups,
    load_analysis_station_ids,
    load_neighbor_map,
    load_station_pairs,
    open_readonly_db,
)


class BiasDataTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        rows = []
        for tm, obs, en, ef in [
            ("2024-01-01 00:00:00", 10.0, 2.0, 0.5),
            ("2024-02-01 12:00:00", 12.0, 0.0, -0.5),
        ]:
            for method, err in [("none", en), ("fixed_lapse", ef)]:
                rows.append({
                    "TM": tm, "STN": "T", "OBSERVED": obs,
                    "PREDICTED": obs + err, "ERROR": err,
                    "ABS_ERROR": abs(err), "NEIGHBOR_COUNT": 3,
                    "interpolable": 1, "NEIGHBOR_SET_ID": "sig",
                    "COORD_EPOCH": "2020-01-01", "MONTH": tm[:7].replace("-", ""),
                    "METHOD": method,
                })
        pd.DataFrame(rows).to_sql("raw", self.con, index=False)
        pd.DataFrame([{
            "NEIGHBOR_SET_ID": "sig",
            "neighbor_ids": "N1-N2-N3", "n_neighbors": 3,
        }]).to_sql("neighbor_set_map", self.con, index=False)

    def tearDown(self):
        self.con.close()

    def test_loads_only_paired_rows(self):
        pairs = load_station_pairs(self.con, "T")
        self.assertEqual(len(pairs), 2)
        self.assertListEqual(pairs["error_none"].tolist(), [2.0, 0.0])
        self.assertListEqual(pairs["error_fixed"].tolist(), [0.5, -0.5])

    def test_station_population_and_neighbor_map(self):
        self.assertEqual(load_analysis_station_ids(self.con), ["T"])
        self.assertEqual(load_neighbor_map(self.con), {"sig": ["N1", "N2", "N3"]})

    def test_aggregate_month(self):
        pairs = load_station_pairs(self.con, "T")
        pairs["dz_geom_km"] = [-2.0 / -6.5, -0.5 / -6.5]
        monthly = aggregate_bias_groups(pairs, ["STN", "MONTH"])
        self.assertEqual(len(monthly), 2)
        self.assertTrue({"BIAS_none", "BIAS_fixed", "DELTA_ABS_BIAS",
                         "mean_dz_geom_km"}.issubset(monthly.columns))

    def test_observation_mismatch_raises(self):
        self.con.execute(
            "UPDATE raw SET OBSERVED=99 WHERE rowid=("
            "SELECT rowid FROM raw WHERE METHOD='fixed_lapse' LIMIT 1)"
        )
        with self.assertRaisesRegex(ValueError, "OBSERVED mismatch"):
            load_station_pairs(self.con, "T")

    def test_neighbor_signature_mismatch_raises(self):
        self.con.execute(
            "UPDATE raw SET NEIGHBOR_SET_ID='other' WHERE rowid=("
            "SELECT rowid FROM raw WHERE METHOD='fixed_lapse' LIMIT 1)"
        )
        with self.assertRaisesRegex(ValueError, "neighbor signature mismatch"):
            load_station_pairs(self.con, "T")

    def test_coordinate_epoch_mismatch_raises(self):
        self.con.execute(
            "UPDATE raw SET COORD_EPOCH='2024-01-01' WHERE rowid=("
            "SELECT rowid FROM raw WHERE METHOD='fixed_lapse' LIMIT 1)"
        )
        with self.assertRaisesRegex(ValueError, "coordinate epoch mismatch"):
            load_station_pairs(self.con, "T")

    def test_rejects_duplicate_analysis_method_timestamp(self):
        self.con.execute(
            "INSERT INTO raw SELECT * FROM raw WHERE METHOD='none' LIMIT 1"
        )
        with self.assertRaisesRegex(ValueError, "duplicate analysis rows"):
            load_station_pairs(self.con, "T")

    def test_rejects_timestamp_missing_from_one_method(self):
        self.con.execute(
            "DELETE FROM raw WHERE METHOD='fixed_lapse' AND TM='2024-02-01 12:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            load_station_pairs(self.con, "T")

    def test_pairs_equivalent_timestamp_text_across_methods(self):
        self.con.execute(
            "UPDATE raw SET TM='2024-01-01T00:00:00' "
            "WHERE METHOD='fixed_lapse' AND TM='2024-01-01 00:00:00'"
        )
        pairs = load_station_pairs(self.con, "T")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs.loc[0, "TM"], pd.Timestamp("2024-01-01 00:00:00"))

    def test_rejects_semantic_duplicate_timestamp_within_method(self):
        self.con.execute(
            "INSERT INTO raw (TM, STN, OBSERVED, PREDICTED, ERROR, ABS_ERROR, "
            "NEIGHBOR_COUNT, interpolable, NEIGHBOR_SET_ID, COORD_EPOCH, MONTH, METHOD) "
            "SELECT '2024-01-01T00:00:00', STN, OBSERVED, PREDICTED, ERROR, ABS_ERROR, "
            "NEIGHBOR_COUNT, interpolable, NEIGHBOR_SET_ID, COORD_EPOCH, MONTH, METHOD "
            "FROM raw WHERE METHOD='none' AND TM='2024-01-01 00:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "duplicate analysis rows"):
            load_station_pairs(self.con, "T")

    def test_rejects_invalid_analysis_timestamp(self):
        self.con.execute(
            "UPDATE raw SET TM='not-a-timestamp' "
            "WHERE METHOD='none' AND TM='2024-01-01 00:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "invalid analysis timestamp"):
            load_station_pairs(self.con, "T")

    def test_station_population_rejects_invalid_pairing(self):
        self.con.execute(
            "DELETE FROM raw WHERE METHOD='fixed_lapse' AND TM='2024-02-01 12:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "invalid analysis station T"):
            load_analysis_station_ids(self.con)

    def test_station_population_rejects_duplicate_pairing(self):
        self.con.execute(
            "INSERT INTO raw SELECT * FROM raw WHERE METHOD='none' LIMIT 1"
        )
        with self.assertRaisesRegex(ValueError, "invalid analysis station T"):
            load_analysis_station_ids(self.con)

    def test_station_population_rejects_metadata_mismatch(self):
        self.con.execute(
            "UPDATE raw SET NEIGHBOR_SET_ID='other' WHERE rowid=("
            "SELECT rowid FROM raw WHERE METHOD='fixed_lapse' LIMIT 1)"
        )
        with self.assertRaisesRegex(ValueError, "invalid analysis station T"):
            load_analysis_station_ids(self.con)

    def test_rejects_null_neighbor_signature_even_when_both_methods_match(self):
        self.con.execute(
            "UPDATE raw SET NEIGHBOR_SET_ID=NULL WHERE TM='2024-01-01 00:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "neighbor signature missing"):
            load_station_pairs(self.con, "T")

    def test_rejects_null_coordinate_epoch_even_when_both_methods_match(self):
        self.con.execute(
            "UPDATE raw SET COORD_EPOCH=NULL WHERE TM='2024-01-01 00:00:00'"
        )
        with self.assertRaisesRegex(ValueError, "coordinate epoch missing"):
            load_station_pairs(self.con, "T")

    def test_opens_encoded_path_readonly_and_rejects_writes(self):
        with tempfile.NamedTemporaryFile(
            suffix=" space #%.sqlite", dir=os.environ.get("BIAS_TEST_TMPDIR"),
            delete=False,
        ) as tmpfile:
            path = Path(tmpfile.name)
        self.addCleanup(path.unlink, missing_ok=True)
        writer = sqlite3.connect(path)
        writer.execute("CREATE TABLE sample (value INTEGER)")
        writer.execute("INSERT INTO sample VALUES (7)")
        writer.commit()
        writer.close()

        con = open_readonly_db(path)
        self.addCleanup(con.close)
        self.assertEqual(con.execute("SELECT value FROM sample").fetchone()[0], 7)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("INSERT INTO sample VALUES (8)")

    def test_aggregate_all_rows_reports_hand_checked_metrics_and_valid_counts(self):
        pairs = pd.DataFrame({
            "error_none": [1.0, 3.0],
            "error_fixed": [0.0, 1.0],
            "dz_geom_km": [0.2, -0.4],
            "nearest_distance_km": [2.0, np.nan],
        })
        result = aggregate_bias_groups(pairs, [])
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["n_hours"], 2)
        self.assertAlmostEqual(row["BIAS_none"], 2.0)
        self.assertAlmostEqual(row["BIAS_fixed"], 0.5)
        self.assertAlmostEqual(row["DELTA_ABS_BIAS"], -1.5)
        self.assertAlmostEqual(row["mean_dz_geom_km"], -0.1)
        self.assertAlmostEqual(row["mean_absdz_geom_km"], 0.3)
        self.assertAlmostEqual(row["sd_dz_geom_km"], 0.3)
        self.assertEqual(row["n_valid_dz_geom_km"], 2)
        self.assertEqual(row["n_valid_nearest_distance_km"], 1)
        self.assertAlmostEqual(row["mean_nearest_distance_km"], 2.0)

    def test_aggregate_rejects_nonfinite_required_geometry(self):
        pairs = pd.DataFrame({
            "STN": ["T"],
            "error_none": [1.0],
            "error_fixed": [0.5],
            "dz_geom_km": [np.nan],
        })
        with self.assertRaisesRegex(ValueError, "dz_geom_km must be finite"):
            aggregate_bias_groups(pairs, ["STN"])

    def test_aggregate_rejects_infinite_optional_geometry(self):
        pairs = pd.DataFrame({
            "STN": ["T"],
            "error_none": [1.0],
            "error_fixed": [0.5],
            "dz_geom_km": [0.1],
            "nearest_distance_km": [np.inf],
        })
        with self.assertRaisesRegex(ValueError, "nearest_distance_km must be finite"):
            aggregate_bias_groups(pairs, ["STN"])

    def test_aggregate_rejects_nonnumeric_optional_geometry(self):
        pairs = pd.DataFrame({
            "STN": ["T"],
            "error_none": [1.0],
            "error_fixed": [0.5],
            "dz_geom_km": [0.1],
            "nearest_distance_km": ["bad"],
        })
        with self.assertRaisesRegex(ValueError, "nearest_distance_km must be finite"):
            aggregate_bias_groups(pairs, ["STN"])


if __name__ == "__main__":
    unittest.main()
