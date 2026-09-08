import unittest

import numpy as np
import pandas as pd

from bias_geometry import compute_geometry_metrics, reconstruct_geometry_for_pairs


def make_meta(rows):
    return pd.DataFrame([
        {
            "지점": str(stn), "위도": lat, "경도": lon,
            "노장해발고도(m)": elev,
            "시작일": pd.Timestamp(start), "종료일": pd.NaT,
        }
        for stn, lat, lon, elev, start in rows
    ])


class GeometryMetricTests(unittest.TestCase):
    def setUp(self):
        self.meta = make_meta([
            ("T", 37.00, 127.00, 500.0, "2020-01-01"),
            ("N1", 37.02, 127.00, 100.0, "2020-01-01"),
            ("N2", 37.00, 127.02, 100.0, "2020-01-01"),
            ("N3", 36.98, 127.00, 100.0, "2020-01-01"),
        ])

    def test_equal_neighbor_elevations_produce_400m_delta_z(self):
        metrics = compute_geometry_metrics("T", ["N1", "N2", "N3"], self.meta)
        self.assertAlmostEqual(metrics.dz_geom_km, 0.4, places=12)
        self.assertEqual(metrics.neighbor_count_geom, 3)

    def test_nan_neighbor_uses_zero_correction_term(self):
        meta = make_meta([
            ("T", 0.00, 0.00, 500.0, "2020-01-01"),
            ("N1", 0.01, 0.00, np.nan, "2020-01-01"),
            ("N2", -0.01, 0.00, 100.0, "2020-01-01"),
            ("N3", 0.00, 0.01, 100.0, "2020-01-01"),
        ])
        metrics = compute_geometry_metrics("T", ["N1", "N2", "N3"], meta)
        self.assertAlmostEqual(metrics.dz_geom_km, 0.8 / 3, places=12)

    def test_nan_target_uses_zero_correction_term(self):
        meta = self.meta.copy()
        meta.loc[meta["지점"] == "T", "노장해발고도(m)"] = np.nan
        metrics = compute_geometry_metrics("T", ["N1", "N2", "N3"], meta)
        self.assertAlmostEqual(metrics.dz_geom_km, 0.0, places=12)

    def test_zero_distance_uses_first_neighbor_and_keeps_all_neighbor_count(self):
        meta = make_meta([
            ("T", 37.00, 127.00, 500.0, "2020-01-01"),
            ("N0", 37.00, 127.00, 100.0, "2020-01-01"),
            ("N1", 37.01, 127.00, 900.0, "2020-01-01"),
        ])
        metrics = compute_geometry_metrics("T", ["N0", "N1"], meta)
        self.assertAlmostEqual(metrics.dz_geom_km, 0.4, places=12)
        self.assertEqual(metrics.neighbor_count_geom, 2)
        self.assertEqual(metrics.nearest_distance_km, 0.0)
        self.assertEqual(metrics.weighted_distance_km, 0.0)
        self.assertEqual(metrics.neighbor_elev_sd_m, 0.0)

    def test_distance_and_elevation_dispersion_metrics_match_idw_weights(self):
        meta = make_meta([
            ("T", 0.00, 0.00, 0.0, "2020-01-01"),
            ("N1", 0.01, 0.00, 100.0, "2020-01-01"),
            ("N2", 0.02, 0.00, 300.0, "2020-01-01"),
        ])
        metrics = compute_geometry_metrics("T", ["N1", "N2"], meta)
        one_degree_cent = 6371 * np.pi / 18000
        self.assertAlmostEqual(metrics.nearest_distance_km, one_degree_cent, places=12)
        self.assertAlmostEqual(metrics.weighted_distance_km, 1.2 * one_degree_cent, places=12)
        self.assertAlmostEqual(metrics.neighbor_elev_sd_m, 80.0, places=12)

    def test_symmetric_neighbors_have_small_centroid_offset(self):
        meta = make_meta([
            ("T", 37.00, 127.00, 0.0, "2020-01-01"),
            ("N1", 37.01, 127.00, 0.0, "2020-01-01"),
            ("N2", 36.99, 127.00, 0.0, "2020-01-01"),
            ("N3", 37.00, 127.0125, 0.0, "2020-01-01"),
            ("N4", 37.00, 126.9875, 0.0, "2020-01-01"),
        ])
        metrics = compute_geometry_metrics("T", ["N1", "N2", "N3", "N4"], meta)
        self.assertLess(metrics.centroid_offset_km, 0.02)

    def test_reconstruction_honors_metadata_epoch(self):
        meta = make_meta([
            ("T", 37.00, 127.00, 500.0, "2020-01-01"),
            ("T", 37.00, 127.00, 700.0, "2024-07-01"),
            ("N1", 37.02, 127.00, 100.0, "2020-01-01"),
            ("N2", 37.00, 127.02, 100.0, "2020-01-01"),
            ("N3", 36.98, 127.00, 100.0, "2020-01-01"),
        ])
        meta.loc[0, "종료일"] = pd.Timestamp("2024-07-01")
        pairs = pd.DataFrame({
            "TM": pd.to_datetime(["2024-06-01 00:00", "2024-08-01 00:00"]),
            "STN": ["T", "T"],
            "NEIGHBOR_SET_ID": ["sig", "sig"],
            "COORD_EPOCH": [pd.Timestamp("2020-01-01"), "2024-07-01"],
        })
        out = reconstruct_geometry_for_pairs(
            pairs, {"sig": ["N1", "N2", "N3"]}, meta
        )
        self.assertAlmostEqual(out.loc[0, "dz_geom_km"], 0.4, places=12)
        self.assertAlmostEqual(out.loc[1, "dz_geom_km"], 0.6, places=12)

    def test_reconstruction_rejects_metadata_epoch_mismatch(self):
        pairs = pd.DataFrame({
            "TM": pd.to_datetime(["2024-06-01 00:00"]),
            "STN": ["T"],
            "NEIGHBOR_SET_ID": ["sig"],
            "COORD_EPOCH": ["2021-01-01"],
        })
        with self.assertRaisesRegex(ValueError, "COORD_EPOCH mismatch"):
            reconstruct_geometry_for_pairs(
                pairs, {"sig": ["N1", "N2", "N3"]}, self.meta
            )

    def test_reconstruction_rejects_unknown_neighbor_signature(self):
        pairs = pd.DataFrame({
            "TM": pd.to_datetime(["2024-06-01 00:00"]),
            "STN": ["T"],
            "NEIGHBOR_SET_ID": ["unknown"],
            "COORD_EPOCH": ["2020-01-01"],
        })
        with self.assertRaisesRegex(ValueError, "unknown neighbor signature"):
            reconstruct_geometry_for_pairs(pairs, {}, self.meta)

    def test_reconstruction_rejects_target_missing_from_valid_metadata(self):
        meta = self.meta.copy()
        meta.loc[meta["지점"] == "T", "시작일"] = pd.Timestamp("2025-01-01")
        pairs = pd.DataFrame({
            "TM": pd.to_datetime(["2024-06-01 00:00"]),
            "STN": ["T"],
            "NEIGHBOR_SET_ID": ["sig"],
            "COORD_EPOCH": ["2020-01-01"],
        })
        with self.assertRaisesRegex(ValueError, "target metadata count != 1"):
            reconstruct_geometry_for_pairs(
                pairs, {"sig": ["N1", "N2", "N3"]}, meta
            )

    def test_reconstruction_rejects_neighbor_missing_from_valid_metadata(self):
        meta = self.meta.copy()
        meta.loc[meta["지점"] == "N1", "시작일"] = pd.Timestamp("2025-01-01")
        pairs = pd.DataFrame({
            "TM": pd.to_datetime(["2024-06-01 00:00"]),
            "STN": ["T"],
            "NEIGHBOR_SET_ID": ["sig"],
            "COORD_EPOCH": ["2020-01-01"],
        })
        with self.assertRaisesRegex(ValueError, "neighbor missing from valid metadata"):
            reconstruct_geometry_for_pairs(pairs, {"sig": ["N1"]}, meta)


if __name__ == "__main__":
    unittest.main()
