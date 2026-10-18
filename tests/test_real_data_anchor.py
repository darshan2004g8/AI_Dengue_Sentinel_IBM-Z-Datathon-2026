"""tests/test_real_data_anchor.py
Validation suite verifying real data provenance and anchoring across Dengue Sentinel:
  (a) Replay-season monthly totals correlate above 0.9 with real OpenDengue monthly counts.
  (b) Weather values in streamed events match raw Open-Meteo weather CSV for identical dates.
  (c) Every ward id in generated events strictly exists in the real BBMP ward GeoJSON.
"""

import os
import unittest
import json
import numpy as np
import pandas as pd


class TestRealDataAnchor(unittest.TestCase):
    """
    Test suite enforcing strict empirical data anchoring and provenance.
    """

    @classmethod
    def setUpClass(cls):
        cls.real_monthly_path = "data/processed/india_monthly_2024.csv"
        cls.replay_events_path = "data/processed/events/season_replay.csv"
        cls.replay_gt_path = "data/processed/ground_truth/season_replay.csv"
        cls.replay_off_path = "data/processed/official/season_replay.csv"
        cls.raw_weather_path = "data/raw/bengaluru_weather_2018_2025.csv"
        cls.geojson_path = "dashboard/assets/bbmp_wards.geojson"

        # Verify prerequisite files exist
        for p in [cls.real_monthly_path, cls.replay_events_path, cls.raw_weather_path, cls.geojson_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Prerequisite file missing for test: {p}")

        cls.real_monthly_df = pd.read_csv(cls.real_monthly_path)
        cls.replay_events_df = pd.read_csv(cls.replay_events_path)
        cls.raw_weather_df = pd.read_csv(cls.raw_weather_path)
        with open(cls.geojson_path, "r", encoding="utf-8") as f:
            cls.geojson_data = json.load(f)

    def test_replay_season_monthly_correlation(self):
        """(a) Replay-season monthly totals correlate above 0.9 with real monthly counts."""
        real_totals = self.real_monthly_df["dengue_total"].values

        # Group replay-season daily case_index by month (YYYY-MM)
        gt_df = pd.read_csv(self.replay_gt_path)
        gt_df["month"] = gt_df["date"].str[:7]
        replay_monthly_ci = gt_df.groupby("month")["case_index"].sum().values

        corr_ci = float(np.corrcoef(real_totals, replay_monthly_ci)[0, 1])
        print(f"\n[Test Anchor (a)] Pearson correlation (real monthly vs replay case_index): {corr_ci:.4f}")
        self.assertGreater(
            corr_ci,
            0.90,
            f"Correlation between real monthly totals and replay case_index ({corr_ci:.4f}) must be > 0.90",
        )

        # Also verify correlation on official delayed cases
        off_df = pd.read_csv(self.replay_off_path)
        off_df["month"] = off_df["date"].str[:7]
        replay_monthly_off = off_df.groupby("month")["official_cases"].sum().values
        corr_off = float(np.corrcoef(real_totals, replay_monthly_off)[0, 1])
        print(f"[Test Anchor (a)] Pearson correlation (real monthly vs replay official_cases): {corr_off:.4f}")
        self.assertGreater(
            corr_off,
            0.90,
            f"Correlation between real monthly totals and replay official_cases ({corr_off:.4f}) must be > 0.90",
        )

    def test_weather_values_match_raw_csv(self):
        """(b) Weather values in events match raw weather CSV for the same dates."""
        raw_w = self.raw_weather_df.set_index("date")
        weather_signals = ["rainfall", "temp", "humidity"]

        for sig in weather_signals:
            ev_sig = self.replay_events_df[self.replay_events_df["signal_type"] == sig]
            self.assertFalse(ev_sig.empty, f"No events found for weather signal '{sig}'")

            # Verify for every ward that values match raw weather CSV exactly
            merged = pd.merge(ev_sig, raw_w[[sig]].reset_index(), on="date")
            max_diff = float(np.abs(merged["value"] - merged[sig]).max())
            self.assertLess(
                max_diff,
                1e-4,
                f"Weather signal '{sig}' differs from raw weather CSV by up to {max_diff}",
            )
            print(f"[Test Anchor (b)] Weather signal '{sig}': max diff vs raw weather CSV = {max_diff:.6f} (MATCH)")

    def test_every_ward_id_exists_in_geojson(self):
        """(c) Every ward id in the events exists in the GeoJSON."""
        geojson_wards = set()
        for feat in self.geojson_data.get("features", []):
            wid = feat.get("properties", {}).get("ward_id")
            if wid:
                geojson_wards.add(wid)

        self.assertEqual(len(geojson_wards), 20, "Expected exactly 20 wards in GeoJSON")

        event_wards = set(self.replay_events_df["ward_id"].unique())
        print(f"\n[Test Anchor (c)] Event wards: {len(event_wards)}, GeoJSON wards: {len(geojson_wards)}")
        missing_wards = event_wards - geojson_wards
        self.assertEqual(
            len(missing_wards),
            0,
            f"Ward IDs in events missing from GeoJSON: {missing_wards}",
        )
        self.assertTrue(event_wards.issubset(geojson_wards))

    def test_risk_scores_fallback_wards_in_pilot_geojson(self):
        """(d) Every ward id in risk_scores_fallback.csv strictly exists in the pilot GeoJSON."""
        fallback_path = "data/processed/risk_scores_fallback.csv"
        self.assertTrue(os.path.exists(fallback_path), f"Fallback CSV missing: {fallback_path}")
        fallback_df = pd.read_csv(fallback_path)
        csv_wards = set(fallback_df["ward_id"].unique())
        geojson_wards = {
            feat["properties"]["ward_id"]
            for feat in self.geojson_data.get("features", [])
            if feat.get("properties", {}).get("ward_id")
        }
        missing = csv_wards - geojson_wards
        self.assertEqual(
            len(missing),
            0,
            f"Ward IDs in risk_scores_fallback.csv missing from pilot GeoJSON: {missing}",
        )
        self.assertEqual(len(csv_wards), 20, f"Expected exactly 20 pilot wards, got {len(csv_wards)}")
        self.assertEqual(csv_wards, geojson_wards)


if __name__ == "__main__":
    unittest.main()
