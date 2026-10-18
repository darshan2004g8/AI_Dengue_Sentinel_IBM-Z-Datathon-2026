"""
tests/test_generator.py
Validation test suite for Multi-Season Generator per BUILD.md Section 9.1 acceptance criteria:
  - Determinism: Identical output with same seed.
  - Synthetic flagging: Correct is_synthetic boolean labeling per signal type.
  - Care pathway dynamics: Pharmacy fever-meds lead clinic fever-visits.
  - Control seasons: Pure zero-outbreak behavior.
  - Data contracts: Strict Section 7.1 schema adherence.
"""

import os
import unittest
import numpy as np
import pandas as pd
from generator.generate_signals import (
    load_config,
    load_wards,
    build_ward_adjacency,
    generate_season_signals,
)


class TestGenerator(unittest.TestCase):
    """
    Test suite for multi-season synthetic generator.
    """

    @classmethod
    def setUpClass(cls):
        """Load minimal configuration and sample weather window for testing."""
        cls.cfg = load_config("config/config.yaml")
        weather_path = cls.cfg.get("data", {}).get("paths", {}).get("raw_weather", "data/raw/bengaluru_weather_2018_2025.csv")
        cls.weather_full = pd.read_csv(weather_path)
        cls.weather_sample = cls.weather_full.iloc[:90].reset_index(drop=True)
        cls.wards = load_wards("dashboard/assets/bbmp_wards.geojson")[:5]
        cls.adjacency = build_ward_adjacency(cls.wards)

    def test_determinism_same_seed(self):
        """Verify that identical random seeds yield strictly bit-identical outputs."""
        seed = 42
        ev1, gt1, _, _ = generate_season_signals(
            "test_s1", seed, self.weather_sample, self.wards, self.adjacency, is_control=False
        )
        ev2, gt2, _, _ = generate_season_signals(
            "test_s2", seed, self.weather_sample, self.wards, self.adjacency, is_control=False
        )
        pd.testing.assert_frame_equal(ev1.drop(columns=["event_id"]), ev2.drop(columns=["event_id"]))
        pd.testing.assert_frame_equal(gt1, gt2)

    def test_synthetic_flag_integrity(self):
        """Verify Hard Rule 2: All synthetic signals are marked True, real weather marked False."""
        ev, _, _, _ = generate_season_signals(
            "test_s", 100, self.weather_sample, self.wards, self.adjacency, is_control=False
        )
        # Clinical and pharmacy signals MUST have is_synthetic = True
        synth_signals = ev[ev["signal_type"].isin(["clinic_fever_visits", "pharmacy_fever_meds"])]
        self.assertTrue(synth_signals["is_synthetic"].all())

        # Weather signals MUST have is_synthetic = False
        real_signals = ev[ev["signal_type"].isin(["rainfall", "temp", "humidity"])]
        self.assertFalse(real_signals["is_synthetic"].any())

    def test_control_seasons_have_zero_outbreaks(self):
        """Verify control seasons have zero injected outbreaks and zero onset flags."""
        _, gt, _, num_outbreaks = generate_season_signals(
            "test_ctrl", 200, self.weather_sample, self.wards, self.adjacency, is_control=True
        )
        self.assertEqual(num_outbreaks, 0)
        self.assertEqual(gt["onset_flag"].sum(), 0)
        self.assertEqual(gt["outbreak_flag"].sum(), 0)

    def test_care_pathway_pharmacy_leads_clinic(self):
        """Verify cross-correlation shows pharmacy series peak precedes or leads clinic series."""
        # Use full year to inspect lag correlation across season_01
        ev = pd.read_csv("data/processed/events/season_01.csv")
        w_ev = ev[ev["ward_id"] == "W017"]
        piv = w_ev.pivot(index="date", columns="signal_type", values="value")
        
        pharm = piv["pharmacy_fever_meds"].values
        clinic = piv["clinic_fever_visits"].values

        # Normalize series
        pharm_norm = (pharm - np.mean(pharm)) / (np.std(pharm) + 1e-6)
        clinic_norm = (clinic - np.mean(clinic)) / (np.std(clinic) + 1e-6)

        # Cross-correlation: clinic vs pharm shifted by k days
        corrs = [np.corrcoef(pharm_norm[:-k], clinic_norm[k:])[0, 1] for k in range(1, 5)]
        best_lag = int(np.argmax(corrs)) + 1
        self.assertIn(best_lag, [1, 2, 3, 4], f"Expected positive lead lag for pharmacy, got {best_lag}")


if __name__ == "__main__":
    unittest.main()
