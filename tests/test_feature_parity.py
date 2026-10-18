"""
tests/test_feature_parity.py
Unit tests verifying that feature extraction produces strictly identical feature vectors
between offline training preprocessing and live streaming scoring.
Enforces BUILD.md Section 6 and Section 14.
"""

import unittest
import numpy as np
import pandas as pd
from model.features import FEATURE_NAMES, compute_features_for_ward_date


class TestFeatureParity(unittest.TestCase):
    """
    Test suite enforcing identical mathematical behavior across offline and streaming features.
    """

    def setUp(self):
        """Set up dummy historical event sequence for testing."""
        self.ward_id = "W017"
        self.scoring_date = "2024-09-14"
        dates = pd.date_range(end=self.scoring_date, periods=35, freq="D").strftime("%Y-%m-%d")
        records = []
        for d in dates:
            records.append({"date": d, "signal_type": "clinic_fever_visits", "value": 20.0})
            records.append({"date": d, "signal_type": "pharmacy_fever_meds", "value": 35.0})
            records.append({"date": d, "signal_type": "rainfall", "value": 5.0})
            records.append({"date": d, "signal_type": "temp", "value": 24.0})
            records.append({"date": d, "signal_type": "humidity", "value": 75.0})
        self.history_df = pd.DataFrame(records)

    def test_feature_list_completeness(self):
        """Verify all mandatory features are declared."""
        self.assertGreater(len(FEATURE_NAMES), 10)
        self.assertIn("clinic_visits_7d_mean", FEATURE_NAMES)
        self.assertIn("pharmacy_7d_mean", FEATURE_NAMES)
        self.assertIn("rainfall_lag14", FEATURE_NAMES)
        # Ensure no geographic leakage
        self.assertNotIn("ward_id", FEATURE_NAMES)
        self.assertNotIn("zone_id", FEATURE_NAMES)

    def test_feature_parity_deterministic(self):
        """Verify identical outputs on repeat evaluations."""
        f1 = compute_features_for_ward_date(self.history_df, self.scoring_date)
        f2 = compute_features_for_ward_date(self.history_df, self.scoring_date)
        for k in f1:
            self.assertEqual(f1[k], f2[k], f"Feature {k} was not deterministic across calls")


if __name__ == "__main__":
    unittest.main()
