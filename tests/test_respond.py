"""Unit tests for Decide/Respond module and configuration thresholds.

Tests adhere to SPEC.md Section 7.2 and 9.4 requirements:
- Thresholds: low (< 0.40), medium (0.40 to 0.70), high (> 0.70).
- Response team allocation: ranks by risk_score, assigns to alert_level != 'low',
  overflow wards marked on Watchlist.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming.config import get_alert_level, ALERT_LOW_THRESHOLD, ALERT_HIGH_THRESHOLD
from dashboard.respond import allocate_response_teams, generate_ward_action


class TestDengueSentinelRespond(unittest.TestCase):
    def test_alert_thresholds(self):
        """Validates that alert levels follow SPEC.md Section 7.2."""
        self.assertEqual(ALERT_LOW_THRESHOLD, 0.40)
        self.assertEqual(ALERT_HIGH_THRESHOLD, 0.70)
        self.assertEqual(get_alert_level(0.20), "low")
        self.assertEqual(get_alert_level(0.39), "low")
        self.assertEqual(get_alert_level(0.40), "medium")
        self.assertEqual(get_alert_level(0.70), "medium")
        self.assertEqual(get_alert_level(0.71), "high")
        self.assertEqual(get_alert_level(0.82), "high")

    def test_team_allocation_and_watchlist(self):
        """Validates allocation of 3 teams across 5 wards with varied risk."""
        mock_scores = [
            {"ward_id": "W004", "risk_score": 0.30, "alert_level": "low", "confidence": 0.8},
            {"ward_id": "W017", "risk_score": 0.82, "alert_level": "high", "confidence": 0.74},
            {"ward_id": "W009", "risk_score": 0.72, "alert_level": "high", "confidence": 0.70},
            {"ward_id": "W063", "risk_score": 0.60, "alert_level": "medium", "confidence": 0.65},
            {"ward_id": "W055", "risk_score": 0.50, "alert_level": "medium", "confidence": 0.62},
        ]

        result = allocate_response_teams(mock_scores, num_teams=2)
        summary = result["summary"]
        assigned = result["assigned_teams"]
        watchlist = result["watchlist"]

        # Only 2 teams assigned
        self.assertEqual(len(assigned), 2)
        self.assertEqual(summary["teams_deployed"], 2)
        # Top 2 should be W017 (0.82) and W009 (0.72)
        self.assertEqual(assigned[0]["ward_id"], "W017")
        self.assertEqual(assigned[1]["ward_id"], "W009")

        # Remaining actionable wards (W063, W055) must be on watchlist
        self.assertEqual(len(watchlist), 2)
        self.assertEqual(watchlist[0]["ward_id"], "W063")
        self.assertEqual(watchlist[1]["ward_id"], "W055")

        # Low risk ward W004 must not receive a team or watchlist
        self.assertEqual(summary["low_risk_count"], 1)

    def test_plain_language_action(self):
        """Verifies plain-language operational directive for high risk wards."""
        drivers = [
            {"feature": "clinic_visits_7d_change_pct", "value": 41.0, "direction": "up", "text": "Clinic fever visits up 41% over 7 days"}
        ]
        action = generate_ward_action("W017", "Banashankari", 0.82, "high", top_drivers=drivers)
        self.assertIn("fogging", action["operational_action"].lower())
        self.assertIn("W017", action["operational_action"])
        self.assertIn("hospitals", action["hospital_advisory"].lower())
        self.assertEqual(action["urgency"], "IMMEDIATE (24-48h)")


if __name__ == "__main__":
    unittest.main()
