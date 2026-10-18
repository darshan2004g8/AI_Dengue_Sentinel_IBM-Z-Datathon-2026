"""
model/baseline.py
Baseline outbreak detection algorithms per BUILD.md Section 9.2:
- Baseline A: Persistence detector ("same as last period").
- Baseline B: Rolling z-score detector on clinic visits (EARS-style: z > 2.0 against trailing 28 days with 2-day guard band).
"""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """
    Persistence detector that forecasts current outbreak status based directly on recent state.
    Predicts alert = 1 if the ward had an active alert on the previous date.
    """

    def __init__(self):
        """Initialize persistence baseline state dictionary."""
        self.last_alerts = {}

    def score_ward(self, ward_id, previous_state):
        """
        Score a single ward given its previous binary state.
        
        Inputs:
            ward_id (str): Ward identifier.
            previous_state (int or bool): Alert status yesterday.
        Outputs:
            float: Predicted risk score (0.0 or 1.0).
        """
        score = 1.0 if bool(previous_state) else 0.0
        self.last_alerts[ward_id] = score
        return score

    def score_batch(self, previous_flags):
        """
        Vectorized batch scoring for persistence baseline.
        
        Inputs:
            previous_flags (np.ndarray): 1D array of previous state flags.
        Outputs:
            np.ndarray: Predicted risk scores.
        """
        return np.array(previous_flags, dtype=np.float32)


class ZScoreBaseline:
    """
    EARS-style rolling z-score anomaly detector on clinic fever visits.
    Flags an anomaly when z > 2.0 against trailing 28 days with a 2-day guard band.
    """

    def __init__(self, z_threshold=2.0, window_days=28, guard_band_days=2):
        """
        Initialize EARS z-score detector parameters.
        
        Inputs:
            z_threshold (float): Z-score cutoff for triggering an anomaly flag (default: 2.0).
            window_days (int): Baseline trailing window duration (default: 28).
            guard_band_days (int): Days immediately preceding current date excluded from baseline (default: 2).
        """
        self.z_threshold = z_threshold
        self.window_days = window_days
        self.guard_band_days = guard_band_days

    def score_series(self, values):
        """
        Calculate rolling z-score and binary anomaly flag for the most recent observation in a series.
        
        Inputs:
            values (np.ndarray or list): Sequence of daily clinic fever visit counts.
        Outputs:
            tuple: (z_score (float), alert_flag (bool), risk_score (float))
        """
        req_len = self.window_days + self.guard_band_days + 1
        if len(values) < req_len:
            return 0.0, False, 0.0
        
        arr = np.asarray(values, dtype=np.float32)
        # Trailing baseline: [-(28 + 2 + 1) : -(2 + 1)]
        baseline = arr[-req_len : -(self.guard_band_days + 1)]
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        current = float(arr[-1])
        
        z = (current - mean) / (std + 1e-6)
        alert = bool(z >= self.z_threshold)
        # Sigmoid mapping of z-score to pseudo-probability [0, 1]
        risk_score = float(1.0 / (1.0 + np.exp(-1.5 * (z - self.z_threshold))))
        return float(z), alert, risk_score
