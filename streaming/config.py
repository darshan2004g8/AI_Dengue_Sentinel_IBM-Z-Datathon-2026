"""
streaming/config.py
Configuration and thresholds for Dengue Sentinel streaming service.
Loads thresholds from model/artifacts/thresholds.json when available,
with default fallbacks per BUILD.md Section 7.2 and 9.3.
"""

import os
import json

# Kafka Topics and Server Defaults
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_SIGNALS = "signals"
TOPIC_RISK_SCORES = "risk-scores"
FALLBACK_CSV_PATH = "data/processed/risk_scores_fallback.csv"

# Streaming Replayer Settings
DEFAULT_REPLAY_SPEED = 5  # Simulated days per real second

# Persistence & Alert Lifecycle Parameters (BUILD.md Section 7.2)
ALERT_LOW_THRESHOLD = 0.40  # T_MEDIUM default
ALERT_HIGH_THRESHOLD = 0.70 # T_HIGH default
PERSISTENCE_DAYS = 2        # Consecutive days at/above threshold to confirm alert
COOLDOWN_HOURS = 72         # No repeat notification for same ward and level
DOWNGRADE_DAYS = 5          # Consecutive days below (threshold - margin) to clear
DOWNGRADE_MARGIN = 0.05     # Hysteresis safety margin


def load_alert_thresholds(artifacts_path="model/artifacts/thresholds.json"):
    """
    Load tuned alert thresholds from model artifacts, or fall back to BUILD.md defaults.
    
    Inputs:
        artifacts_path (str): Path to thresholds.json.
    Outputs:
        tuple: (t_medium, t_high)
    """
    if os.path.exists(artifacts_path):
        try:
            with open(artifacts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("t_medium", 0.40)), float(data.get("t_high", 0.70))
        except Exception:
            pass
    return 0.40, 0.70


def get_alert_level(risk_score, t_medium=0.40, t_high=0.70):
    """
    Classify a calibrated risk score into an alert level.
    
    Inputs:
        risk_score (float): Probability between 0.0 and 1.0.
        t_medium (float): Threshold for medium alert.
        t_high (float): Threshold for high alert.
    Outputs:
        str: 'low', 'medium', or 'high'.
    """
    if risk_score > t_high:
        return "high"
    elif risk_score >= t_medium:
        return "medium"
    return "low"
