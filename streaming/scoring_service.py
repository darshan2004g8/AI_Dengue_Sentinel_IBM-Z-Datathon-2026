"""
streaming/scoring_service.py
Real-time scoring daemon consuming streamed signals and publishing risk predictions.
Adheres strictly to BUILD.md Section 7.2 and 9.3:
  - Deduplicates incoming signals by event_id.
  - Maintains rolling observation buffers in memory per ward.
  - Performs vectorized per-date batch scoring when daily signals complete.
  - Runs Isotonic-Calibrated XGBoost (via ONNX Runtime) and EARS z-score baseline.
  - Applies alert lifecycle rules:
      * 2-day persistence confirmation rule.
      * 72-hour notification cooldown per ward and level.
      * 5-day hysteresis downgrade rule.
  - Publishes to Kafka topic 'risk-scores' and appends to data/processed/risk_scores_fallback.csv.
  - Operates as standalone Kafka daemon or direct in-process engine (--no-kafka mode).
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from streaming.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_SIGNALS,
    TOPIC_RISK_SCORES,
    FALLBACK_CSV_PATH,
    PERSISTENCE_DAYS,
    COOLDOWN_HOURS,
    DOWNGRADE_DAYS,
    DOWNGRADE_MARGIN,
    load_alert_thresholds,
)
from model.features import FEATURE_NAMES, compute_features_for_ward_dataframe
from model.baseline import ZScoreBaseline
from model.predict import DenguePredictor


class WardStateManager:
    """
    Tracks historical observations, consecutive alert counters, and cooldown timestamps per ward.
    """

    def __init__(self, buffer_days=40):
        """
        Initialize in-memory state manager.
        
        Inputs:
            buffer_days (int): Maximum daily records retained in rolling buffer.
        """
        self.buffer_days = buffer_days
        self.seen_event_ids = set()
        self.ward_buffers = {}  # ward_id -> list of dicts
        
        # Lifecycle state: ward_id -> state dict
        self.lifecycle = {}

    def is_duplicate(self, event_id):
        """
        Check if event_id was already ingested.
        
        Inputs:
            event_id (str): Unique event key (ward_id|date|signal_type).
        Outputs:
            bool: True if duplicate.
        """
        if event_id in self.seen_event_ids:
            return True
        self.seen_event_ids.add(event_id)
        # Cap set size to prevent unbounded memory growth in long streams
        if len(self.seen_event_ids) > 250000:
            self.seen_event_ids.clear()
        return False

    def append_event(self, event):
        """
        Append a single valid event to the corresponding ward's buffer.
        
        Inputs:
            event (dict): Event conforming to Section 7.1.
        Outputs:
            None
        """
        wid = event["ward_id"]
        if wid not in self.ward_buffers:
            self.ward_buffers[wid] = []
        self.ward_buffers[wid].append({
            "date": event["date"],
            "signal_type": event["signal_type"],
            "value": float(event["value"]),
        })

    def prune_buffers(self, current_date):
        """
        Prune records older than buffer_days relative to current_date.
        
        Inputs:
            current_date (str or pd.Timestamp): Current simulation date.
        Outputs:
            None
        """
        cutoff = pd.Timestamp(current_date) - pd.Timedelta(days=self.buffer_days)
        for wid in self.ward_buffers:
            self.ward_buffers[wid] = [
                r for r in self.ward_buffers[wid]
                if pd.Timestamp(r["date"]) >= cutoff
            ]

    def get_ward_history_df(self, ward_id):
        """
        Retrieve structured historical dataframe for a ward.
        
        Inputs:
            ward_id (str): Ward identifier.
        Outputs:
            pd.DataFrame: Table with columns ['date', 'clinic_fever_visits', ...].
        """
        records = self.ward_buffers.get(ward_id, [])
        if not records:
            return pd.DataFrame()
        raw_df = pd.DataFrame(records)
        piv = raw_df.pivot_table(index="date", columns="signal_type", values="value").reset_index()
        for col in ["clinic_fever_visits", "pharmacy_fever_meds", "rainfall", "temp", "humidity"]:
            if col not in piv.columns:
                piv[col] = 0.0
        piv["date"] = pd.to_datetime(piv["date"])
        piv.sort_values("date", inplace=True)
        return piv


class DirectScoringPipeline:
    """
    Self-contained scoring pipeline supporting both live Kafka streaming and direct offline execution.
    """

    def __init__(self, artifacts_dir="model/artifacts", fallback_csv_path=FALLBACK_CSV_PATH):
        """
        Initialize scoring models, state managers, and thresholds.
        
        Inputs:
            artifacts_dir (str): Location of trained models.
            fallback_csv_path (str): Destination for offline risk score appending.
        """
        self.state = WardStateManager(buffer_days=40)
        self.predictor = DenguePredictor(artifacts_dir=artifacts_dir)
        self.baseline_zscore = ZScoreBaseline(z_threshold=2.0, window_days=28, guard_band_days=2)
        self.fallback_csv_path = fallback_csv_path
        
        t_med, t_high = load_alert_thresholds(os.path.join(artifacts_dir, "thresholds.json"))
        self.t_medium = t_med
        self.t_high = t_high
        
        os.makedirs(os.path.dirname(self.fallback_csv_path), exist_ok=True)
        # Clear or initialize fallback CSV
        if not os.path.exists(self.fallback_csv_path):
            pd.DataFrame(columns=[
                "ward_id", "date", "risk_score", "confidence", "alert_level",
                "alert_confirmed", "baseline_flag", "top_drivers", "backend",
                "model_version", "scored_at_ms", "published_at_ms"
            ]).to_csv(self.fallback_csv_path, index=False)

    def apply_lifecycle_rules(self, ward_id, candidate_level, risk_score, current_date):
        """
        Enforce persistence confirmation, cooldown, and downgrade rules per Section 7.2:
          - alert_confirmed: score >= threshold on 2 consecutive days.
          - cooldown: no repeat trigger within 72 simulated hours.
          - downgrade: 5 consecutive days below (threshold - downgrade_margin).
        
        Inputs:
            ward_id (str): Ward identifier.
            candidate_level (str): 'low', 'medium', or 'high'.
            risk_score (float): Calibrated risk score.
            current_date (str): Current ISO date string.
        Outputs:
            bool: alert_confirmed.
        """
        curr_dt = pd.Timestamp(current_date)
        if ward_id not in self.state.lifecycle:
            self.state.lifecycle[ward_id] = {
                "active_confirmed_level": "low",
                "consecutive_above": 0,
                "consecutive_below": 0,
                "last_confirmed_date": None,
            }

        w_state = self.state.lifecycle[ward_id]
        threshold_for_candidate = self.t_high if candidate_level == "high" else (self.t_medium if candidate_level == "medium" else 0.0)

        # Persistence check
        if candidate_level in ("medium", "high") and risk_score >= threshold_for_candidate:
            w_state["consecutive_above"] += 1
            w_state["consecutive_below"] = 0
        else:
            w_state["consecutive_above"] = 0
            w_state["consecutive_below"] += 1

        alert_confirmed = False

        # Confirmation condition: at least 2 consecutive days at/above candidate threshold
        if w_state["consecutive_above"] >= PERSISTENCE_DAYS and candidate_level in ("medium", "high"):
            # Check 72-hour cooldown: no repeat notification for same ward and level
            can_notify = True
            if w_state["last_confirmed_date"] is not None:
                elapsed_hours = (curr_dt - w_state["last_confirmed_date"]).total_seconds() / 3600.0
                if elapsed_hours < COOLDOWN_HOURS and w_state["active_confirmed_level"] == candidate_level:
                    can_notify = False

            if can_notify:
                alert_confirmed = True
                w_state["active_confirmed_level"] = candidate_level
                w_state["last_confirmed_date"] = curr_dt

        # Downgrade condition: 5 consecutive days below (threshold - margin)
        active_lvl = w_state["active_confirmed_level"]
        if active_lvl != "low":
            active_thresh = self.t_high if active_lvl == "high" else self.t_medium
            if risk_score < (active_thresh - DOWNGRADE_MARGIN):
                if w_state["consecutive_below"] >= DOWNGRADE_DAYS:
                    w_state["active_confirmed_level"] = "low"

        return alert_confirmed

    def ingest_day_events(self, events_list, date_str, kafka_producer=None):
        """
        Ingest an entire day's batch of events, trigger vectorized scoring across all wards,
        and emit output records conforming to Section 7.2.
        
        Inputs:
            events_list (list of dict): All signal events for date_str.
            date_str (str): Target scoring date.
            kafka_producer (KafkaProducer, optional): Connected Kafka producer.
        Outputs:
            list of dict: Emitted risk score records.
        """
        # Deduplicate and append
        for ev in events_list:
            if not self.state.is_duplicate(ev["event_id"]):
                self.state.append_event(ev)

        self.state.prune_buffers(date_str)
        ward_ids = sorted(list(self.state.ward_buffers.keys()))
        if not ward_ids:
            return []

        # Build feature matrix across all active wards
        feature_rows = []
        baseline_flags = []
        completeness_scores = []
        pub_ms = events_list[0].get("published_at_ms", int(time.time() * 1000)) if events_list else int(time.time() * 1000)

        for wid in ward_ids:
            w_hist = self.state.get_ward_history_df(wid)
            if len(w_hist) == 0:
                feature_rows.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
                baseline_flags.append(False)
                completeness_scores.append(0.0)
                continue

            # Compute features vectorially
            feat_df = compute_features_for_ward_dataframe(w_hist)
            latest_feat = feat_df.iloc[-1].values
            feature_rows.append(latest_feat)
            completeness_scores.append(float(latest_feat[FEATURE_NAMES.index("completeness")]))

            # Run EARS z-score baseline
            clinic_counts = w_hist["clinic_fever_visits"].values
            _, b_flag, _ = self.baseline_zscore.score_series(clinic_counts)
            baseline_flags.append(b_flag)

        X_matrix = np.array(feature_rows, dtype=np.float32)

        # Execute vectorized model prediction in one call
        risk_outputs = self.predictor.score_feature_matrix(
            feature_matrix=X_matrix,
            ward_ids=ward_ids,
            date_str=date_str,
            completeness_scores=completeness_scores,
            published_at_ms=pub_ms,
            baseline_flags=baseline_flags,
        )

        # Apply persistence and cooldown lifecycle logic
        for item in risk_outputs:
            wid = item["ward_id"]
            lvl = item["alert_level"]
            score = item["risk_score"]
            is_confirmed = self.apply_lifecycle_rules(wid, lvl, score, date_str)
            item["alert_confirmed"] = is_confirmed

        # Publish to Kafka or save to fallback CSV
        if kafka_producer is not None:
            for item in risk_outputs:
                kafka_producer.send(TOPIC_RISK_SCORES, key=item["ward_id"], value=item)
            kafka_producer.flush()

        # Append to fallback CSV for offline dashboard consumption
        out_df = pd.DataFrame(risk_outputs)
        # Convert list of drivers to JSON string for CSV compatibility
        out_df["top_drivers"] = out_df["top_drivers"].apply(json.dumps)
        out_df.to_csv(self.fallback_csv_path, mode="a", header=not os.path.exists(self.fallback_csv_path) or os.stat(self.fallback_csv_path).st_size == 0, index=False)

        return risk_outputs


def run_kafka_scoring_daemon(artifacts_dir="model/artifacts"):
    """
    Run continuous Kafka consumer daemon ingesting signals and publishing risk scores.
    
    Inputs:
        artifacts_dir (str): Artifacts directory path.
    Outputs:
        None
    """
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        print("[SCORING SERVICE] kafka-python not installed. Exiting.")
        return

    print("==================================================")
    print("DENGUE SENTINEL SCORING SERVICE (KAFKA DAEMON)")
    print(f"Connecting to broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Subscribed to topic: {TOPIC_SIGNALS} -> Publishing to: {TOPIC_RISK_SCORES}")
    print("==================================================")

    pipeline = DirectScoringPipeline(artifacts_dir=artifacts_dir)

    try:
        consumer = KafkaConsumer(
            TOPIC_SIGNALS,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="dengue-sentinel-scorer",
        )
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
        )
    except Exception as e:
        print(f"[SCORING SERVICE] Failed to connect to Kafka ({e}). Use --no-kafka mode instead.")
        return

    current_date = None
    batch_events = []

    print("[SCORING SERVICE] Ready. Listening for inbound signal stream...")

    try:
        for message in consumer:
            ev = message.value
            ev_date = ev.get("date")

            if current_date is None:
                current_date = ev_date

            if ev_date != current_date:
                # Previous day complete: score the batch
                pipeline.ingest_day_events(batch_events, current_date, kafka_producer=producer)
                current_date = ev_date
                batch_events = [ev]
            else:
                batch_events.append(ev)

            # If 100 events arrived (20 wards x 5 signals), date is complete
            if len(batch_events) >= 100:
                pipeline.ingest_day_events(batch_events, current_date, kafka_producer=producer)
                batch_events = []
    except KeyboardInterrupt:
        print("\n[SCORING SERVICE] Shutting down daemon gracefully.")
    finally:
        consumer.close()
        producer.close()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Dengue Sentinel Scoring Service")
    parser.add_argument("--no-kafka", action="store_true", help="Run in self-test direct mode without Kafka")
    parser.add_argument("--artifacts-dir", type=str, default="model/artifacts", help="Artifacts directory")
    args = parser.parse_args()

    if args.no_kafka:
        print("[SCORING SERVICE] Running self-test in direct mode (--no-kafka)...")
        pipeline = DirectScoringPipeline(artifacts_dir=args.artifacts_dir)
        print("Scoring pipeline initialized successfully with ONNX Runtime backend and tuned thresholds.")
    else:
        run_kafka_scoring_daemon(artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    main()
