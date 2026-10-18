"""
streaming/replayer.py
Historical event stream replayer simulating real-time Kafka feed.
Adheres strictly to BUILD.md Section 7.1, 9.3, and Hard Rule 3:
  - Replays historical daily events chronologically by date.
  - Injects published_at_ms at publish time.
  - Publishes to Kafka topic 'signals' with key = ward_id.
  - Configurable playback speed (--speed in simulated days per real second).
  - Date filtering (--start, --end).
  - Robust --no-kafka mode routing directly to scoring service for guaranteed offline demo resilience.
"""

import os
import sys
import time
import json
import argparse
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from streaming.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_SIGNALS,
    DEFAULT_REPLAY_SPEED,
)


def parse_args():
    """
    Parse command-line arguments for replayer execution.
    
    Inputs:
        None (reads sys.argv).
    Outputs:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dengue Sentinel Streaming Replayer")
    parser.add_argument("--season", type=str, default="season_replay", help="Season identifier to replay (default: season_replay)")
    parser.add_argument("--speed", type=float, default=DEFAULT_REPLAY_SPEED, help="Simulated days per real second (default: 5.0)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-kafka", action="store_true", help="Run in direct offline pipeline mode without Kafka broker")
    parser.add_argument("--events-dir", type=str, default="data/processed/events", help="Directory of processed season event CSVs")
    return parser.parse_args()


def load_season_events(season_id, events_dir="data/processed/events"):
    """
    Load and chronologically order all signal events for a designated season.
    
    Inputs:
        season_id (str): Season identifier.
        events_dir (str): Directory containing season_XX.csv files.
    Outputs:
        pd.DataFrame: Ordered events dataframe.
    """
    file_path = os.path.join(events_dir, f"{season_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Season events CSV missing at: {file_path}. Run `make generate` first.")
    df = pd.read_csv(file_path)
    df.sort_values(by=["date", "ward_id", "signal_type"], inplace=True)
    return df


def create_kafka_producer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
    """
    Create a KafkaProducer instance configured for JSON serialization and snappy delivery.
    
    Inputs:
        bootstrap_servers (str): Broker connection address.
    Outputs:
        KafkaProducer or None: Connected producer or None if unreachable.
    """
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            request_timeout_ms=3000,
            max_block_ms=3000,
        )
        return producer
    except Exception as e:
        print(f"[REPLAYER] Kafka producer connection failed ({e}).")
        return None


def run_replayer(season_id="season_replay", speed=5.0, start_date=None, end_date=None, no_kafka=False, events_dir="data/processed/events"):
    """
    Execute day-by-day streaming replay at controlled playback velocity.
    
    Inputs:
        season_id (str): Season identifier.
        speed (float): Playback speed in simulated days per second.
        start_date (str, optional): Start date string.
        end_date (str, optional): End date string.
        no_kafka (bool): Direct dispatch flag.
        events_dir (str): Events directory.
    Outputs:
        int: Total number of events transmitted.
    """
    df = load_season_events(season_id, events_dir=events_dir)
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    dates = df["date"].unique()
    total_dates = len(dates)
    total_events = len(df)
    
    print("==================================================")
    print(f"DENGUE SENTINEL REPLAYER: {season_id}")
    print(f"Total dates: {total_dates} | Total events: {total_events}")
    print(f"Speed: {speed:.1f} simulated days/sec | Mode: {'--no-kafka (Direct)' if no_kafka else 'Kafka Broker'}")
    print("==================================================")

    producer = None
    if not no_kafka:
        producer = create_kafka_producer()
        if producer is None:
            print("[REPLAYER] Falling back to --no-kafka mode automatically.")
            no_kafka = True

    # If in direct mode, initialize direct scoring pipeline
    direct_scorer = None
    if no_kafka:
        from streaming.scoring_service import DirectScoringPipeline
        direct_scorer = DirectScoringPipeline()

    sleep_interval = 1.0 / max(speed, 0.1)
    published_count = 0

    for idx, dt_str in enumerate(dates):
        day_events = df[df["date"] == dt_str].to_dict(orient="records")
        current_time_ms = int(time.time() * 1000)

        # Inject publish timestamp into each event
        for ev in day_events:
            ev["published_at_ms"] = current_time_ms

        if no_kafka:
            # Direct pipeline call
            direct_scorer.ingest_day_events(day_events, dt_str)
        else:
            # Publish to Kafka topic 'signals' with key = ward_id
            for ev in day_events:
                producer.send(TOPIC_SIGNALS, key=ev["ward_id"], value=ev)
            producer.flush()

        published_count += len(day_events)

        if (idx + 1) % 15 == 0 or (idx + 1) == total_dates:
            print(f"  [Replay] Day {idx + 1:03d}/{total_dates:03d} ({dt_str}): {published_count}/{total_events} events dispatched")

        time.sleep(sleep_interval)

    if producer is not None:
        producer.close()

    print(f"\n[REPLAYER] Successfully finished replay of {season_id}: {published_count} events.")
    return published_count


def main():
    """Command-line entry point."""
    args = parse_args()
    run_replayer(
        season_id=args.season,
        speed=args.speed,
        start_date=args.start,
        end_date=args.end,
        no_kafka=args.no_kafka,
        events_dir=args.events_dir
    )


if __name__ == "__main__":
    main()
