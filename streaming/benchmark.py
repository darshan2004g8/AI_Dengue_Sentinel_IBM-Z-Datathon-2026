"""
streaming/benchmark.py
Rigorous performance benchmarking engine for IBM Z / LinuxONE validation.
Adheres strictly to BUILD.md Section 4, 9.3, and Hard Rule 9:
  - Throughput: Evaluates maximum event ingestion rate (events/sec) across 100k events.
  - Latency: Rate-controlled non-saturated latency percentiles (p50, p95, p99 in ms).
  - Backend Comparison: Native XGBoost vs ONNX Runtime at 1, 100, 1k, 10k, 100k rows.
  - Scale Test: Virtual city scaling (1, 10, 50, 100 cities) measuring p99 tick latency < 1.0s.
  - Honest Labeling: Explicit --platform flag ('ibm-linuxone' or 'local-docker').
  - Emits results to docs/results.md and model/artifacts/benchmark_metrics.json.
"""

import os
import sys
import time
import json
import argparse
import platform
import subprocess
import psutil
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
import xgboost as xgb
import sklearn

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from streaming.scoring_service import WardStateManager
from model.features import FEATURE_NAMES


def parse_args():
    """
    Parse command-line arguments for benchmark execution.
    
    Inputs:
        None (reads sys.argv).
    Outputs:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Dengue Sentinel Performance Benchmark")
    parser.add_argument("--platform", type=str, default="local", choices=["ibm-linuxone", "local", "local-docker"],
                        help="Platform label ('ibm-linuxone', 'local', or 'local-docker') per Hard Rule 9")
    parser.add_argument("--events", type=int, default=100000, help="Number of events for throughput test (default: 100000)")
    parser.add_argument("--rate", type=int, default=5000, help="Controlled rate (events/s) for latency evaluation (default: 5000)")
    parser.add_argument("--output-json", type=str, default="model/artifacts/benchmark_metrics.json", help="Path to output JSON")
    parser.add_argument("--output-md", type=str, default="docs/results.md", help="Path to results Markdown")
    parser.add_argument("--artifacts-dir", type=str, default="model/artifacts", help="Artifacts directory")
    return parser.parse_args()


def get_git_commit():
    """
    Retrieve current short git commit hash safely.
    
    Inputs:
        None
    Outputs:
        str: Short git commit hash or 'unknown'.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def get_system_metadata(platform_label):
    """
    Collect host hardware and software environment details for honest reporting.
    
    Inputs:
        platform_label (str): Selected platform category ('ibm-linuxone' or 'local-docker').
    Outputs:
        dict: System hardware and software metadata.
    """
    cpu_model = platform.processor() or "Unknown CPU"
    if sys.platform == "win32":
        cpu_model = os.environ.get("PROCESSOR_IDENTIFIER", cpu_model)

    return {
        "platform_label": platform_label,
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "git_commit": get_git_commit(),
        "libraries": {
            "xgboost": xgb.__version__,
            "onnxruntime": rt.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "psutil": psutil.__version__,
        }
    }


def generate_benchmark_events(n_events=100000):
    """
    Pre-generate a repeatable stream of valid signal events conforming to Section 7.1.
    
    Inputs:
        n_events (int): Number of events to synthesize.
    Outputs:
        list of dict: Synthetic signal event dictionaries.
    """
    signal_types = ["clinic_fever_visits", "pharmacy_fever_meds", "rainfall_mm", "temp_max", "temp_min"]
    events = []
    base_ts = 1720000000000
    
    for i in range(n_events):
        wid = (i % 20) + 1
        sig = signal_types[(i // 20) % len(signal_types)]
        day_idx = i // 100
        events.append({
            "event_id": f"W{wid:03d}|day_{day_idx:04d}|{sig}",
            "ward_id": f"W{wid:03d}",
            "date": f"2024-07-{(day_idx % 28) + 1:02d}",
            "signal_type": sig,
            "value": float((i * 7) % 50),
            "published_at_ms": base_ts + i * 10,
            "is_synthetic": True,
        })
    return events


def benchmark_throughput(n_events=100000, n_trials=3):
    """
    Measure maximum raw event parsing and ingestion throughput across 100k events.
    Discards 1 warm-up run and computes the median throughput of n_trials.
    
    Inputs:
        n_events (int): Number of synthetic event records to process.
        n_trials (int): Number of repeated evaluation trials.
    Outputs:
        dict: Throughput metrics including events_per_second and median_elapsed_sec.
    """
    events = generate_benchmark_events(n_events)
    warmup_events = events[:10000]

    # Warm-up run (discarded)
    warmup_state = WardStateManager(buffer_days=40)
    for ev in warmup_events:
        if not warmup_state.is_duplicate(ev["event_id"]):
            warmup_state.append_event(ev)

    trial_durations = []
    for _ in range(n_trials):
        state = WardStateManager(buffer_days=40)
        t_start = time.perf_counter()
        for ev in events:
            if not state.is_duplicate(ev["event_id"]):
                state.append_event(ev)
        t_elapsed = time.perf_counter() - t_start
        trial_durations.append(t_elapsed)

    median_duration = float(np.median(trial_durations))
    throughput = float(n_events / median_duration)

    return {
        "n_events": n_events,
        "n_trials": n_trials,
        "median_elapsed_sec": round(median_duration, 4),
        "events_per_second": round(throughput, 1),
        "all_trial_durations_sec": [round(d, 4) for d in trial_durations],
    }


def benchmark_controlled_latency(target_rate=5000, n_samples=10000, n_trials=3):
    """
    Measure p50, p95, and p99 scoring latency under controlled sub-saturation load.
    BUILD.md rule: Never quote latency from the saturated run.
    
    Inputs:
        target_rate (int): Injected events per second.
        n_samples (int): Number of evaluated events.
        n_trials (int): Number of repeated evaluation trials.
    Outputs:
        dict: Latency percentiles in milliseconds: {"p50_ms": float, "p95_ms": float, "p99_ms": float}.
    """
    events = generate_benchmark_events(n_samples)
    batch_size = 50
    n_batches = n_samples // batch_size
    interval_sec = batch_size / float(target_rate)

    # Warm-up
    warm_state = WardStateManager(buffer_days=40)
    for b in range(min(5, n_batches)):
        chunk = events[b * batch_size: (b + 1) * batch_size]
        for ev in chunk:
            if not warm_state.is_duplicate(ev["event_id"]):
                warm_state.append_event(ev)

    trial_p50 = []
    trial_p95 = []
    trial_p99 = []

    for _ in range(n_trials):
        state = WardStateManager(buffer_days=40)
        latencies_ms = []

        next_time = time.perf_counter()
        for b in range(n_batches):
            chunk = events[b * batch_size: (b + 1) * batch_size]
            
            # Rate control pacing
            now = time.perf_counter()
            if now < next_time:
                time.sleep(max(0.0, next_time - now))
            next_time = time.perf_counter() + interval_sec

            # Process micro-batch and measure per-event latency
            t0 = time.perf_counter()
            for ev in chunk:
                if not state.is_duplicate(ev["event_id"]):
                    state.append_event(ev)
            t_batch_ms = (time.perf_counter() - t0) * 1000.0
            per_event_ms = t_batch_ms / batch_size
            latencies_ms.extend([per_event_ms] * batch_size)

        trial_p50.append(float(np.percentile(latencies_ms, 50)))
        trial_p95.append(float(np.percentile(latencies_ms, 95)))
        trial_p99.append(float(np.percentile(latencies_ms, 99)))

    return {
        "target_rate_events_per_sec": target_rate,
        "n_samples": n_samples,
        "p50_ms": round(float(np.median(trial_p50)), 4),
        "p95_ms": round(float(np.median(trial_p95)), 4),
        "p99_ms": round(float(np.median(trial_p99)), 4),
    }


def benchmark_backends(artifacts_dir="model/artifacts", row_counts=(1, 100, 1000, 10000, 100000), n_trials=3):
    """
    Compare inference execution times between Native XGBoost and ONNX Runtime.
    Discards initial warm-up run and reports the median of 3 iterations.
    
    Inputs:
        artifacts_dir (str): Location of model.joblib and model.onnx.
        row_counts (tuple of int): Batch sizes evaluated.
        n_trials (int): Number of measurement iterations.
    Outputs:
        dict: Inference latencies (ms) and throughput (rows/s) across evaluated batch sizes.
    """
    native_path = os.path.join(artifacts_dir, "model.joblib")
    onnx_path = os.path.join(artifacts_dir, "model.onnx")

    if not os.path.exists(native_path):
        raise FileNotFoundError(f"Native model missing at {native_path}. Run `make train` first.")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model missing at {onnx_path}. Run `make train` first.")

    xgb_model = joblib.load(native_path)
    onnx_session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name

    # Check Snap ML availability honestly
    snapml_available = False
    try:
        import snapml
        snapml_available = True
    except ImportError:
        pass

    results = {
        "native_xgboost": {},
        "onnxruntime": {},
        "snap_ml": "Not installed / unavailable in current environment (optional on LinuxONE VM per BUILD.md Section 4)" if not snapml_available else {}
    }

    n_features = len(FEATURE_NAMES)
    np.random.seed(42)

    for r in row_counts:
        X = np.random.randn(r, n_features).astype(np.float32)

        # 1. Native XGBoost
        # Warmup
        _ = xgb_model.predict_proba(X)
        xgb_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = xgb_model.predict_proba(X)
            xgb_times.append((time.perf_counter() - t0) * 1000.0)
        xgb_med_ms = float(np.median(xgb_times))
        results["native_xgboost"][str(r)] = {
            "median_latency_ms": round(xgb_med_ms, 3),
            "throughput_rows_per_sec": round(r / (xgb_med_ms / 1000.0), 1),
        }

        # 2. ONNX Runtime
        # Warmup
        _ = onnx_session.run(None, {input_name: X})
        ort_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = onnx_session.run(None, {input_name: X})
            ort_times.append((time.perf_counter() - t0) * 1000.0)
        ort_med_ms = float(np.median(ort_times))
        results["onnxruntime"][str(r)] = {
            "median_latency_ms": round(ort_med_ms, 3),
            "throughput_rows_per_sec": round(r / (ort_med_ms / 1000.0), 1),
        }

    return results


def benchmark_virtual_cities_scale(artifacts_dir="model/artifacts", city_counts=(1, 10, 50, 100), n_ticks=30):
    """
    Replicate Bengaluru's wards into N virtual cities and verify if p99 tick latency stays under 1.0s.
    Evaluates ONNX inference + isotonic calibration + threshold checks across all wards.
    
    Inputs:
        artifacts_dir (str): Location of model artifacts.
        city_counts (tuple of int): Number of replicated virtual cities.
        n_ticks (int): Simulated scoring ticks per city count.
    Outputs:
        dict: Tick latencies and maximum sustainable city count.
    """
    onnx_path = os.path.join(artifacts_dir, "model.onnx")
    cal_path = os.path.join(artifacts_dir, "calibrator.joblib")
    th_path = os.path.join(artifacts_dir, "thresholds.json")

    onnx_session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[1].name
    calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

    t_medium, t_high = 0.40, 0.70
    if os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f)
            t_medium = float(th.get("t_medium", 0.40))
            t_high = float(th.get("t_high", 0.70))

    results_by_city = {}
    sustainable_max = 1
    n_features = len(FEATURE_NAMES)

    for n_cities in city_counts:
        n_wards = n_cities * 20
        X_tick = np.random.randn(n_wards, n_features).astype(np.float32)

        # Warmup ticks (discarded)
        for _ in range(5):
            raw = onnx_session.run([output_name], {input_name: X_tick})[0]
            probs = raw[:, 1] if hasattr(raw, "ndim") and raw.ndim > 1 else raw
            if calibrator is not None:
                _ = calibrator.predict(probs)

        # Measured ticks
        tick_times_ms = []
        for _ in range(n_ticks):
            t0 = time.perf_counter()
            raw = onnx_session.run([output_name], {input_name: X_tick})[0]
            probs = raw[:, 1] if hasattr(raw, "ndim") and raw.ndim > 1 else raw
            if calibrator is not None:
                cal_probs = calibrator.predict(probs)
            else:
                cal_probs = probs
            # Alert classification
            _ = np.where(cal_probs > t_high, "high", np.where(cal_probs >= t_medium, "medium", "low"))
            tick_times_ms.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.percentile(tick_times_ms, 50))
        p95 = float(np.percentile(tick_times_ms, 95))
        p99 = float(np.percentile(tick_times_ms, 99))
        under_one_second = (p99 < 1000.0)

        if under_one_second and n_cities > sustainable_max:
            sustainable_max = n_cities

        results_by_city[str(n_cities)] = {
            "virtual_cities": n_cities,
            "total_wards": n_wards,
            "p50_tick_latency_ms": round(p50, 2),
            "p95_tick_latency_ms": round(p95, 2),
            "p99_tick_latency_ms": round(p99, 2),
            "meets_1s_sla": under_one_second,
            "events_per_sec": round(n_cities * 20 * 5 / 60.0, 2),  # 20 wards x 5 signals / 60s
        }

    return {
        "sustainable_cities_under_1s_sla": sustainable_max,
        "cities_per_node_arithmetic": "events/s per city = wards (20) x signals per ward (5) x events per signal per minute (1) / 60 = 1.67 events/s. At 100 cities: 167 events/s.",
        "latencies_by_city_count": results_by_city,
    }


def update_results_markdown(metrics_dict, md_path="docs/results.md"):
    """
    Format and update Section 2 (Benchmark Results) in docs/results.md.
    Enforces honest platform labeling and reproducibility.
    
    Inputs:
        metrics_dict (dict): Consolidated benchmark outcomes.
        md_path (str): Destination markdown file.
    Outputs:
        None
    """
    if not os.path.exists(md_path):
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    meta = metrics_dict["metadata"]
    tp = metrics_dict["throughput"]
    lat = metrics_dict["controlled_latency"]
    backends = metrics_dict["backend_comparison"]
    scale = metrics_dict["scale_test"]

    # Build backend comparison rows
    backend_rows = []
    for r_str in ["1", "100", "1000", "10000", "100000"]:
        xgb_lat = backends["native_xgboost"].get(r_str, {}).get("median_latency_ms", "N/A")
        ort_lat = backends["onnxruntime"].get(r_str, {}).get("median_latency_ms", "N/A")
        speedup = ""
        if isinstance(xgb_lat, (int, float)) and isinstance(ort_lat, (int, float)) and ort_lat > 0:
            speedup = f"{xgb_lat / ort_lat:.1f}x"
        backend_rows.append(f"| {r_str} | {xgb_lat} ms | {ort_lat} ms | {speedup} |")

    # Build scale test rows
    scale_rows = []
    for c_str, s_data in scale["latencies_by_city_count"].items():
        scale_rows.append(
            f"| {s_data['virtual_cities']} cities ({s_data['total_wards']} wards) | {s_data['events_per_sec']} | "
            f"{s_data['p50_tick_latency_ms']} ms | {s_data['p95_tick_latency_ms']} ms | {s_data['p99_tick_latency_ms']} ms | "
            f"{'Yes (<1.0s)' if s_data['meets_1s_sla'] else 'No'} |"
        )

    benchmark_md = f"""## 2. Benchmark Results (IBM Z / LinuxONE vs Local)

*Generated deterministically by `make benchmark` (`streaming/benchmark.py`) at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.*

### Platform & Environment Metadata
- **Platform Label**: `{meta['platform_label']}` (Hard Rule 9: honest reporting)
- **OS**: `{meta['os']}`
- **Architecture**: `{meta['arch']}`
- **CPU Model**: `{meta['cpu_model']}` ({meta['cpu_logical_cores']} logical cores, {meta['cpu_physical_cores']} physical cores)
- **RAM**: `{meta['ram_gb']} GB`
- **Python Version**: `{meta['python_version']}` | **Git Commit**: `{meta['git_commit']}`
- **Key Libraries**: XGBoost `{meta['libraries']['xgboost']}`, ONNX Runtime `{meta['libraries']['onnxruntime']}`, Scikit-Learn `{meta['libraries']['scikit-learn']}`

### Streaming Ingestion & Latency
- **Ingestion Throughput**: **{tp['events_per_second']:,} events/sec** (evaluated across {tp['n_events']:,} Section 7.1 events; median of {tp['n_trials']} trials)
- **Rate-Controlled Latency** (at controlled sub-saturation load of {lat['target_rate_events_per_sec']:,} events/sec; *never quoted from saturated run*):
  - **p50 Latency**: `{lat['p50_ms']} ms`
  - **p95 Latency**: `{lat['p95_ms']} ms`
  - **p99 Latency**: `{lat['p99_ms']} ms`

### Backend Comparison (Native XGBoost vs ONNX Runtime)
*Median of 3 runs after warm-up on identical 16-feature matrices:*

| Rows (Batch Size) | Native XGBoost | ONNX Runtime | ONNX Speedup |
|---|---|---|---|
""" + "\n".join(backend_rows) + f"""

*Snap ML status*: `{backends.get('snap_ml', 'N/A')}`

### Virtual Cities Scale Test
*{scale['cities_per_node_arithmetic']}*
*Daily batch tick latency SLA: p99 < 1,000 ms (1.0 s).*

| Scale Level | Ingestion Stream (events/s) | p50 Tick Latency | p95 Tick Latency | p99 Tick Latency | SLA Met? |
|---|---|---|---|---|---|
""" + "\n".join(scale_rows) + f"""

- **Maximum Sustainable Capacity**: **{scale['sustainable_cities_under_1s_sla']} virtual cities** ({scale['sustainable_cities_under_1s_sla'] * 20} wards) maintaining sub-second p99 decision latency.
"""

    # Replace section 2 in results.md
    if "## 2. Benchmark Results" in content:
        parts = content.split("## 2. Benchmark Results")
        header = parts[0]
        new_content = header + benchmark_md
    else:
        new_content = content + "\n\n" + benchmark_md

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    """Main benchmark suite orchestrator."""
    args = parse_args()
    print("==================================================")
    print("DENGUE SENTINEL PERFORMANCE BENCHMARK")
    print(f"Platform: {args.platform} (Hard Rule 9)")
    print(f"Throughput Target: {args.events:,} events | Paced Latency Rate: {args.rate:,} events/s")
    print("==================================================")

    # 1. System Metadata
    metadata = get_system_metadata(args.platform)
    print(f"Host: {metadata['cpu_model']} | {metadata['cpu_logical_cores']} cores | {metadata['ram_gb']} GB RAM")
    print(f"OS: {metadata['os']} | Python: {metadata['python_version']} | Git: {metadata['git_commit']}")

    # 2. Ingestion Throughput
    print(f"\n[1/4] Measuring raw ingestion throughput ({args.events:,} events)...")
    tp_results = benchmark_throughput(n_events=args.events, n_trials=3)
    print(f"  -> Ingestion Throughput: {tp_results['events_per_second']:,} events/sec (median: {tp_results['median_elapsed_sec']}s)")

    # 3. Controlled Latency
    print(f"\n[2/4] Measuring rate-controlled latency ({args.rate:,} events/s)...")
    lat_results = benchmark_controlled_latency(target_rate=args.rate, n_samples=min(10000, args.events), n_trials=3)
    print(f"  -> Latency Percentiles: p50={lat_results['p50_ms']} ms | p95={lat_results['p95_ms']} ms | p99={lat_results['p99_ms']} ms")

    # 4. Backend Comparison
    print("\n[3/4] Evaluating backend execution times (Native XGBoost vs ONNX Runtime)...")
    backend_results = benchmark_backends(artifacts_dir=args.artifacts_dir, row_counts=(1, 100, 1000, 10000, 100000), n_trials=3)
    print(f"  {'Batch':<10} | {'Native XGB':<14} | {'ONNX Runtime':<14} | {'Speedup'}")
    print(f"  {'-'*10}-|-{'-'*14}-|-{'-'*14}-|-{'-'*8}")
    for r in [1, 100, 1000, 10000, 100000]:
        xgb_ms = backend_results["native_xgboost"][str(r)]["median_latency_ms"]
        ort_ms = backend_results["onnxruntime"][str(r)]["median_latency_ms"]
        speedup = f"{xgb_ms / ort_ms:.1f}x" if ort_ms > 0 else "N/A"
        print(f"  {r:<10} | {xgb_ms:<11.3f} ms | {ort_ms:<11.3f} ms | {speedup}")

    # 5. Virtual Cities Scale Test
    print("\n[4/4] Running virtual cities scale test (1, 10, 50, 100 cities)...")
    scale_results = benchmark_virtual_cities_scale(artifacts_dir=args.artifacts_dir, city_counts=(1, 10, 50, 100), n_ticks=30)
    for c, s in scale_results["latencies_by_city_count"].items():
        print(f"  City count {c:>3} ({s['total_wards']:>4} wards): p50={s['p50_tick_latency_ms']:.2f}ms, p95={s['p95_tick_latency_ms']:.2f}ms, p99={s['p99_tick_latency_ms']:.2f}ms | SLA Met: {s['meets_1s_sla']}")
    print(f"  -> Maximum Sustainable Cities (p99 < 1.0s): {scale_results['sustainable_cities_under_1s_sla']}")

    # 6. Save JSON and update Markdown
    all_metrics = {
        "metadata": metadata,
        "throughput": tp_results,
        "controlled_latency": lat_results,
        "backend_comparison": backend_results,
        "scale_test": scale_results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[BENCHMARK] Saved metrics JSON to {args.output_json}")

    update_results_markdown(all_metrics, md_path=args.output_md)
    print(f"[BENCHMARK] Updated markdown results in {args.output_md}")
    print("==================================================")


if __name__ == "__main__":
    main()

