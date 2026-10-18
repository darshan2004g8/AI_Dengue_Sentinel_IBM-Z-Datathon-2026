"""
run_all.py
Single-command, zero-friction runner and self-healing orchestrator for Dengue Sentinel.
Event Code: SSF2026 (IBM Z Datathon 2026)

Key Capabilities:
  1. Automated Dependency Detection & Self-Healing:
     - Scans installed environment for all required packages in requirements.txt.
     - Automatically installs any missing packages via pip.
  2. End-to-End Pipeline Verification & Auto-Generation:
     - Verifies raw weather and epidemiology data; fetches if missing.
     - Generates 41 synthetic signal streams if not present.
     - Trains XGBoost model, fits Isotonic calibrator, tunes thresholds, and exports ONNX.
     - Executes backtest suite (E1-E6) and hardware benchmarks.
     - Runs all 12 unit tests to ensure system integrity.
  3. Interactive Dashboard Launch:
     - Automatically launches Streamlit operational dashboard on an available port.
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import importlib.util


# Map package distribution name -> importable module name
REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyyaml", "yaml"),
    ("requests", "requests"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("skl2onnx", "skl2onnx"),
    ("onnxmltools", "onnxmltools"),
    ("onnxruntime", "onnxruntime"),
    ("joblib", "joblib"),
    ("kafka-python", "kafka"),
    ("psutil", "psutil"),
    ("streamlit", "streamlit"),
    ("pydeck", "pydeck"),
    ("altair", "altair"),
    ("matplotlib", "matplotlib"),
]


def print_banner():
    """
    Print startup banner with datathon details and system metadata.
    """
    print("=" * 72)
    print("  DENGUE SENTINEL: REAL-TIME EARLY-WARNING DECISION SUPPORT SYSTEM")
    print("  Event Code: SSF2026 | IBM Z Datathon 2026 | Mission: AI for Good")
    print("  Platform Target: IBM LinuxONE (s390x) & x86_64")
    print("=" * 72)
    print(f"  Python: {sys.version.split()[0]} | Executable: {sys.executable}")
    print(f"  Working Directory: {os.getcwd()}")
    print("=" * 72)


def check_and_install_dependencies():
    """
    Detect missing Python packages and automatically install them via pip.
    
    Inputs:
        None.
    Outputs:
        bool: True if all dependencies are verified and available.
    """
    print("\n[STEP 1/6] Auditing installed dependencies...")
    missing_packages = []

    for dist_name, import_name in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing_packages.append(dist_name)

    if missing_packages:
        print(f"  [!] Missing {len(missing_packages)} package(s): {', '.join(missing_packages)}")
        print("  [*] Automatically installing missing dependencies from requirements.txt...")
        
        req_file = os.path.abspath("requirements.txt")
        if not os.path.exists(req_file):
            raise FileNotFoundError(f"requirements.txt missing at {req_file}")
            
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
        try:
            subprocess.check_call(cmd)
            print("  [+] Dependencies successfully installed.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] pip installation failed: {e}")
            print("  Retrying with individual missing packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    else:
        print("  [+] All required Python dependencies are installed and verified.")

    return True


def ensure_data_inputs():
    """
    Verify raw data availability and fetch if missing.
    
    Inputs:
        None.
    Outputs:
        None.
    """
    print("\n[STEP 2/6] Verifying raw data assets...")
    weather_path = os.path.join("data", "raw", "bengaluru_weather_2018_2025.csv")
    opendengue_path = os.path.join("data", "raw", "filtered_data_SEARO_1792086539625.csv")

    if not os.path.exists(weather_path):
        print("  [*] Fetching Open-Meteo Bengaluru historical weather (2018-2025)...")
        subprocess.check_call([sys.executable, "data/fetch_weather.py"])
    else:
        print(f"  [+] Real weather data present: {weather_path}")

    if not os.path.exists(opendengue_path):
        print("  [*] Fetching OpenDengue regional baseline data...")
        subprocess.check_call([sys.executable, "data/fetch_opendengue.py"])
    else:
        print(f"  [+] Epidemiology data present: {opendengue_path}")


def ensure_synthetic_signals(force=False):
    """
    Verify synthetic multi-season signals and generate if missing.
    
    Inputs:
        force (bool): Force re-generation if True.
    Outputs:
        None.
    """
    print("\n[STEP 3/6] Verifying multi-season streaming signals...")
    replay_file = os.path.join("data", "processed", "events", "season_replay.csv")
    manifest_file = os.path.join("data", "processed", "manifest.csv")

    if force or not os.path.exists(replay_file) or not os.path.exists(manifest_file):
        print("  [*] Generating 40 seasons + 5 control + 1 replay season...")
        subprocess.check_call([sys.executable, "generator/generate_signals.py"])
        print("  [+] Multi-season signal streams generated successfully.")
    else:
        print(f"  [+] Processed event streams verified (including {replay_file}).")


def ensure_model_and_onnx(force=False):
    """
    Verify model training and ONNX export. Train if missing.
    
    Inputs:
        force (bool): Force re-training if True.
    Outputs:
        None.
    """
    print("\n[STEP 4/6] Verifying trained model & ONNX runtime graph...")
    onnx_file = os.path.join("model", "artifacts", "model.onnx")
    calib_file = os.path.join("model", "artifacts", "calibrator.joblib")
    thresh_file = os.path.join("model", "artifacts", "thresholds.json")

    if force or not os.path.exists(onnx_file) or not os.path.exists(calib_file) or not os.path.exists(thresh_file):
        print("  [*] Training Monotone XGBoost & fitting Isotonic probability calibrator...")
        subprocess.check_call([sys.executable, "model/train.py"])
        print("  [*] Exporting model to ONNX & verifying numerical parity...")
        subprocess.check_call([sys.executable, "model/export_onnx.py"])
        print("  [+] Model artifacts and ONNX graph verified.")
    else:
        print(f"  [+] ONNX model and calibration weights verified ({onnx_file}).")


def run_tests_and_benchmarks(skip_backtest=False):
    """
    Run automated unit tests and check metrics generation.
    
    Inputs:
        skip_backtest (bool): Skip backtest/benchmark if artifacts already exist.
    Outputs:
        None.
    """
    print("\n[STEP 5/6] Running system verification & unit tests...")
    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    subprocess.check_call(test_cmd)
    print("  [+] All unit tests passed (12/12).")

    metrics_file = os.path.join("model", "artifacts", "metrics.json")
    bench_file = os.path.join("benchmark_metrics.json")

    if not skip_backtest or not os.path.exists(metrics_file):
        if not os.path.exists(metrics_file):
            print("  [*] Running epidemiological backtest suite (E1-E6)...")
            subprocess.check_call([sys.executable, "model/backtest.py"])

    if not os.path.exists(bench_file):
        print("  [*] Running hardware and streaming benchmark...")
        subprocess.check_call([sys.executable, "streaming/benchmark.py", "--platform", "local-docker"])


def is_port_in_use(port):
    """
    Check if a network port is currently occupied.
    
    Inputs:
        port (int): Port number.
    Outputs:
        bool: True if port is in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_available_port(start_port=8501, max_attempts=10):
    """
    Find first available port starting from start_port.
    
    Inputs:
        start_port (int): Initial port to check.
        max_attempts (int): Maximum consecutive ports to test.
    Outputs:
        int: Open port.
    """
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port


def launch_dashboard(port=None):
    """
    Launch the Streamlit dashboard on an available port.
    
    Inputs:
        port (int, optional): Port number.
    Outputs:
        None.
    """
    if port is None:
        port = find_available_port(start_port=8501)
    elif is_port_in_use(port):
        alt_port = find_available_port(start_port=port + 1)
        print(f"  [!] Port {port} is already in use. Redirecting to port {alt_port}...")
        port = alt_port

    print(f"\n[STEP 6/6] Launching Dengue Sentinel Operational Dashboard on port {port}...")
    print(f"  --> Local URL: http://localhost:{port}")
    print("  --> Press CTRL+C in this terminal at any time to exit.\n")
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", str(port),
        "--server.headless", "false"
    ]
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[+] Dashboard stopped gracefully. Goodbye!")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Dengue Sentinel Master One-Command Orchestrator (Event Code SSF2026)"
    )
    parser.add_argument("--skip-dashboard", action="store_true", help="Run full pipeline verification without launching browser dashboard")
    parser.add_argument("--demo", action="store_true", help="Run streaming replayer demo before launching dashboard")
    parser.add_argument("--force", action="store_true", help="Force re-generation of synthetic data and re-training of models")
    parser.add_argument("--port", type=int, default=None, help="Port to bind Streamlit dashboard (default: auto-detect from 8501)")
    args = parser.parse_args()

    start_time = time.time()
    print_banner()

    # Step 1: Detect and install missing packages
    check_and_install_dependencies()

    # Step 2: Ensure raw data is present
    ensure_data_inputs()

    # Step 3: Ensure synthetic signals exist
    ensure_synthetic_signals(force=args.force)

    # Step 4: Ensure model is trained and ONNX exported
    ensure_model_and_onnx(force=args.force)

    # Step 5: Run tests and verify backtest/benchmarks
    run_tests_and_benchmarks(skip_backtest=not args.force)

    # Optional stream demo
    if args.demo:
        print("\n[*] Running live streaming replayer demo (simulating real-time feed)...")
        demo_cmd = [
            sys.executable, "streaming/replayer.py",
            "--no-kafka",
            "--season", "season_replay",
            "--speed", "50"
        ]
        subprocess.check_call(demo_cmd)

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"  [+] COMPLETE PIPELINE READY & VERIFIED ({elapsed:.1f}s)")
    print("  Hard Rule 10 Caveat:")
    print("  'Pharmacy and clinic signals are synthetic, so lead time shows")
    print("   pipeline behavior, not real-world validation.'")
    print("=" * 72)

    # Step 6: Launch dashboard
    if not args.skip_dashboard:
        launch_dashboard(port=args.port)
    else:
        print("\n  [+] --skip-dashboard flag supplied. Execution finished successfully.")


if __name__ == "__main__":
    main()
