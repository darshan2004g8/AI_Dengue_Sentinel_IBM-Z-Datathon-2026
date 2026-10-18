# Dengue Sentinel - Makefile
# Automated pipeline targets per BUILD.md Section 5 and Section 14

PYTHON ?= python

.PHONY: all help data generate train backtest run benchmark demo reset test smoke-test run-all

all: run-all

run-all:
	$(PYTHON) run_all.py

help:
	@echo "Dengue Sentinel Build Targets (BUILD.md):"
	@echo "  make (or make all) - One-command self-healing runner: audits packages, checks data, runs tests, launches UI"
	@echo "  make data        - Check and fetch raw data inputs"
	@echo "  make generate    - Generate multi-season synthetic signals and ground truth"
	@echo "  make train       - Train XGBoost model, isotonic calibrator, and export ONNX"
	@echo "  make backtest    - Run E1-E6 evaluations and generate docs/results.md"
	@echo "  make run         - Run scoring service and dashboard"
	@echo "  make benchmark   - Run latency, throughput, backend comparison, and scale tests"
	@echo "  make demo        - Execute end-to-end replay and verification"
	@echo "  make test        - Run feature parity, ONNX parity, and allocator unit tests"
	@echo "  make smoke-test  - Run hour-0 environment check (scripts/smoke_test.sh)"
	@echo "  make reset       - Clear all generated artifacts and reset workspace"

data:
	$(PYTHON) -c "import os, sys; sys.exit(0 if os.path.exists('data/raw/bengaluru_weather_2018_2025.csv') else 1)" || $(PYTHON) data/fetch_weather.py
	$(PYTHON) -c "import os, sys; sys.exit(0 if os.path.exists('data/raw/filtered_data_SEARO_1792086539625.csv') else 1)" || $(PYTHON) data/fetch_opendengue.py
	@echo "Data verification complete."

generate:
	$(PYTHON) generator/generate_signals.py

train:
	$(PYTHON) model/train.py

backtest:
	$(PYTHON) model/backtest.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

run:
	$(PYTHON) streaming/scoring_service.py --no-kafka &
	streamlit run dashboard/app.py

benchmark:
	$(PYTHON) streaming/benchmark.py --platform local

demo:
	$(PYTHON) streaming/replayer.py --no-kafka --season season_replay --speed 10

smoke-test:
	bash scripts/smoke_test.sh

reset:
	$(PYTHON) -c "import shutil, glob, os; [os.remove(f) for f in glob.glob('data/processed/events/*.csv')]; [os.remove(f) for f in glob.glob('data/processed/ground_truth/*.csv')]; [os.remove(f) for f in glob.glob('data/processed/official/*.csv')]; [os.remove(f) for f in glob.glob('model/artifacts/*.*') if not f.endswith('.gitkeep')]; print('Reset complete.')"
