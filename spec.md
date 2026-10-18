# SPEC.md: Dengue Sentinel

> Every human and every AI agent must read this file fully before writing code.
> If a task conflicts with this file, stop and ask. Do not improvise.

## 1. Project summary

**Dengue Sentinel** is a real-time early-warning system for dengue outbreaks in Indian cities, piloted for **Bengaluru**. It detects emerging hotspots at ward level from streaming signals, ranks wards by risk, and recommends a prioritized response.

- **Hackathon:** IBM Z Datathon 2026
- **Theme:** Real-Time AI for Critical Decisions (Detect. Decide. Respond.)
- **Mission:** AI for Good
- **Deadline:** Sunday 18 Oct 2026. Feature freeze: Saturday 17 Oct, 12:00. Submit early.
- **Team:** 4 people (see Section 12)

### The pitch in three lines
1. **Detect:** weather, pharmacy fever-medicine sales and clinic fever visits stream in; a model scores every ward.
2. **Decide:** wards are ranked, each alert shows its top drivers in plain language.
3. **Respond:** the system allocates a limited number of fogging/inspection teams to the highest-risk wards.

### Why IBM Z / LinuxONE (must be demonstrated, not just claimed)
- Ingestion and scoring service runs on LinuxONE (Linux on Z), or locally in Docker as a clearly labeled fallback.
- Benchmark: events per second and p50/p99 scoring latency, reported on whichever platform it ran on.
- Model exported to ONNX, run with onnxruntime (Snap ML / Telum acceleration only if the environment supports it).
- Data protection story: health signals are sensitive; mention encryption and secure execution as platform strengths. Do not claim features we did not use.

## 2. Goals and non-goals

### Goals (must ship)
1. Replay of a real dengue season at accelerated speed through Kafka.
2. Ward-level risk scoring with a z-score baseline and one XGBoost model.
3. Live dashboard: ward heatmap, alert feed, ward detail panel, Respond panel.
4. One backtest result: how many days earlier we flag a rise than the official counts, plus the false alarm rate.
5. One benchmark result: throughput and latency, labeled with the platform used.
6. Submission deck, README, and a backup demo video.

### Non-goals (do NOT build)
- Google Trends or any extra data source.
- Deep learning, GNNs, LSTMs, Prophet.
- User accounts, authentication, or a production database.
- Real patient data of any kind.
- Anything not listed under Goals.

## 3. Hard rules

1. **Never invent results.** Every number in the deck must come from code in the repo that can be rerun with one command.
2. **Label synthetic data everywhere.** Every synthetic row has `is_synthetic = true`. The dashboard shows a visible "Simulated ward-level signals" badge.
3. **No data leakage.** The scoring service and model may only use streamed signals. The hidden ground-truth file is used only by the backtest and evaluation scripts. Train/test split is by time, never shuffled.
4. **Do not fake dataset loading.** If a real dataset file is missing, fail loudly with a clear error. Never silently substitute a made-up sample.
5. **Keep it simple.** Prefer the simplest working version. No new libraries without team approval.
6. **Fixed random seed (`SEED = 42`)** in all generators and models so results reproduce.
7. **Do not edit files outside your own folder** without telling the owner (see Section 12).
8. **Honest labeling of benchmarks:** "Local Docker" and "IBM LinuxONE" results must never be mixed or mislabeled.

## 4. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, geopandas (or plain json for GeoJSON) |
| ML | scikit-learn, xgboost, shap (optional), skl2onnx or onnxmltools, onnxruntime |
| Streaming | Apache Kafka via Docker Compose; `confluent-kafka` (preferred) or `kafka-python` |
| Dashboard | Streamlit + pydeck (or Folium); reads from Kafka topic `risk-scores` or a fallback CSV |
| Weather API | Open-Meteo historical API (no key needed) |
| Version control | Git, feature branches, merge to `main` |

Keep a single `requirements.txt` at repo root. Pin versions once things work.

## 5. Repository structure

```
dengue-sentinel/
├── SPEC.md
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Makefile                  # make data | generate | train | backtest | run | benchmark
├── data/
│   ├── raw/                  # downloaded files (opendengue, weather, wards geojson) - not committed if large
│   ├── processed/            # cleaned outputs
│   └── DATA_NOTES.md         # column meanings and source URLs, licences
├── generator/
│   └── generate_signals.py
├── model/
│   ├── features.py           # shared feature-building code (used by training AND streaming)
│   ├── baseline.py           # rolling z-score detector
│   ├── train.py
│   ├── predict.py            # predict(features) -> risk dict
│   ├── backtest.py
│   └── artifacts/            # model.joblib, model.onnx
├── streaming/
│   ├── replayer.py
│   ├── scoring_service.py
│   ├── benchmark.py
│   └── config.py
├── dashboard/
│   ├── app.py
│   ├── respond.py            # Decide/Respond rules
│   └── assets/
├── docs/
│   ├── architecture.md
│   ├── results.md            # backtest + benchmark numbers (generated by code)
│   └── deck/
└── tests/
```

## 6. Architecture

```
Real data: OpenDengue cases + Open-Meteo weather + ward GeoJSON
        |
        v
generator/ -> data/processed/events.csv        (streamed signals, is_synthetic flag)
           -> data/processed/ground_truth.csv  (hidden truth, NEVER read by scoring)
        |
        v
streaming/replayer.py  -- publishes at configurable speed -->  Kafka topic: signals
        |
        v
streaming/scoring_service.py
   - maintains rolling per-ward feature windows
   - runs z-score baseline and XGBoost (ONNX) model
   - publishes -->  Kafka topic: risk-scores
        |
        v
dashboard/app.py  (heatmap, alert feed, ward detail, Respond panel)
```

Kafka topics: `signals` (key = ward_id), `risk-scores` (key = ward_id).

## 7. Data contracts (the most important section)

### 7.1 Signal event (topic `signals`, and `events.csv`)
```json
{
  "ward_id": "W017",
  "date": "2024-09-14",
  "signal_type": "clinic_fever_visits",
  "value": 23.0,
  "is_synthetic": true
}
```
- `signal_type` is exactly one of: `rainfall`, `temp`, `humidity`, `pharmacy_fever_meds`, `clinic_fever_visits`.
- `rainfall`, `temp`, `humidity` are **real** (Open-Meteo, city-level, copied to every ward) and have `is_synthetic = false`.
- `pharmacy_fever_meds` and `clinic_fever_visits` are **synthetic** and have `is_synthetic = true`.
- `date` is ISO `YYYY-MM-DD`. Events are daily per ward per signal.
- `ward_id` is a string taken from the ward GeoJSON property chosen in `DATA_NOTES.md` and must match everywhere.

### 7.2 Risk output (topic `risk-scores`)
```json
{
  "ward_id": "W017",
  "date": "2024-09-14",
  "risk_score": 0.82,
  "confidence": 0.74,
  "alert_level": "high",
  "baseline_flag": true,
  "top_drivers": [
    {"feature": "clinic_visits_7d_change_pct", "value": 41.0, "direction": "up", "text": "Clinic fever visits up 41% over 7 days"},
    {"feature": "rainfall_lag14", "value": 62.5, "direction": "up", "text": "Heavy rainfall 14 days ago (62 mm)"}
  ],
  "scored_at_ms": 1760500000000,
  "event_time_ms": 1760499999800
}
```
- `risk_score`: float 0 to 1, probability that the ward enters an outbreak within the next 14 days.
- `confidence`: float 0 to 1. Define as agreement/strength measure documented in `model/predict.py` (for example based on distance from the decision threshold plus data completeness). Document the exact formula in code comments.
- `alert_level`: `low` (< 0.40), `medium` (0.40 to 0.70), `high` (> 0.70). Thresholds live in `streaming/config.py` only.
- `top_drivers`: 1 to 3 items, human-readable `text` required (used directly by the dashboard).

## 8. Data sources

| Dataset | Use | Real/Synthetic | Notes |
|---|---|---|---|
| OpenDengue | Historical dengue case counts (India / Karnataka) | Real | Check coverage level (state vs district). Record source URL and licence in `DATA_NOTES.md`. |
| Open-Meteo historical weather | Rainfall, temp, humidity for Bengaluru (lat 12.97, lon 77.59) | Real | Daily. No API key. |
| Bengaluru ward boundaries (GeoJSON) | Map and ward ids | Real | Use whichever ward file is available; record version and ward count. |
| Pharmacy fever-med sales per ward | Streamed signal | **Synthetic** | Generated. |
| Clinic fever visits per ward | Streamed signal | **Synthetic** | Generated. |

If OpenDengue only has state or district-level data, the pitch wording is: "real district-level trends, ward-level simulation."

## 9. Module specs

### 9.1 generator/ (owner: Person 1)
Script: `generator/generate_signals.py`

Inputs: processed OpenDengue weekly cases (real season), daily Bengaluru weather, ward list.

Behavior:
1. Pick one past dengue season (for example the most recent full season with usable data) as the **backbone curve**. Document which in `DATA_NOTES.md`.
2. Convert the weekly real curve to a daily city-level case index (interpolate).
3. Build a per-ward hidden `case_index` = city index x ward weight (random, seeded, lognormal, mean 1) + noise.
4. Inject outbreaks: choose 3 to 5 wards and make their `case_index` rise sharply (for example 3x to 5x) over 10 to 14 days at chosen dates, shortly before the city-level curve rises in the official data. Save the injected wards and dates in `ground_truth.csv`.
5. Derive streamed signals from `case_index`:
   - `clinic_fever_visits` = Poisson(base + k1 * case_index lagged by 3 to 5 days)
   - `pharmacy_fever_meds` = Poisson(base + k2 * case_index lagged by 5 to 8 days)
6. Attach real weather to every ward.

Outputs:
- `data/processed/events.csv` in the Section 7.1 schema.
- `data/processed/ground_truth.csv`: `ward_id, date, case_index, outbreak_flag`. **Never streamed, never used for features.**
- A printed summary: number of wards, date range, injected outbreaks.

Acceptance: running with the same seed gives identical output; all synthetic rows flagged; a plot of one injected ward looks like a believable outbreak.

### 9.2 model/ (owner: Person 2)

**Features (`model/features.py`)** - one function used by both training and streaming so they cannot drift apart:
- `clinic_visits_7d_mean`, `clinic_visits_7d_change_pct`
- `pharmacy_7d_mean`, `pharmacy_7d_change_pct`
- `rainfall_lag7`, `rainfall_lag14`, `rainfall_lag21`, `rainfall_14d_sum`
- `temp_7d_mean`, `humidity_7d_mean`
- ward-relative z-scores of clinic and pharmacy signals vs the ward's own trailing 28-day history

**Target:** `outbreak_in_next_14d` (0/1), derived from `ground_truth.csv` outbreak_flag. Used for training/evaluation only.

**Models:**
1. **Baseline A:** "same as last period" persistence.
2. **Baseline B:** rolling z-score detector on clinic visits (flag if z > 2 versus the trailing 28 days).
3. **Main:** XGBoost classifier. Time-based split (train earlier dates, validate on the later portion, test on the final season segment). No shuffling.

**`predict.py`:** `predict(feature_row) -> {risk_score, confidence, top_drivers}`. Uses ONNX runtime if `model.onnx` exists, otherwise joblib. Top drivers from SHAP values or XGBoost feature importance times the ward's feature deviation; must produce plain-language `text`.

**Backtest (`model/backtest.py`)** - the headline result. Definitions (do not change without team agreement):
- **Official detection date:** the first date the *real* weekly case count in the replayed season exceeds (mean + 2 standard deviations) of the preceding 8 weeks, plus an assumed **reporting delay of 7 days**. State this assumption in `results.md`.
- **Our detection date:** the first date the system raises an alert (risk > 0.70) in the corresponding period.
- **Days earlier** = official date minus our date. Report for the city and for each injected ward.
- **False alarm rate:** alerts (ward-days at `high`) in wards/periods with no outbreak within the following 14 days, divided by total `high` alerts. Also report precision and recall.
- Report the same metrics for baseline B so improvement is visible.
- Output: `docs/results.md` (auto-generated) and one chart `docs/backtest.png`.

State clearly in `results.md`: ward-level outbreaks are simulated, so the days-earlier result demonstrates the pipeline's behavior on realistic seasonality, not a validated real-world forecast.

### 9.3 streaming/ (owner: Person 3)
- **`docker-compose.yml`:** Kafka (KRaft mode, single broker) plus any UI-free helpers. One command to start: `docker compose up -d`.
- **`replayer.py`:** reads `events.csv`, groups by date, publishes to `signals` with key = ward_id. CLI args: `--speed` (simulated days per real second, default 5), `--start`, `--end`. Must be restartable.
- **`scoring_service.py`:** consumes `signals`, keeps per-ward rolling windows in memory, builds features with `model/features.py`, calls `model/predict.py`, publishes Section 7.2 messages to `risk-scores`. Scores a ward once all signals for that ward-day have arrived. Includes `scored_at_ms` and `event_time_ms` for latency measurement.
- **`benchmark.py`:** pushes N events (default 100,000) at maximum speed and reports:
  - throughput (events/sec)
  - scoring latency p50, p95, p99 (ms)
  - platform label (`--platform "local-docker"` or `--platform "ibm-linuxone"`), CPU model, core count, Python and library versions
  - Writes to `docs/results.md` under a "Benchmark" heading.
- Fallback: if Kafka breaks, support `--no-kafka` mode (replayer writes straight to the scoring function) so the demo never dies.

### 9.4 dashboard/ (owner: Person 4)
Streamlit app (`dashboard/app.py`), reads `risk-scores` from Kafka, or from `data/processed/risk_scores_fallback.csv` if `--offline` is set.

Layout:
1. **Header:** project name, current simulated date, replay speed control, a "Simulated ward-level signals" badge.
2. **Heatmap:** ward polygons colored by `risk_score` (green / amber / red per alert_level), tooltip with ward name, score, confidence.
3. **Alert feed:** newest alerts first: ward, level, one-line top driver text.
4. **Ward detail panel** (on selecting a ward): time series of clinic visits, pharmacy sales and rainfall; risk score over time; top drivers in plain language.
5. **Respond panel** (`dashboard/respond.py`):
   - Input: current risk scores. Control: number of available fogging/inspection teams (default 5, slider 1 to 15).
   - Rank wards by `risk_score`; assign teams to the top N wards with `alert_level != low`.
   - Each assignment shows a plain-language action, for example: "Schedule fogging and larval source inspection in W017 within 48 hours. Notify nearby hospitals to prepare for increased fever cases."
   - Wards above the team cap appear as "Watchlist".
   - Rules are simple, deterministic, and documented in `respond.py` docstrings.
6. **Footer:** "Real: weather, historical case counts, ward boundaries. Simulated: ward-level pharmacy and clinic signals."

Acceptance: runs fully on mock data first (matching Section 7.2), then on live Kafka data with no code changes except the data source.

## 10. Benchmark and results reporting
- All numbers in the deck come from `docs/results.md`, which is generated by `make backtest` and `make benchmark`.
- Never hand-edit numbers in `results.md`.
- Include in the deck: one backtest chart, one benchmark table (with platform label), one honest limitations slide.

## 11. Git workflow
- `main` is always runnable. Never push broken code to `main`.
- Branches: `data/*`, `model/*`, `streaming/*`, `dashboard/*`.
- Commit after every working step with a clear message.
- Merge to `main` at least twice a day (Fri midday, Fri night, Sat morning). Person 4 resolves conflicts.
- Large raw data files go in `.gitignore`; keep download instructions in `data/DATA_NOTES.md`.

## 12. Ownership and schedule

| Person | Role | Owns | Also responsible for |
|---|---|---|---|
| 1 | Data and generator | `data/`, `generator/` | Deck and README (from Fri), dataset slide |
| 2 | ML and backtest | `model/` | `docs/results.md` backtest section, ML slide |
| 3 | Streaming and Z | `streaming/`, `docker-compose.yml` | Benchmark, "Why IBM Z" slide |
| 4 | Dashboard and Respond | `dashboard/` | Integration, demo script, backup demo video |

| Day | Goal |
|---|---|
| **Thu 15 Oct** | Repo + this SPEC committed. Data downloaded. Generator v1 running. Kafka up with a test producer/consumer. Dashboard on mock data. |
| **Fri 16 Oct** | First end-to-end run (generator -> Kafka -> scoring -> dashboard). XGBoost plugged in. Backtest numbers computed. |
| **Sat 17 Oct** | Benchmark (LinuxONE if available). Polish. Deck, README, demo video. **Feature freeze 12:00.** |
| **Sun 18 Oct** | Buffer. Submit early. No new features. |

**Fallbacks (use immediately if a task stalls for more than half a day):**
- XGBoost fails -> submit z-score baseline only, say so honestly.
- Kafka fails -> `--no-kafka` mode.
- LinuxONE access unavailable -> run local Docker benchmark and label it "local baseline".
- ONNX export fails -> use joblib model.

## 13. Definition of done
- [ ] `make data generate train backtest run` works from a fresh clone, following README steps.
- [ ] Dashboard shows live ward heatmap, alert feed, ward detail, Respond panel.
- [ ] Synthetic data is labeled in code, in the dashboard, and in the deck.
- [ ] `docs/results.md` has backtest (days earlier, false alarm rate, precision, recall, baseline comparison) and benchmark (throughput, p50/p95/p99, platform label).
- [ ] Each team member can explain their module without an AI agent.
- [ ] Backup demo video recorded.
- [ ] Submission form fields match the problem statement, solution, audience and datasets text.

## 14. Instructions for AI coding agents
1. Read this entire file before starting. State a short plan and wait for approval before writing code.
2. Work only in the folder your task belongs to. Do not change schemas in Section 7.
3. Write small, runnable pieces. After each, show how to run it and what output to expect.
4. Add a short docstring to every function explaining inputs and outputs in plain language.
5. Never hardcode results, sample numbers, or fake "successful" output. If something cannot be computed, say so.
6. Never read `ground_truth.csv` outside `model/train.py`, `model/backtest.py`, and evaluation scripts.
7. When you hit an error, show the full error and propose the smallest fix.
8. Do not add dependencies, features, or files that are not in this spec without asking.