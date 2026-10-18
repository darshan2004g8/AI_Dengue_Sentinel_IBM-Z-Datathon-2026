# BUILD.md: Dengue Sentinel

> Replaces SPEC.md. Every human and every AI agent must read this file fully before writing code.
> If a task conflicts with this file, stop and ask. Do not improvise.
> The PRD and Architecture document explains the reasoning behind these choices. This file is the build contract.

## 0. What changed from SPEC.md

| Area | SPEC.md | BUILD.md |
|---|---|---|
| Schedule | Freeze Sat 12:00 (2 h after kick-off) | Kick-off Sat 10:00, freeze Sun 02:00, submit by Sun 08:30 (hard close 10:30 IST) |
| Platform | LinuxONE optional, Docker fallback | LinuxONE VM is the primary target. Docker is a laptop dev aid only |
| Data | One season, 3 to 5 outbreaks, time split | Many synthetic seasons, split by season, control seasons, held-out replay season |
| Generator | Pharmacy lags clinic, plain Poisson | Pharmacy leads clinic, negative-binomial noise, confounders, lag sweep |
| Outbreak label | Injected wards only | Rule-based onset label (Section 9.1) so seasonal rises are handled uniformly |
| Metrics | Days earlier, false-alarm share, precision, recall | Adds false alerts per 100 ward-weeks, PR-AUC, Brier, ablations, delay sensitivity |
| Alerts | Fixed 0.40 / 0.70 | Thresholds tuned on validation seasons, plus a 2-day persistence rule |
| Scoring | Per ward-day | Per-date batch, one vectorised call for all wards |
| Allocation | Rank by risk | Exposure-weighted greedy if ward population exists, otherwise rank by risk |
| Actions | Fogging for every non-low ward | Inspection first, targeted fogging only for high-risk clusters |
| Benchmark | 100k events at max rate | Adds rate-controlled latency, backend comparison, virtual-city scale test |
| Dashboard | 4 panels | Adds Benchmark and Validation tabs, official-detection marker, offline-safe map |
| Dependencies | confluent-kafka preferred, Streamlit fixed | kafka-python default, Streamlit gated by the hour-0 smoke test |

## 1. Project summary

**Dengue Sentinel** is a real-time early-warning system for dengue outbreaks in Indian cities, piloted for **Bengaluru**. It detects emerging hotspots at ward level from streaming signals, ranks wards by risk, and recommends a prioritized response.

- **Hackathon:** IBM Z Datathon 2026. Theme: Real-Time AI for Critical Decisions (Detect. Decide. Respond.). Mission: AI for Good.
- **Event code (put in the project description):** `SSF2026`
- **Team:** 4 people (Section 12)

### Dates (IST)
- Fri 16 Oct, 19:30: mandatory LinuxONE session (join the Zoom link from the participant instructions).
- Sat 17 Oct, 10:00: kick-off at GAT campus. Team works virtually from about 17:00.
- Sun 18 Oct, 02:00: **feature freeze**. 08:30: **target submission**. 10:30: hard close on Hack2Skill.
- Check the participant guidebook: if code must be written during the event, only do non-code prep before 10:00 on Sat (abstract, data download, this file).

### The pitch in three lines
1. **Detect:** weather, pharmacy fever-medicine sales and clinic fever visits stream in; a model scores every ward.
2. **Decide:** wards are ranked, each alert shows its top drivers in plain language.
3. **Respond:** the system allocates a limited number of inspection and fogging teams to the highest-risk wards.

### Why IBM Z / LinuxONE (must be demonstrated, not just claimed)
- Ingestion and scoring run on the team's LinuxONE VM. The final project is uploaded there.
- Benchmark: events per second, p50/p95/p99 latency, backend comparison, scale test, all labeled with the platform.
- Model exported to ONNX and run with onnxruntime. Snap ML or accelerator use only if the VM supports it. Say plainly what was and was not used.
- Data protection: only claim controls that were enabled and tested (Section 11).

## 2. Goals and non-goals

### Goals (must ship)
1. Replay of a real dengue season at accelerated speed through Kafka (on the LinuxONE VM).
2. Ward-level risk scoring with two baselines and one calibrated XGBoost model.
3. Dashboard: ward heatmap, alert feed, ward detail, Respond panel, Benchmark tab, Validation tab.
4. Backtest results: days earlier than official counts, false alerts, precision, recall, ablations.
5. Benchmark results: throughput, latency, backend comparison, scale test.
6. Submission deck, README, model card, and a backup demo video.

### Non-goals (do NOT build)
- Google Trends or any extra data source.
- Deep learning, GNNs, LSTMs, Prophet.
- RAG assistant, SMS, WhatsApp, ambulance integration (roadmap slide only).
- User accounts or a database. Files only.
- Real patient data of any kind.

## 3. Hard rules

1. **Never invent results.** Every number in the deck comes from code in the repo that reruns with one command.
2. **Label synthetic data everywhere.** Every synthetic row has `is_synthetic = true`. The dashboard shows a visible "Simulated ward-level signals" badge.
3. **No data leakage.** Scoring and features use streamed signals only. Ground truth is read only by `generator/`, `model/train.py`, `model/backtest.py`. Splits are by season, never shuffled.
4. **Never tune on test or replay data.** Calibration, thresholds and early stopping use validation seasons only. The replay season is generated once and touched only after the model is frozen.
5. **Do not fake dataset loading.** If a real dataset file is missing, fail loudly. Never substitute made-up data.
6. **Keep it simple.** No new libraries without team approval.
7. **Fixed seed (`SEED = 42`)**; season `s` uses `SEED + s`.
8. **Do not edit files outside your own folder** without telling the owner (Section 12).
9. **Honest benchmark labels.** "Local Docker" and "IBM LinuxONE" results are never mixed or mislabeled.
10. **Report the caveat.** Pharmacy and clinic signals are synthetic, so lead time shows pipeline behavior, not real-world validation. This sentence appears in `results.md`, the dashboard footer and the deck.

## 4. Tech stack

| Area | Choice | Notes |
|---|---|---|
| Language | Python (use the VM's version; check at hour 0) | Do not force 3.11 if the VM differs |
| Data | pandas, numpy; geopandas only if it installs | Else precompute adjacency and areas on a laptop and ship JSON |
| ML | scikit-learn, xgboost, onnxmltools or skl2onnx, onnxruntime | shap not needed: use XGBoost `pred_contribs=True` |
| Streaming | Apache Kafka (KRaft, single broker), `kafka-python` | `confluent-kafka` only if it installs on the VM. Docker Compose for laptops only |
| Dashboard | Streamlit + pydeck or plain Plotly choropleth | Gate: if Streamlit/pyarrow fail the smoke test, switch to FastAPI + static Leaflet page |
| Weather | Open-Meteo historical API | Pull once, cache in `data/raw/` |
| Metrics/perf | psutil | For CPU and memory in the benchmark |
| VCS | Git, feature branches, merge to `main` | |

Single `requirements.txt` at the repo root. Pin versions once things work.

### s390x notes
LinuxONE is s390x (big-endian). Not every wheel exists. Start from the event's Jupyter Lab ML image and check what it ships. Fallbacks:

| Component | Fallback |
|---|---|
| xgboost missing | Train on a laptop and export ONNX, or use scikit-learn HistGradientBoosting |
| onnxruntime missing | Build from source if time allows, else serve with native XGBoost and say so |
| Kafka does not run | `--no-kafka` mode (Section 9.3) |
| Streamlit/pyarrow missing | FastAPI + static Leaflet page, CSV/JSON only (no Parquet) |
| Docker unavailable on VM | Plain virtualenv plus tmux |

## 5. Repository structure

```
dengue-sentinel/
├── BUILD.md
├── README.md
├── requirements.txt
├── docker-compose.yml        # laptop dev only
├── Makefile                  # data | generate | train | backtest | run | benchmark | demo | reset
├── scripts/
│   └── smoke_test.sh         # hour-0 environment check (Section 8)
├── config/
│   └── config.yaml           # seeds, season counts, lags, thresholds, team capacities
├── data/
│   ├── raw/                  # downloaded files, not committed if large
│   ├── processed/
│   │   ├── events/           # season_XX.csv (streamed signals)
│   │   ├── ground_truth/     # season_XX.csv (hidden truth)
│   │   ├── official/         # real city curve and simulated ward-level official counts
│   │   └── manifest.csv      # season_id, split, has_outbreaks, seed
│   └── DATA_NOTES.md         # column meanings, URLs, licences, ward file version
├── generator/
│   └── generate_signals.py
├── model/
│   ├── features.py           # one feature function, used by training AND streaming
│   ├── baseline.py           # persistence and rolling z-score (EARS-style)
│   ├── train.py              # train, calibrate, choose thresholds on validation seasons
│   ├── export_onnx.py        # ONNX export plus parity check
│   ├── predict.py            # predict(feature_matrix) -> risk dicts
│   ├── backtest.py
│   └── artifacts/            # model.joblib, model.onnx, calibrator, thresholds.json, metrics.json
├── streaming/
│   ├── replayer.py
│   ├── scoring_service.py
│   ├── benchmark.py
│   └── config.py             # topics, thresholds loaded from model/artifacts, tick settings
├── dashboard/
│   ├── app.py
│   ├── respond.py
│   └── assets/
├── docs/
│   ├── PRD_and_Architecture  # link or copy of the PRD document
│   ├── results.md            # generated by backtest and benchmark
│   ├── model_card.md
│   └── deck/
└── tests/
    ├── test_feature_parity.py   # offline vs streaming features identical
    ├── test_onnx_parity.py      # ONNX vs native probabilities within 1e-4
    └── test_allocator.py        # greedy vs brute force on small cases
```

## 6. Architecture

```
Real data: OpenDengue cases + Open-Meteo weather + ward GeoJSON
        |
        v
generator/  -> data/processed/events/season_XX.csv       (streamed signals, is_synthetic flag)
            -> data/processed/ground_truth/season_XX.csv (hidden truth, NEVER read by scoring)
            -> data/processed/official/                  (official counts with reporting delay)
        |
        v
streaming/replayer.py --publishes at configurable speed--> Kafka topic: signals
        |
        v
streaming/scoring_service.py
   - maintains rolling per-ward windows
   - when a date is complete: builds one feature matrix for all wards
   - runs z-score baseline and calibrated XGBoost (ONNX) in one batch call
   - explains medium/high wards (XGBoost pred_contribs), applies persistence rule
   - publishes --> Kafka topic: risk-scores
        |
        v
dashboard/app.py (heatmap, alert feed, ward detail, Respond, Benchmark, Validation)
```

Topics: `signals` (key = ward_id), `risk-scores` (key = ward_id). Fallback: `--no-kafka` calls the scoring function directly.

**Design principle: replay = training = demo.** `model/features.py` builds the training matrix by replaying history through the same function the live service uses. `tests/test_feature_parity.py` enforces this.

## 7. Data contracts (the most important section)

### 7.1 Signal event (topic `signals`, and `events/season_XX.csv`)
```json
{
  "event_id": "W017|2024-09-14|clinic_fever_visits",
  "ward_id": "W017",
  "date": "2024-09-14",
  "signal_type": "clinic_fever_visits",
  "value": 23.0,
  "is_synthetic": true,
  "published_at_ms": 1760499999800
}
```
- `signal_type` is exactly one of: `rainfall`, `temp`, `humidity`, `pharmacy_fever_meds`, `clinic_fever_visits`.
- `rainfall`, `temp`, `humidity` are **real** (Open-Meteo, city-level, copied to every ward), `is_synthetic = false`.
- `pharmacy_fever_meds` and `clinic_fever_visits` are **synthetic**, `is_synthetic = true`.
- `date` is ISO `YYYY-MM-DD`; events are daily per ward per signal.
- `event_id` is `ward_id|date|signal_type`; the scorer drops duplicates by it.
- `published_at_ms` is added by the replayer at publish time (not stored in CSV). Latency is measured from it.
- `ward_id` is a string from the ward GeoJSON property named in `DATA_NOTES.md`; it must match everywhere.

### 7.2 Risk output (topic `risk-scores`)
```json
{
  "ward_id": "W017",
  "date": "2024-09-14",
  "risk_score": 0.82,
  "confidence": 0.74,
  "alert_level": "high",
  "alert_confirmed": true,
  "baseline_flag": true,
  "top_drivers": [
    {"feature": "clinic_visits_7d_change_pct", "value": 41.0, "direction": "up", "text": "Clinic fever visits up 41% over 7 days"},
    {"feature": "rainfall_lag14", "value": 62.5, "direction": "up", "text": "Heavy rainfall 14 days ago (62 mm)"}
  ],
  "backend": "onnxruntime",
  "model_version": "xgb-2026-10-17-a",
  "scored_at_ms": 1760500000000,
  "published_at_ms": 1760499999800
}
```
- `risk_score`: calibrated probability (0 to 1) that the ward has an outbreak onset within the next 14 days. Isotonic calibration fit on validation seasons.
- `confidence`: 0 to 1. Must combine data completeness (share of the 5 signals fresh for that ward-day) with distance from the nearest alert threshold. Document the exact formula in `model/predict.py`.
- `alert_level`: `low`, `medium`, `high`. Thresholds `T_MEDIUM` and `T_HIGH` come from `model/artifacts/thresholds.json`, chosen on validation seasons to meet the false-alert budget (Section 9.2). Defaults before tuning: 0.40 and 0.70.
- `alert_confirmed`: true only after the persistence rule is met (score at or above the level threshold on 2 consecutive days). Cooldown: no repeat notification for the same ward and level within 72 simulated hours. Downgrade only after 5 days below the threshold minus a margin.
- `top_drivers`: 1 to 3 items with plain-language `text`, computed only for `medium` and `high` wards. Template examples: "Fever visits up {x}% in {n} days", "Heavy rainfall {n} weeks ago ({mm} mm in 14 days)", "Fever-medicine sales {x}% above this ward's normal".
- Latency = `scored_at_ms - published_at_ms`.

## 8. Data sources and hour-0 smoke test

| Dataset | Use | Real/Synthetic | Notes |
|---|---|---|---|
| OpenDengue | Historical case counts (India / Karnataka) | Real | Check coverage (state vs district) and latest year. Record URL and licence in `DATA_NOTES.md` |
| Open-Meteo historical weather | Rainfall, temp, humidity for Bengaluru (lat 12.97, lon 77.59) | Real | Daily, no key. Cache before the event. Attribution required |
| Ward boundaries (GeoJSON) | Map and ward ids | Real | Record version, ward count and licence. Boundaries have changed more than once |
| Ward population/density | Exposure weighting in Respond | Real if found | If not found, Respond ranks by risk only and says so |
| Pharmacy sales per ward | Streamed signal | **Synthetic** | Generated |
| Clinic fever visits per ward | Streamed signal | **Synthetic** | Generated |

If OpenDengue is district-level only, the pitch wording is: "real district-level trends, ward-level simulation."

### Hour-0 smoke test (`scripts/smoke_test.sh`, Sat 10:00 to 11:00)
Record in the README: `nproc`, `free -h`, `df -h`, Python and Java versions. Install and import numpy, pandas, scikit-learn, xgboost, onnxruntime, kafka-python, streamlit. Start a single-node Kafka with a small heap (about 512 MB). Train a tiny model off the VM, load it on the VM and compare predictions (tolerance 1e-6 native, 1e-4 ONNX). Decide the dashboard path (Streamlit or FastAPI + Leaflet) and the fallbacks in Section 4.

## 9. Module specs

### 9.1 generator/ (owner: Person 1)
Script: `generator/generate_signals.py`

Inputs: weekly real case curve (backbone season), daily Bengaluru weather, ward list (plus population and adjacency if available), `config/config.yaml`.

Behavior, per season `s` (default 40 seasons, seed `SEED + s`):
1. Backbone: the real weekly curve for the chosen past season, interpolated to a daily city index. Document which season in `DATA_NOTES.md`.
2. Per-ward hidden `case_index` = city index x ward weight x noise. Weight = population share if available, else seeded lognormal (mark as synthetic weights in `DATA_NOTES.md`).
3. Inject 3 to 8 outbreaks (none in control seasons): random seed wards and start dates, ramp 10 to 28 days to 3x to 8x baseline. Spread to adjacent wards with decay if adjacency exists. Dates are random, **not** aligned to precede the official curve.
4. Confounders in every season: a non-dengue viral-fever wave across all wards, and 0 to 3 random fever spikes in wards with no dengue behind them.
5. Signals from `case_index` (care-pathway lags, all configurable):
   - `pharmacy_fever_meds` ~ NegBin(base + k2 x case_index lagged 0 to 2 days)
   - `clinic_fever_visits` ~ NegBin(base + k1 x case_index lagged 2 to 4 days)
   - day-of-week effects and per-ward baselines
6. Official counts: ward-level official series with a reporting delay of 7 to 14 days, batched weekly, with under-reporting (secondary lead-time measure). Headline official detection uses the **real** city-level weekly curve (Section 9.2).
7. Attach real weather to every ward.

**Outbreak label (ground truth).** A ward-day is `outbreak_flag = 1` when its 7-day `case_index` is at least 2x its trailing 8-week baseline and above an absolute floor, for at least 3 consecutive days. Onset = first day of such a run. `outbreak_in_next_14d` = 1 if an onset falls in the next 14 days, and wards already in outbreak at that date are excluded from training rows.

Seasons in `manifest.csv`: split by season 70% train, 15% validation, 15% test; at least 5 control seasons spread across splits; one extra `replay` season (the real backbone curve, fixed seed) never used for training, calibration or thresholds.

Outputs: `events/season_XX.csv` (Section 7.1 schema, without `published_at_ms`), `ground_truth/season_XX.csv` (`ward_id, date, case_index, outbreak_flag, onset_flag`), `manifest.csv`, printed summary (wards, date range, outbreaks per season).

Acceptance: same seed gives identical output; all synthetic rows flagged; a plot of one injected ward looks like a believable outbreak; the pharmacy series visibly leads the clinic series.

### 9.2 model/ (owner: Person 2)

**Features (`model/features.py`)** (one function for training and streaming), all computed from data up to the scoring date only:
- `clinic_visits_7d_mean`, `clinic_visits_7d_change_pct`, `clinic_visits_zscore_28d`
- `pharmacy_7d_mean`, `pharmacy_7d_change_pct`, `pharmacy_zscore_28d`
- `rainfall_lag7`, `rainfall_lag14`, `rainfall_lag21`, `rainfall_14d_sum`, `rainfall_28d_sum`
- `temp_7d_mean`, `humidity_7d_mean`
- calendar: sine and cosine of day of year
- completeness: fraction of the 5 signals fresh
- **No ward ID or zone ID as a feature.**
- Upgrade if time allows (needs adjacency and a generator that spreads outbreaks): mean and max clinic z-score of adjacent wards.

**Models:**
1. **Baseline A:** persistence ("same as last period").
2. **Baseline B:** rolling z-score on clinic visits (EARS-style: z > 2 against the trailing 28 days, with a 2-day guard band).
3. **Main:** XGBoost binary classifier, class weights for imbalance, monotone constraints on growth features, early stopping on validation seasons. Calibrate with isotonic regression on validation seasons. Choose `T_MEDIUM` and `T_HIGH` on validation seasons to meet a false-alert budget (target: at most 5 false alerts per 100 ward-weeks), save to `thresholds.json`.

**Export:** `export_onnx.py` writes `model.onnx` and runs the parity check (max probability difference under 1e-4 on a held-out matrix). The calibrator is applied after the ONNX call.

**`predict.py`:** `predict(feature_matrix) -> list of risk dicts` for all wards in one call. Uses ONNX Runtime if `model.onnx` exists, else joblib. Drivers come from the native XGBoost model (`pred_contribs=True`) for `medium` and `high` wards only, mapped to plain-language templates. Both artifacts must come from the same training run.

**Backtest (`model/backtest.py`)**, definitions (do not change without team agreement):
- **Official detection date (headline):** first date the *real* weekly city-level count exceeds mean + 2 SD of the preceding 8 weeks, plus a reporting delay of 7 days. Also report sensitivity at 10 and 14 days.
- **Our detection date:** first date a ward has a *confirmed* `high` alert that is a **true alert** (a real onset follows within 14 days in that ward or an adjacent ward). False alerts are counted separately and can never produce an early detection.
- **Days earlier** = official date minus our date. Report for the replay season (city level) and for each injected outbreak in the test seasons (ward level, against the simulated ward-level official series).
- **False alarms:** report both (a) share of confirmed `high` ward-days that are false (equals 1 minus precision) and (b) false alert episodes per 100 ward-weeks, including control seasons.
- Also report precision, recall (outbreak-level), PR-AUC (ward-day), Brier score and a reliability curve.
- **Experiments:** E1 held-out test seasons; E2 replay season through the full pipeline; E3 ablation (no pharmacy, no clinic, no weather); E4 robustness sweep (signal lag 1 to 5 days, noise x0.5 to x2, signal strength); E5 control seasons only (no outbreaks); E6 calibration.
- Compare detectors in one table: official-delayed (lead 0 by definition), persistence, z-score, XGBoost full, XGBoost clinic+pharmacy only. Use bootstrap confidence intervals over seasons.
- Output: `docs/results.md` (auto-generated), `docs/backtest.png`, `model/artifacts/metrics.json`.

`results.md` must state the reporting delay assumption and the Hard Rule 10 caveat.

### 9.3 streaming/ (owner: Person 3)
- **Kafka on the LinuxONE VM:** single-node KRaft from the Apache tarball with a small heap. `docker-compose.yml` is for laptops only.
- **`replayer.py`:** reads one season's events, groups by date, publishes to `signals` with key = ward_id and sets `published_at_ms`. Args: `--season`, `--speed` (simulated days per real second, default 5), `--start`, `--end`. Restartable.
- **`scoring_service.py`:** consumes `signals`, drops duplicate `event_id`s, keeps per-ward rolling windows in memory, and when all signals for a date have arrived builds one feature matrix for all wards, calls `predict`, applies the persistence and cooldown rules, and publishes Section 7.2 messages to `risk-scores`. Writes the same messages to `data/processed/risk_scores_fallback.csv` for offline mode.
- **`benchmark.py`:**
  - Throughput: push N events (default 100,000) at maximum speed; report events per second.
  - Latency: separate runs at a controlled `--rate` below saturation; report p50, p95, p99 (ms). Never quote latency from the saturated run.
  - Backend comparison: native XGBoost vs ONNX Runtime (vs Snap ML if available) on the same feature matrices at 1, 100, 1,000, 10,000 and 100,000 rows.
  - Scale test: replicate Bengaluru's wards into N virtual cities (1, 10, 50, 100); find the largest N where p99 tick latency stays under 1 second. Cities-per-node arithmetic: events/s per city = wards x signals per ward x events per signal per minute / 60.
  - Method: discard one warm-up run, report the median of three, run in a quiet window with nothing else on the shared VM.
  - Report: `--platform "ibm-linuxone"` or `"local-docker"`, CPU model, core count, RAM, Python and library versions, git commit. Write to `docs/results.md` under "Benchmark".
- **Fallback:** `--no-kafka` mode: the replayer calls the scoring function directly so the demo never dies.

### 9.4 dashboard/ (owner: Person 4)
Streamlit app (`dashboard/app.py`) unless the smoke test forces the FastAPI + Leaflet fallback. A background consumer reads `risk-scores` from Kafka (or `risk_scores_fallback.csv` with `--offline`) into an in-memory store the UI reads.

Layout:
1. **Header:** project name, simulated date, replay speed control, "Simulated ward-level signals" badge.
2. **Heatmap:** ward polygons colored by `risk_score` (green / amber / red per `alert_level`), drawn on a plain background so it works without internet. Tooltip: ward, score, confidence.
3. **Alert feed:** newest confirmed alerts first: ward, level, top driver text.
4. **Ward detail:** clinic visits, pharmacy sales, rainfall; risk score over time; top drivers; alert history.
5. **Respond panel (`respond.py`):**
   - Control: number of available teams (default 5, slider 1 to 15).
   - If ward population exists: exposure = risk_score x population x (1 + vulnerability). Benefit of n visits = exposure x (1 - (1 - r)^n), with r = 0.30 for inspection and larval source removal (illustrative, to confirm with health officers). Assign each team to the ward with the highest marginal benefit (greedy). Otherwise rank by `risk_score`.
   - Only wards with a confirmed `medium` or `high` alert are eligible; wards above the team cap appear as "Watchlist".
   - Action text by level: `medium`: "Schedule inspection and larval source removal in {ward} within 72 hours." `high`: "Prioritise inspection and larval source removal in {ward} within 48 hours; consider targeted fogging at the cluster; notify nearby hospitals to prepare for more fever cases."
   - State in the UI: priority scores rank wards; they do not predict cases averted.
6. **Benchmark tab:** reads the latest benchmark JSON: events/s, latency percentiles, backend comparison, platform label.
7. **Validation tab:** lead-time chart with the official detection date marked, comparison table, false-alert counts, the synthetic-signal caveat.
8. **Footer:** "Real: weather, historical case counts, ward boundaries. Simulated: ward-level pharmacy and clinic signals."

Acceptance: runs fully on mock data first (matching Section 7.2), then on live Kafka data with only the data source changed.

## 10. Results reporting
- All deck numbers come from `docs/results.md` and `model/artifacts/metrics.json`, generated by `make backtest` and `make benchmark`. Never hand-edit.
- Deck: one backtest chart, one comparison table, one benchmark table (with platform label), one limitations slide.

## 11. Security, privacy, ethics (only claim what was enabled and tested)

| Control | Hackathon build | Production target on LinuxONE |
|---|---|---|
| Data minimisation | Aggregate counts only, pseudonymous source IDs | Same, plus legal review (DPDP Act) |
| Encryption in transit | TLS on any exposed endpoint | Mutual TLS, hardware-assisted crypto |
| Encryption at rest | Encrypted volume if the VM allows, else file-level encryption | Pervasive encryption, protected keys |
| Isolation | Dedicated VM, non-root service user | Secure Execution for Linux |
| Secrets | Environment variables, never committed | HSM-backed key management |

Ethics: decision support, not diagnosis; human approval for any outward action; model card in `docs/model_card.md`; equity check across ward-density quintiles (dense low-income wards may have fewer reporting clinics and pharmacies).

## 12. Ownership and schedule (IST)

| Person | Role | Owns | Also responsible for |
|---|---|---|---|
| 1 | Data and generator | `data/`, `generator/`, `config/` | Deck and README, dataset slide |
| 2 | ML and backtest | `model/`, `tests/test_*parity.py` | `results.md` backtest section, ML slide, model card |
| 3 | Streaming and Z | `streaming/`, `scripts/`, `docker-compose.yml` | Benchmark, "Why IBM Z" slide, LinuxONE VM setup |
| 4 | Dashboard and Respond | `dashboard/`, `tests/test_allocator.py` | Integration, demo script, backup video, submission upload |

| Window | Goal | Exit check |
|---|---|---|
| Fri 16 Oct, 19:30 | Mandatory LinuxONE session; confirm VM access and rules | VM login works for all 4 |
| Sat 10:00 to 11:00 | Register with `SSF2026`, smoke test, repo, agree schemas | Fallbacks chosen, dashboard path chosen |
| 11:00 to 14:00 (lunch 13:00 to 14:00) | Generator v1 (multi-season), Kafka up, replayer, baselines, dashboard on mock data | Events flow replayer to Kafka to console |
| 14:00 to 18:00 | Features, labels, first XGBoost on season split, scorer, heatmap v1 | Replay to score to map works |
| 18:00 to 22:00 | Calibration, thresholds, ONNX export and parity, persistence rule, drivers, Respond | Backends verified equal, first benchmark |
| 22:00 to Sun 02:00 | E1 to E6 experiments, benchmark in a quiet window | Comparison table filled from real runs |
| Sun 02:00 | **Feature freeze** | Fixes only |
| 02:00 to 06:00 | Shifts (half rest), README, model card, deck, backup video | 3 clean demo runs, numbers final |
| 06:00 to 08:30 | Final run, upload project to the LinuxONE VM, submit on Hack2Skill | Submitted by 08:30 |
| 08:30 to 10:30 | Buffer for upload problems | Submission confirmed |

Benchmark runs need a quiet VM: train off the VM where possible and agree one benchmark window.

**Fallbacks (use immediately if a task stalls for more than 2 hours):**
- XGBoost fails: submit z-score baseline and say so honestly.
- Kafka fails: `--no-kafka` mode.
- LinuxONE problem on the VM: run the local Docker benchmark and label it "local baseline", and tell the organizers via the helpdesk channels.
- ONNX export or runtime fails: joblib model, state it.
- Streamlit fails: FastAPI + static Leaflet page.

## 13. Git workflow
- `main` is always runnable. Branches: `data/*`, `model/*`, `streaming/*`, `dashboard/*`.
- Commit after every working step. Merge to `main` at Sat 14:00, Sat 22:00, Sun 02:00 (freeze) and Sun 06:00. Person 4 resolves conflicts.
- Large raw data is git-ignored; keep download steps in `data/DATA_NOTES.md`.
- The team shares one LinuxONE VM: use separate directories and virtualenvs, and do not run heavy jobs during the benchmark window.

## 14. Definition of done
- [ ] `make data generate train backtest run` works from a fresh clone; `make demo` runs replay, Kafka, scoring and dashboard unattended 3 times in a row; `make reset` clears state.
- [ ] Dashboard shows heatmap, alert feed, ward detail, Respond, Benchmark and Validation tabs.
- [ ] Synthetic data labeled in code, dashboard and deck; caveat (Hard Rule 10) present.
- [ ] `docs/results.md` has the comparison table, ablations, delay sensitivity, false alerts per 100 ward-weeks, calibration, and the benchmark with platform label, backend comparison and scale test.
- [ ] Feature parity and ONNX parity tests pass.
- [ ] README (setup, assumptions, licences, synthetic data statement) and model card written; project description contains `SSF2026`.
- [ ] Deck and backup demo video done; project uploaded to the LinuxONE VM and submitted on Hack2Skill.
- [ ] Each team member can explain their module without an AI agent.

## 15. Decisions to confirm as a team (defaults in brackets)
1. Model library: XGBoost [yes, settled].
2. Dashboard: Streamlit unless the smoke test fails [Streamlit].
3. Outbreak label: rule-based onset (Section 9.1) [rule-based].
4. Ward file version and population source [decide Sat morning; risk-only ranking if no population].
5. Which past season is the backbone and whether OpenDengue covers it [decide when data is downloaded].
6. Team capacity per zone and effectiveness values `r` [illustrative; one informal check with a health officer if possible].

## 16. Instructions for AI coding agents
1. Read this entire file before starting. State a short plan and wait for approval before writing code.
2. Work only in the folder your task belongs to. Do not change schemas in Section 7.
3. Write small, runnable pieces. After each, show how to run it and what output to expect.
4. Add a short docstring to every function explaining inputs and outputs in plain language.
5. Never hardcode results, sample numbers, or fake "successful" output. If something cannot be computed, say so.
6. Never read ground truth outside `generator/`, `model/train.py` and `model/backtest.py`.
7. Never tune thresholds, calibration or hyperparameters on test or replay seasons.
8. When you hit an error, show the full error and propose the smallest fix.
9. Do not add dependencies, features, or files that are not in this file without asking.
