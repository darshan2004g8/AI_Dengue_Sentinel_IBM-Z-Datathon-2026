# Model Card: Dengue Sentinel Outbreak Classifier

## Model Details
- **Developer**: Dengue Sentinel Team (IBM Z Datathon 2026, Event Code `SSF2026`)
- **Model Type**: Extreme Gradient Boosted Trees (`XGBClassifier`) with Isotonic Probability Calibration (`IsotonicRegression`).
- **Target Platform**: Designed for IBM LinuxONE (s390x) via ONNX Runtime (`onnxruntime`) and native Python execution. Benchmarks in this card were run on a local x86-64 machine; s390x not tested.
- **Model Version**: `xgb-2026-10-17-a`

## Intended Use
- **Primary Use**: Municipal decision-support tool to alert public health authorities to anomalous dengue transmission risks at the municipal ward level 14 days in advance.
- **Out of Scope**: Clinical diagnosis of individual patients, automated pharmaceutical dispensing, or unsupervised intervention without human public health review.

## Factors & Feature Representation
All features are computed strictly from historical and streaming data up to the scoring date without look-ahead bias:
- Clinic fever visits rolling metrics: 7-day mean, 7-day % change, 28-day rolling z-score.
- Pharmacy fever medication sales rolling metrics: 7-day mean, 7-day % change, 28-day rolling z-score.
- Weather signals: 7-day, 14-day, 21-day rainfall lags, 14-day and 28-day cumulative precipitation sums, 7-day mean temperature, 7-day mean relative humidity.
- Temporal seasonality: Sine and Cosine encoding of day of year.
- Signal integrity: Completeness fraction of expected data streams.
- **Explicit Exclusion**: Ward IDs and zone IDs are strictly excluded to ensure geographic generalization.

## Training & Validation Split
- Multi-season split: 70% Training, 15% Validation, 15% Held-Out Testing, plus 5 Control Seasons and 1 dedicated Replay Season.
- Fixed seed `SEED = 42`, with season `s` seeded with `SEED + s`.
- Calibration and alert thresholds (`T_MEDIUM=0.08`, `T_HIGH=0.12`) tuned strictly on validation seasons to respect the 5 false alerts per 100 ward-weeks budget.

## Quantitative Performance Metrics (Evaluated on Held-Out Test Seasons + Replay Season)
- **Early Warning (E1 Held-Out Test Seasons)**: Mean lead time **+13.7 days earlier** than delayed lab confirmations (95% CI: [12.0d, 15.3d]). Outbreak recall 60.8%, ward-day precision 45.6%.
- **Replay Season (E2, Primary)**: Sentinel alerts fired a mean of **+9.8 days before outbreak onset** (+16.8 days vs per-outbreak official notice assuming 7-day reporting delay). 13/16 outbreaks detected (3 same-ward, 10 via adjacent ward, 3 missed). Recall 81.2%, precision 80.0%.
- **False Alert Control (E5)**: **0.36 false alerts per 100 ward-weeks** on pure control seasons (zero true outbreaks), safely within the target municipal budget of $\le 5.0$.
- **Probability Calibration (Brier Score)**: Isotonic regression reduces Brier score error from **0.171 to 0.0279** (~83.7% error reduction). Calibrated on synthetic data.
- **Inference Runtime (Benchmark, local x86-64)**: ONNX Runtime batch=1: ~0.05 ms (~22x faster than native XGBoost ~1.09 ms); batch=100: ~0.21 ms (~6x faster).

## Caveat & Ethical Considerations
- **Hard Rule 10**: Pharmacy and clinic signals are synthetic, so lead time demonstrates architectural and pipeline behavior, not real-world clinical validation.
- **Decision Support Only**: Sentinel provides prioritized risk rankings to guide field vector control teams; it does not diagnose patients or replace laboratory testing.
- **Equity Check**: Ward density and socioeconomic distribution should be audited regularly to prevent under-reporting bias in lower-income or underserved wards with fewer private commercial pharmacies.

