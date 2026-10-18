# Dengue Sentinel: Pitch Deck Outline (IBM Z Datathon 2026)

**Event Code**: `SSF2026`  
**Theme**: Real-Time AI for Critical Decisions (Detect. Decide. Respond.)  
**Mission**: AI for Good  

---

### Slide 1: The Problem & The Vision (Detect. Decide. Respond.)
- **The Challenge**: Dengue outbreaks in urban centers like Bengaluru currently rely on official laboratory-confirmed hospital case reporting, which suffers from an inherent **7 to 14 day administrative reporting delay**. By the time hospital surges are recorded, transmission is widespread.
- **The Solution**: **Dengue Sentinel** — an AI-powered municipal early-warning system that detects anomalous community syndromic signals (over-the-counter fever medicine sales at pharmacies and fever visits at primary health clinics) weeks before hospital admissions peak.
- **The Pitch in 3 Lines**:
  1. **Detect**: Streaming pharmacy sales, clinic visits, and lagged weather are ingested and scored in real-time.
  2. **Decide**: Wards are prioritized by calibrated outbreak probability, with plain-language Tree SHAP explanations and a 2-day persistence confirmation rule to prevent alert fatigue.
  3. **Respond**: An exposure-weighted greedy algorithm allocates inspection and targeted larval source removal teams to the highest-risk clusters.

---

### Slide 2: Comparative Epidemiological Validation (The Comparison Table)
*All numbers generated deterministically by `make backtest` from held-out test seasons (`docs/results.md`):*

| Detector | Lead Time (Days Earlier) | Outbreak Recall | Ward-Day Precision | False Alerts / 100 Ward-Weeks | Brier Score |
|---|---|---|---|---|---|
| Official Lab Surveillance | 0.0 days (baseline) | N/A | N/A | N/A | N/A |
| Baseline A (7d Persistence) | +19.4d [18.9, 20.1] | 98.3% | 20.7% | 29.36 | 0.2903 |
| Baseline B (Rolling Z-Score $z>2$) | +14.0d [7.0, 21.0] | 29.5% | 9.7% | 41.96 | 0.0691 |
| **Dengue Sentinel (Calibrated XGBoost)** | **+13.7d [12.0, 15.3]** | **60.8%** | **45.6%** | **0.91** (Budget $\le 5.0$) | **0.0279** (~83.7% error drop) |
| Ablation: No Weather | +9.1d [3.8, 14.3] | 37.9% | 33.0% | 1.04 | 0.0281 |

- **Key Takeaway (E1 Held-Out Tests)**: Sentinel delivers **13.7 days of proactive warning window**, detecting **60.8% of outbreak waves** while cutting false alarms to **0.91 per 100 ward-weeks** (vs 42.0 for Z-score).
- **Replay Season (E2, Primary)**: Sentinel alerts fired **+9.8 days before outbreak onset** (+16.8 days vs per-outbreak official notice). 13/16 outbreaks detected; a separate city-level aggregate trigger fired earlier, but Sentinel's contribution is **ward-level localization**.

---

### Slide 3: Detection Timeline & Operational Lead Time (The Backtest Chart)
- **Visual Asset**: `docs/backtest.png` (Top Panel: Replay Season Transmission Wave; Bottom Panels: PR-AUC and Calibration Diagram).
- **Key Milestones in an Outbreak Wave**:
  - **Day 0**: Syndromic rise in OTC antipyretic sales at neighborhood pharmacies.
  - **Day +3 to +14**: **Dengue Sentinel Confirmed Alert** fires (2-day persistence validated; top driver: "Pharmacy fever sales up +115%").
  - **Day +7 to +21 vs official notice**: Public health officials gain advance warning before per-outbreak official laboratory notice (onset + assumed 7-day reporting delay).
- **Municipal Benefit**: Vector control officers gain **~10 days of advance warning before clinical onset** to inspect storm drains, deploy larvicides, and alert hospitals before emergency departments are overwhelmed.

---

### Slide 4: Why IBM Z / LinuxONE (The Benchmark Table)
*Platform: `local` (Hard Rule 9 honest reporting; benchmarks run on local x86-64; not measured on s390x)*

Kafka streaming mode is **optional, not used in the demo, and untested on s390x**.

| Metric | Measured Value | Note |
|---|---|---|
| **Rate-Controlled Latency (p50)** | **0.0019 ms** | At 5,000 events/sec sub-saturation load |
| **Rate-Controlled Latency (p99)** | **0.0315 ms** | Deterministic low-jitter execution under real-time load |
| **ONNX Runtime vs XGBoost (batch=1)** | **~22x faster** (~1.09 ms vs ~0.05 ms) | Approximate; machine-dependent; on this local x86-64 machine |
| **ONNX Runtime vs XGBoost (batch=100)** | **~6x faster** (~1.30 ms vs ~0.21 ms) | At batch≥1,000 native XGBoost is faster |
| **Multi-City Scale Test** | **100 virtual cities (2,000 wards)** | p99 tick latency ~6.7 ms, well under 1,000 ms SLA locally |

- **Snap ML Status**: Honestly noted as optional/not installed on local test environment.
- **Demo Architecture**: Direct in-process pipeline (`--no-kafka`); dashboard reads `data/processed/risk_scores_fallback.csv`.

---

### Slide 5: Ethics, Limitations & Governance (Hard Rule 10 Caveat)
- **Mandatory Caveat (Hard Rule 10)**:
  > *"Pharmacy and clinic signals are synthetic, so lead time shows pipeline behavior, not real-world validation."*
- **Decision Support, Not Diagnosis**: Sentinel ranks municipal geographic risks to prioritize municipal vector control teams; it does not diagnose patients or automate pesticide spraying without human health officer approval.
- **Equity & Data Minimization**:
  - No personal identifiable information (PII) is collected; only aggregated daily counts.
  - No geographic identifiers (Ward ID or Zone ID) used in the ML model, ensuring zero spatial memorization bias.
  - Audit recommendations for lower-density wards to prevent under-reporting bias in areas with fewer commercial retail pharmacies.
