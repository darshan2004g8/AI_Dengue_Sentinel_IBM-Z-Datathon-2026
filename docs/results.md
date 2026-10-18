# Dengue Sentinel: Experimental & Benchmark Results

> **Caveat (Hard Rule 10):**
> *Pharmacy and clinic signals are synthetic, so lead time shows pipeline behavior, not real-world validation.*

## 1. Backtest Results (Model Evaluation)

*Generated deterministically by `make backtest` (`model/backtest.py`) at 2026-10-17 17:40:26 UTC.*

### Experimental Design & Definitions
- **Per-Outbreak Official Notice Date** (E2 headline definition): For each localized outbreak onset $O_i$, official clinical notice is defined as $O_i + 7\text{d}$ (assumed 7-day symptom-to-laboratory-confirmation delay). **This is not the city-level surge trigger; it is per-outbreak arithmetic.**
- **Per-Outbreak "Days Before Onset"** (primary metric): $O_i - A_i$, where $A_i$ is the earliest confirmed Sentinel `high` alert (2-day persistence, 72h cooldown) within 14 days prior to $O_i$ in the outbreak ward or a directly adjacent ward ($0 \le O_i - A_i \le 14$).
- **Per-Outbreak "Lead vs Official"** (secondary metric): $(O_i + 7\text{d}) - A_i = (O_i - A_i) + 7\text{d}$. This is purely arithmetic: the E4 delay-sensitivity table varies only the assumed reporting delay, not the Sentinel alert dates.
- **E2 Headline**: Arithmetic mean of per-outbreak lead-vs-official across all detected outbreaks. True alert confirmed if a real onset follows within 14 days in the alerted ward or an adjacent ward.
- **Separate City-Level Surge Trigger**: A city-wide aggregate detector (trailing 8-week mean + 2 SD + 7-day lab delay) fires on its own computed date. In the replay season this trigger fired earlier than Sentinel's first ward-level alert; Sentinel's contribution is **ward-level localization and prioritization**, not beating a city-wide count.
- **No-Skill Baseline**: Theoretical baseline for PR-AUC equal to the empirical background prevalence of positive outbreak ward-days across the test seasons.
- **Evaluation Split**: Multi-season split (70% train, 15% validation, 15% held-out test; 5 control seasons; 1 held-out replay season). No dates shuffled across seasons.

### Summary Comparison Table (E1: Held-out Test Seasons)
*Bootstrap 95% confidence intervals calculated over 1,000 iterations:*

| Detector | Lead Time (Days Earlier) | Outbreak Recall | Ward-Day Precision | False Alerts / 100 Ward-Weeks | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|
| Official (delayed 7d) | 0.0 days (ref) | N/A | N/A | N/A | N/A | N/A |
| No-Skill Baseline (Prevalence) | 0.0 days (ref) | N/A | N/A | N/A | 0.030 | N/A |
| Baseline A (Persistence) | +19.4d [18.9, 20.1] | 98.3% | 20.7% | 29.36 | 0.045 | 0.2903 |
| Baseline B (Rolling Z-Score) | +14.0d [7.0, 21.0] | 29.5% | 9.7% | 41.96 | 0.030 | 0.0691 |
| **Calibrated XGBoost (Full)** | **+13.7d [12.0, 15.3]** | **60.8%** | **45.6%** | **0.91** | **0.087** | **0.0279** |
| Calibrated XGBoost (No Weather)| +9.1d [3.8, 14.3] | 37.9% | 33.0% | 1.04 | 0.077 | 0.0281 |

### Held-out Replay Season (E2: Full Streaming Pipeline)

**Note:** The separate city-level surge trigger fired on `2024-06-01` (aggregate weekly cases crossing 8-week baseline + 2 SD + 7d lab delay)—earlier than Sentinel's first ward-level alert on `2024-06-24`. Sentinel's contribution is **ward-level localization**: identifying which specific wards are at risk rather than issuing a city-wide aggregate alarm.

- **Sentinel Earliest Confirmed Ward Alert**: `2024-06-24` — earliest confirmed high alert that is a true alert (onset within 14 days in that ward or adjacent ward)
- **Primary: Mean Days Before Onset**: **+9.8 days** (mean of $O_i - A_i$ across 13 detected outbreaks)
- **Secondary: Mean Lead vs Official Notice**: **+16.8 days** (arithmetic: days-before-onset + assumed 7-day reporting delay)
- **Detection breakdown**: 13 of 16 outbreaks detected; 3 same-ward, 10 via adjacent ward, 3 missed
- **Outbreak Recall**: `81.2%` (13/16 outbreak clusters detected)
- **Ward-Day Precision**: `80.0%` (12/15 confirmed alerts validated)
- **Visual Evidence**: Saved to `docs/backtest.png`.

#### Replay Season Injected Outbreaks & Detection Performance
| Ward | True Onset Date | First Confirmed Sentinel Alert | Alert Origin Ward | Days Before Onset | Lead vs Official (+7d) | Counts as True Alert? |
|---|---|---|---|---|---|---|
| `W022` | `2024-07-06` | `2024-07-02` | `W132 (adjacent)` | +4d | +11d | Yes (Detected) |
| `W132` | `2024-07-06` | `2024-06-24` | `W123 (adjacent)` | +12d | +19d | Yes (Detected) |
| `W063` | `2024-07-08` | `2024-06-24` | `W123 (adjacent)` | +14d | +21d | Yes (Detected) |
| `W191` | `2024-07-08` | `2024-06-28` | `W191` | +10d | +17d | Yes (Detected) |
| `W055` | `2024-07-09` | `2024-06-27` | `W123 (adjacent)` | +12d | +19d | Yes (Detected) |
| `W145` | `2024-07-09` | `2024-06-27` | `W123 (adjacent)` | +12d | +19d | Yes (Detected) |
| `W009` | `2024-07-14` | `2024-07-10` | `W063 (adjacent)` | +4d | +11d | Yes (Detected) |
| `W184` | `2024-07-16` | `2024-07-09` | `W184` | +7d | +14d | Yes (Detected) |
| `W017` | `2024-07-18` | `2024-07-09` | `W022 (adjacent)` | +9d | +16d | Yes (Detected) |
| `W114` | `2024-07-28` | `2024-07-20` | `W171 (adjacent)` | +8d | +15d | Yes (Detected) |
| `W171` | `2024-07-31` | `2024-07-20` | `W171` | +11d | +18d | Yes (Detected) |
| `W174` | `2024-07-31` | `2024-07-20` | `W171 (adjacent)` | +11d | +18d | Yes (Detected) |
| `W119` | `2024-08-03` | `2024-07-20` | `W171 (adjacent)` | +14d | +21d | Yes (Detected) |
| `W221` | `2024-08-04` | `None` | `None` | — | — | No (Missed) |
| `W123` | `2024-08-15` | `None` | `None` | — | — | No (Missed) |
| `W132` | `2024-08-20` | `None` | `None` | — | — | No (Missed) |

> **Metric Note:** "Lead vs Official" = Days Before Onset + 7-day assumed reporting delay (arithmetic). The E4 sensitivity table varies the delay assumption, not the Sentinel alert dates. The city-level surge trigger (`2024-06-01`) is a separate macroscopic alarm; Sentinel provides ward-level prioritization.

### Feature Ablations (E3)
| Ablation Variant | Lead Time (Days) | Outbreak Recall | Ward-Day Precision | False Alerts / 100 Ward-Weeks | PR-AUC |
|---|---|---|---|---|---|
| Full Model (All 16 Features) | +13.7d | 60.8% | 45.6% | 0.91 | 0.087 |
| **No Calendar Features** | **+9.7d** | **25.9%** | **46.9%** | **0.35** | **0.087** |
| No Weather Features | +9.1d | 37.9% | 33.0% | 1.04 | 0.077 |
| No Clinic Features | +0.0d | 0.0% | N/A | 0.0 | 0.077 |
| No Pharmacy Features | +0.0d | 0.0% | N/A | 0.0 | 0.074 |

#### No Calendar Features Evaluation: Held-out Test vs Replay Season
| Evaluation Split | Variant | Lead Time vs Official | Days Before Onset | Outbreak Recall | Ward-Day Precision | False Alerts / 100 Ward-Weeks | PR-AUC |
|---|---|---|---|---|---|---|---|
| Held-out Test Seasons | Full Model (All 16 Features) | +13.7d | +6.7d | 60.8% | 45.6% | 0.91 | 0.087 |
| Held-out Test Seasons | **No Calendar Features** | **+9.7d** | **+2.7d** | **25.9%** | **46.9%** | **0.35** | **0.087** |
| Replay Season (`season_replay`) | Full Model (All 16 Features) | +16.8d | +9.8d | 81.2% | 80.0% | 0.31 | 0.199 |
| Replay Season (`season_replay`) | **No Calendar Features** | **+14.0d** | **+7.0d** | **6.2%** | **50.0%** | **0.1** | **0.151** |

> **Ablation Methodology Note:**
> The No-Calendar model was retrained on training seasons without `sin_day_of_year` and `cos_day_of_year`, and thresholds were tuned on validation seasons adhering to the budget of $\le 5.0$ false alerts / 100 ward-weeks ($T_{\text{medium}}=0.10, T_{\text{high}}=0.15$). The model was evaluated on held-out test seasons and the 2024 replay season without modifying `model/artifacts/`.

### Official Reporting Delay Sensitivity (E4)
| Reporting Delay Assumption | Lead Time Advantage (Days Earlier) |
|---|---|
| 7 Days (Baseline Public Health Reporting Latency) | **+13.7 days** |
| 10 Days (Intermediate Laboratory Latency) | **+16.7 days** |
| 14 Days (Delayed Confirmation Latency) | **+20.7 days** |

### Control Seasons False Alert Verification (E5)
- **Evaluated Control Seasons**: 5 pure control seasons (zero true outbreaks).
- **Observed False Alert Rate**: **0.36 alerts per 100 ward-weeks**.
- **Budget Compliance**: Met target budget of <= 5.0 false alerts per 100 ward-weeks (PASSED).

### Probability Calibration & Reliability (E6)
- **Raw XGBoost Brier Score**: `0.171`
- **Isotonic Calibrated Brier Score**: `0.0279` (**83.7% error reduction**)
- **Reliability Curve**: Calibrated curve tracks the identity diagonal closely, validating that predicted probabilities correspond directly to empirical outbreak likelihoods.


## 2. Benchmark Results (IBM Z / LinuxONE vs Local)

*Generated deterministically by `make benchmark` (`streaming/benchmark.py`) at 2026-10-17 15:57:22 UTC.*

### Platform & Environment Metadata
- **Platform Label**: `local` (Hard Rule 9: honest reporting)
- **OS**: `Windows-11-10.0.26200-SP0`
- **Architecture**: `AMD64`
- **CPU Model**: `Intel64 Family 6 Model 126 Stepping 5, GenuineIntel` (8 logical cores, 4 physical cores)
- **RAM**: `7.75 GB`
- **Python Version**: `3.13.7` | **Git Commit**: `8dc2db5`
- **Key Libraries**: XGBoost `3.4.1`, ONNX Runtime `1.31.0`, Scikit-Learn `1.7.2`

### Streaming Ingestion & Latency
- **Ingestion Throughput**: **1,306,098.6 events/sec** (evaluated across 100,000 Section 7.1 events; median of 3 trials)
- **Rate-Controlled Latency** (at controlled sub-saturation load of 5,000 events/sec; *never quoted from saturated run*):
  - **p50 Latency**: `0.0019 ms`
  - **p95 Latency**: `0.0057 ms`
  - **p99 Latency**: `0.0315 ms`

### Backend Comparison (Native XGBoost vs ONNX Runtime)
*Median of 3 runs after warm-up on identical 16-feature matrices:*

| Rows (Batch Size) | Native XGBoost | ONNX Runtime | ONNX Speedup |
|---|---|---|---|
| 1 | 1.09 ms | 0.05 ms | 21.8x |
| 100 | 1.302 ms | 0.208 ms | 6.3x |
| 1000 | 1.891 ms | 1.844 ms | 1.0x |
| 10000 | 8.501 ms | 21.625 ms | 0.4x |
| 100000 | 53.613 ms | 202.764 ms | 0.3x |

*Snap ML status*: `Not installed / unavailable in current environment (optional on LinuxONE VM per BUILD.md Section 4)`

### Virtual Cities Scale Test
*events/s per city = wards (20) x signals per ward (5) x events per signal per minute (1) / 60 = 1.67 events/s. At 100 cities: 167 events/s.*
*Daily batch tick latency SLA: p99 < 1,000 ms (1.0 s).*

| Scale Level | Ingestion Stream (events/s) | p50 Tick Latency | p95 Tick Latency | p99 Tick Latency | SLA Met? |
|---|---|---|---|---|---|
| 1 cities (20 wards) | 1.67 | 0.21 ms | 0.45 ms | 0.79 ms | Yes (<1.0s) |
| 10 cities (200 wards) | 16.67 | 0.44 ms | 0.84 ms | 1.1 ms | Yes (<1.0s) |
| 50 cities (1000 wards) | 83.33 | 2.26 ms | 2.91 ms | 3.09 ms | Yes (<1.0s) |
| 100 cities (2000 wards) | 166.67 | 4.39 ms | 5.85 ms | 6.72 ms | Yes (<1.0s) |

- **Maximum Sustainable Capacity**: **100 virtual cities** (2000 wards) maintaining sub-second p99 decision latency.
