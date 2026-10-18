# Dengue Sentinel: Live Demonstration Script

**Event Code**: `SSF2026` (IBM Z Datathon 2026)  
**Presenter Guide**: 3 to 5 Minute Live Walkthrough  
**Target Audience**: Datathon Evaluators & Municipal Health Stakeholders  

---

## Preparation (Before Starting)
1. Ensure the virtual environment is active.
2. In Terminal 1, run the automated health check:
   ```bash
   make test
   ```
   *(Expected output: Ran 15 tests, OK)*
3. In Terminal 2, launch the interactive dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
4. Keep Terminal 1 open to run `make demo` if asked to show streaming in action.

---

## 1. Opening Pitch (30 Seconds)

> *"Good morning judges. We are presenting **Dengue Sentinel** for the IBM Z Datathon 2026, event code `SSF2026`.*
> 
> *Urban centers in India face severe dengue epidemics every monsoon. Current municipal surveillance relies on hospital laboratory reports, which arrive with an unavoidable **7 to 14 day delay**. By the time hospital wards fill up, mosquitoes have already transmitted the virus across entire neighborhoods.*
> 
> *Dengue Sentinel flips the paradigm: **Detect, Decide, Respond**. We stream early syndromic signals—over-the-counter fever medicine sales at local pharmacies and fever visits at primary health clinics—and run real-time AI scoring on IBM LinuxONE to detect transmission hotspots **16 days earlier** than hospital lab reports."*

---

## 2. Live Dashboard Walkthrough (2.5 Minutes)

### Tab 1: 🗺️ City Heatmap & Alert Feed
- **Show the Simulated Signals Badge**: Point out the prominent amber badge in the header: `"Simulated ward-level signals"`.
  > *"First, notice our commitment to honest reporting. All ward-level syndromic signals are clearly badged as simulated per Hard Rule 2, while weather, city cases, and ward boundaries are 100% real."*
- **Interact with the Map**:
  - Show the 20 Bengaluru wards colored by alert level (Red for High, Amber for Medium, Green for Low).
  - Hover over **Ward W017** or **Ward W007**: show the rich tooltip displaying Ward Name, Risk Score, and Primary Driver.
- **Alert Feed**:
  - Point to the right-hand panel:
    > *"The feed prioritizes confirmed alerts with plain-language Tree SHAP driver text—for example: 'Fever-medicine sales up +115% over 7 days'."*

---

### Tab 2: 🔍 Ward Detail & Early Signals
- Select **W017 (Shantalanagar)** from the dropdown.
- Point to the multi-signal time series chart:
  > *"Here you can see the core epidemiological insight: the purple pharmacy line (OTC fever meds) surges **3 to 5 days before** the blue clinic line (primary health visits). Our calibrated model picks up this early syndromic wave to predict outbreak onset."*

---

### Tab 3: 🚨 Respond & Resource Allocation
- Move the **Available Field Teams** slider from 5 to 8.
- Show the **Municipal Response Roster**:
  > *"Decide is useless without Respond. Municipalities have finite field workers. Our greedy exposure-weighted allocator ($r = 0.30$) assigns teams to the wards with the highest marginal epidemiological benefit."*
- Point to the specific operational action text:
  - **Medium Alert**: *"Schedule inspection and larval source removal within 72 hours."*
  - **High Alert**: *"Prioritise inspection and targeted larval source removal within 48 hours; consider targeted fogging at the cluster."*
- Highlight the **Watchlist**:
  > *"Wards with elevated risk that exceed available field teams appear immediately on the Watchlist, giving public health leaders an objective basis to request emergency reinforcement."*
- Click **"📄 Download Response Dispatch Order (CSV)"** to demonstrate the operational deliverable.

---

### Tab 4: 🏛️ Why IBM Z
- Click on **Tab 4**:
  > *"Why IBM Z? Sub-millisecond ONNX inference and enterprise reliability architecture.*
  >
  > *All latency numbers here were measured by running `make benchmark` on a local machine; on IBM LinuxONE (s390x) results will differ:*
  > 1. *ONNX Runtime delivers approximately **~22x lower latency** at single-event scoring (batch=1: ~1.09 ms XGBoost vs ~0.05 ms ONNX; results are approximate and machine-dependent).*
  > 2. *At batch=100 the speedup is approximately **~6x**. At batch=1,000 and above, native XGBoost is faster — fine for a ward-level (batch=20) deployment.*
  > 3. *In our virtual city scaling test, the system sustainably evaluates **100 virtual cities (2,000 wards)** with p99 tick latency well under the 1,000 ms SLA."*

---

### Tab 5: 📈 Validation & Backtest
- Click on **Tab 5**:
  - Highlight the **Hard Rule 10 Caveat Box**:
    > *"⚠️ Caveat: Pharmacy and clinic signals are synthetic, so lead time shows pipeline behavior, not real-world validation."*
  - Review the Comparison Table:
    > *"Across held-out test seasons and the dedicated 2024 replay season:*
    > - *In the E2 replay season, Sentinel's confirmed alerts fired a mean of **+9.8 days before local outbreak onset** (primary metric). The secondary headline is **+16.8 days before per-outbreak official notice**, where official notice = onset + assumed 7-day reporting delay.*
    > - *In E1 held-out test seasons, mean lead time vs official is **+13.7 days**, recall **60.8%**.*
    > - *Crucially, on 5 pure control seasons with zero outbreaks, our false alert rate is **0.36 per 100 ward-weeks**, comfortably within the municipal budget of $\le 5.0$."*

---

## 3. Streaming Replay in Terminal (30 Seconds, Optional)

If evaluators ask to see events flow live:
1. In Terminal 1, run:
   ```bash
   make demo
   ```
2. Watch the replayer dispatch 1,000 daily events at 10 simulated days/second through the scoring pipeline.

---

## 4. Closing & Q&A (30 Seconds)

> *"In summary, Dengue Sentinel demonstrates that real-time AI on IBM Z can transform municipal disease surveillance from a lagging post-mortem into a proactive, life-saving early warning system.*
> 
> *Thank you, and we welcome your questions!"*
