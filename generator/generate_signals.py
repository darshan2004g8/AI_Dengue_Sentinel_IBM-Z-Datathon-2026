"""
generator/generate_signals.py
Dengue Sentinel Multi-Season Synthetic Signal & Ground Truth Generator.
Adheres strictly to BUILD.md Section 7.1, 8, 9.1, and Hard Rules 1, 2, 3, 5, 7.

Features:
  - Multi-season generation (40 seasons default, plus 1 dedicated held-out replay season).
  - Negative Binomial noise (Gamma-Poisson mixture) for overdispersed clinic visits and pharmacy sales.
  - Care-pathway lead dynamics: Pharmacy antipyretic OTC sales lead clinical visits by 1 to 3 days.
  - Realistic Confounders: City-wide seasonal viral fever surge and random non-dengue fever spikes.
  - Rule-based ground truth outbreak & onset labels (trailing 56-day baseline check with 3-day persistence).
  - Manifest generation with season splits (70% train, 15% val, 15% test) and >= 5 control seasons.
  - Generates validation plot of an injected ward confirming pharmacy leads clinic.
"""

import os
import json
import math
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


def load_config(config_path="config/config.yaml"):
    """
    Load configuration parameters and directory paths from config.yaml.
    
    Inputs:
        config_path (str): Path to configuration YAML.
    Outputs:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wards(geojson_path="dashboard/assets/bbmp_wards.geojson"):
    """
    Extract ward identifiers, centroids, and zone properties from GeoJSON.
    
    Inputs:
        geojson_path (str): Path to bengaluru_wards.geojson.
    Outputs:
        list of dict: Ward property metadata.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Ward GeoJSON file missing: {geojson_path}")
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wards = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        wards.append({
            "ward_id": props.get("ward_id"),
            "ward_name": props.get("ward_name"),
            "zone": props.get("zone"),
            "lat": float(props.get("centroid_lat", 12.97)),
            "lon": float(props.get("centroid_lon", 77.59)),
        })
    wards.sort(key=lambda x: x["ward_id"])
    return wards


def build_ward_adjacency(wards, threshold_km=7.5):
    """
    Build spatial adjacency graph based on Haversine distance between ward centroids.
    
    Inputs:
        wards (list of dict): Ward metadata list.
        threshold_km (float): Distance threshold in kilometers.
    Outputs:
        dict: Mapping ward_id -> list of adjacent ward_ids.
    """
    adj = {w["ward_id"]: [] for w in wards}
    n = len(wards)
    for i in range(n):
        for j in range(i + 1, n):
            w1, w2 = wards[i], wards[j]
            # Haversine distance formula
            lat1, lon1 = math.radians(w1["lat"]), math.radians(w1["lon"])
            lat2, lon2 = math.radians(w2["lat"]), math.radians(w2["lon"])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            c = 2 * math.asin(math.sqrt(a))
            dist_km = 6371.0 * c
            if dist_km <= threshold_km:
                adj[w1["ward_id"]].append(w2["ward_id"])
                adj[w2["ward_id"]].append(w1["ward_id"])
    return adj


def sample_negative_binomial(mean_val, dispersion=5.0, rng=None):
    """
    Sample an integer from a Negative Binomial distribution using a Gamma-Poisson mixture.
    Allows modeling overdispersion in clinical visits and pharmacy sales.
    
    Inputs:
        mean_val (float or np.ndarray): Expected Poisson mean.
        dispersion (float): Overdispersion parameter k (smaller = more variance).
        rng (np.random.RandomState): Seeded random number generator.
    Outputs:
        np.ndarray or int: Sampled count(s).
    """
    if rng is None:
        rng = np.random.RandomState()
    mean_val = np.maximum(mean_val, 1e-4)
    # Gamma-Poisson mixture: gamma_val ~ Gamma(shape=dispersion, scale=mean/dispersion)
    gamma_val = rng.gamma(shape=dispersion, scale=mean_val / dispersion)
    return rng.poisson(gamma_val)


def build_replay_city_curve(
    monthly_csv_path="data/processed/india_monthly_2024.csv",
    num_days=335,
    target_mean=15.066,
    target_peak=42.365,
    plot_path="docs/replay_vs_real.png",
):
    """
    Builds the city-level daily case index for season_replay (Year 2024)
    anchored to real monthly OpenDengue totals per SPEC.md Section 9.1:
      1. Converts monthly totals to daily average rates (total / days_in_month).
      2. Places values at the midpoint of each month.
      3. Interpolates daily series using Monotonic Piecewise Cubic Hermite (PCHIP).
      4. Ends replay season at 2024-11-30 (335 days, never inventing December).
      5. Validates monthly re-sum is within 5% of real totals, rescaling if not.
      6. Rescales to match synthetic seasons' mean and peak so model scale stays valid.
      7. Saves verification plot to docs/replay_vs_real.png.
    """
    if not os.path.exists(monthly_csv_path):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Real monthly counts file missing: '{monthly_csv_path}'. "
            "Never generating synthetic fallback per SPEC.md Hard Rule 4. "
            "Run data/prepare_opendengue.py first."
        )

    df_monthly = pd.read_csv(monthly_csv_path)
    req_cols = ["calendar_start_date", "calendar_end_date", "dengue_total"]
    for col in req_cols:
        if col not in df_monthly.columns:
            raise ValueError(f"CRITICAL ERROR: Column '{col}' missing from '{monthly_csv_path}'.")

    df_monthly["calendar_start_date"] = pd.to_datetime(df_monthly["calendar_start_date"])
    df_monthly["calendar_end_date"] = pd.to_datetime(df_monthly["calendar_end_date"])
    df_monthly["days_in_month"] = (df_monthly["calendar_end_date"] - df_monthly["calendar_start_date"]).dt.days + 1
    df_monthly["daily_rate"] = df_monthly["dengue_total"] / df_monthly["days_in_month"]

    start_season = pd.Timestamp("2024-01-01")
    df_monthly["start_day"] = (df_monthly["calendar_start_date"] - start_season).dt.days
    df_monthly["end_day"] = (df_monthly["calendar_end_date"] - start_season).dt.days
    df_monthly["mid_day"] = df_monthly["start_day"] + (df_monthly["days_in_month"] - 1) / 2.0
    df_monthly["mid_date"] = df_monthly["calendar_start_date"] + pd.to_timedelta((df_monthly["days_in_month"] - 1) / 2.0, unit="D")

    all_days = np.arange(num_days)
    pchip = PchipInterpolator(df_monthly["mid_day"].values, df_monthly["daily_rate"].values, extrapolate=True)
    daily_unscaled = pchip(all_days)

    # Step 3 check: Verify monthly re-sum within 5% of real monthly totals
    print("\n" + "=" * 80)
    print("STEP 3: REPLAY SEASON MONTHLY RESUM & RESCALING CHECK (Target: <= 5% error)")
    print("=" * 80)
    comparison_rows = []
    needs_rescale = False

    for idx, r in df_monthly.iterrows():
        s_d = int(r["start_day"])
        e_d = int(r["end_day"])
        raw_sum = float(daily_unscaled[s_d : e_d + 1].sum())
        real_tot = float(r["dengue_total"])
        err_pct = abs(raw_sum - real_tot) / real_tot * 100.0
        if err_pct > 5.0:
            needs_rescale = True
        comparison_rows.append({
            "Month": r["calendar_start_date"].strftime("%Y-%m"),
            "Days": int(r["days_in_month"]),
            "Real Total": int(real_tot),
            "Interp Sum": round(raw_sum, 1),
            "Raw Diff %": round(err_pct, 2),
            "Rescaled": False,
            "Final Sum": round(raw_sum, 1),
            "Final Diff %": round(err_pct, 2),
        })

    # Rescale per month if not within 5%
    if needs_rescale:
        print("[*] Monthly sums for some months exceed 5% tolerance. Applying per-month rescaling...")
        for idx, r in df_monthly.iterrows():
            s_d = int(r["start_day"])
            e_d = int(r["end_day"])
            raw_sum = float(daily_unscaled[s_d : e_d + 1].sum())
            real_tot = float(r["dengue_total"])
            err_pct = abs(raw_sum - real_tot) / real_tot * 100.0
            if err_pct > 5.0:
                scale_factor = real_tot / raw_sum
                daily_unscaled[s_d : e_d + 1] *= scale_factor
                final_sum = float(daily_unscaled[s_d : e_d + 1].sum())
                comparison_rows[idx]["Rescaled"] = True
                comparison_rows[idx]["Final Sum"] = round(final_sum, 1)
                comparison_rows[idx]["Final Diff %"] = round(abs(final_sum - real_tot) / real_tot * 100.0, 4)

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))
    print("=" * 80)

    # Rescale resulting curve to match synthetic seasons' mean and peak:
    # -------------------------------------------------------------------------
    # Code Comment per specification:
    # The existing synthetic seasons use a base city epidemic curve with:
    #   target_mean ≈ 15.07 and target_peak ≈ 42.37 (over 366 days).
    # To prevent distributional drift in feature generation and ensure that
    # the trained XGBoost and rolling z-score baseline models operate within
    # their calibrated risk thresholds, we apply an affine transformation
    # y_scaled = a * y + b that preserves the exact empirical waveform while
    # matching the synthetic curve's mean and peak:
    #   a = (target_peak - target_mean) / (max(y) - mean(y))
    #   b = target_mean - a * mean(y)
    # -------------------------------------------------------------------------
    raw_peak = daily_unscaled.max()
    raw_mean = daily_unscaled.mean()
    a = (target_peak - target_mean) / (raw_peak - raw_mean)
    b = target_mean - a * raw_mean
    base_city_curve = a * daily_unscaled + b

    # Save validation plot docs/replay_vs_real.png
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        date_series = pd.date_range("2024-01-01", "2024-11-30", freq="D")

        fig, ax1 = plt.subplots(figsize=(13, 6))

        # Real monthly bars (height = daily average rate cases/day)
        bar_widths = df_monthly["days_in_month"] * 0.85
        ax1.bar(
            df_monthly["mid_date"],
            df_monthly["daily_rate"],
            width=bar_widths,
            color="#93c5fd",
            edgecolor="#2563eb",
            linewidth=1.2,
            alpha=0.6,
            label="Real Monthly Dengue (Daily Average Rate, cases/day)",
        )

        # Plot PCHIP daily index on top
        ax1.plot(
            date_series,
            daily_unscaled,
            color="#1e3a8a",
            linewidth=2.5,
            label="PCHIP Daily Case Index (Calibrated, cases/day)",
        )

        # Annotate each bar with real monthly total
        for _, r in df_monthly.iterrows():
            ax1.annotate(
                f"{int(r['dengue_total']):,}",
                xy=(r["mid_date"], r["daily_rate"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#0f172a",
            )

        # Secondary axis for Model-Scaled City Index
        ax2 = ax1.twinx()
        ax2.plot(
            date_series,
            base_city_curve,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Model-Scaled City Index (mean={base_city_curve.mean():.1f}, peak={base_city_curve.max():.1f})",
        )
        ax2.set_ylabel("Model-Scaled Index (Target: mean~15, peak~42)", color="#dc2626", fontsize=11)

        ax1.set_title(
            "Replay Season 2024: Real OpenDengue Monthly Totals vs PCHIP Daily Case Index\n"
            "Anchored to Official 2024 India Data (Jan 1 - Nov 30, 2024)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax1.set_xlabel("Date (2024)", fontsize=11)
        ax1.set_ylabel("Official Daily Case Rate (cases / day)", fontsize=11, color="#1e3a8a")
        ax1.grid(True, linestyle="--", alpha=0.4)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[+] Replay vs Real validation plot saved to: {plot_path}")

    return base_city_curve


def generate_season_signals(
    season_id,
    season_seed,
    weather_df,
    wards,
    adjacency,
    is_control=False,
    is_replay=False,
    cfg=None,
):
    """
    Generate synthetic daily signals and ground truth tables for one season.
    
    Inputs:
        season_id (str): Identifier e.g. 'season_01' or 'season_replay'.
        season_seed (int): Deterministic seed (SEED + s).
        weather_df (pd.DataFrame): 365-day weather dataframe (rainfall, temp, humidity).
        wards (list of dict): List of 20 wards.
        adjacency (dict): Ward adjacency graph.
        is_control (bool): If True, zero outbreaks are injected.
        is_replay (bool): If True, fixed seed replay season without training access.
        cfg (dict): Central configuration.
    Outputs:
        tuple: (events_df, ground_truth_df, official_df, num_outbreaks)
    """
    rng = np.random.RandomState(season_seed)
    num_days = len(weather_df)
    dates = weather_df["date"].values
    ward_ids = [w["ward_id"] for w in wards]
    num_wards = len(ward_ids)

    # 1. Base Seasonal Epidemic Curve for City
    day_indices = np.arange(num_days)
    if is_replay:
        # Replay season: anchored to real monthly OpenDengue counts (2024-01-01 to 2024-11-30)
        # Compute target mean and peak from the standard synthetic formula over 366 days
        ref_indices = np.arange(366)
        ref_primary = np.exp(-0.5 * ((ref_indices - 235) / 32.0) ** 2)
        ref_secondary = 0.40 * np.exp(-0.5 * ((ref_indices - 285) / 25.0) ** 2)
        ref_synth = 5.0 + 35.0 * (ref_primary + ref_secondary)
        target_mean = float(ref_synth.mean())
        target_peak = float(ref_synth.max())

        base_city_curve = build_replay_city_curve(
            monthly_csv_path="data/processed/india_monthly_2024.csv",
            num_days=num_days,
            target_mean=target_mean,
            target_peak=target_peak,
            plot_path="docs/replay_vs_real.png",
        )
    else:
        # Monsoon peak around day 230 (mid August) with secondary bump around day 280
        primary_wave = np.exp(-0.5 * ((day_indices - 235) / 32.0) ** 2)
        secondary_wave = 0.40 * np.exp(-0.5 * ((day_indices - 285) / 25.0) ** 2)
        base_city_curve = 5.0 + 35.0 * (primary_wave + secondary_wave)

    # 2. Per-ward baseline population weights (seeded lognormal, mean 1.0)
    ward_weights = {wid: float(rng.lognormal(mean=0.0, sigma=0.35)) for wid in ward_ids}

    # Initialize hidden case_index matrix: shape (num_days, num_wards)
    case_index = np.zeros((num_days, num_wards), dtype=np.float32)
    for w_idx, wid in enumerate(ward_ids):
        # Baseline case activity with slight day-to-day noise
        noise = rng.normal(loc=1.0, scale=0.12, size=num_days)
        noise = np.clip(noise, 0.5, 1.8)
        case_index[:, w_idx] = base_city_curve * ward_weights[wid] * noise * 0.25

    # 3. Inject outbreaks (3 to 8 outbreaks, none in control seasons)
    num_outbreaks = 0
    if not is_control:
        num_outbreaks = rng.randint(3, 9)
        for _ in range(num_outbreaks):
            seed_ward_idx = rng.randint(0, num_wards)
            seed_ward_id = ward_ids[seed_ward_idx]
            # Random outbreak start date (between day 45 and day 310)
            start_day = rng.randint(45, min(num_days - 35, 310))
            ramp_days = rng.randint(12, 28)
            multiplier = rng.uniform(3.5, 7.5)

            # Generate bell-shaped outbreak wave
            peak_day = start_day + ramp_days
            duration = ramp_days * 2
            t_wave = np.arange(num_days)
            wave = np.exp(-0.5 * ((t_wave - peak_day) / (ramp_days / 2.2)) ** 2)
            
            # Baseline level at seed ward
            baseline_val = np.median(case_index[max(0, start_day - 28):start_day, seed_ward_idx])
            injected_boost = baseline_val * (multiplier - 1.0) * wave
            case_index[:, seed_ward_idx] += injected_boost

            # Spatial spread to adjacent wards with distance decay
            adjacent_wards = adjacency.get(seed_ward_id, [])
            for adj_wid in adjacent_wards:
                if adj_wid in ward_ids:
                    adj_idx = ward_ids.index(adj_wid)
                    # Delayed onset and attenuated amplitude
                    adj_delay = rng.randint(4, 9)
                    decay_factor = rng.uniform(0.35, 0.65)
                    adj_wave = np.exp(-0.5 * ((t_wave - (peak_day + adj_delay)) / (ramp_days / 2.0)) ** 2)
                    case_index[:, adj_idx] += baseline_val * (multiplier - 1.0) * decay_factor * adj_wave

    # 4. Inject Confounders
    # (a) City-wide non-dengue viral fever wave (e.g. influenza spike around day 60-90)
    viral_fever_wave = np.exp(-0.5 * ((day_indices - 75) / 18.0) ** 2) * rng.uniform(15.0, 30.0)
    
    # (b) 0 to 3 random isolated non-dengue fever spikes in random wards
    num_spikes = rng.randint(0, 4)
    false_spikes = np.zeros((num_days, num_wards), dtype=np.float32)
    for _ in range(num_spikes):
        spike_w = rng.randint(0, num_wards)
        spike_day = rng.randint(30, num_days - 20)
        spike_wave = np.exp(-0.5 * ((day_indices - spike_day) / 4.0) ** 2) * rng.uniform(25.0, 50.0)
        false_spikes[:, spike_w] += spike_wave

    # 5. Generate Signals (Pharmacy leads clinic by 1-3 days per Section 9.1)
    # Lags: pharmacy 0-2 days; clinic 2-4 days
    pharm_lag = rng.randint(0, 2)     # Pharmacy purchases earliest
    clinic_lag = rng.randint(2, 5)    # Clinic visits follow after home-treatment delay
    
    pharmacy_events = []
    clinic_events = []
    weather_events = []
    ground_truth_records = []
    official_records = []

    # Weather lookup
    rainfall_arr = weather_df["rainfall"].values
    temp_arr = weather_df["temp"].values
    humidity_arr = weather_df["humidity"].values

    for w_idx, wid in enumerate(ward_ids):
        ci = case_index[:, w_idx]
        
        # Ground Truth Outbreak Labeling per Section 9.1:
        # A ward-day is outbreak_flag = 1 when its 7-day rolling case_index is at least 2x
        # its trailing 56-day baseline and above an absolute floor (>= 8.0) for >= 3 consecutive days.
        ci_series = pd.Series(ci)
        roll7 = ci_series.rolling(7, min_periods=1).mean().values
        roll56 = ci_series.rolling(56, min_periods=14).mean().fillna(ci_series.iloc[:14].mean()).values

        candidate_outbreak = (roll7 >= 2.0 * roll56) & (roll7 >= 8.0)
        outbreak_flag = np.zeros(num_days, dtype=int)
        onset_flag = np.zeros(num_days, dtype=int)

        run_len = 0
        for d in range(num_days):
            if candidate_outbreak[d]:
                run_len += 1
                if run_len >= 3:
                    outbreak_flag[d - run_len + 1 : d + 1] = 1
            else:
                run_len = 0

        # Onset = first day of an outbreak run
        for d in range(1, num_days):
            if outbreak_flag[d] == 1 and outbreak_flag[d - 1] == 0:
                onset_flag[d] = 1
        if outbreak_flag[0] == 1:
            onset_flag[0] = 1

        # Target: outbreak_in_next_14d = 1 if an onset falls in the next 14 days
        outbreak_in_next_14d = np.zeros(num_days, dtype=int)
        for d in range(num_days):
            window_end = min(d + 15, num_days)
            if np.any(onset_flag[d + 1 : window_end] == 1):
                outbreak_in_next_14d[d] = 1

        # Official Series: delayed reporting (7 to 14 days) with under-reporting factor
        rep_delay = rng.randint(7, 15)
        delayed_cases = np.zeros(num_days, dtype=np.float32)
        if rep_delay < num_days:
            delayed_cases[rep_delay:] = ci[:-rep_delay] * rng.uniform(0.35, 0.55)

        for d in range(num_days):
            dt_str = str(dates[d])
            dt_obj = pd.Timestamp(dt_str)
            dow = dt_obj.dayofweek  # 6 = Sunday

            # Weekend suppression on clinics (Sundays closed or reduced)
            dow_clinic_factor = 0.45 if dow == 6 else (0.85 if dow == 5 else 1.0)
            # Pharmacy slight surge on Saturdays/Mondays
            dow_pharm_factor = 1.15 if dow in (0, 5) else 1.0

            # Shifted indices for care pathway
            d_pharm = max(0, d - pharm_lag)
            d_clinic = max(0, d - clinic_lag)

            # Pharmacy mean: baseline OTC + scaled dengue cases + viral wave + spikes
            mu_pharm = (
                15.0 * ward_weights[wid]
                + 1.6 * ci[d_pharm]
                + 0.8 * viral_fever_wave[d]
                + false_spikes[d, w_idx]
            ) * dow_pharm_factor
            
            # Clinic mean: baseline visits + scaled dengue cases + viral wave + spikes
            mu_clinic = (
                8.0 * ward_weights[wid]
                + 2.1 * ci[d_clinic]
                + 0.6 * viral_fever_wave[d]
                + false_spikes[d, w_idx] * 0.7
            ) * dow_clinic_factor

            # Sample Negative Binomial observations
            val_pharm = float(sample_negative_binomial(mu_pharm, dispersion=4.0, rng=rng))
            val_clinic = float(sample_negative_binomial(mu_clinic, dispersion=4.0, rng=rng))

            # Streamed events conforming to Section 7.1
            pharmacy_events.append({
                "event_id": f"{wid}|{dt_str}|pharmacy_fever_meds",
                "ward_id": wid,
                "date": dt_str,
                "signal_type": "pharmacy_fever_meds",
                "value": val_pharm,
                "is_synthetic": True,
            })
            clinic_events.append({
                "event_id": f"{wid}|{dt_str}|clinic_fever_visits",
                "ward_id": wid,
                "date": dt_str,
                "signal_type": "clinic_fever_visits",
                "value": val_clinic,
                "is_synthetic": True,
            })
            weather_events.extend([
                {
                    "event_id": f"{wid}|{dt_str}|rainfall",
                    "ward_id": wid,
                    "date": dt_str,
                    "signal_type": "rainfall",
                    "value": float(rainfall_arr[d]),
                    "is_synthetic": False,
                },
                {
                    "event_id": f"{wid}|{dt_str}|temp",
                    "ward_id": wid,
                    "date": dt_str,
                    "signal_type": "temp",
                    "value": float(temp_arr[d]),
                    "is_synthetic": False,
                },
                {
                    "event_id": f"{wid}|{dt_str}|humidity",
                    "ward_id": wid,
                    "date": dt_str,
                    "signal_type": "humidity",
                    "value": float(humidity_arr[d]),
                    "is_synthetic": False,
                },
            ])

            # Hidden ground truth record
            ground_truth_records.append({
                "ward_id": wid,
                "date": dt_str,
                "case_index": round(float(ci[d]), 3),
                "outbreak_flag": int(outbreak_flag[d]),
                "onset_flag": int(onset_flag[d]),
                "outbreak_in_next_14d": int(outbreak_in_next_14d[d]),
            })

            # Official counts record
            official_records.append({
                "ward_id": wid,
                "date": dt_str,
                "official_cases": round(float(delayed_cases[d]), 2),
            })

    # Combine all signal events
    all_events = pharmacy_events + clinic_events + weather_events
    events_df = pd.DataFrame(all_events)
    events_df.sort_values(by=["date", "ward_id", "signal_type"], inplace=True)

    ground_truth_df = pd.DataFrame(ground_truth_records)
    ground_truth_df.sort_values(by=["date", "ward_id"], inplace=True)

    official_df = pd.DataFrame(official_records)
    official_df.sort_values(by=["date", "ward_id"], inplace=True)

    return events_df, ground_truth_df, official_df, num_outbreaks


def plot_sample_injected_ward(events_df, ground_truth_df, target_ward="W017", output_path="docs/sample_injected_ward.png"):
    """
    Generate an illustrative time-series plot of one injected ward confirming:
      1. Outbreak profile is epidemiologically believable.
      2. Pharmacy sales visibly lead clinic visits by several days.
      3. Rainfall antecedents precede the outbreak surge.
    
    Inputs:
        events_df (pd.DataFrame): Streamed events dataset.
        ground_truth_df (pd.DataFrame): Ground truth dataset.
        target_ward (str): Ward identifier to plot (default: 'W017').
        output_path (str): Destination PNG path.
    Outputs:
        str: Path to saved figure.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter to target ward
    w_events = events_df[events_df["ward_id"] == target_ward].copy()
    w_gt = ground_truth_df[ground_truth_df["ward_id"] == target_ward].copy()

    # Pivot events for plotting
    piv = w_events.pivot(index="date", columns="signal_type", values="value").reset_index()
    piv["date"] = pd.to_datetime(piv["date"])
    w_gt["date"] = pd.to_datetime(w_gt["date"])
    merged = pd.merge(piv, w_gt, on="date", how="inner").sort_values("date")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 2, 1.2]})

    # Subplot 1: Early Warning Care Pathway (Pharmacy vs Clinic)
    ax1.plot(merged["date"], merged["pharmacy_fever_meds"], label="Pharmacy Fever Meds (OTC Antipyretics)", color="#f59e0b", linewidth=2.0)
    ax1.plot(merged["date"], merged["clinic_fever_visits"], label="Clinic Fever Visits", color="#ef4444", linewidth=2.0)
    ax1.plot(merged["date"], merged["case_index"], label="Hidden True Dengue Case Index", color="#1e293b", linestyle="--", alpha=0.7)
    
    # Highlight Ground Truth Outbreak Windows
    outbreak_dates = merged[merged["outbreak_flag"] == 1]["date"]
    if not outbreak_dates.empty:
        for od in outbreak_dates:
            ax1.axvspan(od - pd.Timedelta(hours=12), od + pd.Timedelta(hours=12), color="#fee2e2", alpha=0.25)
    
    onset_dates = merged[merged["onset_flag"] == 1]["date"]
    for od in onset_dates:
        ax1.axvline(od, color="#b91c1c", linestyle=":", linewidth=2.0, label="True Outbreak Onset" if od == onset_dates.iloc[0] else "")

    ax1.set_title(f"Dengue Sentinel Early Signal Dynamics — Ward {target_ward}\nNotice: Pharmacy OTC Fever-Meds Visibly Lead Clinical Visits", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Daily Signal Volume", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Weather Drivers (Rainfall & Temperature)
    ax2.bar(merged["date"], merged["rainfall"], color="#38bdf8", alpha=0.65, label="Daily Rainfall (mm)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged["date"], merged["temp"], color="#f97316", linewidth=1.5, label="Mean Temp (°C)")
    ax2.set_ylabel("Precipitation (mm)", color="#0284c7", fontsize=11)
    ax2_twin.set_ylabel("Temp (°C)", color="#ea580c", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="upper right", frameon=True)

    # Subplot 3: Binary Ground Truth Outbreak & 14-day Anticipation Flag
    ax3.plot(merged["date"], merged["outbreak_in_next_14d"], color="#8b5cf6", linewidth=1.8, label="Outbreak in Next 14 Days (Target)")
    ax3.fill_between(merged["date"], 0, merged["outbreak_flag"], color="#ef4444", alpha=0.3, label="Active Outbreak")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Normal", "Outbreak Window"])
    ax3.set_ylabel("Status", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Sample injected ward validation plot saved to: {output_path}")
    return output_path


def generate_all_seasons(config_path="config/config.yaml"):
    """
    Generate complete multi-season corpus (train, val, test, control, replay).
    Writes output files and updates manifest.csv per BUILD.md Section 9.1.
    
    Inputs:
        config_path (str): Path to system configuration file.
    Outputs:
        pd.DataFrame: Manifest dataframe.
    """
    cfg = load_config(config_path)
    base_seed = int(cfg.get("system", {}).get("base_seed", 42))
    gen_cfg = cfg.get("generator", {})
    paths = cfg.get("data", {}).get("paths", {})

    total_seasons = int(gen_cfg.get("total_seasons", 40))
    train_split = float(gen_cfg.get("train_split", 0.70))
    val_split = float(gen_cfg.get("val_split", 0.15))
    test_split = float(gen_cfg.get("test_split", 0.15))
    control_seasons_count = int(gen_cfg.get("control_seasons_count", 5))

    events_dir = paths.get("events_dir", "data/processed/events")
    gt_dir = paths.get("ground_truth_dir", "data/processed/ground_truth")
    official_dir = paths.get("official_dir", "data/processed/official")
    manifest_path = paths.get("manifest_csv", "data/processed/manifest.csv")
    weather_path = paths.get("raw_weather", "data/raw/bengaluru_weather_2018_2025.csv")
    wards_path = paths.get("wards_geojson", "dashboard/assets/bbmp_wards.geojson")

    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(official_dir, exist_ok=True)

    # Load backbone inputs
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Weather dataset missing at: {weather_path}. Run `make data` first.")
    
    weather_full = pd.read_csv(weather_path)
    # Use 2024 season as primary backbone (366 days in leap year 2024)
    weather_2024 = weather_full[weather_full["date"].str.startswith("2024")].reset_index(drop=True)
    if weather_2024.empty:
        weather_2024 = weather_full.iloc[:365].reset_index(drop=True)

    wards = load_wards(wards_path)
    adjacency = build_ward_adjacency(wards)

    # Save ward adjacency for downstream models
    adj_output_path = os.path.join(os.path.dirname(manifest_path), "ward_adjacency.json")
    with open(adj_output_path, "w", encoding="utf-8") as f:
        json.dump(adjacency, f, indent=2)

    # Determine season splits:
    # 40 seasons: 28 train, 6 val, 6 test.
    # Control seasons (at least 5): distribute evenly (3 in train, 1 in val, 1 in test).
    num_train = int(round(total_seasons * train_split))
    num_val = int(round(total_seasons * val_split))
    num_test = total_seasons - num_train - num_val

    splits = ["train"] * num_train + ["val"] * num_val + ["test"] * num_test
    # Fixed control indices: e.g. 0, 10, 20 (train), 29 (val), 35 (test)
    control_indices = {0, 10, 20, num_train + 1, num_train + num_val + 1}

    manifest_rows = []
    print(f"Generating {total_seasons} multi-season datasets (Base Seed = {base_seed})...")

    sample_injected_events = None
    sample_injected_gt = None

    for s in range(total_seasons):
        season_id = f"season_{s:02d}"
        split = splits[s]
        is_control = s in control_indices
        season_seed = base_seed + s

        ev_df, gt_df, off_df, num_outbreaks = generate_season_signals(
            season_id=season_id,
            season_seed=season_seed,
            weather_df=weather_2024,
            wards=wards,
            adjacency=adjacency,
            is_control=is_control,
            is_replay=False,
            cfg=cfg,
        )

        ev_path = os.path.join(events_dir, f"{season_id}.csv")
        gt_path = os.path.join(gt_dir, f"{season_id}.csv")
        off_path = os.path.join(official_dir, f"{season_id}.csv")

        ev_df.to_csv(ev_path, index=False)
        gt_df.to_csv(gt_path, index=False)
        off_df.to_csv(off_path, index=False)

        manifest_rows.append({
            "season_id": season_id,
            "season_type": "synthetic",
            "split": split,
            "has_outbreaks": not is_control,
            "seed": season_seed,
            "num_outbreaks": num_outbreaks,
            "num_events": len(ev_df),
        })

        if not is_control and sample_injected_events is None:
            sample_injected_events = ev_df
            sample_injected_gt = gt_df

    # Generate held-out Replay Season per Section 9.1 & Hard Rule 4
    # Fixed seed base_seed + 9999, never used for training, calibration or thresholds
    replay_seed = base_seed + 9999
    replay_id = "season_replay"
    print(f"Generating held-out replay season: {replay_id} (Seed = {replay_seed})...")
    # For replay season, end at 2024-11-30 per SPEC.md (do not invent December)
    weather_replay = weather_2024[weather_2024["date"] <= "2024-11-30"].reset_index(drop=True)
    rev_df, rgt_df, roff_df, r_outbreaks = generate_season_signals(
        season_id=replay_id,
        season_seed=replay_seed,
        weather_df=weather_replay,
        wards=wards,
        adjacency=adjacency,
        is_control=False,
        is_replay=True,
        cfg=cfg,
    )
    rev_df.to_csv(os.path.join(events_dir, f"{replay_id}.csv"), index=False)
    rgt_df.to_csv(os.path.join(gt_dir, f"{replay_id}.csv"), index=False)
    roff_df.to_csv(os.path.join(official_dir, f"{replay_id}.csv"), index=False)

    manifest_rows.append({
        "season_id": replay_id,
        "season_type": "real_anchored",
        "split": "replay",
        "has_outbreaks": True,
        "seed": replay_seed,
        "num_outbreaks": r_outbreaks,
        "num_events": len(rev_df),
    })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Manifest written to: {manifest_path}")

    # Plot sample injected ward to docs/sample_injected_ward.png
    if sample_injected_events is not None:
        plot_sample_injected_ward(sample_injected_events, sample_injected_gt, target_ward="W017")

    print("\nGeneration Summary:")
    print(f"  Total Seasons: {len(manifest_df)} (Train: {num_train}, Val: {num_val}, Test: {num_test}, Replay: 1)")
    print(f"  Control Seasons: {sum(1 for r in manifest_rows if not r['has_outbreaks'])}")
    print(f"  Total Events per Season: {manifest_rows[0]['num_events']} (20 wards x 366 days x 5 signals)")
    print(f"  All synthetic rows labeled with is_synthetic = true (Hard Rule 2).")

    return manifest_df


def main():
    """Command-line execution for multi-season signal generator."""
    parser = argparse.ArgumentParser(description="Dengue Sentinel Multi-Season Generator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    generate_all_seasons(config_path=args.config)


if __name__ == "__main__":
    main()
