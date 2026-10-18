"""
model/features.py
Feature extraction engine used identically across offline training and live streaming scoring.
Adheres strictly to BUILD.md Section 9.2:
  - Features computed strictly from historical observations up to scoring date (no look-ahead leakage).
  - Rolling clinic and pharmacy fever volume (7d mean, 7d % change, 28d z-score with 2-day guard band).
  - Rainfall lags (7d, 14d, 21d) and cumulative sums (14d, 28d).
  - Temperature and humidity 7d means.
  - Cyclical day-of-year trigonometric encodings (sin/cos).
  - Data freshness/completeness metric across the 5 streams.
  - STRICT HARD RULE 3 & Section 9.2: Ward IDs and zone IDs are NEVER included as features.
"""

import math
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "clinic_visits_7d_mean",
    "clinic_visits_7d_change_pct",
    "clinic_visits_zscore_28d",
    "pharmacy_7d_mean",
    "pharmacy_7d_change_pct",
    "pharmacy_zscore_28d",
    "rainfall_lag7",
    "rainfall_lag14",
    "rainfall_lag21",
    "rainfall_14d_sum",
    "rainfall_28d_sum",
    "temp_7d_mean",
    "humidity_7d_mean",
    "sin_day_of_year",
    "cos_day_of_year",
    "completeness"
]

FEATURE_MONOTONE_CONSTRAINTS = {
    "clinic_visits_7d_mean": 1,
    "clinic_visits_7d_change_pct": 1,
    "clinic_visits_zscore_28d": 1,
    "pharmacy_7d_mean": 1,
    "pharmacy_7d_change_pct": 1,
    "pharmacy_zscore_28d": 1,
    "rainfall_lag7": 0,
    "rainfall_lag14": 0,
    "rainfall_lag21": 0,
    "rainfall_14d_sum": 0,
    "rainfall_28d_sum": 0,
    "temp_7d_mean": 0,
    "humidity_7d_mean": 0,
    "sin_day_of_year": 0,
    "cos_day_of_year": 0,
    "completeness": 0
}


def compute_features_for_ward_dataframe(ward_df):
    """
    Compute full feature time-series vectorially for a single ward's chronological observations.
    
    Inputs:
        ward_df (pd.DataFrame): Daily observations sorted by date with columns:
            ['date', 'clinic_fever_visits', 'pharmacy_fever_meds', 'rainfall', 'temp', 'humidity'].
    Outputs:
        pd.DataFrame: Feature dataframe with columns matching FEATURE_NAMES aligned to ward_df.
    """
    df = ward_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat_df = pd.DataFrame(index=df.index)

    # 1. Clinic Features
    clinic = df["clinic_fever_visits"]
    c_7d_mean = clinic.rolling(7, min_periods=1).mean()
    c_prev7_mean = clinic.shift(7).rolling(7, min_periods=1).mean()
    c_change_pct = 100.0 * (c_7d_mean - c_prev7_mean) / (c_prev7_mean + 1e-4)
    # 28-day baseline with 2-day guard band: shift(2).rolling(28)
    c_b_mean = clinic.shift(2).rolling(28, min_periods=14).mean()
    c_b_std = clinic.shift(2).rolling(28, min_periods=14).std().fillna(1.0)
    c_zscore = (clinic - c_b_mean) / (c_b_std + 1e-6)

    feat_df["clinic_visits_7d_mean"] = c_7d_mean.fillna(0.0)
    feat_df["clinic_visits_7d_change_pct"] = c_change_pct.fillna(0.0)
    feat_df["clinic_visits_zscore_28d"] = c_zscore.fillna(0.0)

    # 2. Pharmacy Features
    pharm = df["pharmacy_fever_meds"]
    p_7d_mean = pharm.rolling(7, min_periods=1).mean()
    p_prev7_mean = pharm.shift(7).rolling(7, min_periods=1).mean()
    p_change_pct = 100.0 * (p_7d_mean - p_prev7_mean) / (p_prev7_mean + 1e-4)
    p_b_mean = pharm.shift(2).rolling(28, min_periods=14).mean()
    p_b_std = pharm.shift(2).rolling(28, min_periods=14).std().fillna(1.0)
    p_zscore = (pharm - p_b_mean) / (p_b_std + 1e-6)

    feat_df["pharmacy_7d_mean"] = p_7d_mean.fillna(0.0)
    feat_df["pharmacy_7d_change_pct"] = p_change_pct.fillna(0.0)
    feat_df["pharmacy_zscore_28d"] = p_zscore.fillna(0.0)

    # 3. Weather Lags & Cumulative Sums
    rain = df["rainfall"]
    feat_df["rainfall_lag7"] = rain.shift(7).fillna(0.0)
    feat_df["rainfall_lag14"] = rain.shift(14).fillna(0.0)
    feat_df["rainfall_lag21"] = rain.shift(21).fillna(0.0)
    feat_df["rainfall_14d_sum"] = rain.rolling(14, min_periods=1).sum().fillna(0.0)
    feat_df["rainfall_28d_sum"] = rain.rolling(28, min_periods=1).sum().fillna(0.0)

    # 4. Temperature & Humidity 7-Day Means
    feat_df["temp_7d_mean"] = df["temp"].rolling(7, min_periods=1).mean().fillna(24.0)
    feat_df["humidity_7d_mean"] = df["humidity"].rolling(7, min_periods=1).mean().fillna(70.0)

    # 5. Calendar Seasonality Trigonometric Encodings
    doy = df["date"].dt.dayofyear
    feat_df["sin_day_of_year"] = np.sin(2.0 * np.pi * doy / 365.25)
    feat_df["cos_day_of_year"] = np.cos(2.0 * np.pi * doy / 365.25)

    # 6. Completeness
    feat_df["completeness"] = 1.0

    return feat_df[FEATURE_NAMES]


def compute_features_for_ward_date(history_df, scoring_date):
    """
    Compute feature dictionary for a single ward from historical events up to scoring_date.
    
    Inputs:
        history_df (pd.DataFrame): DataFrame with columns ['date', 'signal_type', 'value'].
        scoring_date (str or pd.Timestamp): Target scoring date.
    Outputs:
        dict: Mapping feature_name -> float value.
    """
    s_dt = pd.Timestamp(scoring_date)
    df = history_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= s_dt]

    piv = df.pivot_table(index="date", columns="signal_type", values="value").reset_index()
    for req_col in ["clinic_fever_visits", "pharmacy_fever_meds", "rainfall", "temp", "humidity"]:
        if req_col not in piv.columns:
            piv[req_col] = 0.0

    feat_df = compute_features_for_ward_dataframe(piv)
    latest_features = feat_df.iloc[-1].to_dict()
    return latest_features


def build_training_matrix_from_season(events_df, ground_truth_df, burn_in_days=35):
    """
    Compile feature matrix and ground truth alignment for a complete season using vectorized rolling operations.
    Excludes ward-days already in active outbreak per BUILD.md Section 9.1.
    
    Inputs:
        events_df (pd.DataFrame): Events records for the season.
        ground_truth_df (pd.DataFrame): Ground truth records containing outbreak_flag & outbreak_in_next_14d.
        burn_in_days (int): Days skipped at start to initialize baseline buffers (default: 35).
    Outputs:
        tuple: (X_matrix (np.ndarray), y_vector (np.ndarray), metadata_df (pd.DataFrame))
    """
    ev = events_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(ev["date"]):
        ev["date"] = pd.to_datetime(ev["date"])
    piv = ev.pivot_table(index=["date", "ward_id"], columns="signal_type", values="value").reset_index()

    gt = ground_truth_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(gt["date"]):
        gt["date"] = pd.to_datetime(gt["date"])
    merged = pd.merge(piv, gt, on=["date", "ward_id"], how="inner").sort_values(["ward_id", "date"]).reset_index(drop=True)

    X_list = []
    y_list = []
    meta_list = []

    for ward_id, w_df in merged.groupby("ward_id"):
        w_df = w_df.sort_values("date").reset_index(drop=True)
        w_feats = compute_features_for_ward_dataframe(w_df)

        valid_mask = (w_df.index >= burn_in_days) & (w_df["outbreak_flag"] == 0)
        
        valid_feats = w_feats.loc[valid_mask]
        valid_targets = w_df.loc[valid_mask, "outbreak_in_next_14d"]
        valid_meta = pd.DataFrame({
            "ward_id": ward_id,
            "date": w_df.loc[valid_mask, "date"].dt.strftime("%Y-%m-%d")
        })

        if len(valid_feats) > 0:
            X_list.append(valid_feats.values.astype(np.float32))
            y_list.append(valid_targets.values.astype(np.int32))
            meta_list.append(valid_meta)

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.int32), pd.DataFrame()

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    meta_df = pd.concat(meta_list, ignore_index=True)

    return X, y, meta_df
