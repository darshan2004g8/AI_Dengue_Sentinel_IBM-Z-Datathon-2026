"""
model/train.py
Model training, probability calibration, and threshold optimization for Dengue Sentinel.
Adheres strictly to BUILD.md Section 9.2 and Hard Rules 3, 4, 7, 9:
  - Train dataset: Built strictly from 'train' seasons in manifest.csv.
  - Excludes ward-days already in active outbreak per Section 9.1.
  - Model: Extreme Gradient Boosting (XGBClassifier) with monotone constraints on growth features.
  - Class imbalance handling via scale_pos_weight.
  - Validation-based early stopping on 'val' seasons.
  - Isotonic probability calibration fit strictly on validation seasons (Hard Rule 4).
  - Threshold tuning on validation seasons to satisfy false-alert budget (<= 5 false alerts / 100 ward-weeks).
  - Exports artifacts: model.joblib, calibrator.joblib, thresholds.json, and model.onnx.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc

from model.features import (
    FEATURE_NAMES,
    FEATURE_MONOTONE_CONSTRAINTS,
    build_training_matrix_from_season,
)
from model.export_onnx import export_xgboost_to_onnx, check_onnx_parity


def load_split_dataset(manifest_path, split_name, events_dir="data/processed/events", gt_dir="data/processed/ground_truth"):
    """
    Compile feature matrices and ground truth targets across all seasons belonging to a designated split.
    
    Inputs:
        manifest_path (str): Path to manifest.csv.
        split_name (str): 'train', 'val', or 'test'.
        events_dir (str): Directory of events/season_XX.csv.
        gt_dir (str): Directory of ground_truth/season_XX.csv.
    Outputs:
        tuple: (X (np.ndarray), y (np.ndarray), meta_df (pd.DataFrame))
    """
    manifest = pd.read_csv(manifest_path)
    seasons = manifest[manifest["split"] == split_name]["season_id"].tolist()
    
    if not seasons:
        raise ValueError(f"No seasons found in manifest for split: {split_name}")
        
    X_list, y_list, meta_list = [], [], []
    for sid in seasons:
        ev_file = os.path.join(events_dir, f"{sid}.csv")
        gt_file = os.path.join(gt_dir, f"{sid}.csv")
        if not os.path.exists(ev_file) or not os.path.exists(gt_file):
            continue
        ev_df = pd.read_csv(ev_file)
        gt_df = pd.read_csv(gt_file)
        
        x_s, y_s, m_s = build_training_matrix_from_season(ev_df, gt_df)
        if len(x_s) > 0:
            X_list.append(x_s)
            y_list.append(y_s)
            m_s["season_id"] = sid
            meta_list.append(m_s)
            
    if not X_list:
        raise RuntimeError(f"Could not build feature matrix for split: {split_name}")
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    meta_df = pd.concat(meta_list, ignore_index=True)
    return X, y, meta_df


def train_calibrated_xgboost(
    X_train, y_train, X_val, y_val, random_state=42
):
    """
    Train XGBoost binary classifier with monotone growth constraints and calibrate on validation.
    
    Inputs:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training binary target labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation binary target labels.
        random_state (int): Fixed random seed (42).
    Outputs:
        tuple: (trained_model, isotonic_calibrator)
    """
    # Monotone constraints tuple aligned with FEATURE_NAMES
    monotone_constraints = tuple(FEATURE_MONOTONE_CONSTRAINTS[f] for f in FEATURE_NAMES)
    
    # Class imbalance scale weight
    pos_count = max(int(np.sum(y_train == 1)), 1)
    neg_count = max(int(np.sum(y_train == 0)), 1)
    scale_weight = float(neg_count / pos_count)
    
    print(f"Training XGBoost: {len(y_train)} rows (Positive: {pos_count}, Negative: {neg_count}, Scale: {scale_weight:.2f})")
    
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        monotone_constraints=monotone_constraints,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=random_state,
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Fit Isotonic Regression strictly on validation predictions (Hard Rule 4)
    raw_val_probs = clf.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_val_probs, y_val)
    
    cal_val_probs = calibrator.predict(raw_val_probs)
    brier_raw = brier_score_loss(y_val, raw_val_probs)
    brier_cal = brier_score_loss(y_val, cal_val_probs)
    print(f"Probability Calibration on Validation: Brier Score Raw={brier_raw:.4f} -> Calibrated={brier_cal:.4f}")
    
    return clf, calibrator


def tune_alert_thresholds(
    calibrated_probs, y_val, meta_val, target_budget=5.0
):
    """
    Select alert thresholds T_MEDIUM and T_HIGH on validation seasons to satisfy false-alert budget.
    Target: at most 5 false alerts per 100 ward-weeks.
    
    Inputs:
        calibrated_probs (np.ndarray): Calibrated validation probabilities.
        y_val (np.ndarray): True binary target flags.
        meta_val (pd.DataFrame): Validation metadata containing ['season_id', 'ward_id', 'date'].
        target_budget (float): Maximum allowable false alerts per 100 ward-weeks (default: 5.0).
    Outputs:
        dict: Mapping {"t_medium": float, "t_high": float, "validation_metrics": dict}.
    """
    # Total ward-weeks observed in validation set
    total_ward_days = len(y_val)
    total_ward_weeks = total_ward_days / 7.0
    rate_factor = 100.0 / max(total_ward_weeks, 1e-3)
    
    # Sweep candidates for T_HIGH from high to low across calibrated probability distribution
    max_c = float(np.percentile(calibrated_probs, 99.9))
    candidates_high = np.linspace(max_c, 0.02, 60)
    best_t_high = round(max_c, 2)
    for t in candidates_high:
        pred_high = (calibrated_probs >= t)
        false_positives = np.sum(pred_high & (y_val == 0))
        fa_per_100_ww = false_positives * rate_factor
        if fa_per_100_ww <= target_budget:
            best_t_high = round(float(t), 2)
        else:
            break
            
    # Select T_MEDIUM strictly below T_HIGH
    best_t_med = round(max(0.02, best_t_high * 0.65), 2)
    
    # Compute Precision & PR-AUC on validation
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, calibrated_probs)
    pr_auc = float(auc(recall_vals, precision_vals))
    brier = float(brier_score_loss(y_val, calibrated_probs))
    
    val_metrics = {
        "t_medium": best_t_med,
        "t_high": best_t_high,
        "target_false_alert_budget": target_budget,
        "validation_pr_auc": round(pr_auc, 4),
        "validation_brier_score": round(brier, 4),
        "validation_ward_weeks": round(total_ward_weeks, 1)
    }
    print(f"Threshold Optimization on Validation: T_MEDIUM={best_t_med}, T_HIGH={best_t_high} (Budget <= {target_budget} FA/100ww)")
    return val_metrics


def run_training_pipeline(config_path="config/config.yaml"):
    """
    Execute full training, calibration, threshold tuning, and ONNX export.
    
    Inputs:
        config_path (str): Path to system config.yaml.
    Outputs:
        dict: Training summary and artifact paths.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    paths = cfg.get("data", {}).get("paths", {})
    manifest_path = paths.get("manifest_csv", "data/processed/manifest.csv")
    events_dir = paths.get("events_dir", "data/processed/events")
    gt_dir = paths.get("ground_truth_dir", "data/processed/ground_truth")
    artifacts_dir = paths.get("artifacts_dir", "model/artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    base_seed = int(cfg.get("system", {}).get("base_seed", 42))
    
    # 1. Load Training and Validation Datasets
    print("Loading training seasons...")
    X_train, y_train, meta_train = load_split_dataset(manifest_path, "train", events_dir, gt_dir)
    print(f"Loaded train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    print("Loading validation seasons...")
    X_val, y_val, meta_val = load_split_dataset(manifest_path, "val", events_dir, gt_dir)
    print(f"Loaded validation set: {X_val.shape[0]} samples")
    
    # 2. Train XGBoost and Calibrate
    model, calibrator = train_calibrated_xgboost(
        X_train, y_train, X_val, y_val, random_state=base_seed
    )
    
    # 3. Tune Thresholds on Validation
    raw_val_probs = model.predict_proba(X_val)[:, 1]
    cal_val_probs = calibrator.predict(raw_val_probs)
    thresholds_info = tune_alert_thresholds(cal_val_probs, y_val, meta_val, target_budget=5.0)
    
    # 4. Save Native Artifacts
    model_path = os.path.join(artifacts_dir, "model.joblib")
    calibrator_path = os.path.join(artifacts_dir, "calibrator.joblib")
    thresholds_path = os.path.join(artifacts_dir, "thresholds.json")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    features_path = os.path.join(artifacts_dir, "features.json")
    
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump({
            "t_medium": thresholds_info["t_medium"],
            "t_high": thresholds_info["t_high"]
        }, f, indent=2)
        
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_info, f, indent=2)
        
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
        
    print(f"Saved artifacts to {artifacts_dir}: model.joblib, calibrator.joblib, thresholds.json")
    
    # 5. Export to ONNX and verify parity
    onnx_path = os.path.join(artifacts_dir, "model.onnx")
    export_xgboost_to_onnx(model, X_val[:1], output_path=onnx_path)
    
    passed, max_diff = check_onnx_parity(model, onnx_path, X_val[:500], tolerance=1e-4)
    print(f"ONNX Export Parity Verification: max diff = {max_diff:.6e} (Passed: {passed})")
    if not passed:
        raise RuntimeError(f"ONNX Parity Check Failed! Discrepancy {max_diff} exceeds 1e-4 tolerance.")
        
    return {
        "model_path": model_path,
        "onnx_path": onnx_path,
        "calibrator_path": calibrator_path,
        "thresholds_path": thresholds_path,
        "onnx_parity_passed": passed
    }


if __name__ == "__main__":
    run_training_pipeline()
