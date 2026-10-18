"""
model/export_onnx.py
ONNX model conversion and numerical parity validation.
Converts the native XGBoost model to model.onnx and verifies predictions match within 1e-4 tolerance.
Adheres strictly to BUILD.md Section 4 and Section 9.2.
"""

import os
import numpy as np
import joblib
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt


def export_xgboost_to_onnx(model, sample_input, output_path="model/artifacts/model.onnx"):
    """
    Export trained XGBoost classifier to ONNX graph format using onnxmltools.
    
    Inputs:
        model: Trained XGBoost classifier.
        sample_input (np.ndarray): 2D sample feature matrix to define input schema.
        output_path (str): Destination path for serialized ONNX graph.
    Outputs:
        str: Absolute path to the exported ONNX file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_features = sample_input.shape[1]
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onx = convert_xgboost(model, initial_types=initial_types)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    return output_path


def check_onnx_parity(native_model, onnx_path, test_matrix, tolerance=1e-4):
    """
    Verify numerical parity between native XGBoost predictions and ONNX Runtime evaluation.
    
    Inputs:
        native_model: Fitted native XGBoost model.
        onnx_path (str): Path to exported .onnx model file.
        test_matrix (np.ndarray): Held-out evaluation feature matrix.
        tolerance (float): Maximum allowable absolute discrepancy (default 1e-4).
    Outputs:
        tuple: (passed (bool), max_diff (float))
    """
    prob_native = native_model.predict_proba(test_matrix)[:, 1]
    
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name
    prob_onnx_raw = sess.run([label_name], {input_name: test_matrix.astype(np.float32)})[0]
    
    if isinstance(prob_onnx_raw, list) and isinstance(prob_onnx_raw[0], dict):
        prob_onnx = np.array([p[1] for p in prob_onnx_raw])
    elif isinstance(prob_onnx_raw, np.ndarray):
        prob_onnx = prob_onnx_raw[:, 1] if prob_onnx_raw.ndim > 1 else prob_onnx_raw
    else:
        prob_onnx = np.array(prob_onnx_raw)
        
    diff = float(np.max(np.abs(prob_native - prob_onnx)))
    passed = diff <= tolerance
    return passed, diff


def main():
    """Execute ONNX conversion and verify parity on artifacts."""
    print("Dengue Sentinel ONNX Export & Parity Validator (Phase 0 Scaffold)")


if __name__ == "__main__":
    main()
