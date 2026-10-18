"""
tests/test_onnx_parity.py
Unit tests verifying numerical parity between native XGBoost classifier and exported ONNX model.
Adheres to BUILD.md Section 5 and Section 9.2 (tolerance <= 1e-4).
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt


class TestONNXParity(unittest.TestCase):
    """
    Test suite verifying prediction equivalence between native XGBoost and ONNX Runtime.
    """

    def setUp(self):
        """Prepare synthetic dataset and train temporary test classifier."""
        self.X, self.y = make_classification(n_samples=100, n_features=8, random_state=42)
        self.clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        self.clf.fit(self.X, self.y)

    def test_onnx_runtime_parity(self):
        """Test that ONNX Runtime predictions match native probabilities within 1e-4."""
        prob_native = self.clf.predict_proba(self.X)[:, 1]

        # Convert to ONNX in-memory
        initial_types = [("input", FloatTensorType([None, self.X.shape[1]]))]
        onx = convert_xgboost(self.clf, initial_types=initial_types)
        sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name

        prob_onnx_raw = sess.run([label_name], {input_name: self.X.astype(np.float32)})[0]
        if isinstance(prob_onnx_raw, list) and isinstance(prob_onnx_raw[0], dict):
            prob_onnx = np.array([p[1] for p in prob_onnx_raw])
        elif isinstance(prob_onnx_raw, np.ndarray):
            prob_onnx = prob_onnx_raw[:, 1] if prob_onnx_raw.ndim > 1 else prob_onnx_raw
        else:
            prob_onnx = np.array(prob_onnx_raw)

        max_discrepancy = np.max(np.abs(prob_native - prob_onnx))
        self.assertLessEqual(max_discrepancy, 1e-4, f"Discrepancy {max_discrepancy} exceeded 1e-4")


if __name__ == "__main__":
    unittest.main()
