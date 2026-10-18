#!/usr/bin/env bash
# scripts/smoke_test.sh - Hour-0 environment verification script
# BUILD.md Section 8 and Section 12

set -e

echo "=================================================="
echo "DENGUE SENTINEL - HOUR-0 ENVIRONMENT SMOKE TEST"
echo "=================================================="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

echo "--- 1. System Resources ---"
python -c "
import os, sys, platform, psutil
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
print(f'Byte order: {sys.byteorder}')
print(f'CPU count: {os.cpu_count()}')
mem = psutil.virtual_memory()
print(f'Total Memory: {mem.total / (1024**3):.2f} GB')
print(f'Available Memory: {mem.available / (1024**3):.2f} GB')
disk = psutil.disk_usage('.')
print(f'Disk Free: {disk.free / (1024**3):.2f} GB')
" || echo "System resource check failed via psutil"

echo ""
echo "--- 2. Runtime Versions ---"
python --version 2>&1 || echo "Python not found"
java -version 2>&1 | head -n 2 || echo "Java not found (Optional if using Docker or --no-kafka mode)"

echo ""
echo "--- 3. Required Library Imports ---"
python -c "
libs = ['numpy', 'pandas', 'sklearn', 'xgboost', 'onnxruntime', 'onnxmltools', 'kafka', 'yaml', 'psutil', 'streamlit']
failed = []
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, '__version__', 'installed')
        print(f'  [OK] {lib}: {ver}')
    except ImportError as e:
        print(f'  [FAIL] {lib}: {e}')
        failed.append(lib)

if failed:
    print(f'\nFAILED LIBRARIES: {failed}')
    sys.exit(1)
else:
    print('\nAll core dependencies imported successfully.')
"

echo ""
echo "--- 4. Tiny ONNX Parity Test (Tolerance 1e-4) ---"
python -c "
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt

# Generate tiny synthetic matrix
X, y = make_classification(n_samples=50, n_features=6, random_state=42)
clf = XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
clf.fit(X, y)
prob_native = clf.predict_proba(X)[:, 1]

# Export to ONNX via onnxmltools
initial_types = [('input', FloatTensorType([None, 6]))]
onx = convert_xgboost(clf, initial_types=initial_types)
sess = rt.InferenceSession(onx.SerializeToString(), providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[1].name
prob_onnx = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
if isinstance(prob_onnx, list) and isinstance(prob_onnx[0], dict):
    prob_onnx_arr = np.array([p[1] for p in prob_onnx])
elif isinstance(prob_onnx, np.ndarray):
    prob_onnx_arr = prob_onnx[:, 1] if prob_onnx.ndim > 1 else prob_onnx
else:
    prob_onnx_arr = np.array(prob_onnx)

max_diff = np.max(np.abs(prob_native - prob_onnx_arr))
print(f'Max probability discrepancy between Native and ONNX: {max_diff:.6e}')
assert max_diff < 1e-4, f'Discrepancy {max_diff} exceeded tolerance 1e-4'
print('  [OK] ONNX parity check passed within 1e-4 tolerance.')
"

echo ""
echo "--- 5. Kafka Connectivity Check ---"
python -c "
import socket
from kafka import KafkaProducer

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect(('localhost', 9092))
    s.close()
    producer = KafkaProducer(bootstrap_servers='localhost:9092', max_block_ms=1000)
    print('  [OK] Kafka broker reachable and responsive on localhost:9092')
    producer.close()
except Exception as e:
    print(f'  [NOTE] Kafka broker not currently active ({e}).')
    print('         Will use --no-kafka mode or start via docker-compose / tarball.')
"

echo ""
echo "=================================================="
echo "SMOKE TEST COMPLETED SUCCESSFULLY"
echo "=================================================="
