# Dengue Sentinel: PRD and Architecture

Refer to [BUILD.md](../BUILD.md) for the active build contract.

## 1. Executive Summary
Dengue Sentinel is a real-time early-warning decision support system designed for Indian municipal authorities (piloted on Bengaluru). It detects emerging dengue outbreaks at ward level up to 14 days before official delayed case surge detection, ranks wards by outbreak risk, and optimizes vector control team allocations (inspection and targeted fogging).

## 2. Key Architecture Pillars
1. **Detect**: Streaming real weather + synthetic clinical fever and pharmacy fever medication sales per ward.
2. **Decide**: Vectorized batch scoring per date using an Isotonic-Calibrated XGBoost model exported to ONNX, accompanied by baseline detectors (Persistence and rolling Z-score).
3. **Respond**: Exposure-weighted greedy resource allocator assigning municipal inspection and fogging teams under strict operational capacity constraints.

## 3. IBM Z / LinuxONE Relevance
Ingestion, batch scoring, and latency-benchmarked streaming pipelines run on LinuxONE VM (s390x architecture), demonstrating high-throughput, low-latency, deterministic AI execution without external proprietary cloud services.
