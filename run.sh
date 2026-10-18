#!/usr/bin/env bash
# Dengue Sentinel: One-Command Runner (Bash / Linux / macOS / IBM LinuxONE s390x)
# Event Code: SSF2026 (IBM Z Datathon 2026)

set -e

echo "========================================================================"
echo "  DENGUE SENTINEL: MASTER ONE-COMMAND ORCHESTRATOR"
echo "  Event Code: SSF2026 | IBM Z Datathon 2026 | IBM LinuxONE (s390x)"
echo "========================================================================"

PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_BIN="python"
fi

$PYTHON_BIN run_all.py "$@"
