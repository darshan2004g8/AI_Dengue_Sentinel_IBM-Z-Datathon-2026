# Dengue Sentinel: One-Command Runner (PowerShell)
# Event Code: SSF2026 (IBM Z Datathon 2026)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  DENGUE SENTINEL: MASTER ONE-COMMAND ORCHESTRATOR" -ForegroundColor Cyan
Write-Host "  Event Code: SSF2026 | IBM Z Datathon 2026" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

python run_all.py $args
