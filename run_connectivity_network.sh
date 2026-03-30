#!/bin/bash
source /home/edm9fd/sandbox/source-analytics/.venv/bin/activate

echo "=== Starting vertex_connectivity at $(date) ===" >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1
source-analytics run --study /mnt/d/research/EEG/FORGE/analysis_wholebrain.yaml --analysis vertex_connectivity >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1
echo "=== vertex_connectivity finished at $(date) ===" >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1

echo "=== Starting network at $(date) ===" >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1
source-analytics run --study /mnt/d/research/EEG/FORGE/analysis_wholebrain.yaml --analysis network >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1
echo "=== network finished at $(date) ===" >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1

echo "=== ALL DONE at $(date) ===" >> /home/edm9fd/sandbox/source-analytics/bg_analyses.log 2>&1
