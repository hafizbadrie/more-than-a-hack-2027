#!/usr/bin/env bash
# Regenerate the submission file from the data bundle.   Usage:  bash run_all.sh
# Run from the smart-tags folder, inside the challenge repository, with the data bundle in place.
set -euo pipefail
cd "$(dirname "$0")"

TEAM="${TEAM:-Merah Putih}"                     # the registered team name; the form names the board row after this file
TRACKING=../data/tactical/BAR-ATH_tactical_ds.csv
TAGS=../data/tactical/BAR-ATH_smart_tagging_ds.csv

# 0. data bundle -> two CSVs
python prepare_data.py

# 1. possession file.  Every threshold below was derived from the full-match tracking and tags, never from the
#    public ground truth: see TECHNICAL.md ("how the thresholds were set") and restart_offset.py,
#    speed_threshold_calibration.py.
python combined_submission.py "$TRACKING" "$TAGS" -o "${TEAM}.csv" \
  --window 50 --clear-m 3.0 --post-window 50 --min-run-s 3.0 --slow-ms 0.7 \
  --sp-cutoff "corner kick=5.6,free kick=5.2,throw-in=5.8,kick off (after goal)=4.0"

# 2. format check (the official scorer's own validation)
python ../evaluation/validate_submission.py "${TEAM}.csv"
