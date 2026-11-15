#!/usr/bin/env bash
# Run the HMM + hybrid pipeline, copy the result to the evaluation folder, evaluate it, open the scoreboard in VS Code.
# Run from the data-pipeline/ folder (the one with main.py):   bash run_hybrid_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

# python main.py
python main.py --pipeline hmm
mkdir -p ../evaluation/results
cp output/hybrid_results.csv ../evaluation/results/hybrid_results.csv
python ../evaluation/run_evaluation.py --bootstrap 1000
python visualization/plot_overlap.py --pred output/hybrid_results.csv --gt ../evaluation/gt/BAR-ATH_possession_gt_public.csv --tags output/BAR-ATH_tactical_smart_tagging.csv --out output/overlap.png --html output/overlap.html

SCOREBOARD="../evaluation/scoreboard.csv"
if [ ! -f "$SCOREBOARD" ]; then
  echo "scoreboard not found: $SCOREBOARD" >&2
  exit 1
fi
if command -v code >/dev/null 2>&1; then
  code "$SCOREBOARD"            # open in VS Code
else
  echo "'code' command not found (VS Code: Cmd+Shift+P > 'Shell Command: Install code command in PATH')"
  echo "the scoreboard is at $SCOREBOARD"
fi