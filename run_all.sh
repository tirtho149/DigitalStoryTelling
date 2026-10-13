#!/usr/bin/env bash
# Reproduce every number and figure in the paper from the raw data.
# Run from the repository root:   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/code"

echo "==> [1/5] Split survey into per-condition raw scores"
python3 make_raw_scores.py

echo "==> [2/5] Mann-Whitney U tests  (paper Table 2 & Table 3)"
python3 mann_whitney_analysis.py

echo "==> [3/5] Experiment figures      (paper Fig 2, 3, 5, 6, 7)"
python3 generate_figures.py

echo "==> [4/5] Country-ranking survey  (supplementary OLS/OLR)"
python3 country_ranking.py

echo "==> [5/5] Video A/B survey         (supplementary)"
python3 video_ab.py

echo
echo "NOTE: The Random Forest (paper Fig 4) is heavy (1000 iterations x 2 datasets,"
echo "      ~9 h on CPU). Do NOT run it on a login node. Submit it to Slurm:"
echo "          cd code && python3 random_forest.py"
echo "      Pre-computed reference results are in results/rf_*_FINAL.csv and figures/random_forest/."
echo
echo "Done. Figures -> figures/   Stats -> results/"
