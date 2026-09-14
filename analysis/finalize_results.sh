#!/bin/bash
# finalize_results.sh
# Reproduce every number and figure reported in the manuscript, from the
# LISFLOOD-FP output grids to the compiled PDF. Run from anywhere.
#
#   bash analysis/scripts/finalize_results.sh
#
# Inputs (must already be present):
#   analysis/data/lisflood_96h/runs/<stem>/results/6hr-*.{Qx,Qy,wd}  LISFLOOD-FP output
#   analysis/data/lisflood_96h/rain/*.nc                             forcing rainfall
#
# Stages:
#   1  extract_discharge_line_integral.py  ->  analysis/data/lisflood_96h/line_integral/*_station_Q.csv
#   1b extract_efas_discharge.py          ->  analysis/data/lisflood_96h/efas/efas_station_Q.csv
#   2  catchment_hyetograph.py             ->  analysis/data/pgfplots/hyetograph.csv, forcing_intensity.csv  (Table 1)
#   3  basin_mean_rainfall.py              ->  basin- vs domain-mean rainfall  (Table 1 caption)
#   4  plot_and_stats.py                   ->  analysis/data/pgfplots/*_merged.csv, averaged_metrics.csv,
#                                              per-gauge metrics printed to stdout, Taylor diagram
#   5  pdflatex + bibtex                   ->  WRF_and_AI_models_LISFLOOD.pdf
#
# The bar-chart values in the manuscript are inlined in the .tex; they must match
# analysis/data/pgfplots/averaged_metrics.csv, which stage 4 rewrites.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ANALYSIS_DIR="$(dirname "$SCRIPT_DIR")"
PY="${PY:-python3}"
TEXBIN="${TEXBIN:-$HOME/texlive/2022/bin/universal-darwin}"

echo "=== 1. Discharge and depth at each gauge (line integral, surveyed widths) ==="
"$PY" "$SCRIPT_DIR/extract_discharge_line_integral.py"

echo "=== 1b. EFAS benchmark discharge over the same 96 h window ==="
"$PY" "$SCRIPT_DIR/extract_efas_discharge.py" --verify

echo "=== 2. Catchment-mean rainfall diagnostics (Table 1) ==="
"$PY" "$SCRIPT_DIR/catchment_hyetograph.py"

echo "=== 3. Basin-mean vs domain-mean rainfall (Table 1 caption) ==="
"$PY" "$SCRIPT_DIR/basin_mean_rainfall.py"

echo "=== 4. Metrics, merged CSVs and figures ==="
"$PY" "$SCRIPT_DIR/plot_and_stats.py"

echo "=== 5. Compiling the manuscript ==="
cd "$PAPER_DIR/manuscript"
export PATH="$TEXBIN:$PATH"
pdflatex -interaction=nonstopmode WRF_and_AI_models_LISFLOOD.tex >/dev/null
bibtex WRF_and_AI_models_LISFLOOD >/dev/null || true
pdflatex -interaction=nonstopmode WRF_and_AI_models_LISFLOOD.tex >/dev/null
pdflatex -interaction=nonstopmode WRF_and_AI_models_LISFLOOD.tex >/dev/null

echo "=== Done. WRF_and_AI_models_LISFLOOD.pdf updated. ==="
