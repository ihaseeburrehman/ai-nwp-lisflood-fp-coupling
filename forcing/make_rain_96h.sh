#!/bin/bash
#SBATCH --job-name=gc_rain96
#SBATCH --qos=nf --account=luuni
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G --time=04:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/logs/make_rain_96h_%j.log
set -euo pipefail
R=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected
G=/ec/res4/scratch/lux0804/ai_rerun_2026/v2_tisr_fix/output/2021_event/graphcast
/usr/local/apps/ai-models/0.49/env-jax/bin/python $R/ai_to_lisflood_rain_96h.py \
    graphcast "$G" "$R/inputs/Alzette_sub_basin_10m_bridge_burn.asc" \
    "$R/inputs/rain_graphcast_v2_96h.nc"
