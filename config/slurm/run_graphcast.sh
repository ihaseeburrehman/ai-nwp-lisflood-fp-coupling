#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=ng
#SBATCH --account=luuni
#SBATCH --job-name=lf96_graphcast_18
#SBATCH --output=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/logs/graphcast_corrected_18utc_%j.log

set -euo pipefail
module purge
module load prgenv/gnu gcc/8.5.0 cuda/11.6 netcdf4/4.9.2 cmake/3.28.3
ROOT=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected
BIN=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_fp_8_2/build_patched_backup_v2/lisflood
mkdir -p "${ROOT}/graphcast_corrected_18utc/results" "${ROOT}/logs"
"${BIN}" "${ROOT}/graphcast_corrected_18utc.par"
