#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=ng
#SBATCH --account=luuni
#SBATCH --job-name=lf96_wrf12_afterda_18
#SBATCH --output=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/logs/wrf12_afterda_18utc_%j.log

set -euo pipefail
module purge
module load prgenv/gnu gcc/8.5.0 cuda/11.6 netcdf4/4.9.2 cmake/3.28.3
mkdir -p "/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/wrf12_afterda_18utc/results"
"/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_fp_8_2/build_patched_backup_v2/lisflood" "/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/wrf12_afterda_18utc.par"
