#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --qos=ng
#SBATCH --job-name=lisflood_wrf12
#SBATCH --output=/scratch/lux0804/AI_flood_2021/runs/wrf12/slurm_%j.out
#SBATCH --error=/scratch/lux0804/AI_flood_2021/runs/wrf12/slurm_%j.err
set -e
cd /scratch/lux0804/AI_flood_2021/runs/wrf12
module load gcc/8.5.0 cuda/11.6 netcdf4/4.9.2
echo === LISFLOOD-FP wrf12 on ac6-100.bullx ===
nvidia-smi -L
time /scratch/lux0804/AI_flood_2021/lisflood wrf12.par
echo === DONE wrf12 ===
