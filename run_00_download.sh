#!/bin/bash
#SBATCH --job-name=nd_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=%HOME/nd_pipeline/logs/slurm_download_%j.log
#SBATCH --error=%HOME/nd_pipeline/logs/slurm_download_%j.err

echo "[$(date)] Запуск: скачивание данных"
mkdir -p "$HOME/nd_pipeline/logs"
bash "$HOME/nd_pipeline/scripts/00_download_data.sh"
echo "[$(date)] Скачивание завершено"
