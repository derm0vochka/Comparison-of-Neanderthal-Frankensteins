#!/bin/bash
#SBATCH --job-name=nd_pipelineA
#SBATCH --cpus-per-task=30
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --output=%HOME/nd_pipeline/logs/slurm_pipelineA_%j.log
#SBATCH --error=%HOME/nd_pipeline/logs/slurm_pipelineA_%j.err

echo "[$(date)] Запуск Pipeline A"
mkdir -p "$HOME/nd_pipeline/logs"

NIS="$HOME/nd_pipeline/data/raw/IBS.YRI.grch37.chr6.em.tsv"
if [ ! -f "$NIS" ]; then
    echo "tsv-файл не найден: $NIS"
    echo "Скопируйте файл: $NIS"
    exit 1
fi
bash "$HOME/nd_pipeline/scripts/01_pipeline_A_preprocess.sh"
echo "[$(date)] Pipeline A завершен"
echo "Результаты: $HOME/nd_pipeline/results/pipeline_A/"
