#!/bin/bash
#SBATCH --job-name=nd_visualize
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=%HOME/nd_pipeline/logs/slurm_visualize_%j.log
#SBATCH --error=%HOME/nd_pipeline/logs/slurm_visualize_%j.err

echo "[$(date)] Запуск визуализации результатов"
mkdir -p "$HOME/nd_pipeline/logs"

# Проверяем результаты анализа
if [ ! -f "$HOME/nd_pipeline/results/analysis/isfs.tsv" ]; then
    echo "Результаты анализа не найдены"
    echo "Сначала sbatch run_03_analysis.sh"
    exit 1
fi
python3 "$HOME/nd_pipeline/scripts/04_visualize.py"
echo "[$(date)] Визуализация завершена"
echo "Рисунки: $HOME/nd_pipeline/results/figures/"
ls -lh "$HOME/nd_pipeline/results/figures/"
