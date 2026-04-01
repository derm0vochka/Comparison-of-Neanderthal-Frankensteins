#!/bin/bash
# Скрипт для запуска полного пайплайна для одной хромосомы
# CHR=N bash run_chromosome_v2.sh

set -euo pipefail

# Переменные окружения
CHR=${CHR:-6}
WORKDIR="$HOME/nd_pipeline"
SCRIPTS_DIR="$WORKDIR/scripts"
CHR_RESULTS_DIR="$WORKDIR/results/chr${CHR}"

echo "=========================================="
echo "Запуск пайплайна для chr${CHR}"
echo "=========================================="
echo "Рабочая директория: $WORKDIR"
echo "Директория результатов: $CHR_RESULTS_DIR"
echo "=========================================="

# Создаём директории
mkdir -p "$CHR_RESULTS_DIR/logs"

echo ""
echo "Pipeline A: Препроцессинг"
CHR=${CHR} bash "${SCRIPTS_DIR}/01_pipeline_A_preprocess_universal_v2.sh" 2>&1 | tee "${CHR_RESULTS_DIR}/logs/pipeline_A_chr${CHR}.log"

# Проверка успешности
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ОШИБКА: Pipeline A завершился с ошибкой!"
    exit 1
fi

echo "Pipeline A завершён успешно"

echo ""
echo "Pipeline B: Субпопуляции + поляризация Vindija"
CHR=${CHR} bash "${SCRIPTS_DIR}/02_pipeline_B_subpop_vindija_universal_v2.sh" 2>&1 | tee "${CHR_RESULTS_DIR}/logs/pipeline_B_chr${CHR}.log"

# Проверка успешности
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ОШИБКА: Pipeline B завершился с ошибкой!"
    exit 1
fi

echo "Pipeline B завершён успешно"

echo ""
echo "Pipeline 03: Основной анализ"
CHR=${CHR} python3 "${SCRIPTS_DIR}/03_main_analysis_universal_v2.py" 2>&1 | tee "${CHR_RESULTS_DIR}/logs/pipeline_03_chr${CHR}.log"

# Проверка успешности
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ОШИБКА: Pipeline 03 завершился с ошибкой!"
    exit 1
fi

echo "Pipeline 03 завершён успешно"

echo ""
echo "Pipeline 04: Визуализация"
CHR=${CHR} python3 "${SCRIPTS_DIR}/04_visualize_universal_v2.py" 2>&1 | tee "${CHR_RESULTS_DIR}/logs/pipeline_04_chr${CHR}.log"

# Проверка успешности
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ОШИБКА: Pipeline 04 завершился с ошибкой!"
    exit 1
fi

echo "Pipeline 04 завершён успешно"

echo ""
echo "Пайплайн для chr${CHR} завершён успешно!"
echo "Результаты сохранены в: $CHR_RESULTS_DIR"
echo "Логи сохранены в: $CHR_RESULTS_DIR/logs"
