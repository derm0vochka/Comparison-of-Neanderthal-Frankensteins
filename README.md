# Пайплайн eQTL-влияние неандертальской интрогрессии

## пару слов про DAIseg и получение файлов IBS.YRI.grch37.chrN.em.tsv

## Структура файлов
```
~/nd_pipeline/                        # Рабочая директория пайплайна
├── data/
│   ├── raw/                          # IBS.YRI...
│   ├── hapmap/                       # Карты рекомбинации HapMap II
│   ├── gtex/                         # GTEx v7 cis-eQTL
│   ├── vindija/                      # Vindija 33.19 VCF
│   └── gencode/                      # GENCODE v19 GTF
├── scripts/                          # Скрипты пайплайна
│   ├── 01_pipeline_A_preprocess_universal_v2.sh
│   ├── 02_pipeline_B_subpop_vindija_universal_v2.sh
│   ├── 03_main_analysis_universal_v2.py
│   └── 04_visualize_universal_v2.py
├── results/
│   ├── chr1/
│   │   ├── pipeline_A/
│   │   ├── pipeline_B/
│   │   ├── analysis/
│   │   ├── figures/
│   │   └── logs/
│   ├── chr2/
│   └── ...
└── logs/                             # Общие логи
```
## 0. Подготовка

### Создание структуры директории
```bash
mkdir -p ~/nd_pipeline/{data/{raw,hapmap,gtex,vindija,gencode},scripts,results,logs}
```
### Копирование tsv-файла на сервер

## 1. Скачивание данных

```bash
cd ~/nd_pipeline
```
### Выполнить все, что указано в 00_download_data.sh

## Установить права на выполнение

```bash
cd ~/nd_pipeline
chmod +x scripts/*_universal_v2.sh
chmod +x scripts/*_universal_v2.py
chmod +x run_chromosome_v2.sh
chmod +x run_all_chromosomes_v2.sh
```
## тестовый запуск
```bash
cd ~/nd_pipeline
CHR=6 bash run_chromosome_v2.sh 2>&1 | tee results/chr6/logs/chr6_test.log
```

## 2. Pipeline A (препроцессинг)

```bash
cd ~/nd_pipeline
sbatch run_01_pipeline_A.sh
tail -f logs/slurm_pipelineA_<job_id>.log
# Или
bash run_01_pipeline_A.sh 2>&1 | tee pipeline_A.log
```

```
Подготовка данных для основного анализа 
1. Нарезаем chr6 на окна 1000 bp
2. Вычисляем глобальную частоту интрогрессии Fw для каждого окна
   - Индикатор I=1 если гаплотип покрывает окно >= 50% (>= 500 bp)
   - Fw=(число гаплотипов с I=1) / H
3. Присваиваем бины частот: Zero / Rare / Low / Intermediate / High / Very_High
4. Фильтруем GTEx v7 eQTL: только chr6, FDR < 0.05, биаллельные SNP
5. Пересекаем eQTL с окнами
6. Proxy-clumping: для каждой тройки (окно, ткань, ген) -> lead SNP с max|Z|
7. Агрегируем Sw,t = max_g |Z_lead|
8. Извлекаем TSS из GENCODE v19, вычисляем D_TSS
9. Категоризируем окна: Promoter (<5 kb) / Near (5–50 kb) / Distal (>=50 kb)
10. Вычисляем среднюю скорость рекомбинации из HapMap II
Итоговый файл: results/pipeline_A/chr6_windows_full.tsv
```

## 3. Pipeline B (субпопуляции и поляризация Vindija)

```bash
cd ~/nd_pipeline
sbatch run_02_pipeline_B.sh
tail -f logs/slurm_pipelineB_<job_id>.log
# Или
bash run_02_pipeline_B.sh 2>&1 | tee pipeline_B.log
```

```
Подготовка данных для cпецанализа
1. Группируем гаплотипы по субпопуляциям
2. Вычисляем субпопуляционные частоты F^(k)_w для каждой популяции
3. Пересекаем eQTL с Vindija 33.19 VCF
4. Hard Filtering Vindija:
   - GQ > 30 (качество генотипирования)
   - DP > 10 (глубина покрытия)
   - Исключает транзиции C/T и G/A 
5. Определяем знак D_w,t = sign(beta_neand):
   - +1 если неандертальский аллель повышает экспрессию
   - −1 если понижает
   - NA если аллель неизвестен
Итоговый файл: results/pipeline_B/chr6_windows_final.tsv
```

## 4. Основной анализ

```bash
cd ~/nd_pipeline
sbatch run_03_analysis.sh
tail -f logs/slurm_analysis_<job_id>.log
# Или
bash run_03_analysis.sh
```

```
2.1 Спектр частот (iSFS)
- Гистограмма распределения окон по бинам частот
- Manhattan plot: позиция*Fw
2.2 Двухэтапная модель (очищающий отбор)
Логистическая регрессия
logit(P(Sw > 0)) ~ Fw + log(1 + D_TSS) + log(recomb)
Линейная регрессия (только окна с eQTL)
Sw ~ Fw + log(1 + D_TSS) + log(recomb)
2.3 Валидация: Block Bootstrap Mann-Whitney
- Сравнение Sw: интрогрессия (Fw > 0) и контроль (Fw = 0)
- Отдельно для Promoter / Near / Distal
- 10 000 итераций block bootstrap
- Блок = целый NIS-сегмент (для интрогрессии) или геномный участок медианной длины (для контроля)
2.4 Адаптивная интрогрессия
- Кандидаты: Fw > 95 перцентиль, Sw > 95 перцентиль
- Scatter plot Fw * Sw с генами-мишенями
```

## 5. Визуализация
```bash
cd ~/nd_pipeline
sbatch run_04_visualize.sh
# или
bash run_04_visualize.sh
```
```
fig1_iSFS.png/pdf - спектр частот интрогрессии (barplot, log-шкала)
fig2_manhattan.png/pdf - Manhattan plot Fw (chr6)
fig3_violin_boxplot.png/pdf - Violin + Boxplot: Fw_bin и Sw 
fig4_split_violin.png/pdf - Split Violin: интрогрессия и  контроль по D_TSS
fig5_scatter_adaptive.png/pdf - Scatter Fw и Sw
fig6_summary_panel.png/pdf - все результаты
```

## GNU parallel

```bash
which parallel
```
Если не установлен:
```bash
sudo apt-get install parallel
```

## Запустить параллельную обработку

```bash
cd ~/nd_pipeline
bash run_all_chromosomes_v2.sh 2>&1 | tee results/logs/all_chromosomes.log
```


## Теперь можно скачать картинки локально

## TODO
