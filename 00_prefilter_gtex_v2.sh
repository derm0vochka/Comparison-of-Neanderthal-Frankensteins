#!/bin/bash
# Скрипт 00: Предварительная фильтрация GTEx eQTL по хромосомам
# Запускается ОДИН РАЗ перед параллельной обработкой хромосом.
# Результат: data/gtex/by_chr/chr{1..22}_eqtl_raw.tsv

set -euo pipefail

WORKDIR="$HOME/nd_pipeline"
GTEX_DIR="$WORKDIR/data/gtex/GTEx_Analysis_v7_eQTL"
OUT_DIR="$WORKDIR/data/gtex/by_chr"

mkdir -p "$OUT_DIR"

echo "[$(date)] Предобработка GTEx: разбивка по хромосомам"

python3 << 'PYEOF'
import gzip
import os
import sys
import time
import glob

WORKDIR = os.path.expanduser("~/nd_pipeline")
GTEX_DIR = f"{WORKDIR}/data/gtex/GTEx_Analysis_v7_eQTL"
OUT_DIR = f"{WORKDIR}/data/gtex/by_chr"

os.makedirs(OUT_DIR, exist_ok=True)

# Находим файлы
eqtl_files = sorted(glob.glob(f"{GTEX_DIR}/*.signif_variant_gene_pairs.txt.gz"))
if not eqtl_files:
    eqtl_files = sorted(glob.glob(f"{GTEX_DIR}/*.signif_variant_gene_pairs.txt"))
print(f"  Найдено файлов GTEx: {len(eqtl_files)}")

if not eqtl_files:
    print("  ОШИБКА: GTEx файлы не найдены!")
    sys.exit(1)

# Открываем 22 выходных файла
chr_files = {}
header_written = False
header_line = None

for c in range(1, 23):
    chr_files[str(c)] = open(f"{OUT_DIR}/chr{c}_eqtl_raw.tsv", "w")

# Счётчики
chr_counts = {str(c): 0 for c in range(1, 23)}
total_read = 0
start_time = time.time()

for file_idx, fpath in enumerate(eqtl_files):
    tissue = os.path.basename(fpath).replace(".v7.signif_variant_gene_pairs.txt.gz", "") \
                                    .replace(".v7.signif_variant_gene_pairs.txt", "")

    is_gz = fpath.endswith(".gz")
    fh = gzip.open(fpath, "rt") if is_gz else open(fpath, "r")

    for line_idx, line in enumerate(fh):
        line = line.rstrip("\n")

        # Заголовок — первая строка первого файла
        if line_idx == 0:
            if not header_written:
                header_line = line + "\ttissue"
                for c in range(1, 23):
                    chr_files[str(c)].write(header_line + "\n")
                header_written = True
            continue

        total_read += 1

        # Извлекаем хромосому из variant_id (первая колонка до первого _)
        tab_pos = line.find("\t")
        variant_id = line[:tab_pos] if tab_pos > 0 else line
        under_pos = variant_id.find("_")
        chr_raw = variant_id[:under_pos] if under_pos > 0 else variant_id

        # Убираем "chr" если есть
        if chr_raw.startswith("chr"):
            chr_num = chr_raw[3:]
        else:
            chr_num = chr_raw

        # Записываем в соответствующий файл
        if chr_num in chr_files:
            chr_files[chr_num].write(line + "\t" + tissue + "\n")
            chr_counts[chr_num] += 1

    fh.close()

    elapsed = time.time() - start_time
    print(f"  [{file_idx+1}/{len(eqtl_files)}] {tissue}: "
          f"прочитано строк={total_read}, время={elapsed:.0f}с")
    sys.stdout.flush()

# Закрываем файлы
for c in range(1, 23):
    chr_files[str(c)].close()

# Итоги
print(f"\n  Всего прочитано: {total_read}")
print(f"  Время: {time.time() - start_time:.0f} секунд")
print(f"\n  Результаты:")
for c in range(1, 23):
    fpath = f"{OUT_DIR}/chr{c}_eqtl_raw.tsv"
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"    chr{c}: {chr_counts[str(c)]} eQTL ({size_mb:.1f} МБ)")

print(f"\n  Файлы сохранены в: {OUT_DIR}/")
PYEOF

echo "[$(date)] Предобработка GTEx завершена"
