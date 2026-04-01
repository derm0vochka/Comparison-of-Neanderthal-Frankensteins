#!/bin/bash
# Скрипт 02: Pipeline B — Субпопуляционные частоты + поляризация на Vindija (универсальный)
# python3 (pandas, numpy), bcftools, tabix
# CHR=6 bash 02_pipeline_B_subpop_vindija_universal_v2.sh

set -euo pipefail

# Переменные окружения
CHR=${CHR:-6}
WORKDIR="$HOME/nd_pipeline"
DATA="$WORKDIR/data"
RES="$WORKDIR/results"
CHR_RES="$RES/chr${CHR}"
NIS_DIR="$WORKDIR/data/raw"
LOG="$CHR_RES/logs/pipeline_B_chr${CHR}.log"

# Создаём директории
mkdir -p "$CHR_RES/pipeline_B"
mkdir -p "$CHR_RES/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Pipeline B: Старт (chr${CHR})"

NIS="${NIS_DIR}/IBS.YRI.grch37.chr${CHR}.em.tsv"
VINDIJA_VCF="$DATA/vindija/chr${CHR}_mq25_mapab100.vcf.gz"
OUT_A="$CHR_RES/pipeline_A"
OUT_B="$CHR_RES/pipeline_B"

echo "[$(date)] Шаг 1: Вычисление субпопуляционных частот..."

# 1a: Разбиваем NIS по субпопуляциям и создаём BED-файлы 
python3 << 'PYEOF'
import pandas as pd
import numpy as np
import os
import json

CHR = os.environ.get("CHR", "6")
WORKDIR = os.path.expanduser("~/nd_pipeline")
NIS_DIR = f"{WORKDIR}/data/raw"
OUT_A = f"{WORKDIR}/results/chr{CHR}/pipeline_A"
OUT_B = f"{WORKDIR}/results/chr{CHR}/pipeline_B"

# Читаем NIS
nis = pd.read_csv(f"{NIS_DIR}/IBS.YRI.grch37.chr{CHR}.em.tsv", sep=r"\s+")
nis["sample_id"] = nis["Sample"].str.rsplit("_", n=1).str[0]

# Диагностика
unique_samples = nis["sample_id"].unique()
print(f"  Уникальных образцов: {len(unique_samples)}")
print(f"  Примеры sample_id: {sorted(unique_samples)[:10]}")

def get_pop(sample_id):
    if sample_id.startswith("HG"):
        return "IBS"
    elif sample_id.startswith("NA"):
        return "YRI"
    else:
        return "Unknown"

nis["pop"] = nis["sample_id"].apply(get_pop)
unknown_ids = nis[nis["pop"] == "Unknown"]["sample_id"].unique()
if len(unknown_ids) > 0:
    print(f"  ПРЕДУПРЕЖДЕНИЕ: Unknown sample_ids: {unknown_ids[:10]}")
print(f"  Субпопуляции:\n{nis.groupby('pop')['Sample'].nunique()}")

populations = [p for p in nis["pop"].unique() if p != "Unknown"]
print(f"  Популяции для анализа: {populations}")

# Сохраняем Hk для каждой популяции
hk_dict = {}
for pop in populations:
    pop_nis = nis[nis["pop"] == pop]
    Hk = pop_nis["Sample"].nunique()
    hk_dict[pop] = Hk
    print(f"  Популяция {pop}: Hk = {Hk} гаплотипов")
    
    # BED (0-based half-open)
    bed = pop_nis[["Start","End","Sample"]].copy()
    bed.insert(0, "chr", str(CHR))
    bed["bed_start"] = (bed["Start"] - 1).clip(lower=0)
    bed["bed_end"] = bed["End"]
    bed[["chr","bed_start","bed_end","Sample"]].to_csv(
        f"{OUT_B}/nis_{pop}.bed", sep="\t", index=False, header=False)

# Сохраняем Hk
with open(f"{OUT_B}/Hk.json", "w") as f:
    json.dump(hk_dict, f)

print(f"  BED-файлы субпопуляций сохранены в {OUT_B}/")
PYEOF

# 1b: bedtools intersect для каждой субпопуляции
WIN_BED="$OUT_A/chr${CHR}_windows.bed"
sort -k1,1 -k2,2n "$WIN_BED" > "$OUT_B/chr${CHR}_windows_sorted_B.bed"

for POP_BED in "$OUT_B"/nis_*.bed; do
    POP=$(basename "$POP_BED" .bed | sed 's/nis_//')
    echo "  bedtools intersect для популяции: $POP"
    sort -k1,1 -k2,2n "$POP_BED" > "$OUT_B/nis_${POP}_sorted.bed"
    bedtools intersect \
        -a "$OUT_B/chr${CHR}_windows_sorted_B.bed" \
        -b "$OUT_B/nis_${POP}_sorted.bed" \
        -wo \
        > "$OUT_B/windows_x_nis_${POP}.bed"
    echo "    Пересечений: $(wc -l < "$OUT_B/windows_x_nis_${POP}.bed")"
done

# 1c: Вычисляем Fk векторно
python3 << 'PYEOF'
import pandas as pd
import numpy as np
import os
import json

CHR = os.environ.get("CHR", "6")
WORKDIR = os.path.expanduser("~/nd_pipeline")
OUT_A = f"{WORKDIR}/results/chr{CHR}/pipeline_A"
OUT_B = f"{WORKDIR}/results/chr{CHR}/pipeline_B"
WIN_SIZE = 1000
MIN_OVERLAP = WIN_SIZE // 2

with open(f"{OUT_B}/Hk.json") as f:
    hk_dict = json.load(f)

populations = list(hk_dict.keys())

# Читаем окна с win_id
windows = pd.read_csv(f"{OUT_A}/chr{CHR}_windows_full.tsv", sep="\t")

subpop_results = []

for pop in populations:
    Hk = hk_dict[pop]
    ix_file = f"{OUT_B}/windows_x_nis_{pop}.bed"

    # Колонки: chr_win,win_start,win_end,win_id, chr_nis,seg_start,seg_end,hap_id, overlap_bp
    cols = ["chr_win","win_start","win_end","win_id",
            "chr_nis","seg_start","seg_end","hap_id","overlap_bp"]
    ix = pd.read_csv(ix_file, sep="\t", header=None, names=cols)

    # Фильтр: перекрытие >= 500 bp
    ix_filt = ix[ix["overlap_bp"] >= MIN_OVERLAP].copy()

    # Уникальные гаплотипы на окно
    n_introg = (
        ix_filt.groupby("win_id")["hap_id"]
        .nunique()
        .reset_index()
        .rename(columns={"hap_id": "n_hap_introgressed_k"})
    )

    # Объединяем с полным списком окон
    pop_df = windows[["win_id"]].merge(n_introg, on="win_id", how="left")
    pop_df["n_hap_introgressed_k"] = pop_df["n_hap_introgressed_k"].fillna(0).astype(int)
    pop_df["Fk"] = pop_df["n_hap_introgressed_k"] / Hk
    pop_df["pop"] = pop
    pop_df["Hk"] = Hk

    subpop_results.append(pop_df)
    print(f"  {pop}: окон с интрогрессией = {(pop_df['Fk']>0).sum()}")

subpop_df = pd.concat(subpop_results, ignore_index=True)

# Pivot: строки = win_id, колонки = Fk_IBS, Fk_YRI, ...
pivot = subpop_df.pivot_table(index="win_id", columns="pop", values="Fk").reset_index()
pivot.columns = ["win_id"] + [f"Fk_{p}" for p in pivot.columns[1:]]

# Объединяем с основной матрицей
windows_full = windows.merge(pivot, on="win_id", how="left")
windows_full.to_csv(f"{OUT_B}/chr{CHR}_windows_subpop.tsv", sep="\t", index=False)
print(f"  Сохранено: {OUT_B}/chr{CHR}_windows_subpop.tsv")
print(f"  Колонки: {list(windows_full.columns)}")

# Сохраняем детальную таблицу субпопуляций
subpop_df.to_csv(f"{OUT_B}/chr{CHR}_subpop_frequencies.tsv", sep="\t", index=False)
PYEOF

echo "[$(date)] Шаг 1 завершён."

echo "[$(date)] Шаг 2: Поляризация eQTL на Vindija 33.19..."

# Проверяем наличие VCF
if [ ! -f "$VINDIJA_VCF" ]; then
    echo "  ОШИБКА: Vindija VCF не найден: $VINDIJA_VCF"
    echo "  Проверьте скачивание в 00_download_data.sh"
    exit 1
fi

# Индексируем если нет индекса
if [ ! -f "${VINDIJA_VCF}.tbi" ]; then
    tabix -p vcf "$VINDIJA_VCF"
fi

# Извлекаем позиции eQTL chr${CHR} для запроса к VCF
python3 << 'PYEOF'
import pandas as pd
import os

CHR = os.environ.get("CHR", "6")
WORKDIR = os.path.expanduser("~/nd_pipeline")
OUT_A = f"{WORKDIR}/results/chr{CHR}/pipeline_A"
OUT_B = f"{WORKDIR}/results/chr{CHR}/pipeline_B"

eqtl = pd.read_csv(f"{OUT_A}/chr{CHR}_eqtl_all.tsv", sep="\t")

# Создаём BED-файл позиций для bcftools query -T
positions = eqtl["vpos"].drop_duplicates().astype(int).sort_values().values
pos_bed = pd.DataFrame({
    "chr":   str(CHR),
    "start": positions - 1,
    "end":   positions
})
pos_bed.to_csv(f"{OUT_B}/eqtl_positions.bed", sep="\t", index=False, header=False)
print(f"  Уникальных позиций eQTL: {len(positions)}")
print(f"  Пример BED строк:")
print(pos_bed.head(3).to_string(index=False, header=False))
PYEOF

# Извлекаем генотипы Vindija для позиций eQTL
bcftools query \
    -T "$OUT_B/eqtl_positions.bed" \
    -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\t[%GQ]\t[%DP]\n' \
    "$VINDIJA_VCF" \
    > "$OUT_B/vindija_eqtl_sites.txt"

echo "  Vindija сайтов извлечено: $(wc -l < "$OUT_B/vindija_eqtl_sites.txt")"

echo "[$(date)] Шаг 3: Hard Filtering и поляризация..."

python3 << 'PYEOF'
import pandas as pd
import numpy as np
import os

CHR = os.environ.get("CHR", "6")
WORKDIR = os.path.expanduser("~/nd_pipeline")
OUT_A = f"{WORKDIR}/results/chr{CHR}/pipeline_A"
OUT_B = f"{WORKDIR}/results/chr{CHR}/pipeline_B"

# Читаем Vindija сайты
vindija_file = f"{OUT_B}/vindija_eqtl_sites.txt"
if os.path.getsize(vindija_file) == 0:
    print("  ПРЕДУПРЕЖДЕНИЕ: Vindija файл пуст. Поляризация невозможна.")
    pd.DataFrame(columns=["variant_id","D_wt","vindija_allele","polarized"]).to_csv(
        f"{OUT_B}/chr{CHR}_polarized_eqtl.tsv", sep="\t", index=False)
    exit(0)

vindija = pd.read_csv(vindija_file, sep="\t", header=None,
                      names=["chrom","pos","ref","alt","gt","gq","dp"])
print(f"  Vindija сайтов до фильтрации: {len(vindija)}")

# Hard Filtering
TRANSITIONS = {("C","T"), ("T","C"), ("G","A"), ("A","G")}

vindija["gq"] = pd.to_numeric(vindija["gq"], errors="coerce")
vindija["dp"] = pd.to_numeric(vindija["dp"], errors="coerce")

vindija_filt = vindija[
    (vindija["gq"] > 30) &
    (vindija["dp"] > 10) &
    (~vindija.apply(lambda r: (r["ref"], r["alt"]) in TRANSITIONS, axis=1))
].copy()
print(f"  После Hard Filtering: {len(vindija_filt)}")

# Парсим генотип Vindija
def parse_gt(gt_str):
    if pd.isna(gt_str) or gt_str in (".", "./.", ".|."):
        return None
    alleles = gt_str.replace("|","/").split("/")
    try:
        return [int(a) for a in alleles if a != "."]
    except:
        return None

vindija_filt["alleles"] = vindija_filt["gt"].apply(parse_gt)
vindija_filt = vindija_filt[vindija_filt["alleles"].notna()].copy()

# Определяем неандертальский аллель
def get_neandertal_allele(row):
    alleles = row["alleles"]
    if alleles is None:
        return None
    if 1 in alleles:
        return row["alt"]
    return None

vindija_filt["neandertal_allele"] = vindija_filt.apply(get_neandertal_allele, axis=1)
vindija_filt = vindija_filt[vindija_filt["neandertal_allele"].notna()].copy()
print(f"  Сайтов с неандертальским аллелем: {len(vindija_filt)}")

# Создаём ключ для объединения с eQTL
vindija_filt["variant_key"] = f"{CHR}_" + vindija_filt["pos"].astype(str) + "_" + \
                               vindija_filt["ref"] + "_" + vindija_filt["alt"] + "_b37"

# Читаем lead eQTL
lead_eqtl = pd.read_csv(f"{OUT_A}/chr{CHR}_lead_eqtl.tsv", sep="\t")

# Создаём ключ для eQTL
lead_eqtl["variant_key"] = lead_eqtl["variant_id"].str.replace(r"^chr" + CHR + "_", f"{CHR}_", regex=True)

# Переименовываем ref/alt Vindija
vindija_for_merge = vindija_filt[["variant_key","neandertal_allele","ref","alt"]].copy()
vindija_for_merge = vindija_for_merge.rename(columns={
    "ref": "vindija_ref",
    "alt": "vindija_alt"
})

# Объединяем
polarized = lead_eqtl.merge(
    vindija_for_merge,
    on="variant_key", how="inner"
)
print(f"  eQTL с поляризацией: {len(polarized)}")

# Определяем знак D_w,t
def compute_D(row):
    if row["neandertal_allele"] == row["vindija_alt"]:
        return np.sign(row["slope"])
    elif row["neandertal_allele"] == row["vindija_ref"]:
        return -np.sign(row["slope"])
    else:
        return np.nan

polarized["D_wt"] = polarized.apply(compute_D, axis=1)
polarized["polarized"] = True

# Агрегируем D по окну
D_by_window = polarized.groupby("win_id").agg(
    D_wt_mean=("D_wt", "mean"),
    D_wt_sign=("D_wt", lambda x: np.sign(x.mean()) if len(x) > 0 else np.nan),
    n_polarized=("D_wt", "count")
).reset_index()

D_by_window.to_csv(f"{OUT_B}/chr{CHR}_D_by_window.tsv", sep="\t", index=False)
polarized.to_csv(f"{OUT_B}/chr{CHR}_polarized_eqtl.tsv", sep="\t", index=False)
print(f"  Сохранено: {OUT_B}/chr{CHR}_polarized_eqtl.tsv")
print(f"  Сохранено: {OUT_B}/chr{CHR}_D_by_window.tsv")

# Итоговая матрица Pipeline B
windows_subpop = pd.read_csv(f"{OUT_B}/chr{CHR}_windows_subpop.tsv", sep="\t")
final = windows_subpop.merge(D_by_window, on="win_id", how="left")
final.to_csv(f"{OUT_B}/chr{CHR}_windows_final.tsv", sep="\t", index=False)
print(f"  Итоговая матрица Pipeline B: {OUT_B}/chr{CHR}_windows_final.tsv")
print(f"  Размер: {final.shape}")
PYEOF

echo "[$(date)] Шаг 3 завершён."
echo "[$(date)] Pipeline B завершён"
echo "  Итоговые файлы:"
echo "    $OUT_B/chr${CHR}_windows_subpop.tsv   — матрица с субпопуляционными частотами"
echo "    $OUT_B/chr${CHR}_polarized_eqtl.tsv   — поляризованные eQTL"
echo "    $OUT_B/chr${CHR}_D_by_window.tsv      — знак D_w,t по окнам"
echo "    $OUT_B/chr${CHR}_windows_final.tsv    — полная матрица"
