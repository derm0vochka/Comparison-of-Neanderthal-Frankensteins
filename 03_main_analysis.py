#!/usr/bin/env python3
# Основной анализ iSFS, регрессии, bootstrap, адаптивная интрогрессия
# python 03_main_analysis.py
# пререквизиты Pipeline A и B

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
import warnings
import os
import json

warnings.filterwarnings("ignore")

WORKDIR = os.path.expanduser("~/nd_pipeline")
OUT_A   = f"{WORKDIR}/results/pipeline_A"
OUT_B   = f"{WORKDIR}/results/pipeline_B"
OUT     = f"{WORKDIR}/results/analysis"
os.makedirs(OUT, exist_ok=True)

print("eQTL-влияние неандертальской интрогрессии")
print("\n[1] Загрузка данных")

# загрузка матрицы Pipeline B, иначе Pipeline A
final_b = f"{OUT_B}/chr6_windows_final.tsv"
final_a = f"{OUT_A}/chr6_windows_full.tsv"

if os.path.exists(final_b):
    df = pd.read_csv(final_b, sep="\t")
    print(f"  Загружена матрица Pipeline B: {final_b}")
else:
    df = pd.read_csv(final_a, sep="\t")
    print(f"  Загружена матрица Pipeline A: {final_a}")
print(f"  Размер матрицы: {df.shape}")
print(f"  Колонки: {list(df.columns)}")

# ID сегменты для кластеризации ошибок
seg_ids_file = f"{OUT_A}/chr6_window_seg_ids.tsv"
if os.path.exists(seg_ids_file):
    seg_ids = pd.read_csv(seg_ids_file, sep="\t")
    df = df.merge(seg_ids, on="win_id", how="left")
    # Для окон без сегмента - уникальный ID
    max_seg = df["seg_id"].max() if "seg_id" in df.columns else 0
    mask_no_seg = df["seg_id"].isna()
    df.loc[mask_no_seg, "seg_id"] = range(
        int(max_seg) + 1, int(max_seg) + 1 + mask_no_seg.sum()
    )
    df["seg_id"] = df["seg_id"].astype(int)
else:
    # фиктивные seg_id
    df["seg_id"] = df["win_id"]

required_cols = ["win_id", "Fw", "freq_bin", "Sw_max", "has_eqtl", "D_TSS", "recomb_rate"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"  warning: Отсутствуют колонки: {missing}")

# Лог трансформации ковариат
df["log_D_TSS"]    = np.log1p(df["D_TSS"])
df["log_recomb"]   = np.log(df["recomb_rate"].clip(lower=1e-6))
df["chr_factor"]   = 6  # только chr6

# Только окна с Fw > 0
df_introgressed = df[df["Fw"] > 0].copy()
print(f"  Окон с интрогрессией (Fw > 0): {len(df_introgressed)}")
print(f"  Окон без интрогрессии (Fw = 0): {(df['Fw'] == 0).sum()}")

print("\n[2.1] Спектр частот интрогрессии (iSFS)")
freq_bin_order = ["Zero", "Rare", "Low", "Intermediate", "High", "Very_High"]
isfs = df["freq_bin"].value_counts().reindex(freq_bin_order, fill_value=0)
isfs_df = isfs.reset_index()
isfs_df.columns = ["freq_bin", "n_windows"]
isfs_df["freq_bin_order"] = isfs_df["freq_bin"].map(
    {b: i for i, b in enumerate(freq_bin_order)}
)
isfs_df = isfs_df.sort_values("freq_bin_order")
isfs_df.to_csv(f"{OUT}/isfs.tsv", sep="\t", index=False)
print(f"  iSFS:\n{isfs_df[['freq_bin','n_windows']].to_string(index=False)}")

# Manhattan plot
manhattan_df = df[["win_id","win_start","win_end","Fw","freq_bin"]].copy()
manhattan_df.to_csv(f"{OUT}/manhattan_data.tsv", sep="\t", index=False)
print(f"  Manhattan данные сохранены: {OUT}/manhattan_data.tsv")

print("\n[2.2] Двухэтапная модель")
# logit(P(Sw > 0)) ~ Fw + log(1 + D_TSS) + log(recomb) + chr
print("  Этап 1: Логистическая регрессия (наличие eQTL)")
df_logit = df[["has_eqtl","Fw","log_D_TSS","log_recomb","seg_id"]].dropna().copy()

try:
    logit_model = smf.logit(
        "has_eqtl ~ Fw + log_D_TSS + log_recomb",
        data=df_logit
    ).fit(disp=False, maxiter=200)
    cov_clust = cov_cluster(logit_model, df_logit["seg_id"].values)
    logit_results = {
        "coef_Fw":        float(logit_model.params.get("Fw", np.nan)),
        "se_Fw":          float(np.sqrt(cov_clust.diagonal()[
            list(logit_model.params.index).index("Fw")
        ])) if "Fw" in logit_model.params.index else np.nan,
        "pval_Fw":        float(logit_model.pvalues.get("Fw", np.nan)),
        "aic":            float(logit_model.aic),
        "n_obs":          int(logit_model.nobs),
        "pseudo_r2":      float(logit_model.prsquared),
    }
    print(f"    Коэф. Fw: {logit_results['coef_Fw']:.4f} "
          f"(p={logit_results['pval_Fw']:.4e})")
    print(f"    Pseudo-R^2: {logit_results['pseudo_r2']:.4f}")
    
    with open(f"{OUT}/logit_results.json", "w") as f:
        json.dump(logit_results, f, indent=2)
  
    logit_summary = pd.DataFrame({
        "variable": logit_model.params.index,
        "coef":     logit_model.params.values,
        "pval":     logit_model.pvalues.values,
    })
    logit_summary.to_csv(f"{OUT}/logit_summary.tsv", sep="\t", index=False)

except Exception as e:
    print(f"    Ошибка логистической регрессии: {e}")
    logit_results = {}

# Sw ~ Fw + log(1 + D_TSS) + log(recomb) + chr
print("  Этап 2: Линейная регрессия (сила eQTL)")
df_lm = df_introgressed[df_introgressed["Sw_max"] > 0][
    ["Sw_max","Fw","log_D_TSS","log_recomb","seg_id"]
].dropna().copy()

print(f"    Окон для регрессии: {len(df_lm)}")

if len(df_lm) >= 10:
    try:
        lm_model = smf.ols(
            "Sw_max ~ Fw + log_D_TSS + log_recomb",
            data=df_lm
        ).fit()
        cov_clust_lm = cov_cluster(lm_model, df_lm["seg_id"].values)
        se_clustered = np.sqrt(np.diag(cov_clust_lm))
        
        lm_results = {
            "coef_Fw":   float(lm_model.params.get("Fw", np.nan)),
            "se_Fw_clustered": float(se_clustered[
                list(lm_model.params.index).index("Fw")
            ]) if "Fw" in lm_model.params.index else np.nan,
            "pval_Fw":   float(lm_model.pvalues.get("Fw", np.nan)),
            "r2":        float(lm_model.rsquared),
            "n_obs":     int(lm_model.nobs),
        }
        print(f"    Коэф. Fw: {lm_results['coef_Fw']:.4f} "
              f"(p={lm_results['pval_Fw']:.4e})")
        print(f"    R^2: {lm_results['r2']:.4f}")
        
        with open(f"{OUT}/lm_results.json", "w") as f:
            json.dump(lm_results, f, indent=2)
        
        lm_summary = pd.DataFrame({
            "variable":       lm_model.params.index,
            "coef":           lm_model.params.values,
            "se_clustered":   se_clustered,
            "pval":           lm_model.pvalues.values,
        })
        lm_summary.to_csv(f"{OUT}/lm_summary.tsv", sep="\t", index=False)

    except Exception as e:
        print(f"    Ошибка линейной регрессии: {e}")
        lm_results = {}
else:
    print(f"    Недостаточно данных для регрессии (n={len(df_lm)})")
    lm_results = {}

# Boxplot и Violin plot (Fw_bin и Sw)
boxplot_data = df_introgressed[["win_id","freq_bin","Sw_max","Fw"]].copy()
boxplot_data.to_csv(f"{OUT}/boxplot_data.tsv", sep="\t", index=False)
print(f"  Данные для boxplot: {OUT}/boxplot_data.tsv")
# Медианы Sw по бинам
medians = df_introgressed.groupby("freq_bin")["Sw_max"].median().reindex(
    [b for b in freq_bin_order if b != "Zero"]
)
print(f"  Медианы Sw по бинам:\n{medians.to_string()}")
medians.to_csv(f"{OUT}/sw_medians_by_bin.tsv", sep="\t", header=True)

# Корреляция Spearman (Fw и Sw)
df_corr = df_introgressed[["Fw","Sw_max"]].dropna()
if len(df_corr) >= 5:
    rho, pval_rho = stats.spearmanr(df_corr["Fw"], df_corr["Sw_max"])
    print(f"  Spearman ρ(Fw, Sw): {rho:.4f} (p={pval_rho:.4e})")
    with open(f"{OUT}/spearman_Fw_Sw.json", "w") as f:
        json.dump({"rho": float(rho), "pval": float(pval_rho),
                   "n": len(df_corr)}, f, indent=2)

print("\n[2.3] Валидация: Block Bootstrap Mann-Whitney")
# медианная длина сегмента
median_seg_len_file = f"{OUT_A}/median_seg_len.txt"
if os.path.exists(median_seg_len_file):
    with open(median_seg_len_file) as f:
        BLOCK_SIZE = int(f.read().strip())
else:
    BLOCK_SIZE = 124000  # медиана NIS-сегмента
print(f"  Размер геномного блока: {BLOCK_SIZE} bp")

N_BOOTSTRAP = 10000
np.random.seed(42)
dtss_categories = ["Promoter", "Near", "Distal"]
bootstrap_results = {}

def genomic_block_bootstrap(introg_df, ctrl_df, obs_diff, n_boot=10000):
    introg_blocks = introg_df["block_id"].unique()
    ctrl_blocks   = ctrl_df["block_id"].unique()
    n_introg_blocks = len(introg_blocks)
    n_ctrl_blocks   = len(ctrl_blocks)
    # Диагностика блоков
    diag = {
        "n_introg_blocks":      int(n_introg_blocks),
        "n_ctrl_blocks":        int(n_ctrl_blocks),
        "n_introg_windows":     int(len(introg_df)),
        "n_ctrl_windows":       int(len(ctrl_df)),
        "median_introg_block_size": float(
            introg_df.groupby("block_id").size().median()
        ),
        "median_ctrl_block_size": float(
            ctrl_df.groupby("block_id").size().median()
        ),
    }
    if n_introg_blocks == 0 or n_ctrl_blocks == 0:
        print(f"    warning: пустые блоки "
              f"(introg={n_introg_blocks}, ctrl={n_ctrl_blocks}), "
              f"bootstrap пропущен")
        return np.array([obs_diff]), diag
    # словари block_id → массив Sw_max
    introg_block_arrays = {
        b: introg_df[introg_df["block_id"] == b]["Sw_max"].values
        for b in introg_blocks
    }
    ctrl_block_arrays = {
        b: ctrl_df[ctrl_df["block_id"] == b]["Sw_max"].values
        for b in ctrl_blocks
    }
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        if i > 0 and i % 1000 == 0:
            print(f"      Bootstrap: {i}/{n_boot} итераций...")
        # Ресэмплируем блоки с возвращением
        chosen_introg = introg_blocks[
            np.random.randint(0, n_introg_blocks, size=n_introg_blocks)
        ]
        chosen_ctrl = ctrl_blocks[
            np.random.randint(0, n_ctrl_blocks, size=n_ctrl_blocks)
        ]
        boot_introg = np.concatenate([introg_block_arrays[b] for b in chosen_introg])
        boot_ctrl   = np.concatenate([ctrl_block_arrays[b]   for b in chosen_ctrl])

        # Пропускаем вырожденные итерации
        if len(boot_introg) == 0 or len(boot_ctrl) == 0:
            boot_diffs[i] = 0.0
            continue
        # Разница средних рангов
        comb_b = np.concatenate([boot_ctrl, boot_introg])
        ranks_b = stats.rankdata(comb_b)
        boot_diffs[i] = ranks_b[:len(boot_ctrl)].mean() - ranks_b[len(boot_ctrl):].mean()
    return boot_diffs, diag

for cat in dtss_categories:
    print(f"\n  Категория: {cat}")
    df_cat = df[df["dtss_cat"] == cat].copy()
    # интрогрессия (Fw > 0) и контроль (Fw = 0)
    introg_df_cat = df_cat[df_cat["Fw"] > 0][["win_start", "Sw_max"]].dropna().copy()
    ctrl_df_cat   = df_cat[df_cat["Fw"] == 0][["win_start", "Sw_max"]].dropna().copy()

    introg = introg_df_cat["Sw_max"].values
    control = ctrl_df_cat["Sw_max"].values
    print(f"    Интрогрессия: n={len(introg)}, Контроль: n={len(control)}")

    if len(introg) < 5 or len(control) < 5:
        print(f"    Недостаточно данных для {cat}")
        bootstrap_results[cat] = {"error": "insufficient_data"}
        continue

    print(f"    Sw (интрогрессия): median={np.median(introg):.4f}, "
          f"mean={np.mean(introg):.4f}, "
          f"p75={np.percentile(introg, 75):.4f}, "
          f"frac_zero={(introg == 0).mean():.3f}")
    print(f"    Sw (контроль):     median={np.median(control):.4f}, "
          f"mean={np.mean(control):.4f}, "
          f"p75={np.percentile(control, 75):.4f}, "
          f"frac_zero={(control == 0).mean():.3f}")

    # Mann-Whitney U-test
    # H1: Sw(контроль) > Sw(интрогрессия)
    stat_obs, pval_mw = mannwhitneyu(control, introg, alternative="greater")
    print(f"    MW U={stat_obs:.0f}, p={pval_mw:.4e} "
          f"(H1: Sw_ctrl > Sw_introg)")

    # Геномные блоки для bootstrap
    introg_df_cat = introg_df_cat.sort_values("win_start").copy()
    ctrl_df_cat   = ctrl_df_cat.sort_values("win_start").copy()

    introg_df_cat["block_id"] = (introg_df_cat["win_start"] // BLOCK_SIZE).astype(int)
    ctrl_df_cat["block_id"]   = (ctrl_df_cat["win_start"]   // BLOCK_SIZE).astype(int)

    # разница средних рангов (ctrl - introg)
    combined = np.concatenate([control, introg])
    ranks    = stats.rankdata(combined)
    n_ctrl   = len(control)
    obs_diff = ranks[:n_ctrl].mean() - ranks[n_ctrl:].mean()
    print(f"    Наблюдаемая разница рангов (ctrl−introg): {obs_diff:.4f}")

    # Block Bootstrap
    print(f"    Запуск bootstrap ({N_BOOTSTRAP} итераций)...")
    boot_diffs, diag = genomic_block_bootstrap(
        introg_df_cat, ctrl_df_cat, obs_diff, N_BOOTSTRAP
    )
    print(f"    Диагностика блоков:")
    print(f"      Блоков интрогрессии: {diag['n_introg_blocks']} "
          f"(медиана размера: {diag['median_introg_block_size']:.1f} окон)")
    print(f"      Блоков контроля:     {diag['n_ctrl_blocks']} "
          f"(медиана размера: {diag['median_ctrl_block_size']:.1f} окон)")

    # p-value (где boot_diff >= obs_diff)
    # H1: контроль сильнее интрогрессии
    pval_boot = (boot_diffs >= obs_diff).mean()
    # bootstrap-распределения
    boot_mean = float(np.mean(boot_diffs))
    boot_std  = float(np.std(boot_diffs))
    print(f"    Bootstrap: mean_diff={boot_mean:.4f}, std={boot_std:.4f}, "
          f"p={pval_boot:.4e}")
    bootstrap_results[cat] = {
        "n_introgressed":           int(len(introg)),
        "n_control":                int(len(control)),
        "median_introg":            float(np.median(introg)),
        "median_control":           float(np.median(control)),
        "mean_introg":              float(np.mean(introg)),
        "mean_control":             float(np.mean(control)),
        "frac_zero_introg":         float((introg == 0).mean()),
        "frac_zero_ctrl":           float((control == 0).mean()),
        "mw_stat":                  float(stat_obs),
        "pval_mw_standard":         float(pval_mw),
        "obs_rank_diff_ctrl_minus_introg": float(obs_diff),
        "pval_bootstrap":           float(pval_boot),
        "boot_mean_diff":           boot_mean,
        "boot_std_diff":            boot_std,
        "n_bootstrap":              N_BOOTSTRAP,
        "block_size_bp":            BLOCK_SIZE,
        "n_introg_blocks":          diag["n_introg_blocks"],
        "n_ctrl_blocks":            diag["n_ctrl_blocks"],
        "median_introg_block_size": diag["median_introg_block_size"],
        "median_ctrl_block_size":   diag["median_ctrl_block_size"],
    }

with open(f"{OUT}/bootstrap_results.json", "w") as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"  Bootstrap результаты: {OUT}/bootstrap_results.json")

# Split Violin plot
violin_data = df[["win_id","Fw","Sw_max","dtss_cat","freq_bin"]].copy()
violin_data["group"] = np.where(violin_data["Fw"] > 0, "Introgressed", "Control")
violin_data.to_csv(f"{OUT}/violin_data.tsv", sep="\t", index=False)

print("\n[2.4] Поиск кандидатов адаптивной интрогрессии")
# окна с Fw > 0
df_pos = df_introgressed.copy()
# топ 5% по Fw и Sw
Fw_95 = df_pos["Fw"].quantile(0.95)
Sw_95 = df_pos["Sw_max"].quantile(0.95)
print(f"  Порог Fw (95%): {Fw_95:.4f}")
print(f"  Порог Sw (95%): {Sw_95:.4f}")

candidates = df_pos[
    (df_pos["Fw"] >= Fw_95) & 
    (df_pos["Sw_max"] >= Sw_95)
].copy()
print(f"  Кандидатов адаптивной интрогрессии: {len(candidates)}")

if len(candidates) > 0:
    # гены-мишени
    if "nearest_gene" in candidates.columns:
        print(f"  Топ-10 кандидатов:")
        top = candidates.nlargest(10, "Sw_max")[
            ["win_id","win_start","win_end","Fw","Sw_max","nearest_gene","dtss_cat"]
        ]
        print(top.to_string(index=False))
    
    candidates.to_csv(f"{OUT}/adaptive_candidates.tsv", sep="\t", index=False)
    print(f"  Сохранено: {OUT}/adaptive_candidates.tsv")

# Scatter plot (Fw и Sw)
scatter_data = df_pos[["win_id","win_start","win_end","Fw","Sw_max",
                         "freq_bin","nearest_gene","dtss_cat"]].copy()
scatter_data["is_candidate"] = (
    (scatter_data["Fw"] >= Fw_95) & 
    (scatter_data["Sw_max"] >= Sw_95)
).astype(int)
scatter_data.to_csv(f"{OUT}/scatter_data.tsv", sep="\t", index=False)
thresholds = {"Fw_95": float(Fw_95), "Sw_95": float(Sw_95)}
with open(f"{OUT}/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print(f"  Хромосома: 6")
print(f"  Всего окон: {len(df)}")
print(f"  Окон с интрогрессией: {len(df_introgressed)} "
      f"({100*len(df_introgressed)/len(df):.1f}%)")
print(f"  Окон с eQTL: {df['has_eqtl'].sum()}")
print(f"  Кандидатов адаптивной интрогрессии: {len(candidates)}")
print(f"\n  Файлы результатов в: {OUT}/")
summary = {
    "chr": 6,
    "n_windows_total":       int(len(df)),
    "n_windows_introgressed": int(len(df_introgressed)),
    "n_windows_with_eqtl":   int(df["has_eqtl"].sum()),
    "n_adaptive_candidates": int(len(candidates)),
    "Fw_95_threshold":       float(Fw_95),
    "Sw_95_threshold":       float(Sw_95),
    "block_size_bp":         BLOCK_SIZE,
    "n_bootstrap":           N_BOOTSTRAP,
}
with open(f"{OUT}/analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nАнализ завершен. Все результаты в: {OUT}/")
