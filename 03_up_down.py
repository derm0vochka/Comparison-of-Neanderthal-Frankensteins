#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0, centre - half), min(1, centre + half)


def chrom_key(path):
    match = re.search(r"/chr([^/]+)/pipeline_B/", str(path))
    chrom = match.group(1)
    return (0, int(chrom)) if chrom.isdigit() else (1, chrom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-workdir", default=str(Path.home() / "nd_pipeline"))
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--mode", default="max_absz")
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    old_root = Path(args.old_workdir).resolve()
    root = Path(args.workdir).resolve()
    outdir = root / "results" / "special_analysis" / args.mode / "up_down"
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        old_root.glob("results/chr*/pipeline_B/chr*_polarized_eqtl.tsv"),
        key=chrom_key,
    )
    if len(files) != 22:
        raise RuntimeError(f"Expected 22 polarized files, found {len(files)}")

    # variant_id -> [сумма D_wt, число ассоц]
    global_stats = defaultdict(lambda: [0.0, 0])

    # (variant_id, tissue) -> [сумма D_wt, число ассоц]
    tissue_stats = defaultdict(lambda: [0.0, 0])

    n_rows_total = 0
    n_rows_polarized = 0
    n_rows_valid_sign = 0

    for path in files:
        print(f"[INFO] {path.name}")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["variant_id", "tissue", "gene_id", "D_wt", "polarized"],
            chunksize=args.chunksize,
            low_memory=False,
        ):
            n_rows_total += len(chunk)

            polarized = chunk["polarized"].astype(str).str.lower().eq("true")
            chunk = chunk.loc[polarized].copy()
            n_rows_polarized += len(chunk)

            chunk["D_wt"] = pd.to_numeric(chunk["D_wt"], errors="coerce")
            chunk = chunk.loc[chunk["D_wt"].isin([-1.0, 1.0])].copy()
            n_rows_valid_sign += len(chunk)

            # одна SNP может быть eQTL для нескольких генов, строки сохраняются для majority sign, а не независимые
            for row in chunk[["variant_id", "tissue", "D_wt"]].itertuples(index=False):
                variant, tissue, sign = row
                global_stats[variant][0] += sign
                global_stats[variant][1] += 1
                tissue_stats[(variant, tissue)][0] += sign
                tissue_stats[(variant, tissue)][1] += 1

    # уникальные SNP, majority sign по всем тканям и генам
    global_rows = []
    n_up = n_down = n_tie = 0

    for variant, (sign_sum, n_assoc) in global_stats.items():
        if sign_sum > 0:
            direction = "Up"
            n_up += 1
            included = True
        elif sign_sum < 0:
            direction = "Down"
            n_down += 1
            included = True
        else:
            direction = "Tie"
            n_tie += 1
            included = False

        global_rows.append(
            {
                "variant_id": variant,
                "n_eqtl_associations": n_assoc,
                "D_wt_sum": sign_sum,
                "majority_direction": direction,
                "included_in_global_test": included,
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values(
        ["included_in_global_test", "n_eqtl_associations"],
        ascending=[False, False],
    )
    global_df.to_csv(outdir / "up_down_unique_snps.tsv", sep="\t", index=False)

    n_tested = n_up + n_down
    test = binomtest(n_up, n=n_tested, p=0.5, alternative="two-sided")

    global_summary = pd.DataFrame(
        [
            {
                "n_unique_snps_total": len(global_df),
                "n_unique_snps_tested": n_tested,
                "n_ties_excluded": n_tie,
                "n_up": n_up,
                "n_down": n_down,
                "p_up": n_up / n_tested,
                "binomial_test_p_value": test.pvalue,
            }
        ]
    )
    global_summary.to_csv(outdir / "up_down_global_summary.tsv", sep="\t", index=False)

    # Tissue-specific barplot
    per_tissue = defaultdict(lambda: {"up": 0, "down": 0, "tie": 0})

    for (_, tissue), (sign_sum, _) in tissue_stats.items():
        if sign_sum > 0:
            per_tissue[tissue]["up"] += 1
        elif sign_sum < 0:
            per_tissue[tissue]["down"] += 1
        else:
            per_tissue[tissue]["tie"] += 1

    tissue_rows = []
    for tissue, values in per_tissue.items():
        n = values["up"] + values["down"]
        low, high = wilson_interval(values["up"], n)

        tissue_rows.append(
            {
                "tissue": tissue,
                "n_unique_snps_tested": n,
                "n_up": values["up"],
                "n_down": values["down"],
                "n_ties_excluded": values["tie"],
                "p_up": values["up"] / n if n else np.nan,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    tissue_df = pd.DataFrame(tissue_rows).sort_values("p_up")
    tissue_df.to_csv(outdir / "up_down_by_tissue.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, max(7, 0.28 * len(tissue_df))))
    y = np.arange(len(tissue_df))
    p = tissue_df["p_up"].to_numpy()
    lower = p - tissue_df["ci95_low"].to_numpy()
    upper = tissue_df["ci95_high"].to_numpy() - p

    ax.errorbar(
        p, y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#3b75af",
        ecolor="#777777",
        elinewidth=1,
        capsize=2,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, tissue_df["tissue"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion Up among unique SNPs")
    ax.set_title("Up/Down by tissue (majority sign within SNP × tissue)")
    fig.tight_layout()
    fig.savefig(outdir / "up_down_by_tissue.png", dpi=250)
    plt.close(fig)

    with open(outdir / "up_down_qc.txt", "w") as handle:
        handle.write(f"pipeline_B_source={old_root}\n")
        handle.write(f"n_chromosomes={len(files)}\n")
        handle.write(f"n_rows_total={n_rows_total}\n")
        handle.write(f"n_rows_polarized={n_rows_polarized}\n")
        handle.write(f"n_rows_valid_D_wt={n_rows_valid_sign}\n")
        handle.write("main_test_unit=unique SNP; majority sign across all eQTL associations\n")
        handle.write("tissue_barplot_unit=unique SNP within each tissue; descriptive only\n")
        handle.write("ties_excluded=True\n")

    print("[DONE]", outdir)


if __name__ == "__main__":
    main()
