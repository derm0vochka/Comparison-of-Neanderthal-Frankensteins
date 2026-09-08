#!/usr/bin/env python3
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


POPS = ["IBS", "TSI", "GBR", "FIN"]
KEYS = ["chrom", "win_start", "win_end"]


def chrom_key(chrom):
    chrom = str(chrom)
    return (0, int(chrom)) if chrom.isdigit() else (1, chrom)


def load_valid_windows(root, mode, pop, chrom, keep_win_id=False):
    path = root / "results" / f"pipeline_A_{pop}" / mode / f"chr{chrom}_windows_full_valid.tsv"
    cols = KEYS + ["Fw", "is_valid"] + (["win_id"] if keep_win_id else [])
    x = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False)
    x = x.loc[x["is_valid"] == 1].drop(columns="is_valid")

    if x.duplicated(KEYS).any():
        raise RuntimeError(f"Duplicate valid windows: {path}")

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--mode", default="max_absz")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    outdir = root / "results" / "special_analysis" / args.mode / "enrichment_ratio"
    outdir.mkdir(parents=True, exist_ok=True)

    gbr_dir = root / "results" / "pipeline_A_GBR" / args.mode
    suffix = "_windows_full_valid.tsv"
    chroms = sorted(
        [p.name[len("chr"):-len(suffix)] for p in gbr_dir.glob(f"chr*{suffix}")],
        key=chrom_key,
    )

    first_sw = gbr_dir / f"chr{chroms[0]}_Sw_per_tissue.tsv"
    tissues = [x for x in pd.read_csv(first_sw, sep="\t", nrows=0).columns if x != "win_id"]
    n_tissues = len(tissues)

    block_n = []
    block_sw_sum = []
    block_fw_sum = []
    block_weighted_sum = []

    for chrom in chroms:
        print(f"[INFO] chr{chrom}")

        # храним win_id нужный для Sw
        common = load_valid_windows(root, args.mode, "GBR", chrom, keep_win_id=True)
        common = common.rename(columns={"Fw": "Fw_GBR"})

        for pop in ["IBS", "TSI", "FIN"]:
            x = load_valid_windows(root, args.mode, pop, chrom)
            x = x.rename(columns={"Fw": f"Fw_{pop}"})
            common = common.merge(x, on=KEYS, how="inner", validate="one_to_one")

        sw_path = gbr_dir / f"chr{chrom}_Sw_per_tissue.tsv"
        sw = pd.read_csv(sw_path, sep="\t", low_memory=False)

        joined = common[["win_id"] + [f"Fw_{pop}" for pop in POPS]].merge(
            sw[["win_id"] + tissues],
            on="win_id",
            how="left",
            validate="one_to_one",
        )

        values = (
            joined[tissues]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        block_n.append(len(joined))
        block_sw_sum.append(values.sum(axis=0))

        fw_this_block = []
        weighted_this_block = []

        for pop in POPS:
            fw = joined[f"Fw_{pop}"].to_numpy(dtype=float)
            fw_this_block.append(fw.sum())
            weighted_this_block.append(values.T @ fw)

        block_fw_sum.append(fw_this_block)
        block_weighted_sum.append(weighted_this_block)

    block_n = np.asarray(block_n, dtype=float) # chr
    block_sw_sum = np.asarray(block_sw_sum, dtype=float) # chr × tissue
    block_fw_sum = np.asarray(block_fw_sum, dtype=float) # chr × pop
    block_weighted_sum = np.asarray(block_weighted_sum, dtype=float) # chr × pop × tissue

    # оценка по всем хромосомам
    genome_mean = block_sw_sum.sum(axis=0) / block_n.sum()
    point_er = np.zeros((len(POPS), n_tissues))

    for i, pop in enumerate(POPS):
        intro_mean = block_weighted_sum[:, i, :].sum(axis=0) / block_fw_sum[:, i].sum()
        point_er[i, :] = intro_mean / genome_mean

    # бутстреп - все хромосомы выбираются с возвращением
    rng = np.random.default_rng(args.seed)
    boot_er = np.empty((args.n_bootstrap, len(POPS), n_tissues), dtype=float)

    for b in range(args.n_bootstrap):
        chosen = rng.integers(0, len(chroms), size=len(chroms))

        boot_genome_mean = block_sw_sum[chosen].sum(axis=0) / block_n[chosen].sum()

        for i in range(len(POPS)):
            boot_intro_mean = (
                block_weighted_sum[chosen, i, :].sum(axis=0)
                / block_fw_sum[chosen, i].sum()
            )
            boot_er[b, i, :] = boot_intro_mean / boot_genome_mean

    rows = []
    for i, pop in enumerate(POPS):
        lower = np.quantile(boot_er[:, i, :], 0.025, axis=0)
        upper = np.quantile(boot_er[:, i, :], 0.975, axis=0)
        p_le_1 = (boot_er[:, i, :] <= 1).mean(axis=0)
        p_ge_1 = (boot_er[:, i, :] >= 1).mean(axis=0)
        p_two_sided = np.minimum(1.0, 2 * np.minimum(p_le_1, p_ge_1))

        for tissue, er, lo, hi, p_value in zip(
            tissues, point_er[i], lower, upper, p_two_sided
        ):
            rows.append(
                {
                    "tissue": tissue,
                    "population": pop,
                    "ER": er,
                    "CI95_low": lo,
                    "CI95_high": hi,
                    "bootstrap_p_two_sided": p_value,
                    "n_bootstrap": args.n_bootstrap,
                    "block_definition": "chromosome",
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        outdir / "ER_tissue_by_population_chromosome_bootstrap.tsv",
        sep="\t",
        index=False,
    )

    print("[DONE]", outdir / "ER_tissue_by_population_chromosome_bootstrap.tsv")


if __name__ == "__main__":
    main()
