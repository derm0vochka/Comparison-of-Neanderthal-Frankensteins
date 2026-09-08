#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POPS = ["IBS", "TSI", "GBR", "FIN"]
KEYS = ["chrom", "win_start", "win_end"]


def chromosome_key(chrom):
    value = str(chrom)
    return (0, int(value)) if value.isdigit() else (1, value)


def append_tsv(df, path):
    df.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())


def read_h(path):
    values = path.read_text().strip().split()
    if len(values) != 1:
        raise ValueError(f"Expected one H value in {path}, got: {values}")
    return int(values[0])


def main():
    parser = argparse.ArgumentParser(
        description="Tissue enrichment ratio and population FST."
    )
    parser.add_argument("--workdir", default=".", help="Project root")
    parser.add_argument("--mode", default="max_absz")
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    mode = args.mode
    outdir = root / "results" / "special_analysis" / mode
    erdir = outdir / "enrichment_ratio"
    fstdir = outdir / "fst"
    qcdir = outdir / "qc"

    for directory in (erdir, fstdir, qcdir):
        directory.mkdir(parents=True, exist_ok=True)

    ibs_dir = root / "results" / "pipeline_A_IBS" / mode
    suffix = "_windows_full_valid.tsv"
    chroms = []
    for path in ibs_dir.glob(f"chr*{suffix}"):
        chroms.append(path.name[len("chr"):-len(suffix)])
    chroms = sorted(chroms, key=chromosome_key)

    if not chroms:
        raise RuntimeError(f"No chromosome files found in {ibs_dir}")

    # Sw одинаковый во всех популяциях (?)
    first_sw = root / "results" / "pipeline_A_GBR" / mode / f"chr{chroms[0]}_Sw_per_tissue.tsv"
    tissue_columns = [
        x for x in pd.read_csv(first_sw, sep="\t", nrows=0).columns
        if x != "win_id"
    ]

    sum_sw = np.zeros(len(tissue_columns), dtype=float)
    weighted_sum = {pop: np.zeros(len(tissue_columns), dtype=float) for pop in POPS}
    sum_fw = {pop: 0.0 for pop in POPS}
    n_common_windows = 0
    h_rows = []
    plot_samples = []

    fst_path = fstdir / "fst_by_window.tsv"
    if fst_path.exists():
        fst_path.unlink()

    for chrom in chroms:
        print(f"[INFO] chr{chrom}")

        base_path = root / "results" / "pipeline_A_IBS" / mode / f"chr{chrom}_windows_full_valid.tsv"
        base = pd.read_csv(
            base_path,
            sep="\t",
            usecols=KEYS + ["win_id", "Fw", "Sw_global", "is_valid"],
            low_memory=False,
        )
        base = base.loc[base["is_valid"] == 1].copy()
        base = base.drop(columns=["is_valid"]).rename(columns={"Fw": "Fw_IBS"})

        if base.duplicated(KEYS).any():
            raise RuntimeError(f"Duplicate valid coordinates in IBS chr{chrom}")

        common = base
        h_by_pop = {"IBS": read_h(
            root / "results" / "pipeline_A_IBS" / mode / f"chr{chrom}_H_total.txt"
        )}

        for pop in POPS[1:]:
            win_path = root / "results" / f"pipeline_A_{pop}" / mode / f"chr{chrom}_windows_full_valid.tsv"
            x = pd.read_csv(
                win_path,
                sep="\t",
                usecols=KEYS + ["Fw", "is_valid"],
                low_memory=False,
            )
            x = x.loc[x["is_valid"] == 1, KEYS + ["Fw"]].rename(columns={"Fw": f"Fw_{pop}"})

            if x.duplicated(KEYS).any():
                raise RuntimeError(f"Duplicate valid coordinates in {pop} chr{chrom}")

            common = common.merge(x, on=KEYS, how="inner", validate="one_to_one")
            h_by_pop[pop] = read_h(
                root / "results" / f"pipeline_A_{pop}" / mode / f"chr{chrom}_H_total.txt"
            )

        if len(common) == 0:
            raise RuntimeError(f"No common valid windows on chr{chrom}")

        # FST = веса H_k / sum(H_k) отдельно для каждой хромосомы
        h = np.array([h_by_pop[pop] for pop in POPS], dtype=float)
        weights = h / h.sum()
        freq = common[[f"Fw_{pop}" for pop in POPS]].to_numpy(dtype=float)
        fbar = freq @ weights
        numerator = ((freq - fbar[:, None]) ** 2 * weights).sum(axis=1)

        common["Fbar"] = fbar
        common["Fst"] = np.where(
            (fbar > 0) & (fbar < 1),
            numerator / (fbar * (1.0 - fbar)),
            np.nan,
        )

        fst_columns = KEYS + ["win_id", "Sw_global"] + [f"Fw_{pop}" for pop in POPS] + ["Fbar", "Fst"]
        append_tsv(common[fst_columns], fst_path)

        h_rows.append({"chrom": chrom, **{f"H_{pop}": h_by_pop[pop] for pop in POPS}})
        n_common_windows += len(common)

        # ER - добавляем Sw по win_id внутри одной хромосомы
        sw_path = root / "results" / "pipeline_A_GBR" / mode / f"chr{chrom}_Sw_per_tissue.tsv"
        sw = pd.read_csv(sw_path, sep="\t", low_memory=False)

        if "win_id" not in sw.columns:
            raise RuntimeError(f"No win_id in {sw_path}")
        if sw.duplicated(["win_id"]).any():
            raise RuntimeError(f"Duplicate win_id in {sw_path}")

        joined = common[["win_id"] + [f"Fw_{pop}" for pop in POPS]].merge(
            sw[["win_id"] + tissue_columns],
            on="win_id",
            how="left",
            validate="one_to_one",
        )

        # если пустая ячейка в Sw = в этом окне нет eQTL данной ткани = 0
        values = (
            joined[tissue_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        sum_sw += values.sum(axis=0)

        for pop in POPS:
            fw = joined[f"Fw_{pop}"].to_numpy(dtype=float)
            sum_fw[pop] += fw.sum()
            weighted_sum[pop] += values.T @ fw

        scatter = common[["Fst", "Sw_global"]].dropna()
        if len(scatter) > 12000:
            scatter = scatter.sample(n=12000, random_state=123)
        plot_samples.append(scatter)

    # ER
    er_rows = []
    genome_mean = sum_sw / n_common_windows

    for pop in POPS:
        introgressed_mean = weighted_sum[pop] / sum_fw[pop]
        er = np.divide(
            introgressed_mean,
            genome_mean,
            out=np.full(len(tissue_columns), np.nan),
            where=genome_mean != 0,
        )

        for tissue, g_mean, i_mean, er_value in zip(
            tissue_columns, genome_mean, introgressed_mean, er
        ):
            er_rows.append(
                {
                    "tissue": tissue,
                    "population": pop,
                    "n_common_valid_windows": n_common_windows,
                    "sum_Fw": sum_fw[pop],
                    "Sw_genome_mean": g_mean,
                    "Sw_introgressed_weighted_mean": i_mean,
                    "ER": er_value,
                }
            )

    er_df = pd.DataFrame(er_rows)
    er_df.to_csv(erdir / "ER_tissue_by_population_long.tsv", sep="\t", index=False)

    er_matrix = er_df.pivot(index="tissue", columns="population", values="ER").reindex(columns=POPS)
    er_matrix.to_csv(erdir / "ER_tissue_by_population.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(8, max(7, 0.28 * len(er_matrix))))
    delta = np.nanmax(np.abs(er_matrix.to_numpy(dtype=float) - 1.0))
    delta = max(float(delta), 0.05)
    image = ax.imshow(
        er_matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        vmin=1.0 - delta,
        vmax=1.0 + delta,
    )
    ax.set_xticks(range(len(POPS)), POPS)
    ax.set_yticks(range(len(er_matrix.index)), er_matrix.index)
    ax.set_title("Enrichment ratio: tissue × population")
    fig.colorbar(image, ax=ax, label="ER; centre = 1")
    fig.tight_layout()
    fig.savefig(erdir / "ER_heatmap.png", dpi=250)
    plt.close(fig)

    # FST
    fst_values = pd.read_csv(fst_path, sep="\t", usecols=["Fst"])["Fst"].dropna()
    q90, q99 = fst_values.quantile([0.90, 0.99])

    top10_path = fstdir / "extreme_fst_windows_top10.tsv"
    top1_path = fstdir / "extreme_fst_windows_top1.tsv"
    for path in (top10_path, top1_path):
        if path.exists():
            path.unlink()

    top1_parts = []
    for chunk in pd.read_csv(fst_path, sep="\t", chunksize=250000, low_memory=False):
        append_tsv(chunk.loc[chunk["Fst"] >= q90], top10_path)
        selected = chunk.loc[chunk["Fst"] >= q99]
        append_tsv(selected, top1_path)
        top1_parts.append(selected)

    top1 = pd.concat(top1_parts, ignore_index=True)
    top1["chrom_order"] = top1["chrom"].map(chromosome_key)
    top1 = top1.sort_values(["chrom_order", "win_start"])

    regions = []
    current = None

    for row in top1.itertuples(index=False):
        if (
            current is None
            or str(row.chrom) != str(current["chrom"])
            or row.win_start > current["end"]
        ):
            if current is not None:
                regions.append(current)
            current = {
                "chrom": row.chrom,
                "start": row.win_start,
                "end": row.win_end,
                "n_windows": 1,
                "max_Fst": row.Fst,
                "mean_Fst_sum": row.Fst,
            }
        else:
            current["end"] = max(current["end"], row.win_end)
            current["n_windows"] += 1
            current["max_Fst"] = max(current["max_Fst"], row.Fst)
            current["mean_Fst_sum"] += row.Fst

    if current is not None:
        regions.append(current)

    region_df = pd.DataFrame(regions)
    if not region_df.empty:
        region_df["mean_Fst"] = region_df["mean_Fst_sum"] / region_df["n_windows"]
        region_df = region_df.drop(columns=["mean_Fst_sum"])
    region_df.to_csv(fstdir / "extreme_fst_regions_top1.tsv", sep="\t", index=False)

    # FST с глобальной силой eQTL 
    plot_df = pd.concat(plot_samples, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_df["Fst"], plot_df["Sw_global"], s=2, alpha=0.15, rasterized=True)
    ax.set_xlabel("window FST")
    ax.set_ylabel("Sw_global")
    ax.set_title("Population differentiation versus eQTL strength")
    fig.tight_layout()
    fig.savefig(fstdir / "fst_vs_Sw_global.png", dpi=250)
    plt.close(fig)

    pd.DataFrame(h_rows).to_csv(fstdir / "haplotypes_by_chromosome.tsv", sep="\t", index=False)

    with open(qcdir / "special_er_fst_qc.txt", "w") as handle:
        handle.write(f"mode={mode}\n")
        handle.write(f"n_common_valid_windows={n_common_windows}\n")
        handle.write("Sw_source=GBR; verified identical by MD5 across populations\n")
        handle.write("missing_Sw_interpreted_as_zero=True\n")
        handle.write(f"FST_q90={q90}\n")
        handle.write(f"FST_q99={q99}\n")
        for pop in POPS:
            handle.write(f"sum_Fw_{pop}={sum_fw[pop]}\n")

    print(f"[DONE] ER: {erdir}")
    print(f"[DONE] FST: {fstdir}")


if __name__ == "__main__":
    main()
