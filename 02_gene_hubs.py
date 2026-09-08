#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
import re

import pandas as pd

POPS = ["IBS", "TSI", "GBR", "FIN"]

def chrom_key(chrom):
    chrom = str(chrom)
    return (0, int(chrom)) if chrom.isdigit() else (1, chrom)


def strip_ensembl_version(gene_id):
    return re.sub(r"\.\d+$", "", str(gene_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--mode", default="max_absz")
    parser.add_argument("--fw-threshold", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    outdir = root / "results" / "special_analysis" / args.mode / "gene_hubs"
    outdir.mkdir(parents=True, exist_ok=True)

    all_population_tables = []

    for pop in POPS:
        print(f"[INFO] {pop}")
        pipeline = root / "results" / f"pipeline_A_{pop}" / args.mode

        suffix = "_lead_eqtl.tsv"
        chroms = sorted(
            [
                p.name[len("chr"):-len(suffix)]
                for p in pipeline.glob(f"chr*{suffix}")
            ],
            key=chrom_key,
        )

        hub_counts = Counter()
        background_genes = set()
        qc = []

        for chrom in chroms:
            window_path = pipeline / f"chr{chrom}_windows_full_valid.tsv"
            lead_path = pipeline / f"chr{chrom}_lead_eqtl.tsv"

            windows = pd.read_csv(
                window_path,
                sep="\t",
                usecols=["win_id", "Fw", "is_valid"],
                low_memory=False,
            )

            valid_ids = set(
                windows.loc[windows["is_valid"] == 1, "win_id"].astype(int)
            )
            introgressed_ids = set(
                windows.loc[
                    (windows["is_valid"] == 1)
                    & (windows["Fw"] > args.fw_threshold),
                    "win_id",
                ].astype(int)
            )

            seen_gene_window = set()
            lead_rows = 0
            lead_rows_introgressed = 0

            for chunk in pd.read_csv(
                lead_path,
                sep="\t",
                usecols=["win_id", "gene_id"],
                chunksize=args.chunksize,
                low_memory=False,
            ):
                chunk = chunk.dropna(subset=["win_id", "gene_id"]).copy()
                chunk["win_id"] = chunk["win_id"].astype(int)
                lead_rows += len(chunk)

                valid = chunk.loc[chunk["win_id"].isin(valid_ids)]
                background_genes.update(valid["gene_id"].astype(str))

                introgressed = chunk.loc[chunk["win_id"].isin(introgressed_ids)]
                lead_rows_introgressed += len(introgressed)

                seen_gene_window.update(
                    zip(
                        introgressed["gene_id"].astype(str),
                        introgressed["win_id"].astype(int),
                    )
                )

            local = Counter(gene for gene, _ in seen_gene_window)
            hub_counts.update(local)

            qc.append(
                {
                    "chrom": chrom,
                    "n_valid_windows": len(valid_ids),
                    "n_introgressed_windows": len(introgressed_ids),
                    "n_lead_eqtl_rows": lead_rows,
                    "n_lead_eqtl_rows_introgressed": lead_rows_introgressed,
                    "n_unique_gene_window_pairs": len(seen_gene_window),
                }
            )

        result = pd.DataFrame(
            [
                {
                    "population": pop,
                    "gene_id": gene,
                    "gene_id_unversioned": strip_ensembl_version(gene),
                    "n_introgressed_windows": count,
                }
                for gene, count in hub_counts.items()
            ]
        ).sort_values("n_introgressed_windows", ascending=False)

        result.to_csv(
            outdir / f"gene_hubs_{pop}.tsv",
            sep="\t",
            index=False,
        )
        result.head(100).to_csv(
            outdir / f"top100_gene_hubs_{pop}.tsv",
            sep="\t",
            index=False,
        )

        # для GO-enrichment
        result.head(100)["gene_id_unversioned"].to_csv(
            outdir / f"go_foreground_top100_{pop}.txt",
            index=False,
            header=False,
        )
        pd.Series(sorted(strip_ensembl_version(x) for x in background_genes)).to_csv(
            outdir / f"go_background_{pop}.txt",
            index=False,
            header=False,
        )

        pd.DataFrame(qc).to_csv(
            outdir / f"gene_hubs_qc_{pop}.tsv",
            sep="\t",
            index=False,
        )
        all_population_tables.append(result)

    all_hubs = pd.concat(all_population_tables, ignore_index=True)
    matrix = (
        all_hubs.pivot_table(
            index=["gene_id", "gene_id_unversioned"],
            columns="population",
            values="n_introgressed_windows",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=POPS, fill_value=0)
        .reset_index()
    )
    matrix["total_introgressed_windows"] = matrix[POPS].sum(axis=1)
    matrix = matrix.sort_values("total_introgressed_windows", ascending=False)

    matrix.to_csv(
        outdir / "gene_hubs_cross_population.tsv",
        sep="\t",
        index=False,
    )
    matrix.head(100).to_csv(
        outdir / "top100_gene_hubs_cross_population.tsv",
        sep="\t",
        index=False,
    )

    print(f"[DONE] {outdir}")


if __name__ == "__main__":
    main()
