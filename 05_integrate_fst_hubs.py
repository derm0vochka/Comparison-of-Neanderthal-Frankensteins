#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--mode", default="max_absz")
    ap.add_argument("--lead-pop", default="GBR")
    ap.add_argument("--chunksize", type=int, default=250000)
    args = ap.parse_args()

    root = Path(args.workdir).resolve()
    outdir = root / "results" / "special_analysis" / args.mode / "integration"
    outdir.mkdir(parents=True, exist_ok=True)

    fst_path = (
        root / "results" / "special_analysis" / args.mode
        / "fst" / "extreme_fst_windows_top1.tsv"
    )
    fst = pd.read_csv(fst_path, sep="\t")
    fst["chrom"] = fst["chrom"].astype(str)

    parts = []

    for chrom, top_windows in fst.groupby("chrom", sort=False):
        print(f"[INFO] chr{chrom}")
        lead_path = (
            root / "results" / f"pipeline_A_{args.lead_pop}" / args.mode
            / f"chr{chrom}_lead_eqtl.tsv"
        )

        selected_ids = set(top_windows["win_id"].astype(int))
        top_windows = top_windows.drop_duplicates("win_id")

        for chunk in pd.read_csv(
            lead_path,
            sep="\t",
            usecols=[
                "win_id", "gene_id", "tissue", "variant_id",
                "slope", "pval_nominal",
            ],
            chunksize=args.chunksize,
            low_memory=False,
        ):
            chunk = chunk.loc[chunk["win_id"].isin(selected_ids)]
            if not chunk.empty:
                chunk = chunk.merge(
                    top_windows,
                    on="win_id",
                    how="inner",
                    validate="many_to_one",
                )
                parts.append(chunk)

    if not parts:
        raise RuntimeError("No lead-eQTL found in top-1% FST windows.")

    eqtl = pd.concat(parts, ignore_index=True)
    eqtl.to_csv(outdir / "fst_top1_lead_eqtl.tsv", sep="\t", index=False)

    gene_summary = (
        eqtl.groupby("gene_id", as_index=False)
        .agg(
            n_top1_fst_windows=("win_id", "nunique"),
            n_lead_eqtl=("variant_id", "size"),
            n_tissues=("tissue", "nunique"),
            max_Fst=("Fst", "max"),
            mean_Fst=("Fst", "mean"),
            max_Fw_IBS=("Fw_IBS", "max"),
            max_Fw_TSI=("Fw_TSI", "max"),
            max_Fw_GBR=("Fw_GBR", "max"),
            max_Fw_FIN=("Fw_FIN", "max"),
        )
    )

    hubs_path = (
        root / "results" / "special_analysis" / args.mode
        / "gene_hubs" / "gene_hubs_cross_population.tsv"
    )
    hubs = pd.read_csv(hubs_path, sep="\t")

    result = gene_summary.merge(hubs, on="gene_id", how="left")
    for col in ["IBS", "TSI", "GBR", "FIN", "total_introgressed_windows"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    result = result.sort_values(
        ["n_top1_fst_windows", "max_Fst", "total_introgressed_windows"],
        ascending=False,
    )

    result.to_csv(outdir / "fst_top1_genes_with_hubs.tsv", sep="\t", index=False)
    result.head(100).to_csv(
        outdir / "top100_fst_top1_genes_with_hubs.tsv",
        sep="\t",
        index=False,
    )

    gene_ids = (
        result["gene_id_unversioned"]
        if "gene_id_unversioned" in result.columns
        else result["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    )
    gene_ids.head(100).to_csv(
        outdir / "go_foreground_top100_fst_top1_genes.txt",
        index=False,
        header=False,
    )

    with open(outdir / "integration_qc.txt", "w") as f:
        f.write(f"n_top1_fst_windows={len(fst)}\n")
        f.write(f"n_lead_eqtl_rows={len(eqtl)}\n")
        f.write(f"n_genes={len(result)}\n")
        f.write(f"lead_eqtl_source_population={args.lead_pop}\n")

    print("[DONE]", outdir)


if __name__ == "__main__":
    main()
