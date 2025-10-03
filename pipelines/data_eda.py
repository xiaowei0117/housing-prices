# eda_report.py
# -*- coding: utf-8 -*-
"""
轻量 EDA 脚本：读取 CSV，生成 Markdown 报告与配图（仅用 pandas + numpy + matplotlib）
Usage:
  python eda_report.py --input train.csv --outdir eda_outputs --target SalePrice --max-plots 8
"""

import os
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_hist(series: pd.Series, title: str, xlabel: str, outpath: str, bins: int = 30):
    x = series.dropna().values
    plt.figure(figsize=(7, 5))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, outpath: str, vline: Optional[float] = None):
    xv = x.values
    yv = y.values
    plt.figure(figsize=(6, 4))
    plt.scatter(xv, yv, s=10)
    if vline is not None:
        plt.axvline(x=vline, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_barh(labels: List[str], values: List[float], title: str, xlabel: str, outpath: str):
    idx = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.barh(idx, values)
    plt.yticks(idx, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_bars_xticks(labels: List[str], values: List[float], title: str, ylabel: str, outpath: str, rotate: int = 90):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=rotate)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_corr_heatmap(corr: pd.DataFrame, title: str, outpath: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------
# Main EDA routine
# -----------------------------
def run_eda(input_csv: str, outdir: str, target: Optional[str], max_plots: int = 8):
    ensure_dir(outdir)
    plotdir = os.path.join(outdir, "plots")
    ensure_dir(plotdir)

    # 1) Read
    df = pd.read_csv(input_csv)

    # 2) Basic structure
    n_rows, n_cols = df.shape
    dtypes_count = df.dtypes.value_counts()
    id_cols = [c for c in ["Id", "id", "ID"] if c in df.columns]

    # Split by dtype
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    for c in id_cols:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    # 3) Missing values
    miss_cnt = df.isnull().sum().sort_values(ascending=False)
    miss_pct = (miss_cnt / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": miss_cnt, "Missing %": miss_pct})
    missing_df_nonzero = missing_df[missing_df["Missing Count"] > 0]
    # Save top 20 missing barh
    topN = min(20, len(missing_df_nonzero))
    if topN > 0:
        miss_plot_data = missing_df_nonzero.head(topN)
        save_barh(
            labels=list(miss_plot_data.index[::-1]),
            values=list(miss_plot_data["Missing %"][::-1].values),
            title=f"Top {topN} Missing Columns (%)",
            xlabel="Missing %",
            outpath=os.path.join(plotdir, "missing_top20.png"),
        )
        miss_plot_relpath = os.path.join("plots", "missing_top20.png")
    else:
        miss_plot_relpath = None

    # 4) Target distribution (if present)
    target_hist_relpath = None
    target_log_hist_relpath = None
    if target and target in df.columns:
        save_hist(df[target], "Target Distribution", target, os.path.join(plotdir, "target_hist.png"))
        target_hist_relpath = os.path.join("plots", "target_hist.png")
        # Log1p,
        save_hist(np.log1p(df[target]), "Log1p(Target) Distribution", f"log1p({target})", os.path.join(plotdir, "target_log_hist.png"))
        target_log_hist_relpath = os.path.join("plots", "target_log_hist.png")

    # 5) Numeric feature distributions (sample up to max_plots)
    num_hist_paths = []
    sample_num = [c for c in num_cols][:max_plots]
    for col in sample_num:
        outp = os.path.join(plotdir, f"num_hist_{col}.png")
        save_hist(df[col], f"{col} Distribution", col, outp)
        num_hist_paths.append(os.path.join("plots", f"num_hist_{col}.png"))

    # 6) Correlations (numeric only)
    corr_with_target = None
    corr_matrix_relpath = None
    scatter_paths = []
    if target and target in df.columns:
        corr_full = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
        if target in corr_full.columns:
            corr_target = corr_full[target].drop(labels=[target]).sort_values(ascending=False)
            corr_with_target = corr_target
            # heatmap
            corr_matrix_rel = os.path.join("plots", "corr_matrix.png")
            save_corr_heatmap(corr_full, "Correlation Matrix (numeric)", os.path.join(plotdir, "corr_matrix.png"))
            corr_matrix_relpath = corr_matrix_rel
            # scatter for top 3
            for col in corr_target.head(3).index.tolist():
                sp_rel = os.path.join("plots", f"scatter_{col}_vs_{target}.png")
                save_scatter(df[col], df[target], col, target, f"{col} vs {target}", os.path.join(plotdir, f"scatter_{col}_vs_{target}.png"))
                scatter_paths.append(sp_rel)

    # 7) Outlier check (example: GrLivArea vs target)
    outlier_relpath = None
    if target and ("GrLivArea" in df.columns):
        outlier_rel = os.path.join("plots", "outlier_grlivarea.png")
        save_scatter(df["GrLivArea"], df[target], "GrLivArea", target,
                     "GrLivArea vs Target (Outlier Check)", os.path.join(plotdir, "outlier_grlivarea.png"),
                     vline=4000)
        outlier_relpath = outlier_rel

    # 8) Categorical summaries & a category-price bar (Neighborhood)
    cat_count_paths = []
    for col in cat_cols[:max_plots]:
        vc = df[col].value_counts(dropna=False)
        labels = [str(x) for x in vc.index[:12]]
        values = list(vc.values[:12])
        rel = os.path.join("plots", f"cat_count_{col}.png")
        save_barh(labels[::-1], list(reversed(values)), f"{col} (Top 12)", "Count", os.path.join(plotdir, f"cat_count_{col}.png"))
        cat_count_paths.append(rel)

    neigh_relpath = None
    if target and ("Neighborhood" in df.columns):
        means = df.groupby("Neighborhood")[target].mean().sort_values(ascending=False)
        rel = os.path.join("plots", "neighborhood_mean_target.png")
        save_bars_xticks(list(means.index), list(means.values), "Neighborhood vs Mean Target", f"Mean {target}", os.path.join(plotdir, "neighborhood_mean_target.png"))
        neigh_relpath = rel

    # 9) Build Markdown report
    lines = []
    lines.append("# EDA Report\n")
    lines.append(f"**Input file**: `{os.path.basename(input_csv)}`\n")
    lines.append(f"**Rows x Cols**: {n_rows} x {n_cols}\n")
    lines.append(f"**Dtypes**: " + ", ".join([f"{k}: {int(v)}" for k, v in dtypes_count.items()]) + "\n")

    # Missing
    lines.append("## Missing Values\n")
    nz = missing_df_nonzero.copy()
    lines.append("Top missing columns:\n")
    for idx, row in nz.head(10).iterrows():
        lines.append(f"- {idx}: {int(row['Missing Count'])} ({row['Missing %']:.2f}%)")
    if miss_plot_relpath:
        lines.append(f"\n![Missing Top 20]({miss_plot_relpath})\n")

    # Target
    if target and target in df.columns:
        lines.append(f"## Target: {target}\n")
        lines.append("- Distribution often right-skewed; consider log1p transform for modeling.\n")
        if target_hist_relpath:
            lines.append(f"![Target]({target_hist_relpath})\n")
        if target_log_hist_relpath:
            lines.append(f"![Log Target]({target_log_hist_relpath})\n")

    # Numeric dists
    if num_hist_paths:
        lines.append("## Numeric Feature Distributions (samples)\n")
        for p in num_hist_paths:
            lines.append(f"![Num]({p})")

    # Correlation
    if corr_with_target is not None:
        lines.append("\n## Correlation with Target (Top)\n")
        for feat, val in corr_with_target.head(10).items():
            lines.append(f"- {feat}: {val:.3f}")
        if corr_matrix_relpath:
            lines.append(f"\n![Correlation matrix]({corr_matrix_relpath})\n")
        if scatter_paths:
            lines.append("### Scatter vs Target (Top correlated)\n")
            for p in scatter_paths:
                lines.append(f"![Scatter]({p})")

    # Outliers
    if outlier_relpath:
        lines.append("\n## Outliers\n")
        lines.append("Visual check suggests potential large-area outliers around GrLivArea > 4000.\n")
        lines.append(f"![Outliers]({outlier_relpath})\n")

    # Categorical
    if cat_count_paths:
        lines.append("## Categorical Features (samples)\n")
        for p in cat_count_paths:
            lines.append(f"![Cat]({p})")
    if neigh_relpath:
        lines.append(f"\n![Neighborhood Mean {target}]({neigh_relpath})\n")

    # Recommendations
    lines.append("\n## Recommendations\n")
    lines.append("- Treat missing-as-absence columns (e.g., PoolQC, Alley, Fence, FireplaceQu, garage/basement quality fields) as categorical with an explicit 'None' level.\n")
    lines.append("- Consider log1p transform of the target for modeling; inspect and potentially cap extreme area values such as GrLivArea.\n")
    lines.append("- Feature engineering: total floor/basement areas (e.g., TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF), age features (YrSold - YearBuilt, YrSold - YearRemodAdd).\n")
    lines.append("- For tree-based models later: impute numeric with median; categorical via one-hot with rare-category grouping.\n")
    # Note about MSSubClass being categorical code (from the data description)
    lines.append("- Note: columns like MSSubClass are categorical codes for dwelling types rather than continuous numbers.\n")

    report_md = "\n".join(lines)
    report_path = os.path.join(outdir, "EDA_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Also save tabular artifacts
    missing_df_path = os.path.join(outdir, "missing_table.csv")
    missing_df.to_csv(missing_df_path)

    print("=== EDA COMPLETE ===")
    print(f"Report: {report_path}")
    print(f"Plots dir: {plotdir}")
    print(f"Missing table: {missing_df_path}")


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Quick EDA report generator")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--outdir", default="eda_outputs", help="Output directory")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--max-plots", type=int, default=8, help="How many numeric/categorical plots to generate at most")
    args = parser.parse_args()

    run_eda(args.input, args.outdir, args.target, max_plots=args.max_plots)


if __name__ == "__main__":
    main()
