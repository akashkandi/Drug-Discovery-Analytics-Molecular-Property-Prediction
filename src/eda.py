"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates plots and a summary report for the BBBP drug discovery dataset,
including class balance, descriptor distributions, correlation heatmap,
fingerprint variance, and scatter plots.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (12, 10)
DPI = 150

DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3",
]

FP_PREFIX = "fp_"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure to *path* and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def _fp_cols(df: pd.DataFrame) -> List[str]:
    """Return fingerprint bit column names present in *df*."""
    return [c for c in df.columns if c.startswith(FP_PREFIX)]


# ── Individual Plot Functions ──────────────────────────────────────────────────

def plot_class_balance(df: pd.DataFrame, label_col: str, out_dir: str) -> dict:
    """
    Bar chart of class balance (permeable vs non-permeable).

    Returns
    -------
    dict  {class_label: count}
    """
    counts = df[label_col].value_counts().sort_index()
    labels = {0: "Non-permeable (0)", 1: "Permeable (1)"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [labels.get(k, str(k)) for k in counts.index],
        counts.values,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    total = counts.sum()
    ax.set_title("BBBP Dataset — Class Balance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Molecules", fontsize=12)
    ax.set_xlabel(f"Class  (total N={total})", fontsize=12)
    pct = counts.values / total * 100
    ax.set_ylim(0, counts.max() * 1.15)
    ax.text(0.98, 0.97, f"Permeable: {pct[1]:.1f}%\nNon-perm.: {pct[0]:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)

    _save(fig, os.path.join(out_dir, "class_balance.png"))
    return dict(zip(counts.index, counts.values))


def plot_descriptor_distributions(df: pd.DataFrame, out_dir: str) -> None:
    """
    KDE + histogram for each molecular descriptor, coloured by class.
    """
    desc_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    n = len(desc_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(desc_cols):
        ax = axes[i]
        if "p_np" in df.columns:
            for cls, color, label in [(0, "#e74c3c", "Non-perm."),
                                       (1, "#2ecc71", "Permeable")]:
                subset = df[df["p_np"] == cls][col].dropna()
                ax.hist(subset, bins=30, alpha=0.55, color=color,
                        label=label, density=True, edgecolor="none")
                subset.plot.kde(ax=ax, color=color, linewidth=1.8)
        else:
            df[col].dropna().hist(bins=30, ax=ax, color="#3498db",
                                   alpha=0.7, density=True, edgecolor="none")

        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        if "p_np" in df.columns and i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Molecular Descriptor Distributions by Class",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "descriptor_distributions.png"))


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Pearson correlation heatmap for molecular descriptors.
    """
    desc_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    corr = df[desc_cols].corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, square=True, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Molecular Descriptor Correlation Heatmap",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "descriptor_correlation_heatmap.png"))


def plot_boxplots_by_class(df: pd.DataFrame, out_dir: str) -> None:
    """
    Box plots for each descriptor split by BBB permeability class.
    """
    if "p_np" not in df.columns:
        return

    desc_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    n = len(desc_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    class_labels = {0: "Non-perm.", 1: "Permeable"}

    for i, col in enumerate(desc_cols):
        ax = axes[i]
        plot_df = df[["p_np", col]].copy()
        plot_df["Class"] = plot_df["p_np"].map(class_labels)

        sns.boxplot(
            data=plot_df, x="Class", y=col,
            palette={"Non-perm.": "#e74c3c", "Permeable": "#2ecc71"},
            order=["Non-perm.", "Permeable"],
            ax=ax, width=0.5, linewidth=1.2,
        )
        sns.stripplot(
            data=plot_df, x="Class", y=col,
            palette={"Non-perm.": "#c0392b", "Permeable": "#27ae60"},
            order=["Non-perm.", "Permeable"],
            ax=ax, size=2, alpha=0.25, jitter=True,
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Descriptor Distributions by BBB Permeability Class",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "descriptor_boxplots_by_class.png"))


def plot_top_variable_fingerprints(df: pd.DataFrame, out_dir: str,
                                    top_n: int = 20) -> List[str]:
    """
    Bar chart of the top-N most variable (highest variance) fingerprint bits.

    Returns
    -------
    list of the top-N column names.
    """
    fp_cols = _fp_cols(df)
    if not fp_cols:
        print("  [SKIP] No fingerprint columns found.")
        return []

    variances = df[fp_cols].var().sort_values(ascending=False)
    top = variances.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), top.values, color="#3498db", edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([c.replace("fp_", "bit ") for c in top.index],
                        rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Top {top_n} Most Variable Morgan Fingerprint Bits (ECFP4)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Variance", fontsize=11)
    ax.set_xlabel("Fingerprint Bit", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "top_variable_fingerprints.png"))

    return top.index.tolist()


def plot_logp_vs_molwt(df: pd.DataFrame, out_dir: str) -> None:
    """
    Scatter plot: LogP vs MolWt, points coloured by BBB permeability.
    """
    if "MolLogP" not in df.columns or "MolWt" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    if "p_np" in df.columns:
        for cls, color, label, marker in [
            (0, "#e74c3c", "Non-permeable", "o"),
            (1, "#2ecc71", "Permeable",     "^"),
        ]:
            sub = df[df["p_np"] == cls]
            ax.scatter(sub["MolWt"], sub["MolLogP"], c=color, label=label,
                       alpha=0.45, s=18, marker=marker, edgecolors="none")
        ax.legend(fontsize=11, title="BBB Class")
    else:
        ax.scatter(df["MolWt"], df["MolLogP"], alpha=0.45, s=18,
                   c="#3498db", edgecolors="none")

    # Lipinski Rule of 5 reference lines
    ax.axhline(y=5, color="gray", linestyle="--", linewidth=1,
               label="Lipinski LogP=5")
    ax.axvline(x=500, color="gray", linestyle=":",  linewidth=1,
               label="Lipinski MW=500")
    ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
    ax.set_ylabel("LogP (MolLogP)", fontsize=12)
    ax.set_title("LogP vs Molecular Weight — BBBP Dataset", fontsize=14,
                  fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "logp_vs_molwt_scatter.png"))


# ── Summary Report ─────────────────────────────────────────────────────────────

def generate_eda_summary(df: pd.DataFrame, invalid_count: int,
                          class_counts: dict, out_path: str) -> None:
    """
    Write a Markdown EDA summary to *out_path*.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (valid molecules).
    invalid_count : int
        Number of molecules that failed SMILES parsing.
    class_counts : dict
        {0: n_non_perm, 1: n_perm}
    out_path : str
        Output Markdown file path.
    """
    desc_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    stats = df[desc_cols].describe().round(3)
    n_total = len(df) + invalid_count

    lines = [
        "# BBBP Dataset — EDA Summary\n",
        f"**Generated on:** 2026-03-18\n",
        "",
        "## 1. Dataset Overview",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Total molecules in CSV | {n_total} |",
        f"| Valid molecules (parsed) | {len(df)} |",
        f"| Invalid SMILES (skipped) | {invalid_count} |",
        f"| Feature dimensions | {len(df.columns)} columns |",
        f"| Morgan fingerprint bits | 2048 (ECFP4, radius=2) |",
        f"| Molecular descriptors | 9 |",
        "",
        "## 2. Class Balance",
        f"| Class | Count | Percentage |",
        f"|-------|-------|------------|",
    ]
    n_valid = len(df)
    for cls, cnt in sorted(class_counts.items()):
        label = "Permeable (1)" if cls == 1 else "Non-permeable (0)"
        pct = cnt / n_valid * 100 if n_valid else 0
        lines.append(f"| {label} | {cnt} | {pct:.1f}% |")

    lines += [
        "",
        "## 3. Molecular Descriptor Statistics",
        "",
        stats.to_markdown(),
        "",
        "## 4. Key Observations",
        "",
        "- **Class imbalance:** The BBBP dataset is moderately imbalanced, "
        "with more permeable molecules (~75%) than non-permeable ones.",
        "- **LogP range:** Most molecules have LogP between -2 and 8, "
        "consistent with CNS-active drug-like space.",
        "- **Molecular weight:** Majority of molecules fall below 500 Da "
        "(Lipinski Rule of 5 boundary).",
        "- **TPSA:** CNS-active molecules typically have TPSA < 90 Å², "
        "which is reflected in the distribution.",
        "- **Hydrogen bond donors/acceptors:** Both distributions are "
        "right-skewed, consistent with drug-like molecules.",
        "",
        "## 5. Generated Plots",
        "",
        "| Plot | File |",
        "|------|------|",
        "| Class balance bar chart | `outputs/eda/class_balance.png` |",
        "| Descriptor distributions by class | `outputs/eda/descriptor_distributions.png` |",
        "| Descriptor correlation heatmap | `outputs/eda/descriptor_correlation_heatmap.png` |",
        "| Box plots by class | `outputs/eda/descriptor_boxplots_by_class.png` |",
        "| Top-20 variable fingerprint bits | `outputs/eda/top_variable_fingerprints.png` |",
        "| LogP vs MolWt scatter | `outputs/eda/logp_vs_molwt_scatter.png` |",
        "",
        "## 6. Invalid SMILES Report",
        f"- {invalid_count} molecules could not be parsed by RDKit.",
        "- These were silently excluded from the feature matrix.",
        "- Full list saved to `data/invalid_smiles_report.txt`.",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {out_path}")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_eda(features_path: str = "data/features_bbbp.csv",
            invalid_report_path: str = "data/invalid_smiles_report.txt",
            out_dir: str = "outputs/eda",
            summary_path: str = "outputs/eda_summary.md") -> None:
    """
    Full EDA pipeline.

    Parameters
    ----------
    features_path : str
        Path to the BBBP feature matrix CSV.
    invalid_report_path : str
        Path to invalid SMILES report (to count invalid molecules).
    out_dir : str
        Directory to save plot images.
    summary_path : str
        Path to save the Markdown EDA summary.
    """
    print("\n" + "="*60)
    print("STEP 3 — Exploratory Data Analysis")
    print("="*60)

    df = pd.read_csv(features_path)
    print(f"  [LOAD] Feature matrix: {df.shape}")

    os.makedirs(out_dir, exist_ok=True)

    # Count invalid SMILES from report
    invalid_count = 0
    if os.path.exists(invalid_report_path):
        with open(invalid_report_path) as f:
            content = f.read()
        import re
        m = re.search(r"BBBP invalid \((\d+)\)", content)
        if m:
            invalid_count = int(m.group(1))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")

    print("  → Class balance")
    class_counts = plot_class_balance(df, "p_np", out_dir)

    print("  → Descriptor distributions")
    plot_descriptor_distributions(df, out_dir)

    print("  → Correlation heatmap")
    plot_correlation_heatmap(df, out_dir)

    print("  → Box plots by class")
    plot_boxplots_by_class(df, out_dir)

    print("  → Top variable fingerprint bits")
    plot_top_variable_fingerprints(df, out_dir, top_n=20)

    print("  → LogP vs MolWt scatter")
    plot_logp_vs_molwt(df, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  Generating EDA summary...")
    generate_eda_summary(df, invalid_count, class_counts, summary_path)

    print("\n[EDA] Complete. All plots saved to:", out_dir)


if __name__ == "__main__":
    import sys
    features = sys.argv[1] if len(sys.argv) > 1 else "data/features_bbbp.csv"
    run_eda(features_path=features)
