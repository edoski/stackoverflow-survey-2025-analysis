"""
Exploratory Data Analysis (EDA) for Stack Overflow survey data.

Generates comprehensive visualizations for both regression and classification tasks:

Regression (EUR subset):
- Univariate: boxplots, histograms for WorkExp and CompTotal
- Bivariate: scatter plot with regression line
- Multivariate: correlation matrix

Classification (AIAccBin):
- Univariate: target distribution, feature histograms
- Bivariate: boxplots by target, stacked bar charts for categorical features
- Multivariate: correlation matrix for numeric features

All plots are saved to plots_eda/ subdirectories for documentation.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_eur, preprocess_aiaccbin

# EDA constants
PLOT_DPI = 150
HISTOGRAM_BINS = 30
TOP_N_CATEGORIES = 10


def main() -> None:
    """Perform exploratory data analysis on survey data."""
    sns.set_theme(style="whitegrid")

    # ---------- EDA SLR (EUR subset) ----------
    raw = pd.read_csv("survey_results_public.csv", low_memory=False)
    eur = preprocess_eur(raw)
    eur_dir = Path("plots_eda/eur")
    eur_dir.mkdir(parents=True, exist_ok=True)

    # Boxplot CompTotal (univariata)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.boxplot(y=eur["CompTotal"].dropna(), ax=ax, color="#72B7B2")
    ax.set_ylabel("CompTotal (EUR)")
    ax.set_title(f"CompTotal (EUR subset)")
    fig.savefig(eur_dir / f"box_CompTotal.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Istogrammi (univariate)
    for col, xlabel in [("WorkExp", "WorkExp (anni)"), ("CompTotal", "CompTotal (EUR)")]:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        sns.histplot(eur[col].dropna(), bins=HISTOGRAM_BINS, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequenza")
        ax.set_title(f"Istogramma: {xlabel}")
        fig.savefig(eur_dir / f"hist_{col}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    # Matrice di correlazione
    corr = eur[["WorkExp", "CompTotal"]].corr()
    n = corr.shape[0]
    fig_w = max(9, 1.6 * n)
    fig_h = max(7, 1.4 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    ax.set_title("Matrice di correlazione (EUR subset)")
    fig.savefig(eur_dir / "corr_numeric.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---------- EDA Classification ----------
    aiaccbin_dir = Path("plots_eda/aiaccbin")
    aiaccbin_dir.mkdir(parents=True, exist_ok=True)
    cols = ["AIAcc", "WorkExp", "YearsCode", "Age", "EdLevel", "Industry", "Employment"]
    dfc = pd.read_csv("survey_results_public.csv", usecols=cols, low_memory=False)
    dfc = preprocess_aiaccbin(dfc)

    # Distribuzione AIAccBin (univariata)
    labels = dfc["AIAccBin"].map({0: "Distrust(0)", 1: "Trust(1)"})
    vc = labels.value_counts().reindex(["Distrust(0)", "Trust(1)"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.barplot(x=vc.index, y=vc.values, ax=ax)
    ax.set_xlabel("AIAccBin")
    ax.set_ylabel("Count")
    ax.set_title("Distribuzione AIAccBin")
    fig.savefig(aiaccbin_dir / "distribuzione_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Istogrammi di WorkExp e YearsCode (univariate)
    for col, xlabel in [("WorkExp", "WorkExp (anni)"), ("YearsCode", "YearsCode (anni)")]:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        sns.histplot(dfc[col].dropna(), bins=HISTOGRAM_BINS, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequenza")
        ax.set_title(f"Istogramma: {xlabel}")
        fig.savefig(aiaccbin_dir / f"hist_{col}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    # Boxplot WorkExp by AIAccBin (bivariata)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sns.boxplot(data=dfc.replace({"AIAccBin": {0: "Distrust(0)", 1: "Trust(1)"}}), x="AIAccBin", y="WorkExp", ax=ax)
    ax.set_xlabel("AIAccBin")
    ax.set_ylabel("WorkExp (anni)")
    ax.set_title("WorkExp by AIAccBin")
    fig.savefig(aiaccbin_dir / "box_WorkExp_by_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Bar plot Employment vs AIAccBin (proporzioni normalizzate, ordine per Trust)
    top_emp = dfc["Employment"].value_counts().head(TOP_N_CATEGORIES).index
    sub = dfc[dfc["Employment"].isin(top_emp)].copy()
    ct = pd.crosstab(sub["Employment"], sub["AIAccBin"], normalize="index").rename(columns={0: "Distrust(0)", 1: "Trust(1)"})
    ct = ct.sort_values("Trust(1)", ascending=False)
    fig_h = max(6.0, 0.5 * len(ct) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
    ct.plot(kind="barh", stacked=True, ax=ax, width=0.85)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Employment")
    ax.set_title(f"AIAccBin by Employment (Top {TOP_N_CATEGORIES})")
    fig.savefig(aiaccbin_dir / "stacked_Employment_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # Matrice di correlazione
    corr = dfc[["WorkExp", "YearsCode", "AgeMapped", "EdLevelOrd"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    ax.set_title("Matrice di correlazione (aiaccbin subset)")
    fig.savefig(aiaccbin_dir / "corr_numeric.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
