from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import Paths, ensure_dirs, get_project_root


def _read_raw(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM raw_data", conn)


def _read_features(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM churn_features", conn)


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_eda_plots(db_path: Path | None = None) -> None:
    sns.set_theme(style="whitegrid")

    paths = Paths(get_project_root())
    db_path = db_path or paths.processed_db
    out_dir = paths.assets_dir / "eda"
    ensure_dirs(out_dir)

    raw = _read_raw(db_path)
    feat = _read_features(db_path)

    # ---- Plot 1: Overall churn distribution (pie) ----
    churn_counts = raw["Churn"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.pie(
        churn_counts,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#4C9BE8", "#E8694C"],
        explode=[0, 0.06],
    )
    ax.set_title("Overall Churn Distribution", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "01_churn_distribution.png")

    # ---- Plot 2: Churn rate by categorical features (Kaggle-style) ----
    cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
    ]
    n = len(cat_cols)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ct = raw.groupby(col)["Churn"].mean().sort_values(ascending=False)
        ct.plot(kind="bar", ax=axes[i], edgecolor="black", width=0.65)
        axes[i].set_title(f"Churn Rate by {col}", fontsize=11)
        axes[i].set_ylabel("Churn Rate")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=25)
        for bar in axes[i].patches:
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        "Churn Rate Across Categorical Features",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    _save(fig, out_dir / "02_churn_rate_categorical.png")

    # ---- Plot 3: Numeric distributions by churn (Kaggle-style KDE) ----
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col in zip(axes, num_cols):
        sns.kdeplot(data=raw, x=col, hue="Churn", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=12)
    fig.suptitle("Numeric Feature Distributions by Churn", fontsize=14, fontweight="bold")
    _save(fig, out_dir / "03_numeric_distributions.png")

    # ---- Plot 4: Correlation heatmap (numeric only; cleaner & less misleading) ----
    numeric = raw[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation (Numeric Features)", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "04_corr_numeric.png")

    # ---- Plot 5: Tenure vs MonthlyCharges scatter (Kaggle-style) ----
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    sns.scatterplot(
        data=raw,
        x="tenure",
        y="MonthlyCharges",
        hue="Churn",
        alpha=0.35,
        s=20,
        ax=ax,
        palette={0: "#4C9BE8", 1: "#E8694C"},
    )
    ax.set_title("Tenure vs Monthly Charges by Churn", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "05_tenure_vs_monthly_scatter.png")

    # ================= Extra plots beyond the Kaggle notebook =================

    # ---- Plot 6: Churn rate by tenure bucket (business-friendly) ----
    tmp = feat.groupby("tenure_bucket", as_index=False)["Churn"].mean()
    tmp["churn_rate_pct"] = tmp["Churn"] * 100
    order = ["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.barplot(data=tmp, x="tenure_bucket", y="churn_rate_pct", order=order, ax=ax)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Tenure bucket")
    ax.set_title("Churn Rate by Tenure Bucket", fontsize=13, fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    _save(fig, out_dir / "06_churn_by_tenure_bucket.png")

    # ---- Plot 7: MonthlyCharges by churn (violin + box) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.violinplot(data=raw, x="Churn", y="MonthlyCharges", inner=None, ax=ax, palette="pastel")
    sns.boxplot(data=raw, x="Churn", y="MonthlyCharges", width=0.25, ax=ax, color="white")
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_title("Monthly Charges vs Churn", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "07_monthly_charges_violin.png")

    # ---- Plot 8: Revenue-at-risk proxy by contract (churned_count * avg_monthly) ----
    by_contract = feat.groupby("Contract").agg(
        customers=("Churn", "size"),
        churn_rate=("Churn", "mean"),
        churned=("Churn", "sum"),
        avg_monthly=("MonthlyCharges", "mean"),
    )
    by_contract["monthly_revenue_at_risk"] = by_contract["churned"] * by_contract["avg_monthly"]
    by_contract = by_contract.sort_values("monthly_revenue_at_risk", ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=by_contract, x="Contract", y="monthly_revenue_at_risk", ax=ax)
    ax.set_ylabel("Monthly revenue at risk (proxy, $)")
    ax.set_title("Revenue at Risk by Contract Type", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir / "08_revenue_at_risk_by_contract.png")

    # ---- Plot 9: Heatmap — churn rate by TechSupport x OnlineSecurity ----
    pivot = feat.pivot_table(
        index="TechSupport",
        columns="OnlineSecurity",
        values="Churn",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="Reds", linewidths=0.5, ax=ax)
    ax.set_title("Churn Rate (%) — TechSupport × OnlineSecurity", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("OnlineSecurity")
    ax.set_ylabel("TechSupport")
    _save(fig, out_dir / "09_support_services_heatmap.png")

    # ---- Plot 10: Payment method — churn rate + customer count (dual) ----
    pay = raw.groupby("PaymentMethod").agg(
        customers=("Churn", "size"),
        churn_rate=("Churn", "mean"),
    ).reset_index()
    pay["churn_rate_pct"] = pay["churn_rate"] * 100
    pay = pay.sort_values("churn_rate_pct", ascending=False)
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=pay, x="PaymentMethod", y="churn_rate_pct", ax=ax1)
    ax1.set_ylabel("Churn rate (%)")
    ax1.set_xlabel("")
    ax1.set_title("Churn by Payment Method (rate + size)", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(pay["PaymentMethod"], pay["customers"], color="black", marker="o", linewidth=2)
    ax2.set_ylabel("Customer count")
    _save(fig, out_dir / "10_payment_method_rate_and_size.png")

    print(f"Saved EDA plots to: {out_dir}")


def main() -> None:
    generate_eda_plots()


if __name__ == "__main__":
    main()
