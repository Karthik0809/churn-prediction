"""Kaplan-Meier survival analysis: when do customers churn?

Treats tenure (months) as time-to-event and churn as the event.
Saves assets/eda/kaplan_meier_by_contract.png

Run from repo root:  python -m src.survival_analysis
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import Paths, ensure_dirs, get_project_root


def kaplan_meier(durations: np.ndarray, events: np.ndarray):
    """Plain Kaplan-Meier estimator (no external dependency).

    Returns (times, survival_probabilities)."""
    order = np.argsort(durations)
    durations, events = durations[order], events[order]
    times, surv = [0.0], [1.0]
    s = 1.0
    for t in np.unique(durations[events == 1]):
        n_at_risk = (durations >= t).sum()
        d_events = ((durations == t) & (events == 1)).sum()
        if n_at_risk > 0:
            s *= 1.0 - d_events / n_at_risk
        times.append(float(t))
        surv.append(s)
    return np.array(times), np.array(surv)


def main() -> None:
    paths = Paths(get_project_root())
    df = pd.read_csv(paths.raw_csv if hasattr(paths, "raw_csv") else "data/raw/telco_churn.csv")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    # tenure 0 rows are brand-new accounts; treat as 0.5 months observed
    df["tenure"] = df["tenure"].replace(0, 0.5)

    out_dir = paths.assets_dir / "eda" if hasattr(paths, "assets_dir") else None
    if out_dir is None:
        from pathlib import Path

        out_dir = Path("assets/eda")
    ensure_dirs(out_dir)

    colors = {"Month-to-month": "#d62728", "One year": "#ff7f0e", "Two year": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    print("Median survival (months until 50% of cohort has churned):")
    for contract, sub in df.groupby("Contract"):
        t, s = kaplan_meier(sub["tenure"].values.astype(float), sub["Churn"].values)
        ax.step(t, s, where="post", label=f"{contract} (n={len(sub):,})",
                color=colors.get(contract), linewidth=2)
        below = np.where(s <= 0.5)[0]
        med = f"{t[below[0]]:.0f} months" if len(below) else "not reached within 72 months"
        print(f"  {contract:15s} {med}")

    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival probability (still a customer)")
    ax.set_title("Kaplan-Meier Retention Curves by Contract Type")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "kaplan_meier_by_contract.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
