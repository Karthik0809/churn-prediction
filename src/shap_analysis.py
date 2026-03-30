from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import Paths, ensure_dirs, get_project_root


def _read_features(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM churn_features", conn)


def run_shap(
    db_path: Path | None = None,
    model_path: Path | None = None,
    sample_size: int = 1000,
) -> None:
    paths = Paths(get_project_root())
    db_path = db_path or paths.processed_db
    model_path = model_path or (paths.models_dir / "xgb_model.pkl")

    ensure_dirs(paths.assets_dir)

    df = _read_features(db_path)
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    with open(model_path, "rb") as f:
        pipe = pickle.load(f)

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # SHAP works on the transformed (one-hot) feature matrix.
    X_s = X.sample(n=min(sample_size, len(X)), random_state=42)
    X_trans = pre.transform(X_s)
    feature_names = list(pre.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Global summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        features=X_trans,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("Top Churn Drivers — Global (XGBoost + one-hot)")
    plt.tight_layout()
    out1 = paths.assets_dir / "shap_summary.png"
    plt.savefig(out1, dpi=160)
    plt.close()

    # Simple "top 5" list for README/dashboard text
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    top5 = [(feature_names[i], float(mean_abs[i])) for i in top_idx]

    out_txt = paths.assets_dir / "top5_shap_drivers.txt"
    out_txt.write_text(
        "\n".join([f"{name}\t{val:.6f}" for name, val in top5]),
        encoding="utf-8",
    )

    print(f"Saved: {out1}")
    print(f"Saved: {out_txt}")
    print("\nTop 5 drivers (mean |SHAP|):")
    for name, val in top5:
        print(f"- {name}: {val:.4f}")


def main() -> None:
    run_shap()


if __name__ == "__main__":
    main()
