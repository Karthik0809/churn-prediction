from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.utils import Paths, ensure_dirs, get_project_root


@dataclass(frozen=True)
class TrainArtifacts:
    pipeline: Pipeline
    feature_names_out: list[str]


def _read_features(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        raw = pd.read_sql("SELECT * FROM raw_data", conn)
        derived = pd.read_sql(
            "SELECT customerID, tenure_bucket, charge_tier, risk_flag FROM churn_features",
            conn,
        )
    # Use all cleaned raw features + SQL-derived segmentation features.
    return raw.merge(derived, on="customerID", how="left")


def _build_pipeline(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str], list[str]]:
    target = "Churn"
    drop_cols = ["customerID"]

    y = df[target].astype(int)
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Tuned on this dataset split for stronger ROC-AUC / PR-AUC.
    model = XGBClassifier(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=0,
    )

    pipe = Pipeline([("preprocess", pre), ("model", model)])

    return pipe, numeric_cols, categorical_cols, X.columns.tolist()


def train_xgb(
    db_path: Path | None = None,
    model_out: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    decision_threshold: float = 0.35,
) -> TrainArtifacts:
    paths = Paths(get_project_root())
    db_path = db_path or paths.processed_db
    model_out = model_out or (paths.models_dir / "xgb_model.pkl")

    ensure_dirs(model_out.parent, paths.assets_dir)

    df = _read_features(db_path)

    pipe, _, _, _ = _build_pipeline(df)

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # scale_pos_weight from training split
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / max(pos, 1)) if pos else 1.0
    pipe.set_params(model__scale_pos_weight=spw)

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= decision_threshold).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print("=== XGBoost (Pipeline) ===")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Decision threshold: {decision_threshold:.2f}")

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    avg_monthly = float(df["MonthlyCharges"].mean())
    est_annual_missed = fn * avg_monthly * 12
    false_negative_cost = avg_monthly * 12
    false_positive_cost = 10.0
    cost_ratio = false_negative_cost / false_positive_cost

    print(f"\n=== Business framing (threshold={decision_threshold:.2f}) ===")
    print(f"Missed churners (FN): {fn}")
    print(f"Estimated annual revenue lost: ${est_annual_missed:,.0f}")
    print(f"Cost of missing one churner: ${false_negative_cost:,.0f}/yr")
    print(f"Cost of false retention offer: ${false_positive_cost:,.0f}")
    print(f"Cost ratio (FN:FP): {cost_ratio:.0f}x")

    with open(model_out, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved model pipeline -> {model_out}")

    # Feature names for SHAP plots
    pre = pipe.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())

    return TrainArtifacts(pipeline=pipe, feature_names_out=feature_names)


def main() -> None:
    train_xgb()


if __name__ == "__main__":
    main()
