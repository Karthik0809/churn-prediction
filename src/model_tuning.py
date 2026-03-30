from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.utils import Paths, ensure_dirs, get_project_root


def _read_training_frame(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        raw = pd.read_sql("SELECT * FROM raw_data", conn)
        derived = pd.read_sql(
            "SELECT customerID, tenure_bucket, charge_tier, risk_flag FROM churn_features",
            conn,
        )
    return raw.merge(derived, on="customerID", how="left")


def tune_and_train_best(
    *,
    n_iter: int = 20,
    random_state: int = 42,
    test_size: float = 0.2,
    decision_threshold: float = 0.35,
) -> None:
    paths = Paths(get_project_root())
    ensure_dirs(paths.models_dir)

    df = _read_training_frame(paths.processed_db)

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / max(pos, 1)) if pos else 1.0

    base = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=0,
        scale_pos_weight=spw,
    )

    pipe = Pipeline([("preprocess", pre), ("model", base)])

    param_distributions = {
        "model__n_estimators": [300, 400, 500, 700, 900],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.015, 0.02, 0.03, 0.05],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 2, 3, 5],
        "model__gamma": [0.0, 0.1, 0.2],
        "model__reg_lambda": [1.0, 1.5, 2.0],
        "model__reg_alpha": [0.0, 0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=1,
        cv=cv,
        verbose=1,
        random_state=random_state,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_pipe: Pipeline = search.best_estimator_

    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= decision_threshold).astype(int)
    roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    rec = recall_score(y_test, pred)

    print("=== 5-fold CV + RandomizedSearchCV ===")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    print("\n=== Holdout metrics with tuned model ===")
    print(f"Recall (threshold={decision_threshold:.2f}): {rec:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    out_path = paths.models_dir / "xgb_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"Saved tuned model -> {out_path}")


def main() -> None:
    tune_and_train_best()


if __name__ == "__main__":
    main()

