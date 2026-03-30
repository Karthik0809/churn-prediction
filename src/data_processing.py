from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.utils import Paths, ensure_dirs, get_project_root


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Match the Kaggle notebook behavior: numeric conversion + fill.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def load_raw_to_sqlite(
    raw_csv: Path | None = None,
    db_path: Path | None = None,
) -> None:
    paths = Paths(get_project_root())
    raw_csv = raw_csv or paths.raw_csv
    db_path = db_path or paths.processed_db

    ensure_dirs(db_path.parent)

    df = pd.read_csv(raw_csv)
    df = _clean_raw(df)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("raw_data", conn, if_exists="replace", index=False)


def run_sql_file(db_path: Path, sql_path: Path) -> None:
    sql_text = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql_text)


def build_feature_table(
    db_path: Path | None = None,
    sql_path: Path | None = None,
) -> None:
    paths = Paths(get_project_root())
    db_path = db_path or paths.processed_db
    sql_path = sql_path or (paths.sql_dir / "create_features.sql")

    run_sql_file(db_path, sql_path)


def main() -> None:
    paths = Paths(get_project_root())
    ensure_dirs(paths.assets_dir, paths.models_dir, paths.processed_db.parent)

    load_raw_to_sqlite(paths.raw_csv, paths.processed_db)
    build_feature_table(paths.processed_db, paths.sql_dir / "create_features.sql")

    print(f"SQLite DB ready: {paths.processed_db}")


if __name__ == "__main__":
    main()
