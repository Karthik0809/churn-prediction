from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from src.data_processing import build_feature_table, load_raw_to_sqlite
from src.utils import Paths, get_project_root


class TestPreprocessingAssumptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths = Paths(get_project_root())
        cls.db_path: Path = paths.processed_db
        load_raw_to_sqlite(paths.raw_csv, cls.db_path)
        build_feature_table(cls.db_path, paths.sql_dir / "create_features.sql")

    def test_totalcharges_is_numeric_and_not_null(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n_null FROM raw_data WHERE TotalCharges IS NULL"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(int(row[0]), 0)

    def test_required_columns_exist_in_feature_table(self) -> None:
        required = {
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "PaymentMethod",
            "InternetService",
            "TechSupport",
            "OnlineSecurity",
            "tenure_bucket",
            "charge_tier",
            "risk_flag",
            "Churn",
        }
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("PRAGMA table_info(churn_features)").fetchall()
        cols = {r[1] for r in rows}
        self.assertTrue(required.issubset(cols))


if __name__ == "__main__":
    unittest.main()

