from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.data_processing import build_feature_table, load_raw_to_sqlite
from src.utils import Paths, ensure_dirs, get_project_root


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def export_all_to_xlsx(
    *,
    db_path: Path,
    mapping: dict[str, Path],
    out_xlsx: Path,
) -> None:
    """
    Write a single Excel workbook with one sheet per export.
    This is the easiest way to open everything at once in Tableau (Excel connector).
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sheet_name, sql_path in mapping.items():
            sql = _read_sql(sql_path)
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql(sql, conn)
            safe_sheet = sheet_name[:31]  # Excel limit
            df.to_excel(writer, sheet_name=safe_sheet, index=False)


def main() -> None:
    paths = Paths(get_project_root())
    # Ensure DB exists (DB builder is idempotent)

    # Ensure DB exists
    if not paths.processed_db.exists():
        load_raw_to_sqlite(paths.raw_csv, paths.processed_db)
        build_feature_table(paths.processed_db, paths.sql_dir / "create_features.sql")

    exports_dir = paths.project_root / "dashboard" / "exports"
    ensure_dirs(exports_dir)

    # One-click Tableau import (Excel workbook with 5 sheets)
    xlsx_mapping = {
        "01_kpis": paths.sql_dir / "export_kpis.sql",
        "02_churn_by_contract": paths.sql_dir / "export_churn_by_contract.sql",
        "03_churn_by_tenure_bucket": paths.sql_dir / "export_churn_by_tenure_bucket.sql",
        "04_support_services_vs_churn": paths.sql_dir / "export_support_services_vs_churn.sql",
        "05_action_segments_top20": paths.sql_dir / "export_action_segments_top20.sql",
    }
    out_xlsx = exports_dir / "tableau_exports.xlsx"
    export_all_to_xlsx(db_path=paths.processed_db, mapping=xlsx_mapping, out_xlsx=out_xlsx)
    print(f"Exported: {out_xlsx}")

    print(f"\nTableau export written to: {exports_dir}")


if __name__ == "__main__":
    main()

