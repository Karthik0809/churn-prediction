from __future__ import annotations

from src.data_processing import main as build_data
from src.eda_plots import generate_eda_plots
from src.export_tableau import main as export_tableau
from src.model import train_xgb
from src.shap_analysis import run_shap


def main() -> None:
    print("1/5 Building SQLite data layer...")
    build_data()

    print("2/5 Generating EDA plots...")
    generate_eda_plots()

    print("3/5 Training model...")
    train_xgb()

    print("4/5 Generating SHAP assets...")
    run_shap()

    print("5/5 Exporting Tableau workbook...")
    export_tableau()

    print("\nAll pipeline steps completed.")


if __name__ == "__main__":
    main()

