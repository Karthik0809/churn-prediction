from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    # Ensure `src` imports work when app is launched from /app on Streamlit Cloud.
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_pipeline() -> object:
    model_path = _project_root() / "models" / "xgb_model.pkl"
    if not model_path.exists():
        # Streamlit Cloud deployments start from repo files only; build missing artifacts.
        from src.data_processing import main as build_data
        from src.model import train_xgb

        build_data()
        train_xgb(model_out=model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _engineer_sql_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match `sql/create_features.sql` derived columns:
    - tenure_bucket
    - charge_tier
    - risk_flag
    """
    df = df.copy()

    # Clean TotalCharges like training
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Tenure bucket (SQL: <12, <24, <48, else)
    tenure = df["tenure"]
    df["tenure_bucket"] = np.where(
        tenure < 12,
        "0-1yr",
        np.where(tenure < 24, "1-2yr", np.where(tenure < 48, "2-4yr", "4+yr")),
    )

    # Charge tier (SQL: <35, <65, else)
    monthly = df["MonthlyCharges"]
    df["charge_tier"] = np.where(monthly < 35, "Low", np.where(monthly < 65, "Medium", "High"))

    # Risk flag (SQL)
    df["risk_flag"] = np.where(
        (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > 65),
        "High Risk",
        np.where(df["Contract"] == "Month-to-month", "Medium Risk", "Low Risk"),
    )

    return df


def _required_feature_columns(pipe: object) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for name, _transformer, cols in pre.transformers:
        if name == "cat":
            cat_cols = list(cols)
        if name == "num":
            num_cols = list(cols)
    return num_cols + cat_cols


def _eda_image_paths() -> list[Path]:
    eda_dir = _project_root() / "assets" / "eda"
    if not eda_dir.exists():
        return []
    return sorted(eda_dir.glob("*.png"))


@st.cache_resource(show_spinner=False)
def _get_explainer_and_names(_pipe: object):
    pre = _pipe.named_steps["preprocess"]
    model = _pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    feature_names = list(pre.get_feature_names_out())
    return explainer, feature_names, pre, model


def main() -> None:
    tableau_url = (
        "https://public.tableau.com/app/profile/karthik.mulugu/viz/"
        "CustomerChurnPrediction_17748873322730/"
        "CustomerChurnPredictionFromRiskSignalstoRetentionActions?publish=yes"
    )

    st.set_page_config(page_title="Churn Prediction System", layout="wide")
    # Streamlit 1.19 + newer pandas/pyarrow can fail on LargeUtf8 Arrow dtypes.
    # Use legacy serialization for cloud compatibility.
    try:
        st.set_option("global.dataFrameSerialization", "legacy")
    except Exception:
        pass
    st.title("Customer Churn Prediction System")
    st.markdown("*Predict, explain, and export high-risk customers*")
    st.info(
        f"Explore the interactive business dashboard in Tableau Public: "
        f"[Open Dashboard]({tableau_url})"
    )

    pipe = _load_pipeline()

    st.sidebar.markdown("### Project overview")
    st.sidebar.caption(
        "This app predicts telecom customer churn risk and translates model output "
        "into business-ready actions.\n\n"
        "**What this project does**\n"
        "- Scores churn probability for each uploaded customer\n"
        "- Estimates revenue at risk based on configurable business assumptions\n"
        "- Explains model behavior globally and per customer using SHAP\n"
        "- Shows exploratory data visuals to support pattern discovery\n\n"
        "**Tabs**\n"
        "- **Predictions**: KPI cards, top high-risk customers, CSV export\n"
        "- **Explainability (SHAP)**: global feature impact + individual driver breakdown\n"
        "- **EDA Plots**: clean gallery of distribution, cohort, and risk visuals\n\n"
        "**Controls**\n"
        "- Threshold changes who is flagged as churn risk\n"
        "- Revenue/cost inputs update business impact metrics\n"
        "- SHAP sample size controls speed vs detail"
    )

    uploaded = st.file_uploader("Upload customer data (CSV)", type=["csv"])
    st.caption("Tip: upload the same Telco churn CSV; the app will ignore the `Churn` column if present.")

    if not uploaded:
        st.stop()

    with st.expander("Model and business controls", expanded=False):
        c_ctrl_1, c_ctrl_2, c_ctrl_3 = st.columns(3)
        with c_ctrl_1:
            threshold = st.slider(
                "Retention aggressiveness (probability threshold)",
                min_value=0.05,
                max_value=0.95,
                value=0.35,
                step=0.05,
            )
        with c_ctrl_2:
            avg_monthly = st.number_input("Avg Monthly Revenue per Customer ($)", value=65.0)
        with c_ctrl_3:
            retention_cost = st.number_input("Retention Campaign Cost per Customer ($)", value=10.0)

    df = pd.read_csv(uploaded)
    df = df.copy()

    # Keep a copy for display/export
    df_out = df.copy()

    # Remove target if present (prediction mode)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
        df_out = df_out.drop(columns=["Churn"], errors="ignore")

    # Ensure SQL-derived features exist (so pipeline input matches training)
    df = _engineer_sql_features(df)

    # Predict
    required_cols = _required_feature_columns(pipe)
    if "customerID" in df.columns:
        X_input = df.drop(columns=["customerID"])
    else:
        X_input = df

    missing = [c for c in required_cols if c not in X_input.columns]
    if missing:
        st.error(
            "Uploaded CSV is missing required feature columns for this model: "
            + ", ".join(missing)
        )
        st.stop()

    X_input = X_input[required_cols]
    proba = pipe.predict_proba(X_input)[:, 1]
    pred = (proba >= threshold).astype(int)

    df_out["Churn_Probability"] = proba
    df_out["Predicted_Churn"] = pred
    df_out["Risk_Level"] = pd.cut(
        proba, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"], include_lowest=True
    )

    explainer, feature_names, pre, _model = _get_explainer_and_names(pipe)

    tab1, tab2, tab3 = st.tabs(["Predictions", "Explainability (SHAP)", "EDA Plots"])

    with tab1:
        st.markdown(
            """
            **Model Card**
            - Model: XGBoost (one-hot encoded features)
            - Decision threshold: `0.35`
            - Recall: `0.88` | ROC-AUC: `0.8462` | PR-AUC: `0.6632`
            - Cost framing: missing 1 churner (`~$777/yr`) vs false offer (`$10`) -> `~78x`
            """
        )

        churners = int(pred.sum())
        revenue_at_risk = churners * float(avg_monthly)
        campaign_cost = churners * float(retention_cost)
        net_saving = revenue_at_risk - campaign_cost

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers at Risk", f"{churners:,}")
        c2.metric("Revenue at Risk (monthly)", f"${revenue_at_risk:,.0f}")
        c3.metric("Campaign Cost", f"${campaign_cost:,.0f}")
        c4.metric("Net Saving Potential", f"${net_saving:,.0f}")

        st.subheader("Top 20 highest-risk customers")
        display_cols = [c for c in ["customerID", "Contract", "MonthlyCharges", "tenure"] if c in df_out.columns]
        display_cols += ["Churn_Probability", "Risk_Level", "Predicted_Churn"]
        top20 = df_out.sort_values("Churn_Probability", ascending=False).head(20)[display_cols].copy()
        # Avoid Arrow serialization path on older Streamlit by rendering HTML.
        for col in top20.columns:
            top20[col] = top20[col].astype(str)
        st.markdown(top20.to_html(index=False, escape=True), unsafe_allow_html=True)

        st.subheader("Export high-risk customers")
        high_risk = df_out[df_out["Risk_Level"] == "High"].sort_values("Churn_Probability", ascending=False)
        csv_bytes = high_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download High-Risk Customers (CSV)",
            data=csv_bytes,
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Global explanation (SHAP) for uploaded file")
        shap_sample_size = st.number_input(
            "SHAP global sample size (for speed)",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
        )

        X_for_shap = X_input
        if len(X_for_shap) > shap_sample_size:
            X_for_shap = X_for_shap.sample(n=shap_sample_size, random_state=42)

        X_trans = pre.transform(X_for_shap)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        import matplotlib.pyplot as plt  # local import to keep startup fast

        if "shap_fullscreen" not in st.session_state:
            st.session_state["shap_fullscreen"] = False

        c_mode_1, c_mode_2 = st.columns([1, 8])
        with c_mode_1:
            if st.button("⤢", key="shap_toggle_full", help="Toggle full-page SHAP view"):
                st.session_state["shap_fullscreen"] = not st.session_state["shap_fullscreen"]

        shap_size = (14, 8) if st.session_state["shap_fullscreen"] else (10, 4.8)
        plt.figure(figsize=shap_size)
        try:
            shap.summary_plot(
                shap_values,
                X_trans,
                feature_names=feature_names,
                show=False,
                max_display=15,
            )
            fig = plt.gcf()
        except ValueError:
            plt.clf()
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:15]
            top_names = [feature_names[i] for i in top_idx][::-1]
            top_vals = [float(mean_abs[i]) for i in top_idx][::-1]
            fig, ax = plt.subplots(figsize=shap_size)
            ax.barh(top_names, top_vals)
            ax.set_title("Top SHAP Drivers")
            ax.set_xlabel("mean(|SHAP value|)")
            plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

        st.subheader("Customer-level explanation (top SHAP drivers)")
        if "customerID" in df_out.columns:
            risk_sorted = df_out.sort_values("Churn_Probability", ascending=False)
            candidate_rows = risk_sorted.head(300).index.tolist()
            picked_pos = st.number_input(
                "Select customer rank (0 = highest risk)",
                min_value=0,
                max_value=max(len(candidate_rows) - 1, 0),
                value=0,
                step=1,
            )
            sel_row = int(candidate_rows[int(picked_pos)])
            selected_id = str(df_out.iloc[sel_row]["customerID"])
            st.caption(f"Selected customerID: {selected_id}")
        else:
            sel_row = st.number_input(
                "Select a row index (0-based)",
                min_value=0,
                max_value=max(len(df_out) - 1, 0),
                value=0,
            )

        X_sel = X_input.iloc[[int(sel_row)]]
        X_sel_trans = pre.transform(X_sel)
        shap_sel = explainer.shap_values(X_sel_trans)
        if isinstance(shap_sel, list):
            shap_vec = shap_sel[1][0]
        else:
            shap_vec = shap_sel[0]

        top_k = 5
        top_idx = np.argsort(np.abs(shap_vec))[::-1][:top_k]
        top_drivers = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in top_idx],
                "SHAP value": [float(shap_vec[i]) for i in top_idx],
            }
        )
        top_drivers["Feature"] = top_drivers["Feature"].astype(str)
        top_drivers["SHAP value"] = top_drivers["SHAP value"].map(lambda x: f"{x:.4f}")
        st.markdown(top_drivers.to_html(index=False, escape=True), unsafe_allow_html=True)

    with tab3:
        st.subheader("EDA plot gallery")
        images = _eda_image_paths()
        if not images:
            st.info("No EDA plots found. Run `python -m src.eda_plots` first.")
        else:
            if "selected_eda_plot" not in st.session_state:
                st.session_state["selected_eda_plot"] = str(images[0])
            names = [img.name for img in images]
            current_name = Path(st.session_state["selected_eda_plot"]).name
            selected_name = st.selectbox(
                "Select plot",
                options=names,
                index=names.index(current_name) if current_name in names else 0,
            )
            selected_path = str(next(img for img in images if img.name == selected_name))
            st.session_state["selected_eda_plot"] = selected_path

            # Main clean full-size view
            st.image(selected_path, caption=selected_name, use_column_width=True)

            # Optional compact overview strip
            with st.expander("Show all plot thumbnails", expanded=False):
                cols = st.columns(3)
                for i, img in enumerate(images):
                    with cols[i % 3]:
                        st.image(str(img), caption=img.name, width=320)


if __name__ == "__main__":
    main()
