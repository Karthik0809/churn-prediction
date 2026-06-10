"""Chi-square significance tests for the README Key Findings.

Run from repo root:  python -m src.significance_tests
"""
import pandas as pd
from scipy.stats import chi2_contingency


def chi2_test(df, name, mask_a, mask_b):
    a, b = df[mask_a]["Churn"], df[mask_b]["Churn"]
    table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
    chi2, p, _, _ = chi2_contingency(table)
    print(f"{name}: {a.mean()*100:.2f}% vs {b.mean()*100:.2f}%  chi2={chi2:.1f}  p={p:.2e}")
    return chi2, p


def main():
    df = pd.read_csv("data/raw/telco_churn.csv")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    chi2_test(df, "Contract M2M vs 2yr", df.Contract == "Month-to-month", df.Contract == "Two year")
    chi2_test(df, "Tenure <12 vs >=12", df.tenure < 12, df.tenure >= 12)
    chi2_test(df, "TechSupport No vs Yes", df.TechSupport == "No", df.TechSupport == "Yes")
    chi2_test(df, "OnlineSecurity No vs Yes", df.OnlineSecurity == "No", df.OnlineSecurity == "Yes")

    seg = (df.Contract == "Month-to-month") & (df.MonthlyCharges > 65)
    chi2_test(df, "HighCharge-M2M vs rest", seg, ~seg)
    share = df[seg]["Churn"].sum() / df["Churn"].sum() * 100
    print(f"HighCharge-M2M share of all churners: {share:.2f}%")


if __name__ == "__main__":
    main()
