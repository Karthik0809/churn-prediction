SELECT
  TechSupport,
  OnlineSecurity,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM churn_features
GROUP BY TechSupport, OnlineSecurity
ORDER BY churn_rate_pct DESC;

