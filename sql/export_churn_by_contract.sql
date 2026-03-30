SELECT
  Contract,
  COUNT(*) AS total_customers,
  SUM(Churn) AS churned,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
  ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM churn_features
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

