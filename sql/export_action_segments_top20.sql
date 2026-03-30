SELECT
  Contract,
  tenure_bucket,
  risk_flag,
  COUNT(*) AS customers,
  SUM(Churn) AS churned,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
  ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk,
  ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END) * 12, 2) AS annual_revenue_at_risk
FROM churn_features
GROUP BY Contract, tenure_bucket, risk_flag
ORDER BY monthly_revenue_at_risk DESC
LIMIT 20;

