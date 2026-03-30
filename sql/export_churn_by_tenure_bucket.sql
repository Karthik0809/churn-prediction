SELECT
  tenure_bucket,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_spend,
  ROUND(AVG(TotalCharges), 2) AS avg_total_charges
FROM churn_features
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

