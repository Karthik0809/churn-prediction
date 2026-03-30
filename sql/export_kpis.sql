SELECT
  ROUND(AVG(Churn) * 100.0, 2) AS overall_churn_rate_pct,
  ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk,
  COUNT(CASE WHEN risk_flag = 'High Risk' THEN 1 END) AS high_risk_customers_count,
  ROUND(
    AVG(CASE WHEN risk_flag = 'High Risk' THEN Churn END) * 100.0,
    2
  ) AS high_risk_churn_rate_pct,
  ROUND(AVG(CASE WHEN Churn = 1 THEN MonthlyCharges END), 2) AS avg_monthly_charges_of_churners
FROM churn_features;

