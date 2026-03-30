-- Extra segment queries useful for Tableau.

-- High-risk segments by contract + tenure bucket
SELECT
  Contract,
  tenure_bucket,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
FROM churn_features
GROUP BY Contract, tenure_bucket
ORDER BY churn_rate_pct DESC, customers DESC;

-- Payment method churn rate
SELECT
  PaymentMethod,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct
FROM churn_features
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- Internet service churn rate
SELECT
  InternetService,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct
FROM churn_features
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
