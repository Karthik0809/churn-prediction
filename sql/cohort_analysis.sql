-- Cohort / segment style queries against `churn_features`.

-- 1) Churn rate by contract type + revenue-at-risk proxy
SELECT
  Contract,
  COUNT(*) AS total_customers,
  SUM(Churn) AS churned,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
  ROUND(SUM(Churn) * AVG(MonthlyCharges), 2) AS monthly_revenue_at_risk
FROM churn_features
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 2) Average spend by tenure bucket + churn rate
SELECT
  tenure_bucket,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
  ROUND(AVG(TotalCharges), 2) AS avg_total_charges
FROM churn_features
GROUP BY tenure_bucket
ORDER BY churn_rate_pct DESC;

-- 3) Revenue at risk by charge tier and risk flag
SELECT
  charge_tier,
  risk_flag,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct,
  ROUND(SUM(Churn) * AVG(MonthlyCharges), 2) AS monthly_revenue_at_risk
FROM churn_features
GROUP BY charge_tier, risk_flag
ORDER BY monthly_revenue_at_risk DESC;

-- 4) "Support impact" proxy: services vs churn
-- (Dataset doesn't have ticket counts; this uses service availability as a proxy.)
SELECT
  TechSupport,
  OnlineSecurity,
  COUNT(*) AS customers,
  ROUND(AVG(Churn) * 100.0, 2) AS churn_rate_pct
FROM churn_features
GROUP BY TechSupport, OnlineSecurity
ORDER BY churn_rate_pct DESC;
