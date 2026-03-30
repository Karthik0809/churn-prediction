-- Create a feature table in SQLite from `raw_data`.
-- Expected schema: `raw_data` matches the IBM Telco churn CSV columns.

DROP TABLE IF EXISTS churn_features;

CREATE TABLE churn_features AS
SELECT
  customerID,
  tenure,
  MonthlyCharges,
  TotalCharges,
  Contract,
  PaymentMethod,
  InternetService,
  TechSupport,
  OnlineSecurity,
  Churn,

  CASE
    WHEN tenure < 12 THEN '0-1yr'
    WHEN tenure < 24 THEN '1-2yr'
    WHEN tenure < 48 THEN '2-4yr'
    ELSE '4+yr'
  END AS tenure_bucket,

  CASE
    WHEN MonthlyCharges < 35 THEN 'Low'
    WHEN MonthlyCharges < 65 THEN 'Medium'
    ELSE 'High'
  END AS charge_tier,

  CASE
    WHEN Contract = 'Month-to-month' AND MonthlyCharges > 65 THEN 'High Risk'
    WHEN Contract = 'Month-to-month' THEN 'Medium Risk'
    ELSE 'Low Risk'
  END AS risk_flag
FROM raw_data;

CREATE INDEX IF NOT EXISTS idx_churn_features_churn ON churn_features(Churn);
CREATE INDEX IF NOT EXISTS idx_churn_features_contract ON churn_features(Contract);
CREATE INDEX IF NOT EXISTS idx_churn_features_tenure_bucket ON churn_features(tenure_bucket);
