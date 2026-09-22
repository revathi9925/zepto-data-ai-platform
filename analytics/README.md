# Zepto Data & AI Platform — Module 2: Analytics & Machine Learning

An end-to-end Machine Learning pipeline for Titanic passenger survival classification and fare estimation using Python and scikit-learn.

======================================================================
1. REPOSITORY STRUCTURE
======================================================================

analytics ml/
├── titanic_ml.py                      # Core ML pipeline script
├── titanic.csv                        # Offline dataset fallback
├── requirements.txt                   # Environment dependencies
├── module2_output.txt                 # Console execution log
├── titanic_survival_pipeline.joblib   # Saved Random Forest classifier
├── fare_regression_pipeline.joblib    # Saved Linear Regression model
└── figures/                           # Generated EDA & evaluation plots

======================================================================
2. QUICK START
======================================================================

1. Install Dependencies:
   pip install -r requirements.txt

2. Run Pipeline:
   python titanic_ml.py

======================================================================
3. SUMMARY OF WRITTEN INTERPRETATIONS
======================================================================

• Data Ingestion & Cleaning:
  Uses Seaborn with local titanic.csv fallback. Applied percentage-based rules: 
  dropped rows for <5% missing (embarked), imputed 5–30% missing (age via median), 
  and dropped columns with >30% missing (deck). Redundant leaky features were removed.

• Exploratory Data Analysis:
  Age is normally distributed (8 IQR outliers), while fare is strongly right-skewed 
  (Mean: $32.10 > Median: $14.45 > Mode: $8.05; 114 outliers). Females (74.2%) and 
  1st-class passengers (62.9%) had higher survival rates. The decision tree root split 
  confirms sex as the primary driver.

• Regression Diagnostics:
  Linear regression on fare yielded MAE of 17.85 and R2 of 0.3838. Residual plots 
  display a fan shape, proving HETEROSCEDASTICITY.

• Model Recommendation:
  The Tuned Random Forest is recommended as it achieved the highest test ROC-AUC (0.8397) 
  and cross-validation ROC-AUC (0.8726).

======================================================================
4. METRIC SUMMARY
======================================================================

[Classification Performance]

Model                      | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1-Score | Accuracy
---------------------------|------------|--------------|-----------|--------|----------|---------
Logistic Regression        | 0.8447     | 0.8596       | 0.7966    | 0.6912 | 0.7402   | 0.8146
Decision Tree              | 0.8402     | 0.8259       | 0.7347    | 0.5294 | 0.6154   | 0.7472
Random Forest (Baseline)   | 0.8629     | 0.8243       | 0.7931    | 0.6765 | 0.7302   | 0.8090
Tuned Random Forest        | 0.8726     | 0.8397       | 0.8333    | 0.6618 | 0.7377   | 0.8202


[Regression Performance (Fare Prediction)]

MAE     | RMSE    | R2 Score | Adjusted R2
--------|---------|----------|------------
17.8530 | 40.5476 | 0.3838   | 0.3546

======================================================================
5. ARTIFACT DEPLOYMENT
======================================================================

Full preprocessing steps (imputation, scaling, one-hot encoding) are bundled 
directly into .joblib files to allow single-step inference on raw dictionary 
inputs without manual data pre-transformation.