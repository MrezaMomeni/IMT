# Medical Insurance Cost Analysis and Prediction

## Project Summary

In this study, we conducted a comprehensive analysis and prediction task on a medical insurance dataset obtained from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset). The goal was to uncover the most influential factors affecting insurance **expenses** and **premium** costs, and to build machine learning models that could accurately predict them.

Through data exploration, feature analysis, and the use of robust regression algorithms (Random Forest and XGBoost), we ensured accurate predictions without excluding outliers—acknowledging their significance in real-world medical datasets.

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview

| Feature  | Count | Mean     | Std      | Min     | 25%     | 50%     | 75%      | Max      |
| -------- | ----- | -------- | -------- | ------- | ------- | ------- | -------- | -------- |
| Age      | 1338  | 39.21    | 14.05    | 18.00   | 27.00   | 39.00   | 51.00    | 64.00    |
| BMI      | 1338  | 30.67    | 6.10     | 16.00   | 26.30   | 30.40   | 34.70    | 53.10    |
| Children | 1338  | 1.09     | 1.21     | 0.00    | 0.00    | 1.00    | 2.00     | 5.00     |
| Expenses | 1338  | 13270.42 | 12110.01 | 1121.87 | 4740.29 | 9382.03 | 16639.92 | 63770.43 |
| Premium  | 1338  | 262.87   | 292.53   | 11.22   | 87.35   | 174.99  | 342.91   | 1983.11  |

### 1.2 Feature Distribution

Categorical and numerical distributions were visualized:

* **Categorical Distribution:**
  ![Categorical Distribution](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Distribution.png)

  > Variables like `region` and `gender` are evenly distributed. However, `discount_eligibility` is skewed: 70% of individuals are ineligible, implying limited access to insurance discounts.

* **Numerical Distribution:**
  ![Numerical Distribution](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Distributions%20of%20Numeric%20Columns.png)

  > All numerical variables show slight skewness, particularly `expenses` and `premium`. BMI shows near-normal distribution but leans slightly right.

### 1.3 Outlier Detection

Using the **IQR Rule**, the following outliers were found:

| Column   | Number of Outliers | Mean of Outliers |
| -------- | ------------------ | ---------------- |
| Age      | 0                  | None             |
| BMI      | 9                  | 49.28            |
| Children | 0                  | None             |
| Expenses | 139                | 42103.95         |
| Premium  | 113                | 1021.56          |

Outlier visualization:
![Outliers Scatter Plot](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Scatter%20Plots%20Highlighting%20Outliers.png)

> These outliers are meaningful, especially in `expenses` and `premium`, where high costs relate to chronic conditions or advanced medical treatments. They were **kept** to preserve real-world variability.

---

## 2. Feature Relationships

### 2.1 Correlation Analysis

* **Heatmap of Numeric Features:**
  ![Correlation Heatmap](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Correlation%20Heatmap%20of%20Numeric%20Features.png)

  > Strong correlation (0.85) exists between `premium` and `expenses`. A moderate correlation exists between `age` and `premium`, implying higher insurance premiums for older individuals.

* **Categorical Insights:**
  ![Average Premium-Expenses](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Average%20Premium-Expenses.png)

  > * People with 2-3 children have higher expenses than those with 4-5, potentially due to discount eligibility.
  > * Surprisingly, those with `discount_eligibility = yes` spend **more**—possibly indicating discounts for high-risk cases.

### 2.2 Deep Feature Exploration

* **Age vs Expenses by Gender:**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Age%20vs%20Expenses%20colored%20by%20Gender.png)

* **BMI vs Premium by Region:**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/BMI%20vs%20Premium%20colored%20by%20Region.png)

* **Premium vs Expenses (clear bands):**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Insurance%20Premium%20vs%20Expenses.png)

* **Pairplot (Gender-based):**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Pairplot%20of%20Numeric%20Columns%20colored%20by%20Gender.png)

* **Pairplot (Discount-based):**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Pairplot%20of%20Numeric%20Columns%20colored%20by%20discount_eligibility.png)

* **Important Features vs Targets:**
  ![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/data_analysis_plots/Scatter%20Plots%20Important%20Features%20vs%20Expenses%20%26%20Premium.png)

## 3. Limitations

One limitation is that the dataset lacks medical history or diagnosis information, which can heavily influence expenses.

---

## 3. Modeling Pipeline

### 3.1 Preprocessing

* **Encoding:** Used `LabelEncoder` and `OneHotEncoder` for categorical variables.
* **Splitting:** Standard train-test split.
* **No normalization/standardization** required as models (RF, XGBoost) are tree-based.

### 3.2 Modeling Scenarios

Three supervised regression tasks:

1. **Joint Prediction**: Predict both `expenses` and `premium` from demographic and region info.
2. **Single Prediction (Expenses)**: Use all features including `premium`.
3. **Single Prediction (Premium)**: Use all features including `expenses`.

Models used:

* Random Forest Regressor
* XGBoost Regressor

  > The reason behind using this model is that I did not want to modify or remove any outliers, and this model is known to be robust to them.

---

## 4. Results and Evaluation

### 4.1 Joint Prediction (expenses & premium)

**Random Forest:**

| Target   | MAE     | MSE         | RMSE    | R2     |
| -------- | ------- | ----------- | ------- | ------ |
| Expenses | 2524.89 | 21303382.92 | 4615.56 | 0.8628 |
| Premium  | 43.76   | 6846.68     | 82.74   | 0.9260 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomForest_join_goal/expenses_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomForest_join_goal/premium_prediction_plot.png)

SHAP Importance:

![SHAP Summary Expenses](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomForest_join_goal/shap_summary_expenses.png)

![SHAP Summary Premium](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomForest_join_goal/shap_summary_premium.png)

**XGBoost:**

| Target   | MAE     | MSE         | RMSE    | R2     |
| -------- | ------- | ----------- | ------- | ------ |
| Expenses | 2524.05 | 20220715.76 | 4496.75 | 0.8698 |
| Premium  | 43.01   | 6768.38     | 82.27   | 0.9269 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_join_goal/expenses_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_join_goal/premium_prediction_plot.png)

SHAP Importance:

![SHAP Summary Expenses](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_join_goal/shap_summary_expenses.png)

![SHAP Summary Premium](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_join_goal/shap_summary_premium.png)

### 4.2 Predicting Only Expenses

**Random Forest:**

| MAE    | MSE       | RMSE   | R2     |
| ------ | --------- | ------ | ------ |
| 197.78 | 944955.07 | 972.09 | 0.9939 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomFores_predict_expenses/expenses_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomFores_predict_expenses/shap_summary_expenses.png)

**XGBoost:**

| MAE    | MSE        | RMSE    | R2     |
| ------ | ---------- | ------- | ------ |
| 224.94 | 1086162.88 | 1042.19 | 0.9930 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_predict_expenses/expenses_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_predict_expenses/shap_summary_expenses.png)

### 4.3 Predicting Only Premium

**Random Forest:**

| MAE  | MSE    | RMSE  | R2     |
| ---- | ------ | ----- | ------ |
| 4.79 | 725.08 | 26.93 | 0.9922 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomFores_predict_premium/premium_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/RandomFores_predict_premium/shap_summary_premium.png)

**XGBoost:**

| MAE  | MSE   | RMSE | R2     |
| ---- | ----- | ---- | ------ |
| 3.01 | 98.57 | 9.93 | 0.9989 |

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_predict_premium/premium_prediction_plot.png)

![](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian_Medical_Insurance/Codes/2-%20Medical_Insurance/soheib_kholedian/results/xgboost_predict_premium/shap_summary_premium.png)

> While model performance is high (R² ≈ 0.99), care should be taken to evaluate overfitting via cross-validation or testing on unseen data.

---

## 5. SHAP-Based Interpretability

* **Premium and Expenses** are most influenced by:

  * **Age**: Older individuals pay more.
  * **Discount Eligibility**: Strong cost driver—those eligible tend to have higher expenses.
  * **Region & BMI**: Affect premium noticeably.

SHAP confirms **non-linear relationships**, especially with `discount_eligibility` and `region`.

---

## 6. Conclusion & Recommendations

* The model successfully captured key relationships in the insurance dataset.
* **XGBoost** slightly outperforms **Random Forest** in most scenarios.
* Discount eligibility and premium are strong predictors of expenses.
* **Recommendation**: Use **separate models** for expenses and premium when maximum accuracy is desired.
* SHAP confirms feature importance and aligns with EDA findings.

> This pipeline is a strong base for insurance companies to automate pricing, understand customer segments, and offer tailored services.

---

**📝 Report by:** *Soheib Khaledian*

---
