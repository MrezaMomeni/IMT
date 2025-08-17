# 🏥 Medical Expenses Prediction Analysis

This project presents a comprehensive end-to-end analysis and machine learning pipeline designed to predict **individual medical expenses** using demographic and lifestyle data. The goal is to generate actionable business insights while evaluating different regression models and addressing key modeling challenges such as data leakage.

---

## 📊 Dataset Overview

The dataset includes the following features for each individual:

- `age`  
- `gender`  
- `bmi`  
- `children`  
- `discount_eligibility`  
- `region`  
- `expenses` (target variable)  
- `premium`

---

## 🔍 Data Exploration

### Descriptive Analysis

- **Distributions**: Features like `expenses` and `premium` are **right-skewed**, indicating a small subset of individuals with significantly higher costs.
- **Visualizations**:
  - Histograms and box plots revealed outliers, especially in `bmi`, `expenses`, and `premium`.
  - Bar plots for `gender`, `discount_eligibility`, and `region` showed categorical frequencies.
- **Correlation Insights**:
  - `premium` has a **very strong positive correlation** with `expenses`.
  - Moderate positive correlations exist for `age` and `bmi` with `expenses`.

---

## 🛠️ Feature Engineering & Preprocessing

- **Categorical Encoding**: `gender`, `discount_eligibility`, and `region` were numerically encoded.
- **Scaling**: `age`, `bmi`, `children`, and `premium` were scaled using `StandardScaler`.
- **Outliers**: Although outliers were identified, they were retained for model training.
- **Missing Values**: None found in the dataset.

---

## 🧠 Modeling Approach

### Task

- **Regression** problem to predict `expenses`.

### Data Split

- 80% training, 20% testing
- Fixed `random_state` for reproducibility

### Models Used

1. **Linear Regression** – Baseline model
2. **Gradient Boosting Regressor**
3. **Random Forest Regressor**

### Pipelines

- Used `ColumnTransformer` to apply scaling to numerical features and pass encoded categorical variables.
- Integrated into scikit-learn pipelines with model-specific estimators.

### Evaluation

- **Metrics**: R², RMSE, MAE
- **Validation**: 5-Fold Cross-Validation on training set

---

## 📈 Results and Interpretation

### Test Set Performance

| Model                   | R²        | RMSE      
|------------------------|------------|-----------|
| Linear Regression       | `0.7542`  | `5931.56`|
| Gradient Boosting       | `0.9896`  | `1156.68`|
| Random Forest           | `0.9792,` | `1572.34`|

- **Gradient Boosting** and **Random Forest** outperformed Linear Regression significantly.
- **Actual vs Predicted** plots show that tree-based models captured higher expense values better.
- **Residual Plots** show Linear Regression struggles with homoscedasticity, while ensemble models handled variance better.

---

## 🔁 Cross-Validation Performance (Training Data)

### Linear Regression

- R² Mean = `0.7542`, Std = `0.0213`
- RMSE Mean = `5931.56`, Std = `246.57`

### Gradient Boosting Regressor

- R² Mean = `0.9896`, Std = `0.0061`
- RMSE Mean = `1156.68`, Std = `337.94`

### Random Forest Regressor

- R² Mean = `0.9792`, Std = `0.0142`
- RMSE Mean = `1572.34`, Std = `693.86`


