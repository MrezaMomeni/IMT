# 🏥 Medical Insurance Cost Prediction

**Project Code:** #3  
**Category:** Machine Learning – Regression  
**Platform:** IMT ChallengeHub  
**Status:** In Progress  

<p>
 <img src="https://www.ostomy.org/wp-content/uploads/2018/10/insurance-blog-photo_web.jpg">
</p>

---

## 📌 Project Overview

This project focuses on predicting **medical insurance charges** using a real-world healthcare dataset.
The dataset includes demographic and lifestyle-related information such as age, gender, BMI, number of children,
smoking-related discount eligibility, and residential region.

The goal is to build and evaluate **regression models** that accurately estimate insurance costs and
extract meaningful insights from the data.

---

## 🎯 Objectives

- Perform **Exploratory Data Analysis (EDA)**
- Visualize relationships between features and insurance charges
- Handle missing values and outliers
- Encode categorical variables
- Apply feature scaling and transformations
- Train and compare multiple **regression models**
- Evaluate models using standard regression metrics

---

## 📊 Dataset Description

- **Source:** Kaggle – Health Insurance Dataset (IMT Team)
- **Type:** Tabular (Real-world)
- **Target Variable:**
  - `charges` – Medical insurance cost
  - ` premium` – Insurance premium charged

### Features Description

| Feature | Description |
|-------|-------------|
| age | Age of the policyholder |
| gender | Gender (male / female) |
| bmi | Body Mass Index |
| children |  Number of children covered by the insurance |
| discount_eligibility | Whether the policyholder is eligible for a discount (yes/no) |
| region | Geographic region (e.g., southeast, northwest) |
| expenses | Actual medical costs incurred by the policyholder (Target number 1) |
| premium | Insurance premium charged (Target number 2) |

📎 **Dataset Link:**  
https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset

---

## 🔍 Exploratory Data Analysis (EDA)

EDA includes:

- Statistical summary of numerical features
- Distribution analysis (histograms, KDE plots)
- Correlation analysis
- Impact of categorical variables on insurance charges
- Outlier detection and skewness analysis

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical variables:
  - One-Hot Encoding
  - Label Encoding (when applicable)
- Feature scaling:
  - StandardScaler
  - MinMaxScaler
- Log transformation for skewed target variable (if required)

---

## 🤖 Machine Learning Models

The following regression models were implemented and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each model was evaluated to identify the best-performing approach.

---

## 📈 Model Evaluation

Models were evaluated using:

- **R² Score**
- **RMSE (Root Mean Squared Error)**

Model comparison highlights the effectiveness of ensemble methods over linear models.

---

## 🧪 Results & Insights

- Discount eligibility (smoking-related) has a **significant impact** on insurance charges
- Age and BMI show a positive correlation with medical costs
- Tree-based ensemble models outperform linear regression models

---

## 📚 References

- **Kaggle Dataset**  
  Health Insurance Dataset – IMT Team  
  https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset

- **IMT ChallengeHub Guidelines**  
  https://github.com/MrezaMomeni/IMT

---

📌 **Challenge Link:**  
https://github.com/MrezaMomeni/IMT/tree/main/ChallengeHub/2-%20Medical_Insurance


---

## 🔱 Fork Information

This project has been **forked and customized** from the following repository:

https://github.com/MrezaMomeni/IMT

---

## 👤 Author

**Hossein Heydari**  
Computer Engineering Student | Junior Data Scientist and Machine Learning Engineer  

GitHub: https://github.com/HosseinHeydari2004 <br>
Kaggle: 

---

## ❤️ Support

If you find this project helpful, please ⭐ the repository and share your feedback.

Happy Learning 🌱✨

