# 🏥 Medical Insurance Cost & Premium Prediction

**Project Code:** #3
**Category:** Machine Learning – Regression
**Status:** Completed / In Progress

---

## 📌 Project Overview

This project focuses on predicting **medical expenses** and **insurance premiums** using a real-world healthcare dataset. The dataset contains demographic and lifestyle information such as age, gender, BMI, number of children, smoking status, and region.

The main goal is to build reliable **regression models** that help estimate:

* **Medical Expenses** – Annual healthcare costs
* **Premium** – Insurance premium amount

This project is part of the **IMT ChallengeHub** and is designed to strengthen practical skills in **data analysis, feature engineering, and machine learning modeling**.

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** to understand data patterns
* Visualize relationships between features and medical costs
* Clean and preprocess data (encoding, scaling, missing values)
* Build and compare multiple **regression models**
* Evaluate model performance using standard metrics
* Document the full workflow and results clearly

---

## 📊 Dataset Description

* **Source:** Kaggle – Health Insurance Dataset
* **Type:** Tabular, Real-world data
* **Target Variables:**

  * `Medical Expenses`
  * `Premium`

### Main Features

| Feature  | Description           |
| -------- | --------------------- |
| age      | Age of the individual |
| sex      | Gender (male/female)  |
| bmi      | Body Mass Index       |
| children | Number of children    |
| smoker   | Smoking status        |
| region   | Residential region    |

📎 **Dataset Link:**
[https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset)

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA phase includes:

* Statistical summary of features
* Distribution analysis of numerical variables
* Correlation analysis
* Impact of categorical variables (e.g., smoker, region) on expenses
* Outlier detection and skewness analysis

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## 🛠️ Data Preprocessing

Key preprocessing steps:

* Handling missing values
* Encoding categorical variables (One-Hot / Label Encoding)
* Feature scaling using **StandardScaler / MinMaxScaler**
* Log transformation for skewed features (if required)

---

## 🤖 Machine Learning Models

The following regression models were implemented and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Each model was trained separately for:

* Medical Expenses prediction
* Premium prediction

---

## 📈 Model Evaluation

Models were evaluated using:

* **R² Score**
* **RMSE (Root Mean Squared Error)**

Performance comparison helps identify the best-performing model for each target variable.

---

## 🧪 Results & Insights

* Smoking status has a **strong impact** on medical expenses
* BMI and age show positive correlation with costs
* Ensemble models outperform linear models in most cases

Detailed results and plots are available in the notebooks.

---

## 📂 Project Structure

```text
Medical_Insurance/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
│
├── models/
│   └── trained_models.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/MrezaMomeni/IMT/tree/main/ChallengeHub/2-%20Medical_Insurance
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebooks step by step.

---

## 📚 References

* Kaggle Dataset
* Scikit-learn Documentation
* IMT ChallengeHub Guidelines

📌 **Challenge Link:**
[https://github.com/MrezaMomeni/IMT/tree/main/ChallengeHub/2-%20Medical_Insurance](https://github.com/MrezaMomeni/IMT/tree/main/ChallengeHub/2-%20Medical_Insurance)

📖 **Presentation Guide:**
[https://t.me/imtcollege/197](https://t.me/imtcollege/197)

---

## 👤 Author

**Hossein Heydari**
Computer Science Student | Machine Learning Enthusiast

GitHub: [https://github.com/HosseinHeydari2004](https://github.com/HosseinHeydari2004)

---

## ❤️ Support

If you find this project helpful, please ⭐ the repository and share your feedback.

Happy Learning! 🌱✨
