# Medical Insurance Cost Prediction

A machine-learning project that predicts individual medical insurance expenses using features.

---

## Overview

Medical costs have been rising globally, making insurance planning more critical than ever. This project uses regression algorithms to estimate individual insurance expenses based on personal attributes such as age, gender, BMI, region, and lifestyle factors.

---

## Dataset

The `medical_insurance.csv` dataset contains simulated records with the following columns:

- **age**: Integer, age of the policyholder  
- **gender**: Categorical, male or female  
- **bmi**: Float, body mass index  
- **children**: Integer, number of dependent children  

- **region**: Categorical region (northeast, northwest, southeast, southwest)  
- **discount_eligibility**: Categorical, yes or no
**premium**: Float

---

## Modeling Approach

1. **Exploratory Data Analysis**  
   - Visualize distributions and correlations  
 
2. **Preprocessing**  
   - Encode categorical variables  
   - Scale numerical features  
3. **Regression Models**  
   - **Linear Regression**  
   - **Random Forest Regressor**  
   - **AdaBoost Regressor**  
4. **Hyperparameter Tuning**  
   - Compare default vs. tuned models using grid search  
5. **Evaluation**  
   - Metrics: RMSE, MAE, R², MSE  
 

---

## Project Structure




---

***
**Project Structure**<br>
├── data<br>
│   └── medical_insurance.csv<br>
├── src<br>
│   ├── data_analysis.ipynb<br>
│   └── main.ipynb<br>
├── README.md<br>
└── requirements.txt
