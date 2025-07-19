1.Introduction

# 🧮 Medical_Insurance

A machine learning project to predict insurance expenses based on demographic and health-related attributes using regression models.

---

## 📌 Project Features

- 📊 Exploratory Data Analysis (EDA)
- ⚙️ Feature Engineering & Preprocessing
- 🧠 Regression Modeling (Decision Tree, K-Neighbors, Random Forest, XGBoost)
- 📈 Model Evaluation (R², rmse, mae, adj_R²)
- 💾 Model Saving & Loading

---

## 📂 Project Structure

Medical_Insurance/

│

├── data/ # Input CSV data

├── models/ # Saved trained model

├── Medical_Insurance_SepehrHariri.ipynb # Data exploration

├── scripts/

│    ├── preprocessing.py # Data cleaning & encoding

│    ├── modeling.py # Model pipeline creation

│    ├── evaluation.py # Metrics & plots

│    └── save_model.py # Model serialization

├── main.py # Training script

├── requirements.txt # Python dependencies

└── README.md # Project overview


---

## 📈 Dataset Columns

- `age`: Age of the policyholder  
- `gender`: Male/Female  
- `bmi`: Body Mass Index  
- `children`: Number of children covered  
- `discount_eligibility`: Yes/No  
- `region`: Geographical area  
- `expenses`: Medical insurance cost (target variable)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SepehrHariri/IMT.git
   cd Codes/2-Medical_Insurance/Sepehr_Hariri

---------------------------------------------------------------------------------------------

2. Install dependencies:

   pip install -r requirements.txt

---------------------------------------------------------------------------------------------

3. Place your dataset in the data/ folder (e.g., medical_insurance.csv)

---------------------------------------------------------------------------------------------

4. Run the training pipeline:

   python main.py

---------------------------------------------------------------------------------------------

🧪 Model Evaluation Example

Premium

   XGboost: RMSE: 9.1036 , R² Score: 0.9945
   Random Forest: RMSE: 29.0428 , R² Score: 0.9443

Expenses

   XGboost: RMSE: 11.8529 , R² Score: 0.9907
   Random Forest: RMSE: 28.7963 , R² Score: 0.9452

📦 Output
Trained model saved to: models/insurance_model.pkl

📌 License
This project is open-source and available under the MIT License.
