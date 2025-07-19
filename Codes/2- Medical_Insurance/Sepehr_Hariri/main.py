import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data
from scripts.modeling import build_model
from scripts.evaluation import evaluate_model
from scripts.save_model import save_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score

from sklearn.datasets import make_regression
from sklearn.discriminant_analysis import StandardScaler

# Load dataset
df = pd.read_csv('data/medical_insurance.csv')


# Clean outliers from dataset
Q1 = df['bmi'].quantile(0.25)  
Q3 = df['bmi'].quantile(0.75)  
IQR = Q3 - Q1 
df_clean = df[(df['bmi'] >= Q1 - 1.5*IQR) & (df['bmi'] <= Q3 + 1.5*IQR)]
outliers = df[(df['bmi'] < Q1 - 1.5*IQR) | (df['bmi'] > Q3 + 1.5*IQR)]
outliers['bmi'] = Q3
df_combined = pd.concat([df_clean, outliers])
df_combined.reset_index(drop=True, inplace=True)




# Create input
X = df_combined.drop(['expenses', 'premium'], axis='columns')
# Premium
y_premium = df_combined[['premium']]



# Separate dataset into Train and Test
X_train, X_test, y_premium_train, y_premium_test = train_test_split(X, y_premium, test_size=0.2, random_state=42)



# Target Transformation ( log-transform )
y_premium_train_log = np.log1p(y_premium_train)


# Preprocessor
preprocessor = preprocess_data(df_combined)

# Model pipeline   
# select regression model :  xgb or rf or linear
model_type='rf'
model = build_model(model_type, preprocessor=preprocessor)



# Train
model_fit = model.fit(X_train, y_premium_train_log)

y_premium_train_pred = model_fit.predict(X_train)

# Train -- Reverse Target Transformation 
y_premium_train_pred = np.expm1(y_premium_train_pred)


# Predict
y_premium_pred = model.predict(X_test)

# Reverse Target Transformation
y_premium_pred = np.expm1(y_premium_pred)

# Evaluate
evaluate_model(y_premium_test, y_premium_pred)

print(f"model_type: {model_type}")
print("Premium :")
print("Train :")
print('score: %.7f '% model_fit.score(X_train, y_premium_train))
print('RMSE: %.7f'% np.sqrt(mean_squared_error(y_premium_train, y_premium_train_pred)))
print('r2: %.7f'% r2_score(y_premium_train, y_premium_train_pred))
print('variance: %.7f'% explained_variance_score(y_premium_train, y_premium_train_pred))
print('='*35)
print('\n')

# Expenses
y_expenses = df_combined[['expenses']]


# Separate dataset into Train and Test
X_train, X_test, y_expenses_train, y_expenses_test = train_test_split(X, y_expenses, test_size=0.2, random_state=42)


# Target Transformation ( log-transform )
y_expenses_train_log = np.log1p(y_expenses_train)


# Preprocessor
preprocessor = preprocess_data(df_combined)

# Model pipeline   
# select regression model :  xgb or rf or linear
model_type='rf'
model = build_model(model_type, preprocessor=preprocessor)



# Train
model_fit = model.fit(X_train, y_expenses_train_log)

y_expenses_train_pred = model_fit.predict(X_train)

# Train -- Reverse Target Transformation 
y_expenses_train_pred = np.expm1(y_expenses_train_pred)


# Predict
y_expenses_pred = model.predict(X_test)

# Reverse Target Transformation
y_expenses_pred = np.expm1(y_expenses_pred)

# Evaluate
evaluate_model(y_expenses_test, y_expenses_pred)

print(f"model_type: {model_type}")
print("Expenses :")
print("Train :")
print('score: %.7f '% model_fit.score(X_train, y_expenses_train))
print('RMSE: %.7f'% np.sqrt(mean_squared_error(y_expenses_train, y_expenses_train_pred)))
print('r2: %.7f'% r2_score(y_expenses_train, y_expenses_train_pred))
print('variance: %.7f'% explained_variance_score(y_expenses_train, y_expenses_train_pred))


# Save model
save_model(model)

print("Training complete and model saved.")
