import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.preprocessing.preprocessing_soheib_khaledian import preprocessing_premium_goal
from src.validation.validation_soheib_khaledian import evaluate_regression
from sklearn.ensemble import RandomForestRegressor 
from joblib import dump

x_train, x_test, y_train, y_test = preprocessing_premium_goal('dataset/medical_insurance.csv')

feater_names = ['age', 'gender', 'bmi', 'children', 'northeast', 'northwest', 'southeast', 'southwest', 'discount_eligibility', 'expenses']

model = RandomForestRegressor()

model.fit(x_train, y_train)

dump(model, 'results/models/RandomFores_predict_premium.joblib')

evaluate_regression(model, x_test, y_test, ['premium'], 'results/RandomFores_predict_premium', feater_names)