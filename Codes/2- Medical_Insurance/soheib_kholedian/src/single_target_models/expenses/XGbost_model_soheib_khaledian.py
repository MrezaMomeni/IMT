import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.preprocessing.preprocessing_soheib_khaledian import preprocessing_expenses_goal
from src.validation.validation_soheib_khaledian import evaluate_regression
from xgboost import XGBRegressor

x_train, x_test, y_train, y_test = preprocessing_expenses_goal('dataset/medical_insurance.csv')

feater_names = ['age', 'gender', 'bmi', 'children', 'northeast', 'northwest', 'southeast', 'southwest', 'discount_eligibility', 'premium']

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

model.fit(x_train, y_train)

model.save_model('results/models/xgboost_predict_expense.json')

evaluate_regression(model, x_test, y_test, ['expenses'], 'results/xgboost_predict_expense', feater_names)