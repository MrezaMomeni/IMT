import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.preprocessing_soheib_khaledian import preprocessing_join
from src.validation.validation_soheib_khaledian import evaluate_regression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump

x_train, x_test, y_train, y_test = preprocessing_join('dataset/medical_insurance.csv')

feater_names = ['age', 'gender', 'bmi', 'children', 'northeast', 'northwest', 'southeast', 'southwest', 'discount_eligibility']

base_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model = MultiOutputRegressor(base_model)

model.fit(x_train, y_train)

dump(model, 'results/models/xgboost_model_join_goal.joblib')

evaluate_regression(model, x_test, y_test, ['expenses', 'premium'], 'results/xgboost_join_goal', feater_names)