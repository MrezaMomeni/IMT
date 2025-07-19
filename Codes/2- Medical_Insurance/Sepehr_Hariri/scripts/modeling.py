from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline


def build_model(model_type='linear', preprocessor=None):
    if model_type == 'linear':
        model = LinearRegression(n_jobs=-1)

    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1,
                                    min_samples_split=3, max_features=5, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=200, learning_rate=0.2, max_depth=2, colsample_bytree=0.5)
        
    else:
        raise ValueError("Unsupported model type")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline