from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

from sklearn.metrics import explained_variance_score


def evaluate_model(y_test, y_pred):
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    p = len(y_test)
    n = 6
    adj_r2 = 1-((1-r2)*((p-1)/(p-n-1)))

    print("Test :")
    print('variance: %.7f'% explained_variance_score(y_test, y_pred))
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"ADJ_R²: {adj_r2:.4f}")

    return mse, rmse, r2, mae, adj_r2