import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.validation.SHAP_soheib_khaledian import shap_plots_multioutput

def evaluate_regression(model, x_test, y_true, output_names, saving_path, feature_names):
    y_pred = model.predict(x_test)

    metrics = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1

    os.makedirs(saving_path, exist_ok=True)
    image_path = saving_path

    if n_outputs == 1:
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['R2'] = r2_score(y_true, y_pred)

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        plt.figure(figsize=(18, 6))
        plt.plot(y_true, label="True Values", linestyle='-', marker='o')
        plt.plot(y_pred, label="Predicted Values", linestyle='--', marker='x')
        plt.xlabel("Sample Index")
        plt.ylabel(output_names[0])
        plt.legend()
        plt.title(f"True vs Predicted - {output_names[0]}")

        plot_file = os.path.join(image_path, f"{output_names[0]}_prediction_plot.png")
        plt.savefig(plot_file)
        plt.close()

        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join(saving_path, "metrics.csv"), index=False)

    else:
        rows = []
        for i in range(n_outputs):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])

            metrics[output_names[i]] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }

            print(f"\n{output_names[i]}:")
            print(f"  MAE : {mae:.4f}")
            print(f"  MSE : {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2  : {r2:.4f}")

            plt.figure(figsize=(18, 4))
            plt.plot(y_true[:, i], label="True", linestyle='-', marker='o')
            plt.plot(y_pred[:, i], label="Predicted", linestyle='--', marker='x')
            plt.xlabel("Sample Index")
            plt.ylabel(output_names[i])
            plt.title(f"True vs Predicted - {output_names[i]}")
            plt.legend()

            plot_file = os.path.join(image_path, f"{output_names[i]}_prediction_plot.png")
            plt.savefig(plot_file)
            plt.close()

            row = {'Feature': output_names[i], 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(saving_path, "metrics.csv"), index=False)
        
    shap_plots_multioutput(model, x_test, saving_path, feature_names, output_names)

    return metrics
