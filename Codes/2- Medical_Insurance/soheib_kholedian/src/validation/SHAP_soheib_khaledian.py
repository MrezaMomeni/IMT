import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_plots_multioutput(model, X_sample, saving_path, feature_names=None, output_names=None):
    X_sample = pd.DataFrame(X_sample).astype(float)

    y_pred = model.predict(X_sample)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_outputs = y_pred.shape[1]

    if output_names is None:
        output_names = [f"Output_{i}" for i in range(n_outputs)]

    if feature_names is None:
        feature_names = [f"Feature_{j}" for j in range(X_sample.shape[1])]

    images_path = saving_path
    tables_path = saving_path

    for i in range(n_outputs):
        print(f"Generating SHAP for: {output_names[i]}")

        if n_outputs == 1:
            predict_i = lambda x: model.predict(x)
        else:
            predict_i = lambda x: model.predict(x)[:, i]

        explainer = shap.Explainer(predict_i, X_sample)
        shap_values = explainer(X_sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary - {output_names[i]}")
        image_file = os.path.join(images_path, f"shap_summary_{output_names[i]}.png")
        plt.savefig(image_file)
        plt.close()

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP value|': mean_abs_shap
        }).sort_values(by='Mean |SHAP value|', ascending=False)

        csv_file = os.path.join(tables_path, f"shap_importance_{output_names[i]}.csv")
        shap_importance.to_csv(csv_file, index=False)
