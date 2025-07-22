import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.preprocessing_pipeline_soheib_khaleddian import load_preprocessing_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluation(test_image, test_label, model, save_dir="results", model_name="MobileNetV2"):
    os.makedirs(save_dir, exist_ok=True)

    pred_probs = model.predict(test_image)
    pred = (pred_probs > 0.5).astype("int32").flatten()

    cm = confusion_matrix(test_label, pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plot_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    accuracy = accuracy_score(test_label, pred) * 100

    _, fp, fn, tp = cm.ravel()
    precision = (tp / (tp + fp)) * 100 if (tp + fp) != 0 else 0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) != 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    metrics = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2)
    }])
    
    csv_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    metrics.to_csv(csv_path, index=False)

    print(f"Saved confusion matrix to {plot_path}")
    print(f"Saved metrics to {csv_path}")
        

if __name__ == "__main__":
    train_image, test_image, train_label, test_label = load_preprocessing_pipeline("dataset")
    
    model = load_model("models/MobileNetV2_classification.h5")
    
    evaluation(test_image, test_label, model)
