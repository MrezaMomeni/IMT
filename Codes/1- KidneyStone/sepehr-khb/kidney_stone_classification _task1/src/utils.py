# src/utils.py
import os
import pandas as pd
import numpy as np
import json
import re

# 🗃 Save & Load Training History
## - Save function
def save_training_history(history_file, full_acc, full_val_acc, full_loss, full_val_loss):
    # history_file = f"{model_name}_history.npz"
    try:
        np.savez(history_file, acc=full_acc, val_acc=full_val_acc, loss=full_loss, val_loss=full_val_loss)
        print("✅ Training history saved!")
    except Exception as e:
        print(f"⚠️ Error saving history: {e}")

## - Load function
def load_training_history(history_file):
    # history_file = f"{model_name}_history.npz"
    if os.path.exists(history_file):
        try:
            data = np.load(history_file)
            print(">>> Previous training history loaded.")
            return (
                list(data['acc']) if 'acc' in data else [],
                list(data['val_acc']) if 'val_acc' in data else [],
                list(data['loss']) if 'loss' in data else [],
                list(data['val_loss']) if 'val_loss' in data else []
            )
        except Exception as e:
            print(f"⚠️ Error loading history: {e}")
            return [], [], [], []
    else:
        print("*** First training run, starting from scratch.")
        return [], [], [], []
 

# 🗃 Save Test Prediction
def save_test_predictions(df_test, test_steps, categorical_predictions, probabilities, model_name, class_labels):
    """
    Save CSV with: id,label,prob_Class0,prob_Class1,...
    """
    def clean_string(s):
        if not isinstance(s,str):
            s = str(s)
        return re.sub(r'[^\x00-\x7F]+', '', s)

    valid_test_count = test_steps * BATCH_SIZE  # or len(categorical_predictions)
    ids = df_test['image_path'].values[:valid_test_count]
    cleaned_ids = [clean_string(x) for x in ids]

    output_df = pd.DataFrame({
        'id': cleaned_ids,
        'label': categorical_predictions.astype(int)
    })

    for i, cls in enumerate(class_labels):
        output_df[f'prob_{cls}'] = probabilities[:, i]

    output_df.to_csv(f"{model_name.replace('.keras','')}_predictions.csv", index=False)
 
 
# 🗃 Save Model Metrics
def save_model_metrics(model_name, y_true, y_pred, results_file="models_evaluation_results.csv"):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        loss = log_loss(y_true, y_pred)
    except:
        loss = None  # In case of error (e.g. binary hard labels)

    # Create results row
    result_row = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "LogLoss": loss
    }

    # If file exists, append, else create new
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])

    # Save back
    df.to_csv(results_file, index=False)
    
 
 # History loading function and visualization "Accuracy" plot

def plot_accuracy(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file): return print("⚠️ No history found.")
    data = np.load(history_file)
    plt.figure(figsize=(8, 6))
    plt.plot(data['acc'], label='Train Accuracy')
    plt.plot(data['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy') 
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid() 
    plt.show()


# 📈 Visualization: Accuracy & Loss

## History loading function and visualization "Loss" plot

def plot_loss(model_name):
    history_file = f"{model_name}_history.npz"
    if not os.path.exists(history_file): return print("⚠️ No history found.")
    data = np.load(history_file)
    plt.figure(figsize=(8, 6))
    plt.plot(data['loss'], label='Train Loss')
    plt.plot(data['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.title(f"{model_name} Loss")
    plt.legend() 
    plt.grid()
    plt.show()

# 📈 Visualization: Compare Multiple Models

## Function to compare the "Accuracy" of multiple models together

def compare_models_accuracy(models):
    plt.figure(figsize=(10, 6))
    for name in models:
        path = f"{name}_history.npz"
        if os.path.exists(path):
            data = np.load(path)
            plt.plot(data['acc'], label=f"{name} - Train")
            plt.plot(data['val_acc'], '--', label=f"{name} - Val")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Epochs") 
    plt.ylabel("Accuracy")
    plt.legend() 
    plt.grid() 
    plt.show()

## Function to compare the "Loss" of multiple models together
def compare_models_loss(models):
    plt.figure(figsize=(10, 6))
    for name in models:
        path = f"{name}_history.npz"
        if os.path.exists(path):
            data = np.load(path)
            plt.plot(data['loss'], label=f"{name} - Train")
            plt.plot(data['val_loss'], '--', label=f"{name} - Val")
    plt.title("Model Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend() 
    plt.grid() 
    plt.show()

