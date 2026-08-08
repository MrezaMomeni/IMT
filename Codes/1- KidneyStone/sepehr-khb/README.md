# 🩺 Kidney Stone Classification with Deep Learning

> **Kaggle notebook:** Kidney Stone Classification — Task 1.  
> Project & notebook: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow

---

## 📌 Project Overview

This repository contains experiments, code and results for **Kidney Stone Classification** (binary classification: `Normal` vs `Stone`).  
The goal is to train and compare multiple convolutional neural networks, evaluate them with rigorous metrics (Accuracy, Precision, Recall, F1, ROC-AUC, LogLoss) and provide reusable artifacts (trained model files, prediction CSVs, training histories).

This work is part of the **IMT Internship** challenge series and is prepared to be reproducible on Kaggle (recommended) or locally.

---

## 🎯 Project Goal

* Binary image classification to detect presence of kidney stone in medical images (two classes: `Normal`, `Stone`).
* Compare several model families (from lightweight custom CNNs to transfer-learning architectures).
* Provide robust evaluation, ensemble experiments (hard & soft voting), and store reusable artifacts (CSV predictions, `.npz` histories, model metadata).

**Notes:** A second phase (object detection using YOLO) is planned for bounding-box detection of stone regions. This repository focuses on Task 1 (classification).

---

## 🗂 Repository Structure 

KidneyStone-Classification/
├── notebooks/ # Kaggle / Jupyter notebooks (exploration, training, evaluation)
│ ├── 01-exploration.ipynb
│ ├── 02-training.ipynb
│ └── kaggle-original.ipynb
├── src/ # Modular code: data loaders, models, train/eval utils
├── assets/ # Sample images, example plots used in README
├── results/ # predictions (.csv), metrics (.csv), training histories (.npz), plots (.png)
│ └── figures/
├── models/ # Saved models (.keras) — large files, typically NOT pushed to GitHub
├── configs/ # Hyperparameter (.yaml/.yml)
├── logs/ # Training/evaluation logs
├── README.md # This file (English)
├── README_FA.md # Persian translation
├── requirements.txt # Python dependencies
└── .gitignore # Files to ignore (models, data, large outputs)

---

## 📊 Dataset

* **Source:** Kidney Stone dataset 
	Kaggle dataset link in the Kaggle notebook: /kaggle/input/kidney-stone-classification-and-object-detection
* Typical structure used in the notebooks:
  * `df_train`, `df_val`, `df_test` dataframes with column `image_path` and `label` (for labeled splits).
  * `testDataset` (unlabeled) for final predictions (no `y_col`), `testDataset_with_labels` when we need labels for evaluation.

**Important design choices**
* `valDataset` and `testDataset_with_labels` are created with `shuffle=False` so predictions align with `df_test` ordering.
* Training used `tf.data` pipelines (converted from `ImageDataGenerator`) with `prefetch(AUTOTUNE)` for performance.
* Models and histories were saved as:
  * Model files: `ModelName_model.keras`
  * Prediction CSVs: `ModelName_predictions.csv`
  * Training history: `ModelName_history.npz`
  * Metadata (train time, params): `ModelName_meta.json`

---

## 🖼️ Sample Data

Example images from the dataset:

<p align="center">
  <img src="assets/normal_sample1.png" width="400"/>
  <img src="assets/stone_sample1.png" width="400"/>
  <img src="assets/sample1s.png" width="400"/>
</p>

---

## 🧪 Models Tested

* `simple_cnn_model` — custom lightweight CNN (first, fast baseline)
* `AlexNet_model` — classic CNN architecture
* `VGGNet_model` — deep conv stack (performed very well)
* `ResNet34_model` — residual network (underperformed on this dataset)
* `EfficientNetB0/B1/B2` — tested but **they predicted only the `Stone` class** in our experiments (see *Notes* below)

---

## ⚙ Training Configuration (as used in notebooks / train script)

* **Optimizer:** Adam  
* **Learning rate:** `1e-4` (used in model compilation)  
* **Loss:** `categorical_crossentropy` (softmax outputs)  
* **Metrics:** `accuracy` (+ evaluation metrics computed offline)  
* **Callbacks:** `ReduceLROnPlateau`, `EarlyStopping(restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`  
* **Data pipeline:** `tf.data` wrapper around `ImageDataGenerator` for performance  
* **Batching:** controlled by `BATCH_SIZE` variable — ensure `len(dataset)` divisible by number of replicas (if using MirroredStrategy)  
* **Prediction saving:** predictions saved with both hard labels (`label`) and softmax probabilities `prob_<ClassName>` columns for each class.

> See `train_model(...)` in `src/` for the exact training pipeline and how artifacts are saved.

---

## 📈 Evaluation Metrics & Results (final summary)

All metrics computed on the labeled test split (`testDataset_with_labels`) — predictions loaded from saved CSVs.

| Model                     | Accuracy | Precision | Recall  | F1      | ROC_AUC | LogLoss |
|---------------------------|---------:|----------:|--------:|--------:|--------:|--------:|
| **AlexNet_model**         | 0.9868   | 0.9802    | 0.9955  | 0.9878  | 0.9996  | 0.0471  |
| **VGGNet_model**          | 0.9844   | 0.9888    | 0.9821  | 0.9854  | 0.9988  | 0.0583  |
| **simple_cnn_model**      | 0.9663   | 0.9412    | 1.0000  | 0.9697  | 0.9999  | 0.1213  |
| **ResNet34_model**        | 0.7236   | 0.7433    | 0.7433  | 0.7433  | 0.8103  | 0.5767  |
| **Ensemble (Hard Voting)**| 0.9808   | 0.9800    | 0.9844  | 0.9822  |  —     |  —      |
| **Ensemble (Soft Voting)**| 0.9880   | 0.9782    | 1.0000  | 0.9890  | 0.9999  | 0.1579  |

**Key takeaways**
* `AlexNet` and `VGGNet` are the top-performing single models; `VGGNet` is very stable and `AlexNet` slightly edges in Accuracy in one run.  
* A small/custom `simple_cnn_model` also performed very well (high recall = 1.0 in this run).  
* `ResNet34` underperformed on this dataset — possible causes: overfitting, mismatch with dataset size/characteristics, or training hyperparameters that need tuning.  
* **Ensembles** improved robustness; soft-voting ensemble achieved the best F1 and excellent overall performance.  
* **Important:** EfficientNet variants in our runs predicted only `Stone` for nearly all inputs — this indicates a dataset / training mismatch for these architectures (class imbalance, insufficient augmentation or different learning-rate / regularization needs).

---

## ⚠ Important Notes and Known Issues

* **EfficientNet behavior:** In our experiments the EfficientNet models (B0–B2) collapsed to predicting a single class (`Stone`). This happened consistently and may be due to:
  * High model capacity combined with dataset imbalance
  * Not enough augmentation for these architectures
  * Different required hyperparameter schedule (lr warmup, weight decay, different preprocessing)
  * Recommendation: if you want to use EfficientNet, try stronger augmentation, learning rate warmup/decay, or layer-wise tuning.
* **Dataset & GitHub:** Do **NOT** push raw image data or large model weights to GitHub. Instead, store lightweight artifacts (CSV predictions, `.npz` histories, small plots) and provide instructions for dataset download via Kaggle.
* **Reproducibility on Kaggle:** use `shuffle=False` for evaluation/test data to keep ordering consistent with CSVs. When you `predict/evaluate` with `tf.data` generators, specify `steps` appropriately.

---

## 🚀 Reproduce This Project (quick)

1. Clone the repo:
```bash
git clone https://github.com/sepehr-khb/KidneyStone-Classification.git
cd KidneyStone-Classification

2. Install dependencies:
pip install -r requirements.txt

3. Run on Kaggle:
Open the notebook: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow
On Kaggle: Run All, then Quick Save, then when ready Share → Public to create a reproducible version.

4. Or run locally (example):
python src/train.py --model VGGNet_model --config configs/hparams.yaml
python src/evaluate.py --predictions_dir results/
```

🗒️ Implementation Highlights

- Conversion of ImageDataGenerator → tf.data.Dataset for speed & MirroredStrategy compatibility.
- Careful handling of steps_per_epoch, validation_steps, and test_steps to avoid infinite generators.
- Saving predictions with both hard labels and softmax probabilities (prob_Normal, prob_Stone) to allow ROC-AUC / log-loss computation later.
- Meta JSON file for each model containing train_time_sec and params (helpful for final reporting).
- Ensemble utilities for hard & soft voting using saved CSVs (so evaluation can be done without re-running heavy models).

🔮 Future Work & Recommendations

- Phase 2 — Object Detection: label images (CVAT / Roboflow) and train YOLOv8/v11 for localization.
- Improve EfficientNet runs with:
	- Stronger augmentation, learning-rate scheduling (warmup), and weight decay.
	- Try transfer learning with pretrained weights + lower learning rate for head layers.
- Experiment with image size increase (e.g., 224×224) for high-capacity models.
- Add interpretability (Grad-CAM) for clinical review and trust.

📁 Files & Artifacts produced (examples)

- `AlexNet_model_predictions.csv`, `VGGNet_model_predictions.csv`, `simple_cnn_model_predictions.csv`, `ResNet34_model_predictions.csv`, `EfficientNetB0_model_predictions.csv`, ...

	- Each CSV contains: id, label (0/1 or class name), `prob_Normal`, `prob_Stone`.

- `ModelName_history.npz` — training/validation loss & accuracy arrays.

- `ModelName_meta.json` — training time and parameters.

- `final_results_summary.csv` — compiled metrics for all models.

🧾 License

This project is released under the MIT License — see `LICENSE` (or include license text).

👨‍💻 Author & Contact

Sepehr Kh. — GitHub: https://github.com/sepehr-khb

Kaggle Notebook: https://www.kaggle.com/code/sepehrkh/kidney-stone-classification-task-1-tensorfollow

Date: September 2025