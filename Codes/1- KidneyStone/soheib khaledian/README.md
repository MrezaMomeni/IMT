# Kidney Stone Classification 🧠🩺

In this project, we used **transfer learning** to train a model on our [dataset](https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection/data?select=Normal) to classify medical images into **two categories**: with kidney stones and without.

We used **MobileNetV2** as our base model because it's lightweight and efficient—perfect for training without access to a GPU.

---

## 📁 Project Structure

In the following sections, we explain the purpose of each folder in this repository and what it contains.

---

### 🧪 Classification

There is only one file in this folder named:

**`kidney_stone_MobileNetV2_classifier_soheib_khaledian.py`**

This is the **main training script**, and it performs the following tasks:

1. Loads and preprocesses the data  
2. Loads and compiles the model  
3. Trains the model  
4. Validates the model

> 🔹 **Note**: Not all of the code is written inside this file. Some parts such as preprocessing and data loading are imported from other files and modules.

---

### ⚙️ Preprocessing

This folder contains two files:

- **`load_pipeline_soheib_khaledian.py`**  
  Loads the data as grayscale images and assigns labels based on the folder structure:
  - `0` for healthy kidneys  
  - `1` for kidneys with stones  
  It also shuffles the data for training.

- **`preprocessing_pipeline_soheib_khaleddian.py`**  
  Applies preprocessing techniques including:
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization)  
  - **Gaussian blur**  
  - **Resizing** the images  
  - **RGB conversion** (since **MobileNetV2** expects 3-channel input)

Both scripts save the processed data as a `.npz` file.  
Before processing, they check if the `.npz` file already exists — if it does, they simply load it to avoid redundant computation.

➡️ **To use the full load pipline and preprocessing pipeline**, just call:

```python
load_preprocessing_pipeline(data_path)
```

---
### 🧰 Utils

This folder contains a single file:

- **`evaluation_soheib_khaledian.py`**  
  This script is used to **evaluate the trained model**.  
  It can be called either directly as a standalone script or from within the training script after the model has been trained.

The script loads the saved model and performs evaluation using the following metrics:

1. **Confusion Matrix**  
2. **Accuracy**  
3. **Precision**  
4. **Recall**

> it save the result as image ans csv file

---
### 🧠 Model

This folder contains:

- A trained model saved as a `.h5` file:  
  **`MobileNetV2_classification.h5`**

You can load the model like this:
```python
from tensorflow.keras.models import load_model

model = load_model("models/MobileNetV2_classification.h5")
```

Feel free to load the model and evaluate it yourself.

The folder also includes:
- An image of the **confusion matrix**
- A `.csv` file containing detailed evaluation metrics

#### 📊 Evaluation Results

| Model       | Accuracy | Precision | Recall |
|-------------|----------|-----------|--------|
| MobileNetV2 | 98.46%   | 98.75%    | 98.24% |

![Confusion Matrix](https://github.com/SoheibKhaledian/IMT/blob/soheib_khaledian/Codes/1-%20KidneyStone/soheib%20khaledian/models/results/MobileNetV2_confusion_matrix.png?raw=true)
---
> note that i have use diffrent shuffel and randomset and the result has been close to each other and it max there was 3 present diffrent 
