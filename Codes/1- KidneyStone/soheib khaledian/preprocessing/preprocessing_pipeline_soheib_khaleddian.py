from preprocessing.load_pipeline_soheib_khaledian import load_data
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    return enhanced_img

def preprocess_image(grayscale_images):
    processed_images = []

    for im in grayscale_images:
        im = apply_clahe(im)

        im = cv2.GaussianBlur(im, (3, 3), 0)

        im = cv2.resize(im, (224, 224))

        rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        rgb = rgb.astype('float32') / 255.0

        processed_images.append(rgb)

    return np.array(processed_images)

def split_data(image, label):
    train_image, test_image, train_label, test_label = train_test_split(image, label, test_size=0.2, random_state=76)
    
    return train_image, test_image, train_label, test_label

def load_preprocessing_pipeline(path: str):
    if os.path.isfile("dataset/preprocessed_classification_data.npz"):
        data = np.load("dataset/preprocessed_classification_data.npz")
        
        train_image, test_image, train_label, test_label = data["a"], data["b"], data["c"], data["d"]
        
        return train_image, test_image, train_label, test_label
    
    else:
        images, labels = load_data(path)
        
        images = preprocess_image(images)
        
        train_image, test_image, train_label, test_label = split_data(images, labels)
        
        np.savez_compressed("dataset/preprocessed_classification_data.npz", a=train_image, b=test_image, c=train_label, d=test_label)
        
        return train_image, test_image, train_label, test_label
