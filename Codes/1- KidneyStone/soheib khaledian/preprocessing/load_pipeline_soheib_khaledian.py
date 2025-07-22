import numpy as np
import cv2
import os

def read_image_data(path) -> np.array:
    images = []
    labels = []
    
    normal_path = os.path.join(path, 'Normal')
    stone_path = os.path.join(path, 'Stone')

    for normal_image in os.listdir(normal_path):
        full_path = os.path.join(normal_path, normal_image)
        images.append(cv2.imread(full_path, cv2.IMREAD_GRAYSCALE))
        labels.append(0)
        
    for stone_image in os.listdir(stone_path):
        full_path = os.path.join(stone_path, stone_image)
        images.append(cv2.imread(full_path, cv2.IMREAD_GRAYSCALE))
        labels.append(1)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)

def shuffle_data(images, labels):
    np.random.seed(42)
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    return images[indices], labels[indices]

def load_data(path):
    if os.path.isfile("dataset/classification_data.npz"):
        data = np.load("dataset/classification_data.npz")
        
        image, label = data["a"], data["b"]
    
    else:
        image, label = read_image_data(path)
        image, label = shuffle_data(image, label)
        
        np.savez_compressed("dataset/classification_data.npz", a=image, b=label)
    
    return image, label
