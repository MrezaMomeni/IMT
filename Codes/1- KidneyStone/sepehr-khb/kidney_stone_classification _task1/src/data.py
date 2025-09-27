# src/data.py
"""
Data utilities: generator creation, tf.data conversion, helpers for flow_from_dataframe.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(df_train, df_val, df_test, img_size=(224,224), batch_size=32, seed=42):
    """
    Returns: train_gen, val_gen, test_gen (test_gen may have class_mode=None)
    df_* expected to have 'image_path' and 'label' columns (for train/val).
    """
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train, x_col='image_path', y_col='label',
        target_size=img_size, batch_size=batch_size, class_mode='categorical',
        shuffle=True, seed=seed)

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df_val, x_col='image_path', y_col='label',
        target_size=img_size, batch_size=batch_size, class_mode='categorical',
        shuffle=False, seed=seed)

    test_gen = val_datagen.flow_from_dataframe(
        dataframe=df_test, x_col='image_path', y_col=None,
        target_size=img_size, batch_size=batch_size, class_mode=None,
        shuffle=False)

    return train_gen, val_gen, test_gen
    
    
 # A function that converts the Generator to a tf.data.Dataset [Usable for training]

def dataset_from_generator(generator, has_labels=True):
    import tensorflow as tf
    if has_labels:
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=(
                tf.TensorSpec(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE, NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)  # ✅ no ".batch()"!
    else:
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=tf.TensorSpec(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
        ).prefetch(tf.data.AUTOTUNE)
