# src/train.py
"""
Training script: build model, compile, fit, save best model and predictions.
This is a thin wrapper that integrates src/models.py and src/data.py
"""

import os
import time
import json
import numpy as np
import tensorflow as tf

from src.models import simple_cnn, vgg_like
from src.utils import save_test_predictions


# Set precision (in case of NaN problem it is better to use float32)
mixed_precision.set_global_policy("float32")

# Strategy setup (once): Define MirroredStrategy only once, outside the `train_model` function
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ GPUs detected: {[gpu.name for gpu in gpus]}")
else:
    print("⚠ No GPU detected! Running on CPU.")
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")

# Train_model function
def train_model(model_name, model_fn, train_gen, val_gen, test_gen, df_test, class_labels, epochs=20, batch_size=32):
    # clear
    tf.keras.backend.clear_session()

    model = model_fn()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    test_steps = len(test_gen)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )

    # load best model and predict
    best_model = tf.keras.models.load_model(model_name)
    predictions = best_model.predict(test_gen, steps=test_steps)
    categorical = np.argmax(predictions, axis=1)

    save_test_predictions(df_test, test_steps, categorical, predictions, model_name, class_labels)

    return best_model, history
