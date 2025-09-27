# src/models.py
"""
Model definitions: small/simple_cnn, VGG-like, wrapper for transfer-learning models.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def simple_cnn(input_shape=(224,224,3), num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def vgg_like(input_shape=(224,224,3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)
    
# 📦 model_fn: Building different CNN models based on the chosen architecture name
def model_fn(model_name, input_shape=(96, 96, 3), num_classes=2):
    
    if model_name == "AlexNet":
        # 📌 AlexNet-inspired architecture
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=3, strides=2),

            layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=3, strides=2),

            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),

            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

    elif model_name == "VGGNet":
        # 📌 VGGNet-style deep CNN
        # ✅ Better VGG block: use ImageNet weights, GAP (GlobalAveragePooling2D), smaller FC
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape) # ❌weights=None
        base_model.trainable = False  # freeze first; can unfreeze last blocks later for fine-tuning
    
        x = layers.GlobalAveragePooling2D()(base_model.output)   # much smaller flatten
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
    
        model = models.Model(inputs=base_model.input, outputs=output)

    # compile after model creation (outside model_fn also ok)


    elif model_name == "ResNet34":
        # 📌 ResNet-style with skip connections (simulated using ResNet50)
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base model: Set False to use pre-trained weights
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)

    elif model_name == "EfficientNetB0":
        # 📌 EfficientNetB0 - lightweight and efficient
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base model: Set False to use pre-trained weights
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)

    elif model_name == "EfficientNetB1":
        # 📌 EfficientNetB1 - slightly larger and deeper than B0
        base_model = EfficientNetB1(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base model: Set False to use pre-trained weights
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)

    elif model_name == "EfficientNetB2":
        # 📌 EfficientNetB2 - deeper and more complex than B1
        base_model = EfficientNetB2(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base model: Set False to use pre-trained weights
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base_model.input, outputs=output)

    else:
        raise ValueError(f"❌ Model '{model_name}' not supported.")

    # ⚙️ Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Suitable for one-hot categorical labels
        metrics=['accuracy']
    )

    return model

