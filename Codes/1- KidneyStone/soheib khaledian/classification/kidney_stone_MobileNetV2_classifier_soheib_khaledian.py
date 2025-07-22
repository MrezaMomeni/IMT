import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.preprocessing_pipeline_soheib_khaleddian import load_preprocessing_pipeline
from utils.evaluation_soheib_khaledian import evaluation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    train_image, test_image, train_label, test_label = load_preprocessing_pipeline("dataset")

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_image, train_label, epochs=10, batch_size=32, validation_split=0.1)

    model.save("models/MobileNetV2_classification.h5")
    
    evaluation(test_image, test_label, model)
    