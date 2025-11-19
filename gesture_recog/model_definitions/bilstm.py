import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def build_bilstm_classifier(input_shape=(128, 3), num_classes=4):
    """
    Simple BiLSTM classifier for IMU (T, 3).
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3)
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3)
    )(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="bilstm_magicwand")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model