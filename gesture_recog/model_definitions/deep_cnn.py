import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_deep_cnn(win=64, num_classes=4, lr=1e-4, l2=1e-3, dropout=0.4):
    """
    Deep CNN with 5 convolutional layers for IMU gesture recognition.
    Designed for complex feature extraction with heavy regularization to prevent overfitting.
    """
    l2_reg = regularizers.l2(l2)

    inp = keras.Input(shape=(win, 3))

    # Conv block 1 - Initial feature extraction
    x = layers.Conv1D(32, 7, padding="same", activation="relu", kernel_regularizer=l2_reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 2 - Intermediate features
    x = layers.Conv1D(64, 5, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 3 - Deeper features
    x = layers.Conv1D(128, 5, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 4 - Complex patterns
    x = layers.Conv1D(256, 3, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 5 - High-level features
    x = layers.Conv1D(512, 3, padding="same", activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)  # Global pooling instead of max pooling

    # Dense head with heavy regularization
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, outputs, name="deep_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model