import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


def build_bilstm_classifier(input_shape=(128, 3), num_classes=4, lr=1e-3, l2=1e-4):
    """
    Simple BiLSTM classifier for IMU (T, 3).
    """
    l2_reg = keras.regularizers.L2(l2)
    
    inputs = keras.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, kernel_regularizer=l2_reg)
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, kernel_regularizer=l2_reg)
    )(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="bilstm_magicwand")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model