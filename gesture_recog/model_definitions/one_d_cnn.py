from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_oned_cnn(win=64, num_classes=4, lr=1e-3, l2=None, dropout=None):
    """
    New CNN architecture:
    - Conv1D 64, kernel 9
    - Conv1D 64, kernel 5
    - MaxPool1D pool_size=2, strides=2
    - Conv1D 64, kernel 3
    - MaxPool1D pool_size=2, strides=2
    - Flatten
    - Dense 512 (with L2 reg and dropout)
    - Dense 128 (with L2 reg and dropout)
    - Dense num_classes, softmax
    """
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    dp = dropout if dropout and dropout > 0 else 0.0

    inputs = keras.Input(shape=(win, 3))

    x = layers.Conv1D(64, kernel_size=9, activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    if dp > 0:
        x = layers.Dropout(dp)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    if dp > 0:
        x = layers.Dropout(dp)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model