from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn(win=64, num_classes=4, lr=1e-3, l2=None, dropout=None):
    # Coerce optional args
    dp = float(dropout) if dropout is not None else 0.0          # LSTM/Dropout need a float
    reg = regularizers.l2(l2) if (l2 is not None and l2 > 0) else None

    inputs = keras.Input(shape=(win, 3))

    x = layers.Conv1D(64, kernel_size=9, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    # Temporal block
    x = layers.Bidirectional(layers.LSTM(96, dropout=dp, return_sequences=False))(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    if dp > 0:
        x = layers.Dropout(dp)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model