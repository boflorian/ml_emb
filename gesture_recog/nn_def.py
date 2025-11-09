from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_imu_model(win=128, num_classes=4, lr=1e-3, l2=None, dropout=None):
    # Coerce optional args
    dp = float(dropout) if dropout is not None else 0.0          # LSTM/Dropout need a float
    reg = regularizers.l2(l2) if (l2 is not None and l2 > 0) else None

    inp = keras.Input(shape=(win, 3))

    # Conv block 1
    x = layers.Conv1D(64, 5, padding="same", activation="relu",
                      kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 2
    x = layers.Conv1D(128, 5, padding="same", activation="relu",
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Temporal block
    x = layers.Bidirectional(layers.LSTM(96, dropout=dp, return_sequences=False))(x)

    # Head
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
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