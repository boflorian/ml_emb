from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_imu_model(win=128, num_classes=4, l2=1e-4, dropout=0.3):
    inp = keras.Input(shape=(win, 3))  # fixed window, channels-last

    # Conv block 1
    x = layers.Conv1D(64, 5, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv block 2
    x = layers.Conv1D(128, 5, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Temporal modeling
    x = layers.Bidirectional(layers.LSTM(96, dropout=dropout, return_sequences=False))(x)

    # Head
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model