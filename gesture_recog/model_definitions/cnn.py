from tensorflow import keras
from tensorflow.keras import layers, regularizers

'''
NOTE: Any type of RNN (bilstm, recurrent layers, temporal block) seems
to kill attempts to convert to TFLite. Do no use for real-time applicaiton
on microcontroller
'''
def _res_block(x, filters, kernel_size, reg=None, dilation_rate=1, residual=False):
    """Two Conv1D + BN + LeakyReLU; optional residual to stabilize gradients."""
    skip = x
    x = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        kernel_regularizer=reg,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        kernel_regularizer=reg,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    if residual and skip.shape[-1] == filters:
        x = layers.Add()([skip, x])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def build_cnn(win=64, num_classes=4, lr=1e-3, l2=None, dropout=None):
    # Coerce optional args
    dp = float(dropout) if dropout is not None else 0.0          # Dropout needs a float
    reg = regularizers.l2(l2) if (l2 is not None and l2 > 0) else None

    inputs = keras.Input(shape=(win, 3))
    x = layers.LayerNormalization(axis=-1)(inputs)

    # Higher-capacity front-end with residual stabilization
    x = _res_block(x, 64, 9, reg=reg, residual=False)
    x = _res_block(x, 64, 5, reg=reg, residual=True)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.SpatialDropout1D(dp * 0.5)(x) if dp > 0 else x

    x = _res_block(x, 96, 3, reg=reg, residual=True)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.SpatialDropout1D(dp * 0.5)(x) if dp > 0 else x

    # Dilated block to widen temporal receptive field without extra pooling
    x = _res_block(x, 128, 3, reg=reg, dilation_rate=2, residual=False)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    if dp > 0:
        x = layers.Dropout(dp)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, out)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=float(lr), weight_decay=1e-4, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
