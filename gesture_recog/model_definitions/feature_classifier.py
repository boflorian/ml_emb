import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feature_classifier(input_dim=30, num_classes=4, hidden_units=[64, 32], dropout=0.3):
    """
    Build a dense neural network classifier for statistical features.
    input_dim: Number of features (default 30 for 10 per axis * 3).
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name="feature_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model