import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pathlib
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import (
    CATEGORIES,
    _make_ds,
    iter_samples,
    segment_windows,
    lowpass_filter,
    normalize_clip,
)

# -----------------------------
# Global config
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 600
NUM_CLASSES = 4

WIN = 128
HOP = 64
LP_WINDOW = 5

# subject split
TRAIN_SUBJECTS = (0, 1, 2, 3, 4, 5)
VAL_SUBJECTS   = (6,)
TEST_SUBJECTS  = (7,)

PICO_ROOT = pathlib.Path("dataset_pico_gestures/processed")


# -----------------------------
# Dataset loading
# -----------------------------
def make_magicwand_ts_splits(batch_size=BATCH_SIZE, augment=True):
    """
    Return (train_ts, val_ts, test_ts) as (B, T, 3) datasets.
    """
    def make_ts_ds(subjects, aug):
        return _make_ds(
            subjects=subjects,
            batch_size=batch_size,
            lp_window=LP_WINDOW,
            win=WIN,
            hop=HOP,
            drop_short=True,
            augment=aug,
            aug_cfg=None,
        )

    train_ts = make_ts_ds(TRAIN_SUBJECTS, augment_flag=True)
    val_ts   = make_ts_ds(VAL_SUBJECTS,   augment_flag=False)
    test_ts  = make_ts_ds(TEST_SUBJECTS,  augment_flag=False)
    return train_ts, val_ts, test_ts


def _make_pico_ts_ds(subjects, batch_size=BATCH_SIZE):
    """
    Pico dataset as time series (B,T,3).
    """
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(),       dtype=tf.int32),
    )

    def gen():
        for x, y in iter_samples(subjects, root_dir=PICO_ROOT):
            yield x, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.flat_map(lambda x, y: segment_windows(x, y, win=WIN, hop=HOP, drop_short=True))

    def preprocess(x, y):
        x = lowpass_filter(x, window=LP_WINDOW)
        x = normalize_clip(x)
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_pico_timeseries(batch_size=BATCH_SIZE):
    pico_subjects = (0, 1, 2, 3)  # adjust to your actual IDs
    return _make_pico_ts_ds(pico_subjects, batch_size=batch_size)


# -----------------------------
# Model: BiLSTM
# -----------------------------
def build_bilstm_classifier(input_shape=(WIN, 3), num_classes=NUM_CLASSES):
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


# -----------------------------
# Training + evaluation
# -----------------------------
def train_bilstm(batch_size=BATCH_SIZE, epochs=EPOCHS):
    train_ds, val_ds, test_ds = make_magicwand_ts_splits(batch_size=batch_size, augment=True)
    input_shape = (WIN, 3)

    model = build_bilstm_classifier(input_shape)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = pathlib.Path("benchmark_models") / f"{run_id}_bilstm"
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": "bilstm",
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": NUM_CLASSES,
        "class_names": CATEGORIES,
        "train_subjects": list(TRAIN_SUBJECTS),
        "val_subjects": list(VAL_SUBJECTS),
        "test_subjects": list(TEST_SUBJECTS),
        "win": WIN,
        "hop": HOP,
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    ckpt_path = run_root / "best_valacc.keras"
    csv_log_path = run_root / "training_log.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(csv_log_path.as_posix())
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[BiLSTM] Magic Wand test accuracy: {test_acc:.4f}")

    final_path = run_root / "final.keras"
    model.save(final_path)

    (run_root / "history.json").write_text(json.dumps(history.history, indent=2))
    metrics = {"test_loss": float(test_loss), "test_acc": float(test_acc)}
    (run_root / "metrics_magicwand.json").write_text(json.dumps(metrics, indent=2))

    return model, run_root


def evaluate_on_dataset(model, ds, class_names=CATEGORIES):
    y_true = []
    y_pred = []

    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(preds)

    if len(y_true) == 0:
        print("WARNING: dataset empty")
        return {}

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float(np.mean(y_true == y_pred))
    print(f"Accuracy: {acc:.4f}")

    labels_present = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    present_names = [class_names[i] for i in labels_present]

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=present_names,
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }


def run_pico_inference(run_root):
    run_root = pathlib.Path(run_root)
    ckpt_path = run_root / "best_valacc.keras"
    model_path = ckpt_path if ckpt_path.exists() else (run_root / "final.keras")

    print(f"Loading {model_path}")
    model = keras.models.load_model(model_path)

    pico_ds = load_pico_timeseries(batch_size=BATCH_SIZE)

    metrics = evaluate_on_dataset(model, pico_ds, class_names=CATEGORIES)
    (run_root / "metrics_pico.json").write_text(json.dumps(metrics, indent=2))


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n=== Training BiLSTM on Magic Wand ===")
    model, run_root = train_bilstm()

    print("\n=== Evaluating BiLSTM on Pico dataset ===")
    run_pico_inference(run_root)
    print(f"Results saved to: {run_root.resolve()}")