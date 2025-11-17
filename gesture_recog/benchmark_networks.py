import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import json
import pathlib
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

from bilstm import build_bilstm_classifier

from data_loader import (
    CATEGORIES,
    build_train_test_datasets,
    _make_ds,
    iter_samples,
    segment_windows,
    lowpass_filter,
    normalize_clip,
)

# -----------------------------
# Global config
# -----------------------------
IMG_SIZE = (96, 96)   # (height, width)
BATCH_SIZE = 64
EPOCHS = 600
PICO_ROOT = pathlib.Path("dataset_pico_gestures/processed")
NUM_CLASSES = 4

# Time-series window config
WIN = 128
HOP = 64
LP_WINDOW = 5

# Explicit subject-based split for Magic Wand
# -> this is your train / validation / test partition
TRAIN_SUBJECTS = (0, 1, 2, 3, 4, 5)
VAL_SUBJECTS   = (6,)
TEST_SUBJECTS  = (7,)


# -----------------------------
# Dataset loading
# -----------------------------
def timeseries_batch_to_image_batch(x, img_size):
    """
    x: (B, T, 3) -> images: (B, H, W, 3)
    """
    x = tf.expand_dims(x, axis=2)         # (B, T, 1, 3)
    x = tf.image.resize(x, img_size)      # (B, H, W, 3)
    return x


def make_image_ds_from_ts_ds(ds, img_size):
    """
    ds: yields (batch, T, 3), labels
    -> ds: yields (batch, H, W, 3), labels
    """
    def _map(x, y):
        x_img = timeseries_batch_to_image_batch(x, img_size)
        return x_img, y

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def make_magicwand_ts_splits(batch_size=BATCH_SIZE, augment=True):
    """
    Return (train_ts, val_ts, test_ts) as time-series datasets with shape (B, T, 3).

    Uses subject-wise split:
      TRAIN_SUBJECTS for training
      VAL_SUBJECTS   for validation
      TEST_SUBJECTS  for final test
    """
    def make_ts_ds(subjects, augment_flag):
        return _make_ds(
            subjects=subjects,
            batch_size=batch_size,
            lp_window=LP_WINDOW,
            win=WIN,
            hop=HOP,
            drop_short=True,
            augment=augment_flag,
            aug_cfg=None,
        )

    train_ts = make_ts_ds(TRAIN_SUBJECTS, augment_flag=augment)
    val_ts   = make_ts_ds(VAL_SUBJECTS,   augment_flag=False)
    test_ts  = make_ts_ds(TEST_SUBJECTS,  augment_flag=False)
    return train_ts, val_ts, test_ts


def load_magicwand_as_images(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Use the existing time-series pipeline and then convert (T, 3) windows
    into (H, W, 3) images for MobileNet/EfficientNet.
    """
    train_ts, val_ts, test_ts = make_magicwand_ts_splits(batch_size=batch_size, augment=True)

    train_ds = make_image_ds_from_ts_ds(train_ts, img_size)
    val_ds   = make_image_ds_from_ts_ds(val_ts,   img_size)
    test_ds  = make_image_ds_from_ts_ds(test_ts,  img_size)

    return train_ds, val_ds, test_ds


def _make_pico_ts_ds(subjects, batch_size, lp_window=LP_WINDOW,
                     win=WIN, hop=HOP, drop_short=True):
    """
    Build a time-series dataset from Pico recordings using iter_samples with root_dir=PICO_ROOT.
    No augmentation, used only for inference/metrics.
    """
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),   # (T, 3)
        tf.TensorSpec(shape=(), dtype=tf.int32),            # label
    )

    def gen():
        for x, y in iter_samples(subjects, root_dir=PICO_ROOT):
            yield x, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # segment into fixed windows so shapes match Magic Wand models
    ds = ds.flat_map(lambda x, y: segment_windows(x, y, win=win, hop=hop, drop_short=drop_short))

    # preprocessing: lowpass + normalize, no augmentation
    def preprocess(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_pico_as_images(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Build a (B, H, W, 3) dataset from Pico recordings, reusing your pipeline.
    """
    pico_subjects = (0, 1, 2, 3)  # adjust to your actual Pico subject IDs

    pico_ts = _make_pico_ts_ds(
        subjects=pico_subjects,
        batch_size=batch_size,
        lp_window=LP_WINDOW,
        win=WIN,
        hop=HOP,
        drop_short=True,
    )

    pico_ds = make_image_ds_from_ts_ds(pico_ts, img_size)
    return pico_ds


def load_pico_timeseries(batch_size=BATCH_SIZE):
    """
    Pico dataset as time series (B, T, 3), for BiLSTM / 1D models.
    """
    pico_subjects = (0, 1, 2, 3)  # adjust to your actual Pico subject IDs

    pico_ts = _make_pico_ts_ds(
        subjects=pico_subjects,
        batch_size=batch_size,
        lp_window=LP_WINDOW,
        win=WIN,
        hop=HOP,
        drop_short=True,
    )
    return pico_ts


# -----------------------------
# Models
# -----------------------------
def build_mobilenetv2_classifier(input_shape, num_classes=NUM_CLASSES):
    """
    MobileNetV2 classifier head on top of tf.keras.applications.MobileNetV2.
    Using weights=None so it's trained from scratch on your domain.
    """
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights=None,             # or "imagenet" if you use proper images
        input_shape=input_shape,
        pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=True)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_magicwand")
    model.compile(
        optimizer=keras.optimizers.Adam(3e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb0_classifier(input_shape, num_classes=NUM_CLASSES):
    """
    EfficientNetB0 classifier head.
    """
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,               # or "imagenet"
        input_shape=input_shape,
        pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=True)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_magicwand")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



# -----------------------------
# Training + evaluation helpers
# -----------------------------
def train_on_magicwand(model_name, img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Train 'mobilenetv2', 'efficientnetb0', or 'bilstm' on Magic Wand.
    Uses the subject-based train/validation/test split defined above.
    """
    # Load datasets and choose input shape
    if model_name in ["mobilenetv2", "efficientnetb0"]:
        train_ds, val_ds, test_ds = load_magicwand_as_images(img_size=img_size, batch_size=batch_size)
        input_shape = (img_size[0], img_size[1], 3)
    elif model_name == "bilstm":
        train_ds, val_ds, test_ds = make_magicwand_ts_splits(batch_size=batch_size, augment=True)
        input_shape = (WIN, 3)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Build model
    if model_name == "mobilenetv2":
        model = build_mobilenetv2_classifier(input_shape)
    elif model_name == "efficientnetb0":
        model = build_efficientnetb0_classifier(input_shape)
    elif model_name == "bilstm":
        model = build_bilstm_classifier(input_shape)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Output directory for this run
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = pathlib.Path("benchmark_models") / f"{run_id}_{model_name}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "model_name": model_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": NUM_CLASSES,
        "class_names": CATEGORIES,
        "win": WIN,
        "hop": HOP,
        "train_subjects": list(TRAIN_SUBJECTS),
        "val_subjects": list(VAL_SUBJECTS),
        "test_subjects": list(TEST_SUBJECTS),
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    # Callbacks
    ckpt_path = run_root / "best_valacc.keras"
    csv_log_path = run_root / "training_log.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(csv_log_path.as_posix())
    ]

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Final test evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[{model_name}] Magic Wand test accuracy: {test_acc:.4f}")

    # Save final model
    final_path = run_root / "final.keras"
    model.save(final_path)

    # Save history + metrics
    (run_root / "history.json").write_text(json.dumps(history.history, indent=2))
    metrics = {"test_loss": float(test_loss), "test_acc": float(test_acc)}
    (run_root / "metrics_magicwand.json").write_text(json.dumps(metrics, indent=2))

    return model, run_root


def evaluate_on_dataset(model, ds, class_names=CATEGORIES):
    """
    Run inference and compute accuracy + metrics.
    Handles the case where only a subset of classes is present in y_true.
    """
    y_true = []
    y_pred = []

    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(preds)

    if len(y_true) == 0:
        print("WARNING: dataset is empty, no samples to evaluate.")
        return {
            "accuracy": None,
            "classification_report": None,
            "confusion_matrix": None,
        }

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = float(np.mean(y_true == y_pred))
    print(f"Accuracy: {acc:.4f}")

    # figure out which labels actually appear
    labels_present = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    present_names = [class_names[i] for i in labels_present]

    # classification report only over the labels that actually occur
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=present_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        digits=4,
        target_names=present_names,
        output_dict=False,
        zero_division=0,
    )

    print("\nClassification report (only classes present in y_true/y_pred):")
    print(report_str)

    # confusion matrix over *all* classes, fixed size
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),   # 0..3 for your 4 classes
    )
    print("Confusion matrix (rows=true, cols=pred; includes missing classes):")
    print(cm)

    return {
        "accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def run_pico_inference_for_run(run_root, img_size=IMG_SIZE):
    """
    Load a trained model from run_root (best_valacc.keras if present, else final.keras)
    and run inference on the Pico dataset.
    Saves metrics_pico.json into the same folder.
    """
    run_root = pathlib.Path(run_root)

    # load config to know which model type this is
    cfg_path = run_root / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        model_name = cfg.get("model_name", "unknown")
    else:
        model_name = "unknown"

    ckpt_path = run_root / "best_valacc.keras"
    final_path = run_root / "final.keras"

    if ckpt_path.exists():
        model_path = ckpt_path
    else:
        model_path = final_path
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # choose the right dataset representation
    if model_name == "bilstm":
        pico_ds = load_pico_timeseries(batch_size=BATCH_SIZE)
    else:
        pico_ds = load_pico_as_images(img_size=img_size, batch_size=BATCH_SIZE)

    metrics_pico = evaluate_on_dataset(model, pico_ds, class_names=CATEGORIES)
    (run_root / "metrics_pico.json").write_text(json.dumps(metrics_pico, indent=2))


# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    # Train all models on Magic Wand with the same subject-wise train/val split
    for model_name in ["mobilenetv2", "efficientnetb0", "bilstm"]:
        print(f"\n=== Training {model_name} on Magic Wand ===")
        model, run_root = train_on_magicwand(model_name)

        print(f"\n=== Evaluating {model_name} on Pico dataset ===")
        run_pico_inference_for_run(run_root)
        print(f"Results saved in: {run_root.resolve()}")