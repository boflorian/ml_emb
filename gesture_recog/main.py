import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pathlib
from datetime import datetime
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from util.data_loader import (
    CATEGORIES,
    iter_samples,
    segment_windows,
    lowpass_filter,
    normalize_clip,
    build_train_test_datasets
)

from model_definitions.cnn import build_cnn
from model_definitions.bilstm import build_bilstm_classifier
from model_definitions.one_d_cnn import build_oned_cnn
from model_definitions.feature_classifier import build_feature_classifier
from model_definitions.deep_cnn import build_deep_cnn
from util.feature_extraction import tf_extract_features
# -----------------------------
# Global config
# -----------------------------
PICO_ROOT = pathlib.Path("dataset_pico_gestures/processed")
MAGIC_WAND_ROOT = pathlib.Path("dataset_magic_wand")

SUBJECTS = list(range(8))  #careful, hardcoded, breaks on pico dataset

MAX_EPOCHS = 2000  # Global max epochs setting
# ==============================
#  Hyperparameters
# =============================
WIN_CNN = 64
HOP_CNN = 64
LP_WINDOW_CNN = 7
BATCH_SIZE_CNN = 64 
LR_CNN = 0.0001  
L2_CNN = 1e-4  
DROPOUT_CNN = 0.25  

WIN_BILSTM = 64
HOP_BILSTM = 64
LP_WINDOW_BILSTM = 5
BATCH_SIZE_BILSTM = 32  
LR_BILSTM = 5e-4  
L2_BILSTM = 1e-4  

WIN_ONE_D_CNN = 64
HOP_ONE_D_CNN = 64
LP_WINDOW_ONE_D_CNN = 7
BATCH_SIZE_ONE_D_CNN = 16  
LR_ONE_D_CNN = 0.0001
L2_ONE_D_CNN = 0.004  
DROPOUT_ONE_D_CNN = 0.25  

PATIENCE = 200  

CNN_CONFIG = dict(
    win=WIN_CNN, hop=HOP_CNN, lp_window=LP_WINDOW_CNN, batch_size=BATCH_SIZE_CNN, epochs=MAX_EPOCHS,
    num_classes=4, lr=LR_CNN, l2=L2_CNN, dropout=DROPOUT_CNN, augment=True 
)

BILSTM_CONFIG = dict(
    win=WIN_BILSTM, hop=HOP_BILSTM, lp_window=LP_WINDOW_BILSTM, batch_size=BATCH_SIZE_BILSTM, epochs=MAX_EPOCHS,
    num_classes=4, lr=LR_BILSTM, l2=L2_BILSTM, augment=True  
)

ONE_D_CNN_CONFIG = dict(
    win=WIN_ONE_D_CNN, hop=HOP_ONE_D_CNN, lp_window=LP_WINDOW_ONE_D_CNN, batch_size=BATCH_SIZE_ONE_D_CNN, epochs=MAX_EPOCHS,
    num_classes=4, lr=LR_ONE_D_CNN, l2=L2_ONE_D_CNN, dropout=DROPOUT_ONE_D_CNN, augment=True
)

WIN_FEATURE = 128
HOP_FEATURE = 64
LP_WINDOW_FEATURE = 5
BATCH_SIZE_FEATURE = 64
HIDDEN_UNITS_FEATURE = [128, 64, 32]  # Deeper network
DROPOUT_FEATURE = 0.3

WIN_DEEP_CNN = 64
HOP_DEEP_CNN = 64
LP_WINDOW_DEEP_CNN = 7
BATCH_SIZE_DEEP_CNN = 16  # Smaller batch size for stability
LR_DEEP_CNN = 5e-5  # Lower learning rate for deep network
L2_DEEP_CNN = 1e-3  # Higher regularization
DROPOUT_DEEP_CNN = 0.4  # Higher dropout

FEATURE_CONFIG = dict(
    win=WIN_FEATURE, hop=HOP_FEATURE, lp_window=LP_WINDOW_FEATURE, batch_size=BATCH_SIZE_FEATURE, epochs=MAX_EPOCHS,
    num_classes=4, hidden_units=HIDDEN_UNITS_FEATURE, dropout=DROPOUT_FEATURE, extract_features=True, augment=True 
)

DEEP_CNN_CONFIG = dict(
    win=WIN_DEEP_CNN, hop=HOP_DEEP_CNN, lp_window=LP_WINDOW_DEEP_CNN, batch_size=BATCH_SIZE_DEEP_CNN, epochs=1000,
    num_classes=4, lr=LR_DEEP_CNN, l2=L2_DEEP_CNN, dropout=DROPOUT_DEEP_CNN, augment=True
)

# =================================================

def find_best_averaged_model(model_name):
    """Find the averaged model with the best (highest) mean validation accuracy."""
    model_dir = pathlib.Path("benchmark_models") / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No models found for {model_name}")
    run_dirs = list(model_dir.glob("*_cv"))
    if not run_dirs:
        raise FileNotFoundError(f"No CV runs found for {model_name}")
    
    best_run = None
    best_mean_acc = -1.0
    for run_dir in run_dirs:
        summaries_path = run_dir / "fold_summaries.json"
        if not summaries_path.exists():
            continue
        try:
            with open(summaries_path, 'r') as f:
                fold_summaries = json.load(f)
            val_accs = [s["best_val_accuracy"] for s in fold_summaries]
            mean_acc = np.mean(val_accs)
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_run = run_dir
        except (json.JSONDecodeError, KeyError):
            continue
    
    if best_run is None:
        raise FileNotFoundError(f"No valid fold summaries found for {model_name}")
    
    model_path = best_run / "averaged_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Averaged model not found in {best_run}")
    
    print(f"Selected {model_name} model with mean val acc: {best_mean_acc:.4f} from {best_run.name}")
    return model_path


def find_best_pico_model(model_name):
    """Find the Pico-trained model with the best validation accuracy."""
    model_dir = pathlib.Path("pico_models") / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No Pico models found for {model_name}")
    run_dirs = list(model_dir.glob("*_pico"))
    if not run_dirs:
        raise FileNotFoundError(f"No Pico runs found for {model_name}")

    best_run = None
    best_val_acc = -1.0
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            val_acc = metrics.get("best_val_accuracy", 0.0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_run = run_dir
        except (json.JSONDecodeError, KeyError):
            continue

    if best_run is None:
        raise FileNotFoundError(f"No valid metrics found for {model_name}")

    model_path = best_run / "final_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found in {best_run}")

    print(f"Selected {model_name} Pico model with val acc: {best_val_acc:.4f} from {best_run.name}")
    return model_path


# ----------------------------------------
# Dataset loading for pico
# ----------------------------------------
def _make_pico_ts_ds(subjects=None, batch_size=64, win=128, hop=64, extract_features=False):
    """
    Pico dataset as time series (B,T,3) or features (B,30).
    """
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(),       dtype=tf.int32),
    )

    def gen():
        for x, y in iter_samples(subjects, root_dir=PICO_ROOT):
            yield x, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.flat_map(lambda x, y: segment_windows(x, y, win=win, hop=hop, drop_short=False))

    def preprocess(x, y):
        x = lowpass_filter(x, window=5)
        x = normalize_clip(x)
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if extract_features:
        ds = ds.map(lambda x, y: (tf_extract_features(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_pico_timeseries(batch_size=64, win=128, hop=64, extract_features=False):
    return _make_pico_ts_ds(batch_size=batch_size, win=win, hop=hop, extract_features=extract_features)


# -----------------------------
# Training + evaluation
# -----------------------------
def run_cv_for_model(model_name, build_model_fn, cfg):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = pathlib.Path("benchmark_models") / model_name / f"{run_id}_cv"
    run_root.mkdir(parents=True, exist_ok=True)

    config = cfg.copy()
    config["model_name"] = model_name
    config["class_names"] = CATEGORIES
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    fold_summaries = []

    for val_subject in SUBJECTS:
        print(f"\n=== {model_name} Fold: Val on subject {val_subject} ===")
        summary = run_one_fold(run_root, val_subject, cfg, build_model_fn, model_name)
        fold_summaries.append(summary)

    # Average weights (weighted by validation accuracy)
    ckpt_paths = [run_root / f"fold_{s}" / "checkpoints" / "best_valacc.keras" for s in SUBJECTS]
    val_accs = [s["best_val_accuracy"] for s in fold_summaries]
    averaged_model = average_weights_from_checkpoints(ckpt_paths, build_model_fn, cfg, model_name, weights=val_accs)

    # Save .keras
    keras_path = run_root / "averaged_model.keras"
    averaged_model.save(keras_path)

    # Save SavedModel (format should be preferable for TFLite conversion)
    savedmodel_dir = run_root / "averaged_model_savedmodel"
    if hasattr(averaged_model, "export"):
        averaged_model.export(savedmodel_dir.as_posix())
    else:
        import tensorflow as tf
        tf.saved_model.save(averaged_model, savedmodel_dir.as_posix())

    # Evaluate averaged model on full training set
    full_train_subjects = SUBJECTS  # all subjects
    full_train_ds, _ = build_train_test_datasets(
        train_subjects=full_train_subjects,
        test_subjects=[],  # no test
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"], hop=cfg["hop"],
        aug_cfg=None,  # no augmentation for evaluation
        augment=False,
        extract_features=cfg.get("extract_features", False)
    )
    full_train_loss, full_train_acc = averaged_model.evaluate(full_train_ds, verbose=0)
    full_train_acc = float(full_train_acc)

    # Save fold summaries
    (run_root / "fold_summaries.json").write_text(json.dumps(fold_summaries, indent=2))

    # Print summary metrics
    val_accs = [s["best_val_accuracy"] for s in fold_summaries]
    mean_val_acc = np.mean(val_accs)
    std_val_acc = np.std(val_accs)
    print(f"\n{model_name.upper()} CV Summary:")
    print(f"Mean Val Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Overall Train Accuracy (Averaged Model): {full_train_acc:.4f}")
    print(f"Individual folds: {val_accs}")

    return averaged_model, run_root


def run_pico_training(model_name, build_model_fn, cfg):
    """Train a model on the full Pico dataset with validation split."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = pathlib.Path("pico_models") / model_name / f"{run_id}_pico"
    run_root.mkdir(parents=True, exist_ok=True)

    config = cfg.copy()
    config["model_name"] = model_name
    config["class_names"] = CATEGORIES
    config["dataset"] = "pico"
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    # Load full Pico dataset
    if cfg.get("extract_features", False):
        train_ds, val_ds = load_pico_train_val_split(
            batch_size=cfg["batch_size"],
            win=cfg["win"],
            hop=cfg["hop"],
            val_split=0.2,
            extract_features=True
        )
    else:
        train_ds, val_ds = load_pico_train_val_split(
            batch_size=cfg["batch_size"],
            win=cfg["win"],
            hop=cfg["hop"],
            val_split=0.2,
            extract_features=False
        )

    # Build model
    if cfg.get("extract_features", False):
        model = build_model_fn(input_dim=30, num_classes=cfg["num_classes"],
                              hidden_units=cfg.get("hidden_units", [64,32]),
                              dropout=cfg.get("dropout", 0.3))
    elif "lr" in cfg and "l2" in cfg:
        model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"],
                              lr=cfg["lr"], l2=cfg["l2"])
    else:
        model = build_model_fn(input_shape=(cfg["win"], 3), num_classes=cfg["num_classes"])

    # Callbacks
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        str(run_root / "best_valacc.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    csv_cb = keras.callbacks.CSVLogger(str(run_root / "history.csv"))
    early_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True
    )
    rlr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=[ckpt_cb, csv_cb, early_cb, rlr_cb],
        verbose=1
    )

    # Save final model
    model.save(run_root / "final_model.keras")

    # Save SavedModel (format should be preferable for TFLite conversion)
    savedmodel_dir = run_root / "finalmodel"
    if hasattr(model, "export"):
        model.export(savedmodel_dir.as_posix())
    else:
        import tensorflow as tf
        tf.saved_model.save(model, savedmodel_dir.as_posix())

    # Get best validation accuracy
    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    val_acc = float(val_acc)

    # Save metrics
    metrics = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": val_acc,
        "final_val_loss": float(val_loss),
        "epochs_trained": len(history.history["val_accuracy"])
    }
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n{model_name.upper()} Pico Training Summary:")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Final Val Accuracy: {val_acc:.4f}")
    print(f"Training completed in {len(history.history['val_accuracy'])} epochs")

    return model, run_root, metrics


def load_pico_train_val_split(batch_size=64, win=128, hop=64, val_split=0.2, extract_features=False):
    """Load Pico dataset and split into train/val sets."""
    # Load all Pico data
    ds = _make_pico_ts_ds(batch_size=1, win=win, hop=hop, extract_features=extract_features)

    # Convert to lists for splitting
    all_samples = []
    for x_batch, y_batch in ds:
        for x, y in zip(x_batch, y_batch):
            all_samples.append((x.numpy(), y.numpy()))

    # Shuffle
    np.random.shuffle(all_samples)

    # Split
    split_idx = int(len(all_samples) * (1 - val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Create datasets
    def create_ds(samples):
        def gen():
            for x, y in samples:
                yield x, y
        output_signature = (
            tf.TensorSpec(shape=(None, 30 if extract_features else 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = create_ds(train_samples)
    val_ds = create_ds(val_samples)

    return train_ds, val_ds


def run_one_fold(run_root, val_subject, cfg, build_model_fn, model_name):
    train_subjects = tuple(s for s in SUBJECTS if s != val_subject)
    test_subjects  = (val_subject,)  

    fold_dir = run_root / f"fold_{val_subject}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # record fold-specific config
    cfg_fold = dict(cfg)
    cfg_fold["split"] = dict(train_subjects=list(train_subjects), val_subjects=[val_subject])
    (fold_dir / "config.json").write_text(json.dumps(cfg_fold, indent=2))

    # augmentation strengths
    aug_cfg = dict(
        rot_deg=30.0,  
        scale_low=0.8, scale_high=1.2,  # wider range
        jitter_std=0.1,  
        shift_max=12,  
        time_mask_prob=0.5, time_mask_max_ratio=0.15,
        mag_warp_strength=0.2, 
        time_warp_ratio=0.25, 
    ) if cfg.get("augment", False) else None # data train uses augment=True internally
    train_ds, val_ds = build_train_test_datasets(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"], hop=cfg["hop"],
        aug_cfg=aug_cfg,
        augment=cfg.get("augment", True),
        extract_features=cfg.get("extract_features", False)
    )

    train_card = tf.data.experimental.cardinality(train_ds).numpy()
    val_card = tf.data.experimental.cardinality(val_ds).numpy()

    if train_card == 0:
        print(f"[WARN] No training data for fold with val_subject={val_subject}. Skipping this fold.")
        summary = dict(
            val_subject=val_subject,
            best_val_accuracy=0.0,
            last_epoch_metrics={}
        )
        (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # model
    if cfg.get("extract_features", False):
        model = build_model_fn(input_dim=30, num_classes=cfg["num_classes"], hidden_units=cfg.get("hidden_units", [64,32]), dropout=cfg.get("dropout", 0.3))
    elif model_name == "deep_cnn" and "lr" in cfg and "l2" in cfg:
        model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"], dropout=cfg.get("dropout", 0.4))
    elif model_name == "bilstm" and "lr" in cfg and "l2" in cfg:
        model = build_model_fn(input_shape=(cfg["win"], 3), num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    elif "lr" in cfg and "l2" in cfg:
        model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    else:
        model = build_model_fn(input_shape=(cfg["win"], 3), num_classes=cfg["num_classes"])

    # callbacks
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        str(fold_dir / "checkpoints" / "best_valacc.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    csv_cb = keras.callbacks.CSVLogger(str(fold_dir / "history.csv"))
    early_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True
    )
    rlr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=[ckpt_cb, csv_cb, early_cb, rlr_cb],
        verbose=1
    )

    # save artifacts
    model.save(fold_dir / "final.keras")

    # summarize metrics for this fold
    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    last = {k: float(v[-1]) for k, v in history.history.items()}
    summary = dict(
        val_subject=val_subject,
        best_val_accuracy=best_val_acc,
        last_epoch_metrics=last
    )
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def average_weights_from_checkpoints(ckpt_paths, build_model_fn, cfg, model_name, weights=None):
    """Element-wise weighted average of weights across several checkpoints."""
    if not all(p.exists() for p in ckpt_paths):
        raise FileNotFoundError("Some checkpoints missing")

    if weights is None:
        weights = [1.0] * len(ckpt_paths)
    else:
        if len(weights) != len(ckpt_paths):
            raise ValueError("Weights length must match checkpoints")
        # Normalize weights to sum to len (for average)
        total_weight = sum(weights)
        weights = [w / total_weight * len(weights) for w in weights]

    # Build a fresh model to know the weight structure
    if cfg.get("extract_features", False):
        base = build_model_fn(input_dim=30, num_classes=cfg["num_classes"], hidden_units=cfg.get("hidden_units", [64,32]), dropout=cfg.get("dropout", 0.3))
    elif model_name == "deep_cnn" and "lr" in cfg and "l2" in cfg:
        base = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"], dropout=cfg.get("dropout", 0.4))
    elif "lr" in cfg and "l2" in cfg:
        base = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    else:
        base = build_model_fn(input_shape=(cfg["win"], 3), num_classes=cfg["num_classes"])
    base_weights = None
    for i, p in enumerate(ckpt_paths):
        model = keras.models.load_model(p)
        if base_weights is None:
            base_weights = [w.numpy() * weights[i] for w in model.weights]
        else:
            for j in range(len(base_weights)):
                base_weights[j] += model.weights[j].numpy() * weights[i]
    # mean
    for i in range(len(base_weights)):
        base_weights[i] /= len(ckpt_paths)
    # put into a fresh instance
    if cfg.get("extract_features", False):
        final_model = build_model_fn(input_dim=30, num_classes=cfg["num_classes"], hidden_units=cfg.get("hidden_units", [64,32]), dropout=cfg.get("dropout", 0.3))
    elif model_name == "deep_cnn" and "lr" in cfg and "l2" in cfg:
        final_model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"], dropout=cfg.get("dropout", 0.4))
    elif "lr" in cfg and "l2" in cfg:
        final_model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    else:
        final_model = build_model_fn(input_shape=(cfg["win"], 3), num_classes=cfg["num_classes"])
    final_model.set_weights(base_weights)
    return final_model


def evaluate_on_dataset(model, ds, class_names=CATEGORIES, threshold=0.5):
    print(f"Using confidence threshold: {threshold}")
    y_true = []
    y_pred = []
    max_probs = []

    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        preds = []
        for prob in probs:
            max_prob = np.max(prob)
            max_probs.append(max_prob)
            if max_prob < threshold:
                pred = 0  # Negative class for low confidence
            else:
                pred = np.argmax(prob)
            preds.append(pred)
        y_true.append(yb.numpy())
        y_pred.append(preds)

    if len(y_true) == 0:
        print("WARNING: dataset empty")
        return {}

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    max_probs = np.array(max_probs)

    acc = float(np.mean(y_true == y_pred))
    # Diagnostics TODO: remove later or comment out 
    print(f"Accuracy: {acc:.4f}")
    print(f"Average max probability: {np.mean(max_probs):.4f}")
    print(f"Median max probability: {np.median(max_probs):.4f}")
    print(f"Min max probability: {np.min(max_probs):.4f}")
    print(f"Max max probability: {np.max(max_probs):.4f}")

    # Print assigned labels summary
    unique_preds, counts = np.unique(y_pred, return_counts=True)
    print("Predicted labels distribution:")
    for label, count in zip(unique_preds, counts):
        print(f"  {CATEGORIES[label]}: {count} samples")

    # Print sample of true and predicted labels
    #t 200): {y_true[:200]}")
    #print(f"Sample pred labels (first 80): {y_pred[:80]}")

    labels_present = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    present_names = [class_names[i] for i in labels_present]

    print("\nClassification report:")
    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=present_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=present_names,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def run_inference(model, run_root, model_name, dataset):
    # Use appropriate window size for each model
    if model_name in ["cnn", "one_d_cnn", "deep_cnn"]:
        win, hop = WIN_CNN, HOP_CNN
        extract_features = False
    elif model_name == "bilstm":
        win, hop = WIN_BILSTM, HOP_BILSTM
        extract_features = False
    elif model_name == "feature":
        win, hop = WIN_FEATURE, HOP_FEATURE
        extract_features = True
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    if dataset == "pico":
        ds = load_pico_timeseries(batch_size=64, win=win, hop=hop, extract_features=extract_features)
        dataset_name = "pico"
    elif dataset == "magic_wand":
        from ml_emb.gesture_recog.util.inference import build_eval_dataset
        ds = build_eval_dataset(dataset_root=pathlib.Path("dataset_magic_wand"), 
                               batch_size=64, win=win)
        dataset_name = "magic_wand"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    metrics = evaluate_on_dataset(model, ds, class_names=CATEGORIES, threshold=0.4)
    (run_root / f"metrics_{dataset_name}_{model_name}.json").write_text(json.dumps(metrics, indent=2))


def train_single_model(model_item):
    """Helper function for parallel training."""
    model_name, model_info, dataset = model_item
    if dataset == "magic_wand":
        print(f"\n=== Training {model_name.upper()} with CV on Magic Wand ===")
        model, run_root = run_cv_for_model(model_name, model_info['build_fn'], model_info['config'])
        # Collect CV summary
        summaries_path = run_root / "fold_summaries.json"
        cv_summary = None
        if summaries_path.exists():
            with open(summaries_path, 'r') as f:
                fold_summaries = json.load(f)
            val_accs = [s["best_val_accuracy"] for s in fold_summaries]
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)
            cv_summary = {
                "mean_val_acc": mean_val_acc,
                "std_val_acc": std_val_acc,
                "individual_folds": val_accs
            }
    else:  # pico
        print(f"\n=== Training {model_name.upper()} on Pico dataset ===")
        model, run_root, metrics = run_pico_training(model_name, model_info['build_fn'], model_info['config'])
        cv_summary = {
            "pico_val_acc": metrics["best_val_accuracy"],
            "pico_val_loss": metrics["final_val_loss"],
            "epochs_trained": metrics["epochs_trained"]
        }
    return model_name, model, run_root, cv_summary


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and/or evaluate CNN and BiLSTM models on gesture recognition.")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both",
                        help="Mode: train (CV training only), inference (pico evaluation only), both (default)")
    parser.add_argument("--models", nargs='*', default=['cnn', 'bilstm', 'one_d_cnn', 'feature'],
                        help="Models to train/infer: cnn, bilstm, one_d_cnn, feature, deep_cnn (default all except deep_cnn)")
    parser.add_argument("--parallel", action="store_true",
                        help="Train multiple models in parallel using multiprocessing")
    parser.add_argument("--dataset", choices=["magic_wand", "pico"], default="magic_wand",
                        help="Dataset to use: magic_wand (default, cross-validation), pico (train on pico dataset)")
    args = parser.parse_args()

    # Create logs directory
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define available models
    available_models = {
        'cnn': {'build_fn': build_cnn, 'config': CNN_CONFIG},
        'bilstm': {'build_fn': build_bilstm_classifier, 'config': BILSTM_CONFIG},
        'one_d_cnn': {'build_fn': build_oned_cnn, 'config': ONE_D_CNN_CONFIG},
        'feature': {'build_fn': build_feature_classifier, 'config': FEATURE_CONFIG},
        'deep_cnn': {'build_fn': build_deep_cnn, 'config': DEEP_CNN_CONFIG}
    }

    # Filter to selected models
    selected_models = {name: available_models[name] for name in args.models if name in available_models}
    if not selected_models:
        raise ValueError("No valid models selected")

    # Initialize model variables
    models = {}
    run_roots = {}
    cv_summaries = {}
    pico_accuracies = {}

    if args.mode in ["train", "both"]:
        if args.parallel:
            num_processes = min(multiprocessing.cpu_count(), len(selected_models))
            print(f"Training {len(selected_models)} models on {args.dataset} dataset in parallel using {num_processes} processes...")
            model_items = [(name, info, args.dataset) for name, info in selected_models.items()]
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(train_single_model, model_items)
            for model_name, model, run_root, cv_summary in results:
                models[model_name] = model
                run_roots[model_name] = run_root
                if cv_summary:
                    cv_summaries[model_name] = cv_summary
        else:
            for model_name, model_info in selected_models.items():
                if args.dataset == "magic_wand":
                    print(f"\n=== Training {model_name.upper()} with CV on Magic Wand ===")
                    model, run_root = run_cv_for_model(model_name, model_info['build_fn'], model_info['config'])
                    # Collect CV summary
                    summaries_path = run_root / "fold_summaries.json"
                    if summaries_path.exists():
                        with open(summaries_path, 'r') as f:
                            fold_summaries = json.load(f)
                        val_accs = [s["best_val_accuracy"] for s in fold_summaries]
                        mean_val_acc = np.mean(val_accs)
                        std_val_acc = np.std(val_accs)
                        cv_summaries[model_name] = {
                            "mean_val_acc": mean_val_acc,
                            "std_val_acc": std_val_acc,
                            "individual_folds": val_accs
                        }
                else:  # pico
                    print(f"\n=== Training {model_name.upper()} on Pico dataset ===")
                    model, run_root, metrics = run_pico_training(model_name, model_info['build_fn'], model_info['config'])
                    cv_summaries[model_name] = {
                        "pico_val_acc": metrics["best_val_accuracy"],
                        "pico_val_loss": metrics["final_val_loss"],
                        "epochs_trained": metrics["epochs_trained"]
                    }
                models[model_name] = model
                run_roots[model_name] = run_root

    if args.mode in ["inference", "both"]:
        for model_name in selected_models:
            if model_name not in models:
                print(f"\n=== Loading best {model_name.upper()} model ===")
                if args.dataset == "pico":
                    model_path = find_best_pico_model(model_name)
                else:
                    model_path = find_best_averaged_model(model_name)
                model = keras.models.load_model(model_path)
                run_root = model_path.parent
                models[model_name] = model
                run_roots[model_name] = run_root
                print(f"Loaded {model_name.upper()} from {model_path}")

        for model_name in selected_models:
            print(f"\n=== Evaluating {model_name.upper()} on {args.dataset.replace('_', ' ').title()} dataset ===")
            run_inference(models[model_name], run_roots[model_name], model_name, args.dataset)

        # Collect dataset accuracies
        dataset_name = args.dataset
        for model_name in selected_models:
            metrics_path = run_roots[model_name] / f"metrics_{dataset_name}_{model_name}.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                pico_accuracies[model_name] = {
                    'accuracy': metrics.get('accuracy', None),
                    'precision': metrics.get('classification_report', {}).get('weighted avg', {}).get('precision', None),
                    'f1_score': metrics.get('classification_report', {}).get('weighted avg', {}).get('f1-score', None)
                }

    # Print final summary if multiple models
    if len(selected_models) > 1:
        print("\n" + "="*60)
        if args.dataset == "magic_wand":
            print("FINAL SUMMARY - Magic Wand CV Training Results")
            print("="*60)
            dataset_display = "Pico"
            print(f"{'Model':<12} {'CV Mean Val Acc':<15} {'CV Std':<10} {'Pico Acc':<10} {'Pico F1':<10} {'Pico Prec':<10}")
            print("-"*75)
            for model_name in selected_models:
                cv_mean = cv_summaries.get(model_name, {}).get('mean_val_acc', 'N/A')
                cv_std = cv_summaries.get(model_name, {}).get('std_val_acc', 'N/A')
                test_acc = pico_accuracies.get(model_name, {}).get('accuracy', 'N/A')
                test_f1 = pico_accuracies.get(model_name, {}).get('f1_score', 'N/A')
                test_prec = pico_accuracies.get(model_name, {}).get('precision', 'N/A')
                cv_mean_str = f"{cv_mean:.4f}" if cv_mean != 'N/A' else 'N/A'
                cv_std_str = f"{cv_std:.4f}" if cv_std != 'N/A' else 'N/A'
                test_acc_str = f"{test_acc:.4f}" if test_acc != 'N/A' else 'N/A'
                test_f1_str = f"{test_f1:.4f}" if test_f1 != 'N/A' else 'N/A'
                test_prec_str = f"{test_prec:.4f}" if test_prec != 'N/A' else 'N/A'
                print(f"{model_name.upper():<12} {cv_mean_str:<15} {cv_std_str:<10} {test_acc_str:<10} {test_f1_str:<10} {test_prec_str:<10}")
        else:  # pico
            print("FINAL SUMMARY - Pico Dataset Training Results")
            print("="*60)
            dataset_display = "Pico"
            print(f"{'Model':<12} {'Pico Val Acc':<12} {'Pico Val Loss':<13} {'Pico Test Acc':<13} {'Pico Test F1':<12} {'Pico Test Prec':<13}")
            print("-"*85)
            for model_name in selected_models:
                val_acc = cv_summaries.get(model_name, {}).get('pico_val_acc', 'N/A')
                val_loss = cv_summaries.get(model_name, {}).get('pico_val_loss', 'N/A')
                test_acc = pico_accuracies.get(model_name, {}).get('accuracy', 'N/A')
                test_f1 = pico_accuracies.get(model_name, {}).get('f1_score', 'N/A')
                test_prec = pico_accuracies.get(model_name, {}).get('precision', 'N/A')
                val_acc_str = f"{val_acc:.4f}" if val_acc != 'N/A' else 'N/A'
                val_loss_str = f"{val_loss:.4f}" if val_loss != 'N/A' else 'N/A'
                test_acc_str = f"{test_acc:.4f}" if test_acc != 'N/A' else 'N/A'
                test_f1_str = f"{test_f1:.4f}" if test_f1 != 'N/A' else 'N/A'
                test_prec_str = f"{test_prec:.4f}" if test_prec != 'N/A' else 'N/A'
                print(f"{model_name.upper():<12} {val_acc_str:<12} {val_loss_str:<13} {test_acc_str:<13} {test_f1_str:<12} {test_prec_str:<13}")
        print("="*60)

    # Plot validation accuracy over epochs for all trained models (only after all processing is complete)
    if args.mode in ["train", "both"] and run_roots:
        print("\n=== Creating validation accuracy plot ===")
        print(f"Found {len(run_roots)} models to plot: {list(run_roots.keys())}")

        plt.figure(figsize=(12, 8))
        
        for model_name in run_roots:
            run_root = run_roots[model_name]
            print(f"Processing model {model_name}, run_root: {run_root}")
            
            if args.dataset == "magic_wand":
                # For Magic Wand: average across folds with range visualization
                fold_histories = []
                for fold_idx in SUBJECTS:
                    history_path = run_root / f"fold_{fold_idx}" / "history.csv"
                    if history_path.exists():
                        try:
                            fold_df = pd.read_csv(history_path)
                            if 'val_accuracy' in fold_df.columns:
                                fold_histories.append(fold_df['val_accuracy'].values)
                            else:
                                print(f"    History file missing val_accuracy column")
                        except Exception as e:
                            print(f"Warning: Could not read history for {model_name} fold {fold_idx}: {e}")
                    else:
                        print(f"    History file does not exist: {history_path}")
                
                if fold_histories:
                    # Calculate statistics across folds for each epoch
                    min_epochs = min(len(h) for h in fold_histories)
                    fold_arrays = np.array([h[:min_epochs] for h in fold_histories])
                    
                    # Calculate mean, min, and max for each epoch
                    avg_val_acc = np.mean(fold_arrays, axis=0)
                    min_val_acc = np.min(fold_arrays, axis=0)
                    max_val_acc = np.max(fold_arrays, axis=0)
                    
                    epochs = range(1, len(avg_val_acc) + 1)
                    
                    # Plot shaded area showing the range of accuracies
                    plt.fill_between(epochs, min_val_acc, max_val_acc, alpha=0.3, 
                                   label=f'{model_name.upper()} CV Range')
                    
                    # Plot the average line on top
                    plt.plot(epochs, avg_val_acc, label=f'{model_name.upper()} CV Mean', 
                           linewidth=2, marker='o', markersize=3)
                    
                    print(f"  Successfully plotted {model_name} CV data (mean + range)")
                    print(f"    Range: {min_val_acc[-1]:.4f} - {max_val_acc[-1]:.4f} (final epoch)")
                else:
                    print(f"  No valid fold histories found for {model_name}")
            else:  # pico
                # For Pico: single training history
                history_path = run_root / "history.csv"
                print(f"  Checking pico history: {history_path}")
                if history_path.exists():
                    try:
                        history_df = pd.read_csv(history_path)
                        if 'val_accuracy' in history_df.columns:
                            val_acc = history_df['val_accuracy'].values
                            epochs = range(1, len(val_acc) + 1)
                            
                            # Apply smoothing to reduce bumpiness
                            if len(val_acc) > 5:  # Only smooth if we have enough data points
                                # Use a simple moving average with window size 3
                                smoothed_val_acc = np.convolve(val_acc, np.ones(3)/3, mode='valid')
                                # Pad the smoothed array to match original length
                                pad_left = 1  # (3-1)//2
                                smoothed_val_acc = np.concatenate([
                                    val_acc[:pad_left], 
                                    smoothed_val_acc, 
                                    val_acc[-pad_left:] if len(val_acc) % 2 == 0 else val_acc[-pad_left-1:]
                                ])
                                plt.plot(epochs, smoothed_val_acc, label=f'{model_name.upper()} (Pico)', 
                                        linewidth=2, alpha=0.8)
                            else:
                                # Not enough points for smoothing, plot raw
                                plt.plot(epochs, val_acc, label=f'{model_name.upper()} (Pico)', 
                                        linewidth=2, marker='s', markersize=3, alpha=0.8)
                            print(f"  Successfully plotted {model_name} pico data")
                        else:
                            print(f"    History file missing val_accuracy column")
                    except Exception as e:
                        print(f"Warning: Could not read history for {model_name}: {e}")
                else:
                    print(f"    History file does not exist: {history_path}")
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        title_suffix = "Magic Wand (CV)" if args.dataset == "magic_wand" else "Pico Dataset"
        plt.title(f'Validation Accuracy Over Epochs - {title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plot_path = logs_dir / f"validation_accuracy_over_epochs_{args.dataset}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved validation accuracy plot to {plot_path}")
        plt.show()

    # Log all metrics and hyperparameters
    log_file = logs_dir / "training_log.json"
    
    # Load existing log if exists
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "models": list(selected_models.keys()),
        "parallel": args.parallel,
        "dataset": args.dataset,
    }
    
    for model_name in selected_models:
        model_log = {
            "model_name": model_name,
            "hyperparameters": selected_models[model_name]['config'],
        }
        if model_name in cv_summaries:
            model_log["cv_metrics"] = cv_summaries[model_name]
        if model_name in pico_accuracies:
            model_log["pico_metrics"] = pico_accuracies[model_name]
        log_entry[model_name] = model_log
    
    log_data.append(log_entry)
    
    # Save log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nLogged run to {log_file}")

    dataset_name = args.dataset.replace('_', ' ').title()
    if args.mode == "train":
        for model_name in selected_models:
            print(f"\n{model_name.upper()} {dataset_name} results saved to: {run_roots[model_name].resolve()}")
    elif args.mode == "inference":
        for model_name in selected_models:
            print(f"\n{model_name.upper()} {dataset_name} metrics saved to: {run_roots[model_name].resolve()}")
    else:
        for model_name in selected_models:
            print(f"\n{model_name.upper()} {dataset_name} results saved to: {run_roots[model_name].resolve()}")


if __name__ == "__main__":
    main()