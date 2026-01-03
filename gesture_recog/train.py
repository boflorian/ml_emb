import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pathlib
import re
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from util.augmentation import augment_sample
import util.data_loader as data_loader
from util.data_loader import (
    parse_gesture_file,
    segment_windows,
    lowpass_filter,
    normalize_clip,
    build_train_test_datasets
)

from model_definitions.cnn import build_cnn
from util.feature_extraction import tf_extract_features
# -----------------------------
# Global config
# -----------------------------
GOOD_ROOT = pathlib.Path("dataset_good")
MODEL_ROOT = pathlib.Path("good_models")
LOGS_ROOT = pathlib.Path("good_logs")

CATEGORIES = ["negative", "ring", "slope", "wave"]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}
TRAIN_FILE_RE = re.compile(r"(?:train|test)_(\d+)\.txt$", re.IGNORECASE)
CV_FOLDS = 5
CV_SEED = 42




MAX_EPOCHS = 500  # Global max epochs setting
# ==============================
#  Hyperparameters
# =============================
WIN_CNN = 286
HOP_CNN = None
LP_WINDOW_CNN = 7
BATCH_SIZE_CNN = 16
LR_CNN = 5e-3
L2_CNN = 5e-2
DROPOUT_CNN = 0.25

PATIENCE = 100 

CNN_CONFIG = dict(
    win=WIN_CNN, hop=HOP_CNN, lp_window=LP_WINDOW_CNN, batch_size=BATCH_SIZE_CNN, epochs=MAX_EPOCHS,
    num_classes=len(CATEGORIES), lr=LR_CNN, l2=L2_CNN, dropout=DROPOUT_CNN, augment=False
)
# =================================================




def _subject_id(path: pathlib.Path):
    match = TRAIN_FILE_RE.search(path.name)
    return int(match.group(1)) if match else None


def list_available_subjects(root):
    subjects = set()
    root_path = pathlib.Path(root)
    for cls in CATEGORIES:
        cls_dir = root_path / cls
        if not cls_dir.exists():
            continue
        for path in cls_dir.glob("*.txt"):
            pid = _subject_id(path)
            if pid is not None:
                subjects.add(pid)
    return sorted(subjects)


def iter_samples(subjects, root_dir=GOOD_ROOT):
    """Yield (seq[T,3] float32, label int32) only for the given subject ids."""
    subj_set = set(subjects) if subjects is not None else None
    root = pathlib.Path(root_dir)
    for cls in CATEGORIES:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for txt in sorted(cls_dir.glob("*.txt")):
            pid = _subject_id(txt)
            if pid is None:
                continue
            if subj_set is not None and pid not in subj_set:
                continue
            for rec in parse_gesture_file(txt):
                x = np.asarray(rec["data"], dtype=np.float32)
                if x.size == 0:
                    continue
                y = np.int32(CAT_TO_ID[cls])
                yield x, y


data_loader.CATEGORIES = CATEGORIES
data_loader.CAT_TO_ID = CAT_TO_ID
data_loader.ROOT = GOOD_ROOT
data_loader.iter_samples = iter_samples
data_loader.list_available_subjects = list_available_subjects


def _collect_all_samples(root_dir=GOOD_ROOT):
    return list(iter_samples(subjects=None, root_dir=root_dir))


def _split_samples_stratified_kfold(samples, num_folds, seed):
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for cross-validation")
    rng = np.random.default_rng(seed)

    class_to_indices = {}
    for idx, (_, y) in enumerate(samples):
        class_to_indices.setdefault(int(y), []).append(idx)

    for indices in class_to_indices.values():
        rng.shuffle(indices)

    folds = [[] for _ in range(num_folds)]
    for indices in class_to_indices.values():
        splits = np.array_split(indices, num_folds)
        for fold_idx, split in enumerate(splits):
            folds[fold_idx].extend(split.tolist())

    return [[samples[i] for i in fold] for fold in folds]


def _make_ds_from_samples(samples, batch_size, lp_window=5,
                          win=None, hop=None, drop_short=False,
                          augment=False, aug_cfg=None, extract_features=False,
                          shuffle=True):
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def gen():
        for x, y in samples:
            yield x, np.int32(y)

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if win is not None and hop is not None:
        ds = ds.flat_map(lambda x, y: segment_windows(x, y, win=win, hop=hop, drop_short=drop_short))

    aug_cfg = aug_cfg or {}

    def preprocess_train(x, y):
        x = augment_sample(x, **aug_cfg)
        # x = lowpass_filter(x, window=lp_window)
        return x, y

    def preprocess_eval(x, y):
        # x = lowpass_filter(x, window=lp_window)
        return x, y

    ds = ds.map(preprocess_train if augment else preprocess_eval,
                num_parallel_calls=tf.data.AUTOTUNE)

    if extract_features:
        ds = ds.map(lambda x, y: (tf_extract_features(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(5000, reshuffle_each_iteration=True)

    if win is not None and hop is not None:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([])),
            padding_values=(tf.constant(0.0, tf.float32), tf.constant(0, tf.int32))
        ).prefetch(tf.data.AUTOTUNE)
    return ds


def _count_samples_by_class(samples):
    counts = {cls: 0 for cls in CATEGORIES}
    for _, y in samples:
        cls = CATEGORIES[int(y)]
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _count_windows_by_class(samples, win, hop):
    if win is None or hop is None:
        return _count_samples_by_class(samples)
    counts = {cls: 0 for cls in CATEGORIES}
    for x, y in samples:
        t = len(x)
        n = int(np.ceil((t - win) / float(hop))) + 1
        if n < 1:
            n = 1
        cls = CATEGORIES[int(y)]
        counts[cls] = counts.get(cls, 0) + n
    return counts


def _compute_class_weights(counts):
    total = sum(counts.values())
    num_classes = len(CATEGORIES)
    weights = {}
    for idx, cls in enumerate(CATEGORIES):
        cls_count = counts.get(cls, 0)
        if cls_count > 0:
            weights[idx] = total / float(num_classes * cls_count)
    return weights



def find_best_ensemble_run(model_name):
    """Find the CV run with the best mean validation accuracy for ensembling."""
    model_dir = MODEL_ROOT / model_name
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
            with open(summaries_path, "r") as f:
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

    print(f"Selected {model_name} ensemble run with mean val acc: {best_mean_acc:.4f} from {best_run.name}")
    return best_run


def load_best_fold_model(run_root):
    summaries_path = run_root / "fold_summaries.json"
    if not summaries_path.exists():
        raise FileNotFoundError(f"No fold summaries found in {run_root}")
    with summaries_path.open("r") as f:
        fold_summaries = json.load(f)
    best_fold = None
    best_acc = -1.0
    for summary in fold_summaries:
        acc = float(summary.get("best_val_accuracy", -1.0))
        fold_idx = summary.get("fold")
        if fold_idx is None:
            continue
        if acc > best_acc:
            best_acc = acc
            best_fold = fold_idx
    if best_fold is None:
        raise FileNotFoundError(f"No valid fold entries in {summaries_path}")
    ckpt_path = run_root / f"fold_{best_fold}" / "checkpoints" / "best_valacc.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best fold checkpoint not found: {ckpt_path}")
    model = keras.models.load_model(ckpt_path)
    print(f"Loaded best fold model: fold {best_fold} (val acc {best_acc:.4f})")
    return model, best_fold, best_acc




# -----------------------------
# Training + evaluation
# -----------------------------
def run_cv_for_model(model_name, build_model_fn, cfg, debug=False, use_class_weights=False):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = MODEL_ROOT / model_name / f"{run_id}_cv"
    run_root.mkdir(parents=True, exist_ok=True)

    all_samples = _collect_all_samples(GOOD_ROOT)
    if not all_samples:
        raise RuntimeError("No samples found in dataset_good.")
    if len(all_samples) < 2:
        raise RuntimeError("Need at least 2 samples for cross-validation.")
    if debug:
        print("Sample counts by class:", _count_samples_by_class(all_samples))
        print("Window counts by class:", _count_windows_by_class(all_samples, cfg["win"], cfg["hop"]))
    class_counts = {}
    for _, y in all_samples:
        class_counts[int(y)] = class_counts.get(int(y), 0) + 1
    min_class_count = min(class_counts.values())
    num_folds = min(CV_FOLDS, len(all_samples), min_class_count)
    folds = _split_samples_stratified_kfold(all_samples, num_folds, CV_SEED)

    config = cfg.copy()
    config["model_name"] = model_name
    config["class_names"] = CATEGORIES
    config["cv"] = {"folds": num_folds, "seed": CV_SEED, "stratified": True}
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    fold_summaries = []

    for fold_idx in range(num_folds):
        print(f"\n=== {model_name} Fold: {fold_idx + 1}/{num_folds} ===")
        val_samples = folds[fold_idx]
        train_samples = [s for i, f in enumerate(folds) if i != fold_idx for s in f]
        summary = run_one_fold(
            run_root,
            fold_idx,
            train_samples,
            val_samples,
            cfg,
            build_model_fn,
            model_name,
            debug=debug,
            use_class_weights=use_class_weights,
        )
        fold_summaries.append(summary)

    # Average weights (weighted by validation accuracy)
    ckpt_paths = [run_root / f"fold_{i}" / "checkpoints" / "best_valacc.keras" for i in range(num_folds)]
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
    full_train_ds = _make_ds_from_samples(
        all_samples,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"],
        hop=cfg["hop"],
        aug_cfg=None,
        augment=False,
        extract_features=cfg.get("extract_features", False),
        shuffle=False
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


def run_one_fold(run_root, fold_idx, train_samples, val_samples, cfg, build_model_fn, model_name, debug=False, use_class_weights=False):
    fold_dir = run_root / f"fold_{fold_idx}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # record fold-specific config
    cfg_fold = dict(cfg)
    cfg_fold["split"] = {
        "fold": fold_idx,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "seed": CV_SEED
    }
    (fold_dir / "config.json").write_text(json.dumps(cfg_fold, indent=2))

    # augmentation strengths
    aug_cfg = dict(
        rot_deg=0.0,
        scale_low=0.95, scale_high=1.05,
        jitter_std=0.03,
        shift_max=6,
        time_mask_prob=0.2, time_mask_max_ratio=0.1,
        mag_warp_strength=0.1,
        time_warp_ratio=0.1,
    ) if cfg.get("augment", False) else None  # data train uses augment=True internally
    train_ds = _make_ds_from_samples(
        train_samples,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"],
        hop=cfg["hop"],
        aug_cfg=aug_cfg,
        augment=cfg.get("augment", True),
        extract_features=cfg.get("extract_features", False)
    )
    val_ds = _make_ds_from_samples(
        val_samples,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"],
        hop=cfg["hop"],
        aug_cfg=None,
        augment=False,
        extract_features=cfg.get("extract_features", False)
    )

    class_weights = None
    if use_class_weights:
        train_window_counts = _count_windows_by_class(train_samples, cfg["win"], cfg["hop"])
        class_weights = _compute_class_weights(train_window_counts)
        if debug:
            print(f"Fold {fold_idx} class weights:", class_weights)

    if debug:
        print(f"Fold {fold_idx} sample counts:", _count_samples_by_class(train_samples))
        print(f"Fold {fold_idx} window counts:", _count_windows_by_class(train_samples, cfg['win'], cfg['hop']))
        print(f"Fold {fold_idx} val sample counts:", _count_samples_by_class(val_samples))
        print(f"Fold {fold_idx} val window counts:", _count_windows_by_class(val_samples, cfg['win'], cfg['hop']))

    train_card = tf.data.experimental.cardinality(train_ds).numpy()
    val_card = tf.data.experimental.cardinality(val_ds).numpy()

    if train_card == 0:
        print(f"[WARN] No training data for fold {fold_idx}. Skipping this fold.")
        summary = dict(
            fold=fold_idx,
            best_val_accuracy=0.0,
            last_epoch_metrics={}
        )
        (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    steps_per_epoch = int(train_card)
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, int(np.ceil(len(train_samples) / float(cfg["batch_size"]))))
    total_steps = None

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
        callbacks=[cb for cb in [ckpt_cb, csv_cb, early_cb, rlr_cb] if cb is not None],
        class_weight=class_weights,
        verbose=1
    )

    # save artifacts
    model.save(fold_dir / "final.keras")

    # summarize metrics for this fold
    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    last = {k: float(v[-1]) for k, v in history.history.items()}
    summary = dict(
        fold=fold_idx,
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


def evaluate_on_dataset(model, ds, class_names=CATEGORIES, threshold=0.0):
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


def build_good_eval_dataset(dataset_root=GOOD_ROOT, batch_size=64, lp_window=LP_WINDOW_CNN, win=WIN_CNN):
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    gen = lambda: iter_samples(subjects=None, root_dir=dataset_root)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def to_fixed_len(x):
        x = x[:win]
        pad = tf.maximum(0, win - tf.shape(x)[0])
        return tf.pad(x, [[0, pad], [0, 0]])

    def preprocess(x, y):
        # x = lowpass_filter(x, window=lp_window)
        x = to_fixed_len(x)
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _ensemble_predict(models, ds):
    y_true = []
    y_pred = []
    max_probs = []

    for xb, yb in ds:
        probs = None
        for model in models:
            p = model.predict(xb, verbose=0)
            probs = p if probs is None else probs + p
        probs = probs / float(len(models))
        preds = np.argmax(probs, axis=1)
        max_probs.extend(np.max(probs, axis=1).tolist())
        y_true.append(yb.numpy())
        y_pred.append(preds)

    if len(y_true) == 0:
        print("WARNING: dataset empty")
        return {}

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    max_probs = np.array(max_probs)

    acc = float(np.mean(y_true == y_pred))
    print(f"Accuracy: {acc:.4f}")
    print(f"Average max probability: {np.mean(max_probs):.4f}")
    print(f"Median max probability: {np.median(max_probs):.4f}")
    print(f"Min max probability: {np.min(max_probs):.4f}")
    print(f"Max max probability: {np.max(max_probs):.4f}")

    unique_preds, counts = np.unique(y_pred, return_counts=True)
    print("Predicted labels distribution:")
    for label, count in zip(unique_preds, counts):
        print(f"  {CATEGORIES[label]}: {count} samples")

    labels_present = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    present_names = [CATEGORIES[i] for i in labels_present]

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

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CATEGORIES)))
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def run_inference(models, run_root, cfg):
    ds = build_good_eval_dataset(
        dataset_root=GOOD_ROOT,
        batch_size=cfg["batch_size"],
        win=cfg["win"],
    )
    metrics = _ensemble_predict(models, ds)
    (run_root / "metrics_good_cnn.json").write_text(json.dumps(metrics, indent=2))


def export_best_fold_model(run_root, cfg):
    model, best_fold, best_acc = load_best_fold_model(run_root)
    export_path = run_root / "best_fold_model.keras"
    model.save(export_path)
    print(f"Saved best fold model to {export_path}")
    savedmodel_dir = run_root / "best_fold_model_savedmodel"
    if hasattr(model, "export"):
        model.export(savedmodel_dir.as_posix())
    else:
        tf.saved_model.save(model, savedmodel_dir.as_posix())
    print(f"Saved best fold model SavedModel to {savedmodel_dir}")

    ds = build_good_eval_dataset(
        dataset_root=GOOD_ROOT,
        batch_size=cfg["batch_size"],
        win=cfg["win"],
    )
    metrics = evaluate_on_dataset(model, ds, class_names=CATEGORIES, threshold=0.0)
    metrics["best_fold"] = best_fold
    metrics["best_fold_val_acc"] = best_acc
    (run_root / "metrics_good_cnn_best_fold.json").write_text(json.dumps(metrics, indent=2))
    return model


def distill_ensemble(models, cfg, run_root, epochs=50, alpha=0.1, temperature=3.0, optimizer_name="adam"):
    all_samples = _collect_all_samples(GOOD_ROOT)
    if not all_samples:
        raise RuntimeError("No samples found in dataset_good for distillation.")

    ds = _make_ds_from_samples(
        all_samples,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"],
        hop=cfg["hop"],
        aug_cfg=None,
        augment=False,
        extract_features=cfg.get("extract_features", False),
        shuffle=True,
    )

    student = build_cnn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=cfg["lr"])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=cfg["lr"], momentum=0.9, nesterov=True)
    hard_loss_fn = keras.losses.SparseCategoricalCrossentropy()
    soft_loss_fn = keras.losses.KLDivergence()

    for epoch in range(1, epochs + 1):
        losses = []
        accs = []
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                teacher_probs = None
                for model in models:
                    p = model(xb, training=False)
                    teacher_probs = p if teacher_probs is None else teacher_probs + p
                teacher_probs = teacher_probs / float(len(models))

                student_probs = student(xb, training=True)
                teacher_soft = tf.nn.softmax(tf.math.log(teacher_probs + 1e-9) / temperature, axis=-1)
                student_soft = tf.nn.softmax(tf.math.log(student_probs + 1e-9) / temperature, axis=-1)
                hard_loss = hard_loss_fn(yb, student_probs)
                soft_loss = soft_loss_fn(teacher_soft, student_soft) * (temperature ** 2)
                loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))

            losses.append(float(loss.numpy()))
            preds = tf.argmax(student_probs, axis=1)
            preds = tf.cast(preds, yb.dtype)
            accs.append(float(tf.reduce_mean(tf.cast(preds == yb, tf.float32)).numpy()))

        print(f"[DISTILL] Epoch {epoch}/{epochs} loss={np.mean(losses):.4f} acc={np.mean(accs):.4f}")

    distilled_path = run_root / "ensemble_distilled.keras"
    student.save(distilled_path)
    print(f"Saved distilled ensemble model to {distilled_path}")

    ds_eval = build_good_eval_dataset(
        dataset_root=GOOD_ROOT,
        batch_size=cfg["batch_size"],
        win=cfg["win"],
    )
    metrics = evaluate_on_dataset(student, ds_eval, class_names=CATEGORIES, threshold=0.0)
    (run_root / "metrics_good_cnn_distilled.json").write_text(json.dumps(metrics, indent=2))
    return student


def run_tiny_overfit_check(all_samples, cfg, build_model_fn, epochs=30, per_class=2):
    per_class = max(1, int(per_class))
    buckets = {i: [] for i in range(len(CATEGORIES))}
    for x, y in all_samples:
        cls = int(y)
        if len(buckets[cls]) < per_class:
            buckets[cls].append((x, y))
    tiny_samples = [s for cls in buckets.values() for s in cls]
    if not tiny_samples:
        print("[DEBUG] No samples available for tiny overfit check.")
        return

    print(f"[DEBUG] Tiny overfit check with {len(tiny_samples)} samples ({per_class} per class max)")
    tiny_ds = _make_ds_from_samples(
        tiny_samples,
        batch_size=min(cfg["batch_size"], len(tiny_samples)),
        lp_window=cfg["lp_window"],
        win=cfg["win"],
        hop=cfg["hop"],
        aug_cfg=None,
        augment=False,
        extract_features=cfg.get("extract_features", False),
        shuffle=True
    )

    model = build_model_fn(win=cfg["win"], num_classes=cfg["num_classes"], lr=cfg["lr"], l2=cfg["l2"])
    history = model.fit(tiny_ds, epochs=epochs, verbose=0)
    final_acc = float(history.history.get("accuracy", [0.0])[-1])
    best_acc = float(max(history.history.get("accuracy", [0.0])))
    print(f"[DEBUG] Tiny overfit acc: final={final_acc:.4f}, best={best_acc:.4f}")


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
    parser = argparse.ArgumentParser(description="Train and/or evaluate a CNN on dataset_good.")
    parser.add_argument("--mode", choices=["train", "inference", "both"], default="both",
                        help="Mode: train (CV training only), inference (evaluation only), both (default)")
    parser.add_argument("--neg-motion-threshold", type=float, default=None,
                        help="If set, remove negatives with motion energy above this threshold before training.")
    parser.add_argument("--neg-motion-move-dir", default="dataset_good/negative_filtered",
                        help="Where to move removed negatives (ignored if --neg-motion-delete).")
    parser.add_argument("--neg-motion-delete", action="store_true",
                        help="Delete removed negatives instead of moving them.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-class counts and run a tiny overfit check.")
    parser.add_argument("--debug-overfit-epochs", type=int, default=30,
                        help="Epochs for tiny overfit check (debug only).")
    parser.add_argument("--debug-overfit-per-class", type=int, default=2,
                        help="Samples per class for tiny overfit check (debug only).")
    parser.add_argument("--class-weights", action="store_true",
                        help="Apply class weights based on window counts.")
    parser.add_argument("--distill-ensemble", action="store_true",
                        help="Distill the ensemble into a single model file.")
    parser.add_argument("--distill-epochs", type=int, default=20,
                        help="Epochs for ensemble distillation.")
    parser.add_argument("--distill-alpha", type=float, default=0.1,
                        help="Weight for hard-label loss during distillation.")
    parser.add_argument("--distill-temperature", type=float, default=3.0,
                        help="Softmax temperature for distillation.")
    parser.add_argument("--distill-optimizer", choices=["adam", "sgd"], default="adam",
                        help="Optimizer for distillation.")
    args = parser.parse_args()

    # Create logs directory
    logs_dir = LOGS_ROOT
    logs_dir.mkdir(exist_ok=True)

    if args.neg_motion_threshold is not None:
        from remove_high_motion_negatives import filter_negative_files
        removed, kept = filter_negative_files(
            GOOD_ROOT,
            args.neg_motion_threshold,
            pathlib.Path(args.neg_motion_move_dir),
            args.neg_motion_delete,
        )
        action = "deleted" if args.neg_motion_delete else f"moved to {args.neg_motion_move_dir}"
        print(f"Removed {removed} negatives ({action}); kept {kept}.")

    if args.debug:
        all_samples = _collect_all_samples(GOOD_ROOT)
        run_tiny_overfit_check(
            all_samples,
            CNN_CONFIG,
            build_cnn,
            epochs=args.debug_overfit_epochs,
            per_class=args.debug_overfit_per_class
        )

    model_name = "cnn"
    model_info = {"build_fn": build_cnn, "config": CNN_CONFIG}
    selected_models = {model_name: model_info}

    # Initialize model variables
    models = {}
    run_roots = {}
    cv_summaries = {}
    dataset_accuracies = {}

    if args.mode in ["train", "both"]:
        print(f"\n=== Training {model_name.upper()} with CV on dataset_good ===")
        model, run_root = run_cv_for_model(
            model_name,
            model_info["build_fn"],
            model_info["config"],
            debug=args.debug,
            use_class_weights=args.class_weights,
        )
        summaries_path = run_root / "fold_summaries.json"
        if summaries_path.exists():
            with open(summaries_path, "r") as f:
                fold_summaries = json.load(f)
            val_accs = [s["best_val_accuracy"] for s in fold_summaries]
            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)
            cv_summaries[model_name] = {
                "mean_val_acc": mean_val_acc,
                "std_val_acc": std_val_acc,
                "individual_folds": val_accs
            }
        models[model_name] = model
        run_roots[model_name] = run_root

    if args.mode in ["inference", "both"]:
        if model_name not in models or not isinstance(models[model_name], list):
            print(f"\n=== Loading best {model_name.upper()} ensemble ===")
            run_root = find_best_ensemble_run(model_name)
            model_paths = sorted((run_root).glob("fold_*/*/best_valacc.keras"))
            if not model_paths:
                model_paths = sorted((run_root).glob("fold_*/checkpoints/best_valacc.keras"))
            if not model_paths:
                raise FileNotFoundError(f"No fold checkpoints found in {run_root}")
            ensemble_models = [keras.models.load_model(p) for p in model_paths]
            models[model_name] = ensemble_models
            run_roots[model_name] = run_root
            print(f"Loaded {len(ensemble_models)} fold models from {run_root}")

        print(f"\n=== Evaluating {model_name.upper()} on dataset_good ===")
        run_inference(models[model_name], run_roots[model_name], CNN_CONFIG)
        export_best_fold_model(run_roots[model_name], CNN_CONFIG)
        if args.distill_ensemble:
            distill_ensemble(
                models[model_name],
                CNN_CONFIG,
                run_roots[model_name],
                epochs=args.distill_epochs,
                alpha=args.distill_alpha,
                temperature=args.distill_temperature,
                optimizer_name=args.distill_optimizer,
            )

        metrics_path = run_roots[model_name] / "metrics_good_cnn.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            dataset_accuracies[model_name] = {
                "accuracy": metrics.get("accuracy", None),
                "precision": metrics.get("classification_report", {}).get("weighted avg", {}).get("precision", None),
                "f1_score": metrics.get("classification_report", {}).get("weighted avg", {}).get("f1-score", None)
            }

    # Plot validation accuracy over epochs for all trained models (only after all processing is complete)
    if args.mode in ["train", "both"] and run_roots:
        print("\n=== Creating validation accuracy plot ===")
        print(f"Found {len(run_roots)} models to plot: {list(run_roots.keys())}")

        plt.figure(figsize=(12, 8))
        
        for model_name in run_roots:
            run_root = run_roots[model_name]
            print(f"Processing model {model_name}, run_root: {run_root}")
            
            # Average across folds with range visualization
            fold_histories = []
            fold_dirs = sorted(run_root.glob("fold_*"))
            for fold_dir in fold_dirs:
                history_path = fold_dir / "history.csv"
                if history_path.exists():
                    try:
                        fold_df = pd.read_csv(history_path)
                        if "val_accuracy" in fold_df.columns:
                            fold_histories.append(fold_df["val_accuracy"].values)
                        else:
                            print("    History file missing val_accuracy column")
                    except Exception as e:
                        print(f"Warning: Could not read history for {model_name} fold {fold_idx}: {e}")
                else:
                    print(f"    History file does not exist: {history_path}")

            if fold_histories:
                min_epochs = min(len(h) for h in fold_histories)
                fold_arrays = np.array([h[:min_epochs] for h in fold_histories])

                avg_val_acc = np.mean(fold_arrays, axis=0)
                min_val_acc = np.min(fold_arrays, axis=0)
                max_val_acc = np.max(fold_arrays, axis=0)

                epochs = range(1, len(avg_val_acc) + 1)

                plt.fill_between(epochs, min_val_acc, max_val_acc, alpha=0.3,
                                 label=f"{model_name.upper()} CV Range")
                plt.plot(epochs, avg_val_acc, label=f"{model_name.upper()} CV Mean",
                         linewidth=2, marker="o", markersize=3)

                print(f"  Successfully plotted {model_name} CV data (mean + range)")
                print(f"    Range: {min_val_acc[-1]:.4f} - {max_val_acc[-1]:.4f} (final epoch)")
            else:
                print(f"  No valid fold histories found for {model_name}")
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title("Validation Accuracy Over Epochs - Dataset Good (CV)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plot_path = logs_dir / f"validation_accuracy_over_epochs_good_{timestamp}.png"
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
        "parallel": False,
        "dataset": "good",
    }
    if args.neg_motion_threshold is not None:
        log_entry["neg_motion_threshold"] = args.neg_motion_threshold
        log_entry["neg_motion_action"] = "delete" if args.neg_motion_delete else "move"
        log_entry["neg_motion_move_dir"] = args.neg_motion_move_dir
    
    for model_name in selected_models:
        model_log = {
            "model_name": model_name,
            "hyperparameters": selected_models[model_name]['config'],
        }
        if model_name in cv_summaries:
            model_log["cv_metrics"] = cv_summaries[model_name]
        if model_name in dataset_accuracies:
            model_log["good_metrics"] = dataset_accuracies[model_name]
        log_entry[model_name] = model_log
    
    log_data.append(log_entry)
    
    # Save log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nLogged run to {log_file}")

    dataset_name = "Dataset Good"
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
