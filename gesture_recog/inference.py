import json 
from pathlib import Path 
import argparse
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 

from main import find_best_model
from data_loader import parse_gesture_file, iter_samples, lowpass_filter, normalize_clip


CATEGORIES = ["negative", "ring", "slope", "wing"]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

def load_best_model(models_root="models"):
    best = find_best_model(models_root)

    run_path = Path(best["path"])
    ckpt = run_path / "checkpoints" / "best_valacc.keras"
    model_path = ckpt if ckpt.exists() else (run_path / "final.keras")

    print("Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model, run_path


def build_eval_dataset(dataset_root: Path, batch_size=64, lp_window=7, win=None):
    """
    dataset_root must look like:
      root/negative/*.txt
      root/ring/*.txt
      root/slope/*.txt
      root/wing/*.txt
    Uses your iter_samples() to yield (seq[T,3], label), then lowpass + normalize.
    If win is given, each sequence is resampled to fixed length (win).
    """
    dataset_root = Path(dataset_root)

    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    gen = lambda: iter_samples(dataset_root)

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def preprocess(x, y): 
        x = lowpass_filter(x)
        x = normalize_clip(x)

        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def evaluate_dataset(model, ds, categories): 
    y_true, y_pred, y_prob = [], [], []

    for xb, yb in ds: 
        probs = model.predict(xb, verbose=1)
        preds = np.argmax(probs, axis=1)

        y_true.append(yb.numpy())
        y_pred.append(preds)
        y_prob.append(probs)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)


    acc = float((y_true == y_pred).mean())
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=len(categories)).numpy().astype(int)


    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = (tp / (tp + fp + eps)).tolist()
    rec  = (tp / (tp + fn + eps)).tolist()
    f1   = (2 * (np.array(prec) * np.array(rec)) / (np.array(prec) + np.array(rec) + eps)).tolist()
    macro_f1 = float(np.mean(f1))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": [
            {"class": categories[i], "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i in range(len(class_names))
        ],
        "confusion_matrix": cm.tolist(),
    }

def predict_file_samples(model, file_path: Path, win=None, lp_window=7):
    """
    Reads one personX/sessionX file, runs deterministic preprocessing,
    optional resample to fixed win, and prints predicted label + confidence.
    """
    items = parse_gesture_file(file_path)
    # Handle both dict-style and raw-array outputs
    sequences = []
    for it in items:
        if isinstance(it, dict) and "data" in it:
            arr = np.asarray(it["data"], dtype=np.float32)
            sid = it.get("sample_id", None)
        else:
            arr = np.asarray(it, dtype=np.float32)
            sid = None
        if arr.size == 0:
            continue
        sequences.append((sid, arr))

    if not sequences:
        print(f"No valid samples in {file_path}")
        return

    print(f"\nPredictions for {file_path.name}:")
    for idx, (sid, x_raw) in enumerate(sequences, start=1):
        # preprocess in TF to keep parity with training functions
        x = tf.convert_to_tensor(x_raw, tf.float32)
        x = lowpass_filter(x)
        x = normalize_clip(x)
        if win is not None:
            x = tf.convert_to_tensor(resample_linear_to_length(x.numpy(), win), tf.float32)

        probs = model.predict(x[None, ...], verbose=0)[0]
        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])
        label = CATEGORIES[pred_id]
        tag = sid if sid else f"sample{idx}"
        print(f"  {tag:>10s} -> {label:>8s}  (conf={conf:.2f})")


p = argparse.ArgumentParser(description="Run inference on a single data file")
p.add_argument("file", type=Path, help="Path to personX.txt")
args = p.parse_args()

samples = parse_gesture_file(args.file)
if not samples:
    raise SystemExit(f"No samples found in {args.file}")

model, run_dir = load_best_model()

ds = build_eval_dataset(args.dataset_root, batch_size=args.batch_size,
                                lp_window=args.lp_window, win=args.win)
metrics = evaluate_dataset(model, ds, CATEGORIES, verbose=0)
print("\nExternal dataset metrics")
print("  accuracy :", f"{metrics['accuracy']:.4f}")
print("  macro F1 :", f"{metrics['macro_f1']:.4f}")








