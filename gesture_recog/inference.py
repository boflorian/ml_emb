import json 
from pathlib import Path 
import argparse
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import re 

from main import find_best_model
from data_loader import parse_gesture_file, iter_samples, lowpass_filter, normalize_clip


CATEGORIES = ["negative", "ring", "slope", "wing"]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

def load_best_model(models_root="models"):
    best = find_best_model(models_root)

    run_path = Path(best["path"])
    ckpt = run_path / "checkpoints" / "best_valacc.keras"
    model_path = ckpt if ckpt.exists() else (run_path / "final" / "final.keras")

    print("Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model, run_path


def build_eval_dataset(dataset_root, subject, batch_size=64, lp_window=7, win=64):
    dataset_root = Path(dataset_root)
    subject = Path(subject)

    # extract person id
    match = re.search(r'person(\d+)\.txt$', subject.name)
    person_id = int(match.group(1))
    subject = (person_id,)

    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),  # variable T
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    gen = lambda: iter_samples(subject, dataset_root)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def to_fixed_len(x):
        # center-crop/trim then pad to win
        x = x[:win]
        pad = tf.maximum(0, win - tf.shape(x)[0])
        x = tf.pad(x, [[0, pad], [0, 0]])
        return x  # [win, 3]

    def preprocess(x, y):
        x = lowpass_filter(x, window=lp_window)  # <- actually use lp_window
        x = normalize_clip(x)
        x = to_fixed_len(x)                      # <- force [win,3]
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)                    # <- batch to [B, win, 3]
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
            for i in range(len(CATEGORIES))
        ],
        "confusion_matrix": cm.tolist(),
    }
    return metrics



p = argparse.ArgumentParser(description="Run inference on a single data file")
p.add_argument("file", type=Path, help="Path to personX.txt")
args = p.parse_args()

samples = parse_gesture_file(args.file)
if not samples:
    raise SystemExit(f"No samples found in {args.file}")

model, run_dir = load_best_model()  

dataset_root = Path('dataset_magic_wand')
s = Path(args.file)

ds = build_eval_dataset(dataset_root, s, batch_size=64,
                                lp_window=7, win=64)
metrics = evaluate_dataset(model, ds, CATEGORIES)


print("\nExternal dataset metrics")
print("  accuracy :", f"{metrics['accuracy']:.4f}")
print("  macro F1 :", f"{metrics['macro_f1']:.4f}")








