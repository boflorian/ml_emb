from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

from main import find_best_model
from data_loader import parse_gesture_file, lowpass_filter, normalize_clip

# fixed defaults
DATASET_ROOT = Path("dataset_magic_wand")
BATCH_SIZE   = 64
WIN          = 64
LP_WINDOW    = 7

CATEGORIES = ["negative", "ring", "slope", "wing"]
CAT_TO_ID  = {c: i for i, c in enumerate(CATEGORIES)}
PERSON_RE  = re.compile(r"person(\d+)\.txt$", re.IGNORECASE)


def load_best_model(models_root="models"):
    best = find_best_model(models_root)
    run_path = Path(best["path"])
    ckpt = run_path / "checkpoints" / "best_valacc.keras"
    model_path = ckpt if ckpt.exists() else (run_path / "final" / "final.keras")
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model, run_path


def iter_samples_all(dataset_root: Path):
    """Yield (x[T,3], y) for every sample in every category, in sorted order."""
    for cat in CATEGORIES:
        for f in sorted((dataset_root / cat).glob("person*.txt")):
            samples = parse_gesture_file(f)
            for s in samples:
                x = np.asarray(s["data"], np.float32)
                y = CAT_TO_ID[cat]
                yield x, y


def build_eval_dataset(dataset_root=DATASET_ROOT, batch_size=BATCH_SIZE,
                       lp_window=LP_WINDOW, win=WIN):
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    gen = lambda: iter_samples_all(dataset_root)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def to_fixed_len(x):
        x = x[:win]
        pad = tf.maximum(0, win - tf.shape(x)[0])
        return tf.pad(x, [[0, pad], [0, 0]])

    def preprocess(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        x = to_fixed_len(x)
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_dataset(model, ds, categories):
    y_true, y_pred, y_prob = [], [], []

    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(preds)
        y_prob.append(probs)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = float((y_true == y_pred).mean())
    cm = tf.math.confusion_matrix(y_true, y_pred,
                                  num_classes=len(categories)).numpy().astype(int)

    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    macro_f1 = float(np.mean(f1))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_true, y_pred, y_prob


def main():
    model, run_dir = load_best_model()
    print("model expects:", model.input_shape)

    ds = build_eval_dataset()
    metrics, y_true, y_pred, _ = evaluate_dataset(model, ds, CATEGORIES)

    print("\nExternal dataset metrics")
    print(f"  accuracy         : {metrics['accuracy']:.4f}")
    print(f"  macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  confusion matrix :")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")

    #print("\nFirst few predictions:")
    #for yt, yp in list(zip(y_true, y_pred))[:10]:
    #    print(f"true={CATEGORIES[int(yt)]:<8}  pred={CATEGORIES[int(yp)]}")


if __name__ == "__main__":
    main()