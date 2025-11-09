import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from data_loader import build_train_test_datasets
from nn_def import build_imu_model

SUBJECTS = list(range(8))  # persons 0..7

def plot_training_curves(history, save_path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="Train Acc")
    if "val_accuracy" in hist: plt.plot(epochs, hist["val_accuracy"], label="Val Acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="Train Loss")
    if "val_loss" in hist: plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_crossval_results(run_root):
    """Aggregate all fold histories under run_root and plot mean ± std accuracy/loss."""
    fold_dirs = sorted((run_root).glob("fold_*"))
    histories = []
    for fd in fold_dirs:
        hist_file = fd / "history.csv"
        if hist_file.exists():
            df = pd.read_csv(hist_file)
            df["fold"] = fd.name
            histories.append(df)
    if not histories:
        print("No fold histories found for summary plot.")
        return

    df_all = pd.concat(histories, ignore_index=True)
    epochs = np.arange(1, df_all["epoch"].max() + 1)

    # group by epoch → mean ± std
    g = df_all.groupby("epoch").agg({
        "accuracy": ["mean", "std"],
        "val_accuracy": ["mean", "std"],
        "loss": ["mean", "std"],
        "val_loss": ["mean", "std"],
    })
    g.columns = ["_".join(c) for c in g.columns]

    plt.figure(figsize=(10, 4))

    # --- accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, g["accuracy_mean"], label="Train acc (mean)")
    plt.fill_between(epochs,
                     g["accuracy_mean"] - g["accuracy_std"],
                     g["accuracy_mean"] + g["accuracy_std"],
                     alpha=0.2)
    plt.plot(epochs, g["val_accuracy_mean"], label="Val acc (mean)")
    plt.fill_between(epochs,
                     g["val_accuracy_mean"] - g["val_accuracy_std"],
                     g["val_accuracy_mean"] + g["val_accuracy_std"],
                     alpha=0.2)
    plt.title("Cross-validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # --- loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, g["loss_mean"], label="Train loss (mean)")
    plt.fill_between(epochs,
                     g["loss_mean"] - g["loss_std"],
                     g["loss_mean"] + g["loss_std"],
                     alpha=0.2)
    plt.plot(epochs, g["val_loss_mean"], label="Val loss (mean)")
    plt.fill_between(epochs,
                     g["val_loss_mean"] - g["val_loss_std"],
                     g["val_loss_mean"] + g["val_loss_std"],
                     alpha=0.2)
    plt.title("Cross-validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    save_path = run_root / "crossval_summary.png"
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Saved cross-validation summary plot: {save_path}")


def run_one_fold(run_root, val_subject, cfg):
    train_subjects = tuple(s for s in SUBJECTS if s != val_subject)
    test_subjects  = (val_subject,)  # validate on this person

    fold_dir = run_root / f"fold_{val_subject}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "tb").mkdir(parents=True, exist_ok=True)

    # record fold-specific config
    cfg_fold = dict(cfg)
    cfg_fold["split"] = dict(train_subjects=list(train_subjects), val_subjects=[val_subject])
    (fold_dir / "config.json").write_text(json.dumps(cfg_fold, indent=2))

    # data (train uses augment=True internally per our earlier data_loader; val False)
    train_ds, val_ds = build_train_test_datasets(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"], hop=cfg["hop"], drop_short=False
    )

    # model (compile with desired lr, l2/dropout handled inside builder)
    model = build_imu_model(win=cfg["win"],
                            num_classes=cfg["num_classes"],
                            lr=float(cfg["lr"]),
                            l2=cfg.get("l2"),
                            dropout=cfg.get("dropout", 0.0))

    # callbacks
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(fold_dir / "checkpoints" / "best_valacc.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    csv_cb = tf.keras.callbacks.CSVLogger(str(fold_dir / "history.csv"))
    tb_cb  = tf.keras.callbacks.TensorBoard(log_dir=str(fold_dir / "tb"), write_graph=False)
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True
    )
    rlr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=[ckpt_cb, csv_cb, tb_cb, early_cb, rlr_cb],
        verbose=1
    )

    # save artifacts
    model.save(fold_dir / "final.keras")
    plot_training_curves(history, fold_dir / "training_curves.png")

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

def main():
    print("Initializing LOSO cross-validation...\n")

    # common config
    config = dict(
        win=128, hop=64, lp_window=7, batch_size=64, epochs=30,
        num_classes=4, lr=1e-3, l2=None, dropout=0.3
    )

    # run root
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_cv"
    run_root = Path("models") / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    # run all folds
    fold_summaries = []
    for val_subject in SUBJECTS:
        print(f"\n=== Fold (val subject = {val_subject}) ===")
        sum_fold = run_one_fold(run_root, val_subject, config)
        fold_summaries.append(sum_fold)
        print(f"Fold {val_subject}: best val acc = {sum_fold['best_val_accuracy']:.4f}")

    # aggregate
    vals = [s["best_val_accuracy"] for s in fold_summaries]
    agg = dict(
        folds=[dict(val_subject=s["val_subject"], best_val_accuracy=s["best_val_accuracy"])
               for s in fold_summaries],
        mean_val_accuracy=float(np.mean(vals) if vals else 0.0),
        std_val_accuracy=float(np.std(vals) if vals else 0.0)
    )
    (run_root / "summary.json").write_text(json.dumps(agg, indent=2))

    # also write CSV for quick comparisons
    import csv
    with open(run_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_subject", "best_val_accuracy"])
        for s in fold_summaries:
            w.writerow([s["val_subject"], f"{s['best_val_accuracy']:.6f}"])
        w.writerow(["mean", f"{agg['mean_val_accuracy']:.6f}"])
        w.writerow(["std",  f"{agg['std_val_accuracy']:.6f}"])

    print("\n=== Cross-validation complete ===")
    print(f"Mean val acc: {agg['mean_val_accuracy']:.4f}  (std: {agg['std_val_accuracy']:.4f})")
    print(f"Artifacts saved to: {run_root.resolve()}")
    plot_crossval_results(run_root)

if __name__ == '__main__':
    main()