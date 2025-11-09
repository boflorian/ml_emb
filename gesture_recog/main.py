import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from data_loader import *
from nn_def import *

def main():
    print('Initializing...\n')

    # ----- config -----
    config = dict(
	    win=128, hop=64, lp_window=7, batch_size=64, epochs=20,
	    split=dict(train_subjects=[0,1,2,3,4,5,6], test_subjects=[7]),
	    model=dict(num_classes=4, lr=1e-3, l2=None, dropout=0.3)  # <- not None
	)

    # ----- make run folder under models/ -----
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("models") / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    # save the config
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ----- data -----
    train_ds, test_ds = build_train_test_datasets(
        train_subjects=tuple(config["split"]["train_subjects"]),
        test_subjects=tuple(config["split"]["test_subjects"]),
        batch_size=config["batch_size"],
        lp_window=config["lp_window"],
        win=config["win"],
        hop=config["hop"]
    )

    # ----- model -----
    print("Using dropout:", config["model"]["dropout"], "l2:", config["model"]["l2"])
    model = build_imu_model(win=config["win"],
                        num_classes=config["model"]["num_classes"],
                        lr=config["model"]["lr"],
                        l2=config["model"]["l2"],
                        dropout=config["model"]["dropout"])
    

    # ----- callbacks -----
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "checkpoints" / "best_valacc.keras"),
        monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=False, verbose=1
    )
    csv_cb = tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"))
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb"), write_graph=False)
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True
    )
    rlr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    # ----- train -----
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config["epochs"],
        callbacks=[ckpt_cb, csv_cb, tb_cb, early_cb, rlr_cb],
        verbose=1
    )

    # ----- save final model & summary -----
    model.save(run_dir / "final.keras")

    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    last = {k: float(v[-1]) for k, v in history.history.items()}
    summary = dict(best_val_accuracy=best_val_acc, last_epoch_metrics=last)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Plot training curves ----
    plot_training_curves(history, run_dir / "training_curves.png")

    print(f"\nArtifacts saved to: {run_dir.resolve()}")
    print(f"Best val acc: {best_val_acc:.4f}")

    # optional final eval (on the same test_ds used as val)
    model.evaluate(test_ds, verbose=2)


def plot_training_curves(history, save_path):
    """Simple loss/accuracy plot for train and validation."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # ---- Accuracy ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="Train Acc")
    if "val_accuracy" in hist:
        plt.plot(epochs, hist["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # ---- Loss ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="Train Loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved training curve: {save_path}")

if __name__ == '__main__':
    main()