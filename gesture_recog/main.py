import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import csv 
from pathlib import Path
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from data_loader import build_train_test_datasets, _make_ds, lowpass_filter, normalize_clip 
from nn_def import build_imu_model

SUBJECTS = list(range(8))  
CONFIG = dict(
        win=256, hop=64, lp_window=7, batch_size=64, epochs=50,
        num_classes=4, lr=1e-3, l2=1e-2, dropout=0.5
    )



def find_best_model(models_root="models"):
    """Scan all previous runs and find the one with highest mean val accuracy."""
    best = None
    best_path = None

    for sub in Path(models_root).glob("*_cv"):
        summ = sub / "summary.json"
        if summ.exists():
            try:
                data = json.loads(summ.read_text())
                acc = data.get("mean_val_accuracy")
                loss = data.get("mean_val_loss", None)
                if acc is not None:
                    if (best is None) or (acc > best["acc"]):
                        best = {"acc": acc, "loss": loss, "path": sub}
                        best_path = sub
            except Exception:
                continue
    return best


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
    dfs = []
    for fd in fold_dirs:
        f = fd / "history.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["fold"] = fd.name
            dfs.append(df)

    if not dfs:
        print("No fold histories found for summary plot.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Some CSVLogger versions use 0-based epoch; keep as-is but sort
    df_all = df_all.sort_values("epoch")

    # Group by epoch; mean/std will naturally ignore missing folds at later epochs
    g = df_all.groupby("epoch", as_index=True).agg({
        "accuracy":      ["mean", "std"],
        "val_accuracy":  ["mean", "std"],
        "loss":          ["mean", "std"],
        "val_loss":      ["mean", "std"],
    })
    g.columns = ["_".join(c) for c in g.columns]
    g = g.sort_index()

    # Use the grouped index as x-axis to avoid shape mismatches
    epochs = g.index.to_numpy()  # this is 0..N based, can display as is

    plt.figure(figsize=(10, 4))

    # --- Accuracy ---
    plt.subplot(1, 2, 1)
    if "accuracy_mean" in g:
        y = g["accuracy_mean"].to_numpy()
        ysd = g["accuracy_std"].to_numpy()
        plt.plot(epochs, y, label="Train acc (mean)")
        plt.fill_between(epochs, y - ysd, y + ysd, alpha=0.2)
    if "val_accuracy_mean" in g:
        yv = g["val_accuracy_mean"].to_numpy()
        yvsd = g["val_accuracy_std"].to_numpy()
        plt.plot(epochs, yv, label="Val acc (mean)")
        plt.fill_between(epochs, yv - yvsd, yv + yvsd, alpha=0.2)
    plt.title("Cross-validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # --- Loss ---
    plt.subplot(1, 2, 2)
    if "loss_mean" in g:
        y = g["loss_mean"].to_numpy()
        ysd = g["loss_std"].to_numpy()
        plt.plot(epochs, y, label="Train loss (mean)")
        plt.fill_between(epochs, y - ysd, y + ysd, alpha=0.2)
    if "val_loss_mean" in g:
        yv = g["val_loss_mean"].to_numpy()
        yvsd = g["val_loss_std"].to_numpy()
        plt.plot(epochs, yv, label="Val loss (mean)")
        plt.fill_between(epochs, yv - yvsd, yv + yvsd, alpha=0.2)
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


    # augmentation strengths 
    aug_cfg = dict(
	    rot_deg=20.0,
	    scale_low=0.9, scale_high=1.1,
	    jitter_std=0.05,
	    shift_max=8,
	    time_mask_prob=0.35, time_mask_max_ratio=0.12,
	    mag_warp_strength=0.15,
	    time_warp_ratio=0.20,
	)


    # data (train uses augment=True internally per our earlier data_loader; val False)
    train_ds, val_ds = build_train_test_datasets(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        batch_size=cfg["batch_size"],
        lp_window=cfg["lp_window"],
        win=cfg["win"], hop=cfg["hop"], 
        aug_cfg=aug_cfg
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
    

    # Early Stopping 
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max", 
        patience=20, 
        restore_best_weights=True
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


# =========== Consolidate CV into one model =============
def load_fold_checkpoints(run_root):
    ckpts = []
    for fd in sorted((run_root).glob("fold_*")):
        c = fd / "checkpoints" / "best_valacc.keras"
        if c.exists():
            ckpts.append(c)
    return ckpts

def average_weights_from_checkpoints(ckpt_paths, build_model_fn, build_kwargs):
    """Element-wise average of weights across several checkpoints."""
    if not ckpt_paths:
        raise ValueError("No checkpoints to average.")
    # Build a fresh model to know the weight structure
    base = build_model_fn(**build_kwargs)
    base_weights = None
    count = 0
    for p in ckpt_paths:
        m = build_model_fn(**build_kwargs)
        m.load_weights(str(p))
        ws = m.get_weights()
        if base_weights is None:
            base_weights = [w.copy() for w in ws]
        else:
            for i in range(len(ws)):
                base_weights[i] += ws[i]
        count += 1
        tf.keras.backend.clear_session()
    # mean
    for i in range(len(base_weights)):
        base_weights[i] /= float(count)
    # put into a fresh instance
    final_model = build_model_fn(**build_kwargs)
    final_model.set_weights(base_weights)
    return final_model


SEED = 7

def build_all_data_trainval(
    subjects, batch_size, lp_window, win, hop,
    val_mod=10, aug_cfg=None, drop_short=False
):
    """
    Build a combined dataset from all subjects, then deterministically
    split windows into train/val using enumeration (≈1/val_mod as val).
    Train uses augmentation; val does not.
    """
    # First, build one big (unbatched) dataset of all subjects without augment
    base = _make_ds(
        subjects,
        batch_size=1,          # use 1 so we can enumerate individual windows
        lp_window=lp_window,
        win=win, hop=hop,
        drop_short=drop_short,
        augment=False,
        aug_cfg=None
    ).unbatch()                # ensure we’re at window level

    enumerated = base.enumerate()  # (index, (x, y))

    val_raw = enumerated.filter(lambda i, xy: tf.equal(tf.math.mod(i, val_mod), 0)).map(lambda i, xy: xy)
    trn_raw = enumerated.filter(lambda i, xy: tf.not_equal(tf.math.mod(i, val_mod), 0)).map(lambda i, xy: xy)

    # Apply preprocess functions consistent with your pipeline
    acfg = aug_cfg or {}

    def preprocess_train(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        return x, y

    def preprocess_eval(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        return x, y

    trn = trn_raw.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    val = val_raw.map(preprocess_eval,  num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    trn = trn.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return trn, val



def main():
    print("Initializing LOSO cross-validation...\n")

    config = CONFIG

    # run root
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_cv"
    run_root = Path("models") / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(config, indent=2))

    # ---- 1) Run all folds ----------------------------------------------------
    fold_summaries = []
    for val_subject in SUBJECTS:
        print(f"\n=== Fold (val subject = {val_subject}) ===")
        sum_fold = run_one_fold(run_root, val_subject, config)
        fold_summaries.append(sum_fold)
        print(f"Fold {val_subject}: best val acc = {sum_fold['best_val_accuracy']:.4f}")

    # CV aggregate (keep for reference)
    vals = [s["best_val_accuracy"] for s in fold_summaries]
    cv_mean = float(np.mean(vals) if vals else 0.0)
    cv_std  = float(np.std(vals) if vals else 0.0)

    # ---- 2) Consolidate (average) -------------------------------------------
    print("\nConsolidating into one model...")
    ckpts = load_fold_checkpoints(run_root)
    if not ckpts:
        print("no checkpoints")
        # Even if no consolidation, still record CV stats
        agg = dict(
            folds=[dict(val_subject=s["val_subject"], best_val_accuracy=s["best_val_accuracy"])
                   for s in fold_summaries],
            cv_mean_val_accuracy=cv_mean,
            cv_std_val_accuracy=cv_std,
            consolidated_model=None,
            consolidated_mean_accuracy=cv_mean,
            consolidated_std_accuracy=cv_std,
        )
        (run_root / "summary.json").write_text(json.dumps(agg, indent=2))
        with open(run_root / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["val_subject", "best_val_accuracy"])
            for s in fold_summaries:
                w.writerow([s["val_subject"], f"{s['best_val_accuracy']:.6f}"])
            w.writerow(["cv_mean", f"{cv_mean:.6f}"])
            w.writerow(["cv_std",  f"{cv_std:.6f}"])
            w.writerow(["consolidated_model", "N/A"])
            w.writerow(["consolidated_mean_accuracy", f"{cv_mean:.6f}"])
            w.writerow(["consolidated_std_accuracy",  f"{cv_std:.6f}"])

        print("\n=== Cross-validation complete ===")
        print(f"CV mean val acc: {cv_mean:.4f}  (std: {cv_std:.4f})")
        print(f"Artifacts saved to: {run_root.resolve()}")
        plot_crossval_results(run_root)
        return

    build_kwargs = dict(
        win=config["win"],
        num_classes=config["num_classes"],
        lr=float(config["lr"]),
        l2=config.get("l2"),
        dropout=config.get("dropout", 0.0),
    )

    print("\nAveraging fold checkpoints...")
    avg_model = average_weights_from_checkpoints(
        ckpt_paths=ckpts,
        build_model_fn=build_imu_model,
        build_kwargs=build_kwargs
    )
    (run_root / "final").mkdir(exist_ok=True, parents=True)
    avg_path = run_root / "final" / "averaged_folds.keras"
    avg_model.save(avg_path)
    print(f"Saved averaged model: {avg_path.resolve()}")

    # ---- 3) Refit on ALL subjects using averaged init -----------------------
    refit_aug_cfg = dict(
        rot_deg=10.0,
        scale_low=0.95, scale_high=1.05,
        jitter_std=0.02,
        shift_max=4,
        time_mask_prob=0.20, time_mask_max_ratio=0.08,
        mag_warp_strength=0.10,
        time_warp_ratio=0.10,
    )

    refit_train_ds, refit_val_ds = build_all_data_trainval(
        subjects=tuple(SUBJECTS),
        batch_size=CONFIG["batch_size"],
        lp_window=CONFIG["lp_window"],
        win=CONFIG["win"], hop=CONFIG["hop"],
        val_mod=10,
        aug_cfg=refit_aug_cfg,
        drop_short=False
    )

    refit_model = build_imu_model(
        win=CONFIG["win"],
        num_classes=CONFIG["num_classes"],
        lr=min(3e-4, float(CONFIG["lr"]) * 0.3),
        l2=CONFIG.get("l2"),
        dropout=CONFIG.get("dropout", 0.0),
    )
    refit_model.set_weights(avg_model.get_weights())

    refit_ckpt = run_root / "final" / "refit_all.keras"
    refit_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(refit_ckpt), monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(run_root / "final" / "refit_history.csv"))
    ]

    print("\nRefitting on ALL subjects...")
    refit_model.fit(
        refit_train_ds,
        validation_data=refit_val_ds,
        epochs=max(15, min(30, CONFIG["epochs"] // 2)),
        callbacks=refit_callbacks,
        verbose=1
    )

    # ---- 4) Evaluate consolidated model AFTER it exists ---------------------
    final_refit = run_root / "final" / "refit_all.keras"
    final_avg   = run_root / "final" / "averaged_folds.keras"

    cons_tag = None
    cons_mean = None
    cons_std = None

    if final_refit.exists() or final_avg.exists():
        model_path = final_refit if final_refit.exists() else final_avg
        cons_tag = model_path.name
        model = tf.keras.models.load_model(model_path)

        accs = []
        for s in SUBJECTS:
            _, test_ds = build_train_test_datasets(
                train_subjects=tuple(x for x in SUBJECTS if x != s),
                test_subjects=(s,),
                batch_size=CONFIG["batch_size"],
                lp_window=CONFIG["lp_window"],
                win=CONFIG["win"], hop=CONFIG["hop"],
                aug_cfg=None  # no aug for eval
            )
            res = model.evaluate(test_ds, verbose=0)
            acc = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else float(res)
            accs.append(float(acc))
        cons_mean = float(np.mean(accs))
        cons_std  = float(np.std(accs))
        print(f"\nConsolidated model ({cons_tag}) LOSO mean acc: {cons_mean:.4f} (std {cons_std:.4f})")
    else:
        print("\n[Warn] No consolidated model found after refit/average.")

    # ---- 5) Persist summary (JSON + CSV) ------------------------------------
    agg = dict(
        folds=[dict(val_subject=s["val_subject"], best_val_accuracy=s["best_val_accuracy"])
               for s in fold_summaries],
        cv_mean_val_accuracy=cv_mean,
        cv_std_val_accuracy=cv_std,
        consolidated_model=cons_tag,  # 'refit_all.keras' or 'averaged_folds.keras'
        consolidated_mean_accuracy=(cons_mean if cons_mean is not None else cv_mean),
        consolidated_std_accuracy=(cons_std if cons_std is not None else cv_std),
    )
    (run_root / "summary.json").write_text(json.dumps(agg, indent=2))

    with open(run_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_subject", "best_val_accuracy"])
        for s in fold_summaries:
            w.writerow([s["val_subject"], f"{s['best_val_accuracy']:.6f}"])
        w.writerow(["cv_mean", f"{cv_mean:.6f}"])
        w.writerow(["cv_std",  f"{cv_std:.6f}"])
        w.writerow(["consolidated_model", cons_tag or "N/A"])
        w.writerow(["consolidated_mean_accuracy", f"{(cons_mean if cons_mean is not None else cv_mean):.6f}"])
        w.writerow(["consolidated_std_accuracy",  f"{(cons_std  if cons_std  is not None else cv_std):.6f}"])

    # ---- 6) Human-readable prints + comparison ------------------------------
    print("\n=== Cross-validation complete ===")
    print(f"CV mean val acc: {cv_mean:.4f}  (std: {cv_std:.4f})")
    if cons_mean is not None:
        print(f"Consolidated model mean acc: {cons_mean:.4f}  (std: {cons_std:.4f})")
    print(f"Artifacts saved to: {run_root.resolve()}")

    # Compare to historical best (proxy: previous CV mean)
    best = find_best_model('models')
    if best:
        current_acc = (cons_mean if cons_mean is not None else cv_mean)
        print("\n=== Model comparison ===")
        print(f"Current run:  {current_acc:.4f}  ({run_root.name})")
        print(f"Best overall (CV mean proxy): {best['acc']:.4f}  ({best['path'].name})")
        if best['loss'] is not None:
            print(f"Best model mean val loss: {best['loss']:.4f}")
        if current_acc < best["acc"]:
            print("→ Current model is not the best; keep best checkpoint from:")
            print(f"   {best['path'].resolve()}")
        else:
            print("→ This model achieved the best accuracy so far (vs CV-best proxy)!")
    else:
        print("No previous summary.json files found — this is the first recorded run.")

    plot_crossval_results(run_root)

if __name__ == '__main__':
    main()