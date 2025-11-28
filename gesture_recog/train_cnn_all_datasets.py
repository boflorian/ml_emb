import pathlib
import shutil

import numpy as np

import main as pipeline
from model_definitions.cnn import build_cnn
from util import data_loader

# Explicit hyperparameter settings reused across the combined run
GLOBAL_HPARAMS = dict(
    win=pipeline.WIN_CNN,
    hop=pipeline.HOP_CNN,
    lp_window=pipeline.LP_WINDOW_CNN,
    batch_size=pipeline.BATCH_SIZE_CNN,
    #epochs=pipeline.MAX_EPOCHS,
    epochs = 2000,
    num_classes=pipeline.CNN_CONFIG["num_classes"],
    lr=pipeline.LR_CNN,
    l2=pipeline.L2_CNN,
    dropout=pipeline.DROPOUT_CNN,
    #patience=pipeline.PATIENCE,
    patience = 200,
    augment=True,
)

# Roots for all three datasets (Magic Wand, Pico processed1, and GAN augmentation)
DATA_ROOTS = [
    pathlib.Path("dataset_magic_wand"),
    pathlib.Path("dataset_pico_gestures/processed1"),
    pathlib.Path("dataset_magic_wand_gan"),
]

# Keep the original iterator so we can wrap it
ORIGINAL_ITER_SAMPLES = data_loader.iter_samples
NEGATIVE_LABEL = data_loader.CAT_TO_ID["negative"]
SYNTH_NEG_PER_SUBJECT = 50
SYNTH_NEG_LENGTH = pipeline.WIN_CNN  # one window worth of minimal movement


def combined_iter_samples(subjects, root_dir=None):
    """Yield samples from every configured dataset root for the requested subjects."""
    # Positive classes from real data (skip stored negatives)
    for root in DATA_ROOTS:
        for x, y in ORIGINAL_ITER_SAMPLES(subjects, root_dir=root):
            if y == NEGATIVE_LABEL:
                continue
            yield x, y

    # Synthetic negatives: minimal/zero movement to represent inactivity
    for subj in subjects or ():
        for _ in range(SYNTH_NEG_PER_SUBJECT):
            x = np.zeros((SYNTH_NEG_LENGTH, 3), dtype=np.float32)
            yield x, NEGATIVE_LABEL


def collect_all_subjects():
    """Union of subject IDs available across all roots."""
    subjects = set()
    for root in DATA_ROOTS:
        subjects.update(data_loader.list_available_subjects(root))
    return sorted(subjects)


def main():
    # Wire in the combined iterator so the existing pipeline loaders pull from all roots
    data_loader.iter_samples = combined_iter_samples

    # Expand the SUBJECTS list used by the main CV routine
    pipeline.SUBJECTS = collect_all_subjects()
    if not pipeline.SUBJECTS:
        raise RuntimeError("No subjects found across the combined datasets.")

    print(f"Running CNN cross-validation across {len(pipeline.SUBJECTS)} subjects")
    print(f"Dataset roots: {', '.join(r.as_posix() for r in DATA_ROOTS)}")
    print(f"Global hyperparameters: {GLOBAL_HPARAMS}")
    print(f"Synthetic negatives: {SYNTH_NEG_PER_SUBJECT} per subject, length {SYNTH_NEG_LENGTH}")

    # Reuse the main pipeline's CV training routine and CNN config
    cnn_config = pipeline.CNN_CONFIG.copy()
    cnn_config.update(GLOBAL_HPARAMS)
    # Ensure callbacks pick up the patience we log above
    pipeline.PATIENCE = GLOBAL_HPARAMS["patience"]
    model, run_root = pipeline.run_cv_for_model("cnn", build_cnn, cnn_config)

    # Persist the run artifacts to a dedicated folder for the combined datasets
    dest_root = pathlib.Path("combined_models") / "cnn" / run_root.name
    if dest_root.exists():
        raise FileExistsError(f"Destination already exists: {dest_root}")
    shutil.copytree(run_root, dest_root)
    print(f"Saved combined-dataset CNN run to {dest_root}")


if __name__ == "__main__":
    main()
