"""
Convert a model to TensorFlow Lite Format for deployment
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import argparse
import re
from pathlib import Path
import tempfile
import shutil
import sys
import tempfile
import shutil
import sys


GOOD_ROOT = Path("dataset_good")
GOOD_CATEGORIES = ["negative", "ring", "slope", "wave"]
WIN_CNN = 286


def representative_dataset(dataset_root=GOOD_ROOT, win=WIN_CNN, max_samples=200):
    """
    Generator function for representative dataset used in quantization.
    Uses preprocessed windows from dataset_good.
    """
    import numpy as np
    from util.data_loader import parse_gesture_file

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Representative dataset root not found: {root}")

    def to_fixed_len(x):
        if x.shape[0] >= win:
            return x[:win]
        pad = np.zeros((win - x.shape[0], x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    count = 0
    for cls in GOOD_CATEGORIES:
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        for txt in sorted(cls_dir.glob("*.txt")):
            for rec in parse_gesture_file(txt):
                x = np.asarray(rec["data"], dtype=np.float32)
                if x.size == 0:
                    continue
                x = to_fixed_len(x)
                x = np.expand_dims(x, axis=0)  # [1, win, 3]
                yield [x]
                count += 1
                if count >= max_samples:
                    return


def try_convert_from_keras(model, quantize=False): 
    print('Trying to convert from keras...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("[INFO]  - enabling INT8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    return converter.convert()


def try_convert_from_savedmodel(saved_model_dir, optimize=False, use_select_tf_ops=False, quantize=False): 
    print('Trying to convert from saved model...')
    print(f"[INFO]  - creating TFLiteConverter.from_saved_model({saved_model_dir})")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if optimize or quantize:
        print("[INFO]  - enabling optimizations")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            print("[INFO]  - enabling INT8 quantization")
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    if use_select_tf_ops:
        print("[INFO]  - enabling SELECT_TF_OPS")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    return converter.convert()


def convert_to_tflite(model_path: Path, tflite_output_path: Path,
                      optimize: bool = False, use_select_tf_ops: bool = False, quantize: bool = False) -> None:
    """
    Load a .keras model or SavedModel directory and convert it to TFLite.
    """
    if model_path.is_dir():
        # SavedModel directory
        print(f"Converting SavedModel from directory: {model_path}")
        tflite_model = try_convert_from_savedmodel(
            str(model_path), optimize=optimize, use_select_tf_ops=use_select_tf_ops, quantize=quantize
        )
    else:
        # .keras file
        print(f"Loading Keras model from {model_path}")
        model = keras.models.load_model(model_path)

        try:
            print("[INFO] Converting (direct from Keras)...")
            tflite_model = try_convert_from_keras(model, quantize=quantize)
        except Exception as e:
            print("[WARN] Direct Keras → TFLite conversion failed:")
            print(f"       {type(e).__name__}: {e}")
            print("[INFO] Trying SavedModel → TFLite fallback...")
            # Fallback: go via SavedModel
            tmpdir = Path(tempfile.mkdtemp(prefix='tflite_export_'))
            try:
                saved_model_dir = tmpdir / "saved_model"
                print(f"[INFO]  - exporting SavedModel to {saved_model_dir}")
                tf.saved_model.save(model, saved_model_dir.as_posix())
                tflite_model = try_convert_from_savedmodel(
                    saved_model_dir.as_posix(), optimize=optimize, use_select_tf_ops=use_select_tf_ops, quantize=quantize
                )
            finally:
                # clean up temp dir
                shutil.rmtree(tmpdir, ignore_errors=True)

    print('Writing output model')
    tflite_output_path.write_bytes(tflite_model)
    print("[Borat Voice:] GREAT SUCCESS")


def main():
    parser = argparse.ArgumentParser(description="Convert a Keras model or SavedModel to TensorFlow Lite format.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the Keras model file (.keras) or SavedModel directory")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable TFLite optimizations")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable INT8 quantization (requires representative dataset)")
    parser.add_argument("--select-tf-ops", action="store_true",
                        help="Use SELECT_TF_OPS for broader op support")

    args = parser.parse_args()

    input_dir = Path('chosen_models')
    output_dir = Path('deployment_models')
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.model)
    
    # First check if it's a SavedModel directory in chosen_models
    model_path = input_dir / raw.name
    if model_path.is_dir():
        # SavedModel directory
        print(f"[INFO] Detected SavedModel directory: {model_path}")
        out_name = raw.name + ".tflite"
    else:
        # Check if it's a .keras file
        if raw.suffix == "":
            keras_path = input_dir / (raw.name + ".keras")
        else:
            keras_path = input_dir / raw.name
            
        if keras_path.exists():
            model_path = keras_path
            out_name = re.sub(r"\.keras$", ".tflite", keras_path.name)
        else:
            print(f"[ERROR] Neither SavedModel directory nor .keras file found: {model_path} or {keras_path}", file=sys.stderr)
            sys.exit(1)

    tflite_output_path = output_dir / out_name

    convert_to_tflite(model_path, tflite_output_path, 
                     optimize=args.optimize, use_select_tf_ops=args.select_tf_ops, quantize=args.quantize)


if __name__ == "__main__":
    main()
