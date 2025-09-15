# common/data.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    import tensorflow as tf  # optional; only needed for tf.data utilities
except Exception:  # pragma: no cover
    tf = None  # type: ignore


# ---------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------
def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert raw uint8/float arrays to float32 in [0,1] and reshape to HWC.
    """
    H, W, C = img_shape
    x = x.astype("float32", copy=False)
    if x.size == 0:
        return x.reshape((-1, H, W, C))
    if x.max() > 1.5:  # typical for 0..255 arrays
        x = x / 255.0
    x = x.reshape((-1, H, W, C))
    return np.clip(x, 0.0, 1.0)


def binarize_01(x01: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarize [0,1] images: x -> {0,1} using a threshold.
    """
    return (x01 >= float(threshold)).astype("float32")


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Ensure labels are one-hot encoded.
    Accepts integer (N,) or already-one-hot (N,K).
    """
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32", copy=False)
    # fallback without TF dependency
    out = np.zeros((y.shape[0], num_classes), dtype="float32")
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


def flatten_images(x_hwc_01: np.ndarray) -> np.ndarray:
    """
    Flatten HWC images to (N, D) where D=H*W*C.
    """
    n = x_hwc_01.shape[0]
    d = int(np.prod(x_hwc_01.shape[1:]))
    return x_hwc_01.reshape((n, d))


# ---------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------
def load_dataset_npy(
    data_dir: Path | str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    *,
    val_fraction: float = 0.5,
    binarize: bool = False,
    binarize_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load USTC-TFC2016 (or similar) from .npy files:
      - train_data.npy, train_labels.npy
      - test_data.npy,  test_labels.npy

    Returns:
      x_train, y_train, x_val, y_val, x_test, y_test
    where images are float32 in [0,1] with shape (N,H,W,C) and labels are one-hot.
    Optionally binarizes images (useful for RBMs).
    """
    data_dir = Path(data_dir)
    H, W, C = img_shape

    # Load arrays
    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test = np.load(data_dir / "test_data.npy")
    y_test = np.load(data_dir / "test_labels.npy")

    # Normalize & reshape
    x_train01 = to_01_hwc(x_train, (H, W, C))
    x_test01 = to_01_hwc(x_test, (H, W, C))

    # Optional binarization (RBM-friendly)
    if binarize:
        x_train01 = binarize_01(x_train01, threshold=binarize_threshold)
        x_test01 = binarize_01(x_test01, threshold=binarize_threshold)

    # One-hot labels
    y_train1h = one_hot(y_train, num_classes)
    y_test1h = one_hot(y_test, num_classes)

    # Split test -> (val, test)
    n_val = int(len(x_test01) * float(val_fraction))
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


# ---------------------------------------------------------------------
# tf.data helpers (optional)
# ---------------------------------------------------------------------
def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    prefetch: bool = True,
) -> "tf.data.Dataset":
    """
    Build a simple tf.data pipeline from numpy arrays.
    Requires TensorFlow to be installed/importable.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available; cannot create tf.data.Dataset.")
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 8192), reshuffle_each_iteration=True)
    ds = ds.batch(int(batch_size), drop_remainder=False)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_tf_splits(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    *,
    batch_size: int = 64,
) -> Tuple["tf.data.Dataset", "tf.data.Dataset", "tf.data.Dataset"]:
    """
    Convenience wrapper to create train/val/test tf.data datasets.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available; cannot create tf.data.Dataset.")
    train_ds = make_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds   = make_tf_dataset(x_val,   y_val,   batch_size=batch_size, shuffle=False)
    test_ds  = make_tf_dataset(x_test,  y_test,  batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, test_ds
