# restrictedboltzmann/train.py

"""
Training utilities for Bernoulli–Bernoulli Restricted Boltzmann Machines (RBMs).

What this module provides
-------------------------
- `build_visible_dataset(...)`: tf.data pipeline for RBM training (visible-only).
- `cd_k_update(...)`: Single Contrastive Divergence (CD-k) parameter update step.
- `train_rbm(...)`: Full training loop with early stopping + checkpoints.

Design choices
--------------
- We use the standard *analytic* CD-k update (positive/negative phase) instead
  of backprop-through-sampling, which is non-differentiable. This is fast and
  stable:
      dW  = <v0^T h0_prob> - <vk^T hk_prob>
      dvb = <v0 - vk>
      dhb = <h0_prob - hk_prob>
- Visible/hidden units are Bernoulli; inputs are expected in [0,1] (we can
  binarize or treat them as probabilities then sample once).
- Checkpoints are saved as Keras-3-style weights: BEST/LAST and periodic epochs.

Conventions
-----------
- Visible vectors are shaped (B, V) where V = H * W * C.
- Images are channels-last (H, W, C) and values in [0,1] (binarized internally).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Helpers
# =============================================================================
def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(-1)[0])


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _binarize01(x: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Threshold to {0,1} with straight casts. Input assumed in [0,1]."""
    return tf.cast(x > threshold, tf.float32)


def _flatten_if_needed(x: tf.Tensor, visible_dim: int) -> tf.Tensor:
    """Reshape (N,H,W,C) -> (N,V) if not already flat."""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if x.shape.rank == 2 and x.shape[-1] == visible_dim:
        return x
    return tf.reshape(x, (-1, visible_dim))


# =============================================================================
# Public tf.data builder
# =============================================================================
def build_visible_dataset(
    x: np.ndarray,
    *,
    img_shape: Tuple[int, int, int],
    batch_size: int,
    shuffle: bool = True,
    binarize: bool = True,
    threshold: float = 0.5,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset that yields only visible vectors (B, V) for RBM.

    Args
    ----
    x : np.ndarray
        Images shaped (N,H,W,C) or flat (N,V); values in [0,1].
    img_shape : (H,W,C)
        Used to compute V when flattening.
    batch_size : int
    shuffle : bool
    binarize : bool
        If True, threshold to {0,1}. If False, keep as probabilities.
    threshold : float
        Threshold used when `binarize=True`.

    Returns
    -------
    tf.data.Dataset yielding tensors (B,V) float32 in {0,1} (if binarize) or [0,1].
    """
    H, W, C = img_shape
    V = H * W * C

    def _prep(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype("float32")
        if arr.max() > 1.5:  # common 0..255 case
            arr = arr / 255.0
        arr = arr.reshape((-1, V))
        if binarize:
            arr = (arr > threshold).astype("float32")
        return arr

    x_flat = _prep(x)
    ds = tf.data.Dataset.from_tensor_slices(x_flat)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x_flat), 8192), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# CD-k update (analytic, vectorized)
# =============================================================================
@tf.function(reduce_retracing=True)
def cd_k_update(
    W: tf.Variable,
    v_bias: tf.Variable,
    h_bias: tf.Variable,
    v0: tf.Tensor,
    *,
    k: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Perform one analytic CD-k update on the parameters.

    Args
    ----
    W, v_bias, h_bias : trainable variables of shape (V,H), (V,), (H,)
    v0 : (B,V) visible batch (in {0,1} ideally), float32
    k : int, number of Gibbs steps (>=1)
    lr : float, learning rate for parameter update
    weight_decay : L2 penalty on weights (applied to update)

    Returns
    -------
    recon_prob : (B,V) final visible probabilities after k steps
    recon_mse  : scalar tensor, MSE between v0 and recon_prob
    """
    # Positive phase (use probabilities for stability)
    h0_prob = tf.nn.sigmoid(tf.matmul(v0, W) + h_bias)  # (B,H)

    # Start chain from h0_prob (sampled) -> v1 -> h1 -> ... k steps
    # We sample both v and h in the negative phase (classic CD).
    h = tf.cast(tf.random.uniform(tf.shape(h0_prob)) < h0_prob, tf.float32)

    v_prob = None
    for _ in range(k):
        v_prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + v_bias)   # (B,V)
        v = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        h_prob = tf.nn.sigmoid(tf.matmul(v, W) + h_bias)                  # (B,H)
        h = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

    # Negative phase expectations (use probabilities, not hard samples)
    vk_prob = v_prob                                  # last v probs (B,V)
    hk_prob = h_prob                                  # last h probs (B,H)

    B = tf.cast(tf.shape(v0)[0], tf.float32)

    # Gradients (positive - negative), averaged over batch
    dW = (tf.matmul(tf.transpose(v0), h0_prob) - tf.matmul(tf.transpose(vk_prob), hk_prob)) / B  # (V,H)
    dvb = tf.reduce_mean(v0 - vk_prob, axis=0)                                                    # (V,)
    dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)                                               # (H,)

    if weight_decay > 0.0:
        dW -= weight_decay * W

    # Parameter updates (in-place)
    W.assign_add(lr * dW)
    v_bias.assign_add(lr * dvb)
    h_bias.assign_add(lr * dhb)

    # Reconstruction loss (monitoring)
    recon_mse = tf.reduce_mean(tf.square(v0 - vk_prob))
    return vk_prob, recon_mse


# =============================================================================
# Training loop with early stopping + checkpoints
# =============================================================================
@dataclass
class RBMTrainConfig:
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    epochs: int = 50
    batch_size: int = 128
    cd_k: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    log_every: int = 10
    # Artifacts
    ckpt_dir: str = "artifacts/rbm/checkpoints"


def train_rbm(
    rbm,
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    *,
    cfg: RBMTrainConfig,
    log_cb: Optional[Callable[[int, float, Optional[float]], None]] = None,
) -> Dict[str, float]:
    """
    Train a Bernoulli–Bernoulli RBM via CD-k with early stopping.

    Args
    ----
    rbm : RBM-like object exposing trainable vars `W`, `v_bias`, `h_bias`.
    x_train : (N,H,W,C) or (N,V) in [0,1]; will be binarized for training.
    x_val : optional validation set (same shape rules). If None, we monitor only train loss.
    cfg : RBMTrainConfig
    log_cb : optional callback(epoch_idx, train_loss, val_loss)

    Returns
    -------
    dict with {"best_val_loss": float, "best_epoch": int, "last_train_loss": float}
    and saves:
      - BEST:  <ckpt_dir>/RBM_best.weights.h5
      - LAST:  <ckpt_dir>/RBM_last.weights.h5
      - EPOCH: <ckpt_dir>/RBM_epoch_XXXX.weights.h5 (periodic)
    """
    H, W, C = cfg.img_shape
    V = H * W * C

    # Prepare datasets
    ds_train = build_visible_dataset(
        x_train, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=True, binarize=True
    )
    ds_val = (
        build_visible_dataset(x_val, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=False, binarize=True)
        if x_val is not None
        else None
    )

    # Ensure variables exist
    # If the RBM is a tf.keras.Model, build by calling once.
    try:
        _ = rbm.W.shape  # will raise if missing
    except Exception:
        # Try to build via a dummy call (supports Keras subclass pattern)
        rbm(tf.zeros((1, V), dtype=tf.float32))

    # Checkpoint directory
    ckpt_dir = Path(cfg.ckpt_dir)
    _ensure_dir(ckpt_dir)
    best_path = ckpt_dir / "RBM_best.weights.h5"
    last_path = ckpt_dir / "RBM_last.weights.h5"

    best_val = np.inf
    best_epoch = -1
    patience_ctr = 0

    for epoch in range(1, cfg.epochs + 1):
        # ------------------------ Train ------------------------
        running = []
        for v0 in ds_train:
            v0 = _flatten_if_needed(v0, V)
            _, mse = cd_k_update(
                rbm.W, rbm.v_bias, rbm.h_bias, v0, k=cfg.cd_k, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            running.append(_as_float(mse))
        train_loss = float(np.mean(running)) if running else np.nan

        # ------------------------ Validate ---------------------
        val_loss = None
        if ds_val is not None:
            v_losses = []
            for vv in ds_val:
                vv = _flatten_if_needed(vv, V)
                # One reconstruction pass using probabilities (no updates)
                h_prob = tf.nn.sigmoid(tf.matmul(vv, rbm.W) + rbm.h_bias)
                v_prob = tf.nn.sigmoid(tf.matmul(h_prob, tf.transpose(rbm.W)) + rbm.v_bias)
                v_losses.append(_as_float(tf.reduce_mean(tf.square(vv - v_prob))))
            val_loss = float(np.mean(v_losses)) if v_losses else np.nan

        # Logging callback (external console/ui logger)
        if log_cb is not None:
            log_cb(epoch, train_loss, val_loss)

        # Periodic epoch checkpoints
        if epoch % max(1, int(cfg.log_every)) == 0 or epoch == 1:
            epoch_path = ckpt_dir / f"RBM_epoch_{epoch:04d}.weights.h5"
            rbm.save_weights(str(epoch_path))

        # Early stopping on validation (if provided), else track train
        monitor_loss = val_loss if val_loss is not None else train_loss
        improved = monitor_loss < best_val if np.isfinite(monitor_loss) else False

        if improved:
            best_val = monitor_loss
            best_epoch = epoch
            patience_ctr = 0
            rbm.save_weights(str(best_path))
        else:
            patience_ctr += 1
            if patience_ctr >= int(cfg.patience):
                # Save last and stop
                rbm.save_weights(str(last_path))
                return {
                    "best_val_loss": float(best_val),
                    "best_epoch": int(best_epoch),
                    "last_train_loss": float(train_loss),
                    "stopped_early": True,
                }

    # Finished all epochs
    rbm.save_weights(str(last_path))
    return {
        "best_val_loss": float(best_val if np.isfinite(best_val) else train_loss),
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(cfg.epochs),
        "last_train_loss": float(train_loss),
        "stopped_early": False,
    }


__all__ = ["RBMTrainConfig", "build_visible_dataset", "cd_k_update", "train_rbm"]
