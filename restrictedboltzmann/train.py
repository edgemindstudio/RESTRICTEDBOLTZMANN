# restrictedboltzmann/train.py
"""
Training utilities for Bernoulli–Bernoulli Restricted Boltzmann Machines (RBMs).

What this module provides
-------------------------
- `build_visible_dataset(...)`: tf.data pipeline for RBM training (visible-only).
- `cd_k_update(...)`: Single Contrastive Divergence (CD-k) parameter update step.
- `train_rbm(...)`: Full training loop with early stopping + checkpoints.
- `train(cfg)`: Per-class trainer used by the GenCyberSynth unified CLI.

Design choices
--------------
- We use the standard *analytic* CD-k update (positive/negative phase),
  not backprop-through-sampling. Fast and stable:
      dW  = <v0^T h0_prob> - <vk^T hk_prob>
      dvb = <v0 - vk>
      dhb = <h0_prob - hk_prob>
- Visible/hidden units are Bernoulli; inputs are expected in [0,1].
- Checkpoints are saved as Keras-3-style weights: BEST/LAST and periodic epochs.

Conventions
-----------
- Visible vectors are shaped (B, V) where V = H * W * C.
- Images are channels-last (H, W, C) with values in [0,1] (binarized internally).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import tensorflow as tf

# Local model class (must expose .W, .v_bias, .h_bias and save_weights)
from .models import BernoulliRBM  # noqa: E402

# Optional shared loader (preferred if present)
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # _load_dataset will fall back to raw .npy files


# =============================================================================
# Helpers
# =============================================================================
def _as_float(x) -> float:
    """Safely cast tensors/arrays/scalars to float."""
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(-1)[0])


def _ensure_dir(p: Path) -> None:
    """mkdir -p."""
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
    img_shape : (H,W,C) used to compute V when flattening.
    batch_size : int
    shuffle : bool
    binarize : bool  (threshold to {0,1} if True)
    threshold : float

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

    Returns
    -------
    recon_prob : (B,V) final visible probabilities after k steps
    recon_mse  : scalar tensor, MSE between v0 and recon_prob
    """
    # Positive phase (use probabilities for stability)
    h0_prob = tf.nn.sigmoid(tf.matmul(v0, W) + h_bias)  # (B,H)

    # Negative phase: Gibbs chain of length k
    h = tf.cast(tf.random.uniform(tf.shape(h0_prob)) < h0_prob, tf.float32)

    v_prob = None
    for _ in range(k):
        v_prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + v_bias)   # (B,V)
        v = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        h_prob = tf.nn.sigmoid(tf.matmul(v, W) + h_bias)                  # (B,H)
        h = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

    # Expectations using probabilities (less noisy)
    vk_prob = v_prob                                  # (B,V)
    hk_prob = h_prob                                  # (B,H)

    B = tf.cast(tf.shape(v0)[0], tf.float32)

    # Gradients (positive - negative), averaged over batch
    dW = (tf.matmul(tf.transpose(v0), h0_prob) - tf.matmul(tf.transpose(vk_prob), hk_prob)) / B  # (V,H)
    dvb = tf.reduce_mean(v0 - vk_prob, axis=0)                                                    # (V,)
    dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)                                               # (H,)

    if weight_decay > 0.0:
        dW -= weight_decay * W

    # SGD updates
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
    # Artifacts (unused in unified CLI path, but kept for standalone use)
    ckpt_dir: str = "artifacts/restrictedboltzmann/checkpoints"


def train_rbm(
    rbm,
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    *,
    cfg: RBMTrainConfig,
    log_cb: Optional[Callable[[int, float, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    """
    Train a Bernoulli–Bernoulli RBM via CD-k with early stopping.

    Returns
    -------
    dict with {"best_val_loss": float, "best_epoch": int, "last_train_loss": float, "stopped_early": bool}
    and saves:
      - BEST:  <ckpt_dir>/RBM_best.weights.h5
      - LAST:  <ckpt_dir>/RBM_last.weights.h5
      - EPOCH: <ckpt_dir>/RBM_epoch_XXXX.weights.h5 (periodic)
    """
    H, W, C = cfg.img_shape
    V = H * W * C

    # Datasets
    ds_train = build_visible_dataset(
        x_train, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=True, binarize=True
    )
    ds_val = (
        build_visible_dataset(x_val, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=False, binarize=True)
        if x_val is not None
        else None
    )

    # Ensure variables exist (supports Keras subclass pattern)
    try:
        _ = rbm.W.shape
    except Exception:
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


# =============================================================================
# Dataset & config helpers
# =============================================================================
def _cfg_get(cfg: Dict, path: str, default=None):
    """Safely read nested dict keys via dotted paths."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_dataset(cfg: Dict, img_shape: Tuple[int, int, int], num_classes: int):
    """Shared loader: try common.data.load_dataset_npy, else raw .npy files."""
    data_dir = Path(_cfg_get(cfg, "DATA_DIR", _cfg_get(cfg, "data.root", "data")))
    if load_dataset_npy is not None:
        return load_dataset_npy(
            data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
        )
    # Fallback loader
    x_train = np.load(data_dir / "train_data.npy").astype("float32")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test = np.load(data_dir / "test_data.npy").astype("float32")
    y_test = np.load(data_dir / "test_labels.npy")

    if x_train.max() > 1.5:
        x_train /= 255.0
        x_test /= 255.0

    H, W, C = img_shape
    x_train = x_train.reshape((-1, H, W, C))
    x_test = x_test.reshape((-1, H, W, C))
    n_val = int(len(x_test) * float(cfg.get("VAL_FRACTION", 0.5)))
    x_val, y_val = x_test[:n_val], y_test[:n_val]
    x_test, y_test = x_test[n_val:], y_test[n_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def _int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept (N,) ints or (N,K) one-hot -> return (N,) ints."""
    return (np.argmax(y, axis=1) if (y.ndim == 2 and y.shape[1] == num_classes) else y).astype(int)


# =============================================================================
# Unified-CLI entrypoint
# =============================================================================
def train(cfg: Dict):
    """
    Per-class RBM trainer used by the unified CLI.

    Saves checkpoints under:
      artifacts/restrictedboltzmann/checkpoints/class_{k}/RBM_best.weights.h5
    Also writes a 1×K preview grid PNG at:
      artifacts/restrictedboltzmann/summaries/rbm_train_preview.png
    """
    # Defaults & shapes
    H, W, C = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    K = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))
    V = H * W * C

    seed = int(cfg.get("SEED", 42))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # RBM hyperparams
    hidden = int(cfg.get("RBM_HIDDEN", 256))
    epochs = int(cfg.get("EPOCHS", cfg.get("RBM_EPOCHS", 50)))
    batch = int(cfg.get("BATCH_SIZE", cfg.get("RBM_BATCH", 128)))
    cd_k = int(cfg.get("CD_K", 1))
    lr = float(cfg.get("LR", 1e-3))
    wd = float(cfg.get("WEIGHT_DECAY", 0.0))
    patience = int(cfg.get("PATIENCE", 10))
    log_every = int(cfg.get("LOG_EVERY", 10))

    # Artifacts
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    model_root = artifacts_root / "restrictedboltzmann"
    ckpt_root = model_root / "checkpoints"
    sums_dir = model_root / "summaries"
    _ensure_dir(ckpt_root)
    _ensure_dir(sums_dir)

    # Data
    x_tr, y_tr, x_va, y_va, x_te, y_te = _load_dataset(cfg, (H, W, C), K)
    y_tr_i = _int_labels(y_tr, K)
    y_va_i = _int_labels(y_va, K) if y_va is not None else None

    # Train one RBM per class
    for k in range(K):
        idx = (y_tr_i == k)
        n_k = int(idx.sum())
        if n_k < 2:
            print(f"[rbm] skip class {k}: too few samples (n={n_k})")
            continue

        rbm = BernoulliRBM(visible_dim=V, hidden_dim=hidden)
        cfg_train = RBMTrainConfig(
            img_shape=(H, W, C),
            epochs=epochs,
            batch_size=batch,
            cd_k=cd_k,
            lr=lr,
            weight_decay=wd,
            patience=patience,
            log_every=log_every,
            ckpt_dir=str(ckpt_root / f"class_{k}"),
        )

        def _log(e, tr, va):
            if (e == 1) or (e % max(1, log_every) == 0):
                msg = f"[rbm][k={k}] epoch={e:04d} train_mse={tr:.5f}"
                if va is not None:
                    msg += f" val_mse={va:.5f}"
                print(msg)

        print(f"[rbm] training class {k} on {n_k} samples (V={V}, H={hidden})")
        _ = train_rbm(
            rbm,
            x_tr[idx],
            x_val=(x_va[y_va_i == k] if (y_va_i is not None) else None),
            cfg=cfg_train,
            log_cb=_log,
        )

    # Optional: tiny preview grid (1 sample per class) if sampler is present
    try:
        from .sample import save_grid_from_checkpoints
        save_grid_from_checkpoints(
            ckpt_root=ckpt_root,
            img_shape=(H, W, C),
            num_classes=K,
            path=sums_dir / "rbm_train_preview.png",
            per_class=1,
        )
        print(f"[preview] wrote {sums_dir / 'rbm_train_preview.png'}")
    except Exception as e:
        print(f"[warn] preview grid failed: {e}")


__all__ = [
    "RBMTrainConfig",
    "build_visible_dataset",
    "cd_k_update",
    "train_rbm",
    "train",
]
