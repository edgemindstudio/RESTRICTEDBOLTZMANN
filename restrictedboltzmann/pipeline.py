# restrictedboltzmann/pipeline.py

"""
Training + synthesis pipeline for the Restricted Boltzmann Machine (RBM) baseline.

Why this exists
---------------
A small, production-friendly wrapper that:
  • trains class-conditional RBMs (one RBM per label) using CD-k or MSE mode
  • saves checkpoints per class (Keras 3-compatible *.weights.h5)
  • synthesizes class-balanced samples and writes evaluator-friendly files

Artifacts written by `.synthesize(...)`
---------------------------------------
ARTIFACTS/restrictedboltzmann/synthetic/
  gen_class_{k}.npy        -> float32 images in [0, 1], shape (Nk, H, W, C)
  labels_class_{k}.npy     -> int labels (k), shape (Nk,)
  x_synth.npy, y_synth.npy -> convenience concatenations

Conventions
-----------
- Images are channels-last (H, W, C) with values in [0, 1].
- Labels may be one-hot or integer; this pipeline converts internally.
- Checkpoints: RBM_class_{k}.weights.h5 (one per class)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from restrictedboltzmann.model import (
    RBMConfig,
    RBM,
    build_rbm,
    to_float01,
    binarize01,
    flatten_images,
    reshape_to_images,
    sample_gibbs,
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Accepts one-hot (N, K) or integer (N,) labels and returns int labels (N,).
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 1:
        return y.astype(np.int64)
    raise ValueError(f"Labels must be (N,) ints or (N,{num_classes}) one-hot; got {y.shape}")


def _is_built(rbm: RBM) -> bool:
    # Keras models are "built" once variables have shapes; we rely on presence of weights.
    try:
        _ = rbm.W.shape  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------
@dataclass
class RBMPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # RBM hyperparameters (per class)
    HIDDEN_UNITS: int = 256
    CD_K: int = 1
    LR: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    TRAIN_MODE: str = "cd"  # {"cd", "mse"}

    # Optimization (MSE mode only; CD mode uses manual SGD-style updates)
    BATCH_SIZE: int = 256
    EPOCHS: int = 10
    PATIENCE: int = 10

    # Data preprocessing
    BINARIZE: bool = True
    BIN_THRESHOLD: float = 0.5
    VAL_SPLIT: float = 0.1  # used if val set not provided

    # Synthesis
    SAMPLES_PER_CLASS: int = 1000
    SAMPLE_K_STEPS: Optional[int] = None  # defaults to CD_K if None

    # Reproducibility
    SEED: Optional[int] = 42

    # Artifacts
    ARTIFACTS: Dict[str, str] = None  # filled in __init__ of pipeline


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class RBMPipeline:
    """
    Orchestrates training and synthesis for class-conditional RBMs.
    """

    def __init__(self, cfg: Dict):
        # Map dict -> dataclass (with defaults)
        self.cfg = RBMPipelineConfig(
            IMG_SHAPE=tuple(cfg.get("IMG_SHAPE", (40, 40, 1))),
            NUM_CLASSES=int(cfg.get("NUM_CLASSES", 9)),
            HIDDEN_UNITS=int(cfg.get("HIDDEN_UNITS", 256)),
            CD_K=int(cfg.get("CD_K", 1)),
            LR=float(cfg.get("LR", 1e-3)),
            WEIGHT_DECAY=float(cfg.get("WEIGHT_DECAY", 0.0)),
            TRAIN_MODE=str(cfg.get("TRAIN_MODE", "cd")),
            BATCH_SIZE=int(cfg.get("BATCH_SIZE", 256)),
            EPOCHS=int(cfg.get("EPOCHS", 10)),
            PATIENCE=int(cfg.get("PATIENCE", 10)),
            BINARIZE=bool(cfg.get("BINARIZE", True)),
            BIN_THRESHOLD=float(cfg.get("BIN_THRESHOLD", 0.5)),
            VAL_SPLIT=float(cfg.get("VAL_SPLIT", 0.1)),
            SAMPLES_PER_CLASS=int(cfg.get("SAMPLES_PER_CLASS", 1000)),
            SAMPLE_K_STEPS=cfg.get("SAMPLE_K_STEPS", None),
            SEED=cfg.get("SEED", 42),
            ARTIFACTS=cfg.get("ARTIFACTS", {}),
        )

        # Artifacts
        arts = self.cfg.ARTIFACTS or {}
        self.ckpt_dir = Path(arts.get("checkpoints", "artifacts/restrictedboltzmann/checkpoints"))
        self.synth_dir = Path(arts.get("synthetic", "artifacts/restrictedboltzmann/synthetic"))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional external logger callback: cb(stage: str, message: str)
        self.log_cb = cfg.get("LOG_CB", None)

        # Placeholder for in-memory RBMs (one per class)
        self.models: List[Optional[RBM]] = [None] * self.cfg.NUM_CLASSES

        # Set seeds for reproducibility (best-effort)
        if self.cfg.SEED is not None:
            np.random.seed(self.cfg.SEED)
            tf.random.set_seed(self.cfg.SEED)

    # ----------------------- Logging -----------------------
    def _log(self, stage: str, msg: str) -> None:
        if self.log_cb:
            try:
                self.log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    # ----------------------- tf.data helpers ----------------
    def _make_dataset(self, x_flat: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """
        Create a dataset of visible vectors only: returns v0 batches (float32).
        """
        ds = tf.data.Dataset.from_tensor_slices(x_flat.astype("float32"))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(x_flat), 8192), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    # ----------------------- Training (per class) ----------------
    def _train_single_class(
        self,
        Xk: np.ndarray,
        *,
        class_id: int,
        x_val: Optional[np.ndarray] = None,
    ) -> RBM:
        """
        Train one RBM on data from a single class (flattened [0,1] vectors).
        Uses CD-k (manual updates) or MSE mode (optimizer) depending on config.
        Saves best checkpoint under RBM_class_{k}.weights.h5.
        """
        V = int(np.prod(self.cfg.IMG_SHAPE))
        H = self.cfg.HIDDEN_UNITS

        # Build RBM (variables are created immediately)
        rbm_cfg = RBMConfig(
            visible_units=V,
            hidden_units=H,
            cd_k=self.cfg.CD_K,
            learning_rate=self.cfg.LR,
            weight_decay=self.cfg.WEIGHT_DECAY,
            train_mode=self.cfg.TRAIN_MODE,
            seed=self.cfg.SEED,
        )
        rbm = build_rbm(rbm_cfg)

        # Train/val split if explicit val not provided
        if x_val is None and self.cfg.VAL_SPLIT > 0.0 and len(Xk) > 1:
            n_val = max(1, int(len(Xk) * self.cfg.VAL_SPLIT))
            x_val = Xk[:n_val]
            Xk = Xk[n_val:]

        train_ds = self._make_dataset(Xk, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        val_ds = self._make_dataset(x_val, batch_size=self.cfg.BATCH_SIZE, shuffle=False) if x_val is not None else None

        patience = int(self.cfg.PATIENCE)
        best_val = np.inf
        best_epoch = 0
        wait = 0

        ckpt_path = self.ckpt_dir / f"RBM_class_{class_id}.weights.h5"

        self._log(
            "train",
            f"[class {class_id}] RBM(H={H}) mode={self.cfg.TRAIN_MODE} "
            f"cd_k={self.cfg.CD_K} lr={self.cfg.LR:g} wd={self.cfg.WEIGHT_DECAY:g}",
        )

        if self.cfg.TRAIN_MODE.lower() == "mse":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.LR)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            # ---- Train ----
            losses = []
            for v0 in train_ds:
                if self.cfg.TRAIN_MODE.lower() == "cd":
                    loss = rbm.train_step_cd(
                        v0,
                        k=self.cfg.CD_K,
                        lr=self.cfg.LR,
                        weight_decay=self.cfg.WEIGHT_DECAY,
                    )
                else:
                    loss = rbm.train_step_mse(v0, optimizer=optimizer, k=self.cfg.CD_K)
                losses.append(float(loss))

            train_loss = float(np.mean(losses)) if losses else float("nan")

            # ---- Validate ----
            if val_ds is not None:
                vlosses = []
                for v in val_ds:
                    v_prob = rbm(v, training=False)
                    vloss = tf.reduce_mean(tf.square(v - v_prob))
                    vlosses.append(float(vloss))
                val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            else:
                val_loss = train_loss  # best effort

            self._log("train", f"[class {class_id}] epoch {epoch:03d}: train={train_loss:.4f} | val={val_loss:.4f}")

            # ---- Early stopping on val reconstruction loss ----
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                rbm.save_weights(str(ckpt_path))
            else:
                wait += 1
                if wait >= patience:
                    self._log("train", f"[class {class_id}] early stop at epoch {epoch} (best={best_val:.4f} @ {best_epoch}).")
                    break

        # Ensure weights are loaded from best epoch
        if ckpt_path.exists():
            rbm.load_weights(str(ckpt_path))
            self._log("ckpt", f"Saved {ckpt_path.name}")
        else:
            # If never improved, save final anyway
            rbm.save_weights(str(ckpt_path))
            self._log("ckpt", f"Saved {ckpt_path.name} (final)")

        return rbm

    # ----------------------- Public training API ----------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> List[RBM]:
        """
        Fit one RBM per class on binarized [0,1] images (robust early-stopped training).

        Args
        ----
        x_train : (N,H,W,C) or (N,D) float32 in [0,1] (0..255 allowed; will be scaled)
        y_train : (N,) int or (N,K) one-hot
        x_val, y_val : optional validation sets (used per class if provided)

        Returns
        -------
        list of fitted RBM instances (length <= NUM_CLASSES)
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES
        V = H * W * C

        # Prepare training data
        x = x_train
        x = to_float01(x.reshape((-1, H, W, C)))
        if self.cfg.BINARIZE:
            x = binarize01(x, thresh=self.cfg.BIN_THRESHOLD)
        X = x.reshape((-1, V)).astype("float32")

        y_ids = _labels_to_int(y_train, K)

        # Optional validation
        if x_val is not None:
            xv = to_float01(x_val.reshape((-1, H, W, C)))
            if self.cfg.BINARIZE:
                xv = binarize01(xv, thresh=self.cfg.BIN_THRESHOLD)
            Xv = xv.reshape((-1, V)).astype("float32")
            yv_ids = _labels_to_int(y_val, K) if y_val is not None else None
        else:
            Xv, yv_ids = None, None

        self._log("train", f"Training class-conditional RBMs on {len(X)} samples (dim={V}), classes={K}")

        for k in range(K):
            idx = (y_ids == k)
            Xk = X[idx]
            if Xk.size == 0:
                self._log("warn", f"[class {k}] no training samples; skipping.")
                self.models[k] = None
                continue

            # Optional per-class val set if provided
            Xk_val = None
            if Xv is not None and yv_ids is not None:
                Xk_val = Xv[yv_ids == k]
                if Xk_val.size == 0:
                    Xk_val = None

            rbm_k = self._train_single_class(Xk, class_id=k, x_val=Xk_val)
            self.models[k] = rbm_k

        # Save a final "last ok" marker
        (self.ckpt_dir / "RBM_LAST_OK").write_text("ok", encoding="utf-8")

        # Return all non-None models
        return [m for m in self.models if m is not None]

    # Backwards-friendly alias (some runners may call .fit)
    def fit(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    # ----------------------- Checkpoint I/O -----------------------
    def _load_model_for_class(self, k: int) -> Optional[RBM]:
        """
        Load a class RBM from disk if not already in memory.
        """
        if 0 <= k < len(self.models) and self.models[k] is not None and _is_built(self.models[k]):  # type: ignore[arg-type]
            return self.models[k]

        # Rebuild a fresh RBM and load weights
        V = int(np.prod(self.cfg.IMG_SHAPE))
        rbm = build_rbm(
            RBMConfig(
                visible_units=V,
                hidden_units=self.cfg.HIDDEN_UNITS,
                cd_k=self.cfg.CD_K,
                learning_rate=self.cfg.LR,
                weight_decay=self.cfg.WEIGHT_DECAY,
                train_mode=self.cfg.TRAIN_MODE,
                seed=self.cfg.SEED,
            )
        )
        path = self.ckpt_dir / f"RBM_class_{k}.weights.h5"
        if path.exists():
            rbm.load_weights(str(path))
            self.models[k] = rbm
            return rbm
        return None

    # ----------------------- Synthesis -----------------------
    def synthesize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic dataset.

        Returns
        -------
        x_synth : float32, shape (N_total, H, W, C), values in [0, 1]
        y_synth : float32, shape (N_total, NUM_CLASSES), one-hot labels
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES
        per_class = int(self.cfg.SAMPLES_PER_CLASS)
        k_steps = int(self.cfg.SAMPLE_K_STEPS or self.cfg.CD_K)

        xs, ys = [], []
        _ensure_dir(self.synth_dir)

        for k in range(K):
            rbm = self._load_model_for_class(k)
            if rbm is None:
                raise FileNotFoundError(
                    f"No RBM checkpoint for class {k} found in {self.ckpt_dir}. "
                    f"Train first (app: `python app/main.py train`)."
                )

            # Draw samples via k-step block Gibbs (start from Bernoulli(0.5))
            imgs = sample_gibbs(
                rbm,
                num_samples=per_class,
                k=k_steps,
                init=None,
                img_shape=(H, W, C),
                binarize_init=False,
            )
            # imgs are in {0,1}; cast to float32 and keep as [0,1]
            imgs = imgs.astype("float32", copy=False)

            # Per-class dumps (contract used by evaluator)
            np.save(self.synth_dir / f"gen_class_{k}.npy", imgs)
            np.save(self.synth_dir / f"labels_class_{k}.npy", np.full((per_class,), k, dtype=np.int32))

            xs.append(imgs)
            y1h = np.zeros((per_class, K), dtype=np.float32)
            y1h[:, k] = 1.0
            ys.append(y1h)

        x_synth = np.concatenate(xs, axis=0).astype(np.float32)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32)

        # Sanity: drop any non-finite rows (extremely unlikely)
        mask = np.isfinite(x_synth).all(axis=(1, 2, 3))
        if not mask.all():
            dropped = int((~mask).sum())
            self._log("warn", f"Dropping {dropped} non-finite synthetic samples.")
            x_synth = x_synth[mask]
            y_synth = y_synth[mask]

        # Combined convenience dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        self._log("synthesize", f"{x_synth.shape[0]} samples ({per_class} per class, k={k_steps}) -> {self.synth_dir}")
        return x_synth, y_synth


__all__ = ["RBMPipeline", "RBMPipelineConfig"]
