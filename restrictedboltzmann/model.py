# restrictedboltzmann/model.py
"""
TensorFlow/Keras Restricted Boltzmann Machine (RBM) utilities.

What you get
------------
- RBMConfig:  Dataclass of core hyperparameters.
- RBM:        Minimal TF2/Keras RBM (Bernoulli visible/hidden) with:
                * prop-up/prop-down
                * Gibbs sampling (CD-k)
                * free energy
                * two train modes:
                    - 'cd':    classical contrastive-divergence update (manual)
                    - 'mse':   reconstruction MSE with autograd
- build_rbm(): Convenience constructor (seeding + variable build).
- Image helpers: to_float01, binarize01, flatten_images, reshape_to_images.
- Sampling helper: sample_gibbs() to draw images from the model.
- BernoulliRBM: Thin wrapper exposing expected methods for samplers:
                sample_h_given_v, sample_v_given_h.

Conventions
-----------
- Images are channels-last (H, W, C), values in [0,1]; binarization is optional
  but typical for Bernoulli RBMs.
- All computations are float32 by default; CD updates are applied with
  `assign_add` (optimizer-free) to match the classic algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Config
# =============================================================================
@dataclass
class RBMConfig:
    """
    Hyperparameters for the RBM.

    visible_units : int
        Flattened input dimensionality (H*W*C).
    hidden_units : int
        Number of hidden (latent) binary units.
    cd_k : int
        Steps of Gibbs sampling per update (CD-k).
    learning_rate : float
        Step size for manual CD updates (ignored in 'mse' mode if you pass your
        own optimizer).
    weight_decay : float
        L2 decay on weights in manual CD updates.
    train_mode : str
        'cd'  -> use classical CD-k update (no optimizer object needed)
        'mse' -> use autograd on v->v̂ reconstruction (requires optimizer)
    seed : Optional[int]
        Optional global seed for reproducibility.
    """
    visible_units: int
    hidden_units: int = 256
    cd_k: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_mode: str = "cd"  # {'cd', 'mse'}
    seed: Optional[int] = 42


# =============================================================================
# Small helpers
# =============================================================================
def to_float01(x: np.ndarray) -> np.ndarray:
    """Ensure float32 in [0,1] (handles 0..255)."""
    x = x.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def binarize01(x: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Deterministically binarize values in [0,1] at `thresh`.
    For stochastic binarization, do `(np.random.rand(*x.shape) < x).astype(np.float32)`.
    """
    return (x >= float(thresh)).astype("float32")


def flatten_images(
    x: np.ndarray, img_shape: Tuple[int, int, int], *, assume_01: bool = True
) -> np.ndarray:
    """Reshape images (N,H,W,C) -> (N,D), optionally converting to [0,1]."""
    H, W, C = img_shape
    x = x.reshape((-1, H, W, C))
    if not assume_01:
        x = to_float01(x)
    return x.reshape((-1, H * W * C)).astype("float32", copy=False)


def reshape_to_images(x_flat: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """(N,D) -> (N,H,W,C)."""
    H, W, C = img_shape
    return x_flat.reshape((-1, H, W, C)).astype("float32", copy=False)


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# RBM core
# =============================================================================
class RBM(tf.keras.Model):
    """
    Bernoulli-Bernoulli RBM with minimal API for training & sampling.

    Variables
    ---------
    W : (V,H)   Visible-to-hidden weights.
    h_bias : (H,)  Hidden bias.
    v_bias : (V,)  Visible bias.
    """

    def __init__(self, visible_units: int, hidden_units: int = 256, name: str = "rbm"):
        super().__init__(name=name)
        self.visible_units = int(visible_units)
        self.hidden_units = int(hidden_units)

        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.W = tf.Variable(init(shape=(self.visible_units, self.hidden_units)), name="W")
        self.h_bias = tf.Variable(tf.zeros([self.hidden_units]), name="h_bias")
        self.v_bias = tf.Variable(tf.zeros([self.visible_units]), name="v_bias")

    # ----- Probability transforms -----
    @tf.function(jit_compile=False)
    def _sigmoid(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.sigmoid(x)

    @tf.function(jit_compile=False)
    def _bernoulli_sample(self, probs: tf.Tensor) -> tf.Tensor:
        """Sample 0/1 from probabilities (same shape)."""
        rnd = tf.random.uniform(tf.shape(probs), dtype=probs.dtype)
        return tf.cast(rnd < probs, probs.dtype)

    # ----- Up/Down passes -----
    @tf.function(jit_compile=False)
    def propup(self, v: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """v -> h (logits, probs)"""
        logits = tf.linalg.matmul(v, self.W) + self.h_bias
        probs = self._sigmoid(logits)
        return logits, probs

    @tf.function(jit_compile=False)
    def propdown(self, h: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """h -> v (logits, probs)"""
        logits = tf.linalg.matmul(h, tf.transpose(self.W)) + self.v_bias
        probs = self._sigmoid(logits)
        return logits, probs

    # ----- Gibbs sampling -----
    @tf.function(jit_compile=False)
    def gibbs_k(self, v0: tf.Tensor, k: int = 1) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run k steps of block-Gibbs: v0 -> h0 -> v1 -> ... -> vk.
        Returns (vk_sample, hk_prob, v_prob_last).
        """
        v = v0
        _, h_prob = self.propup(v)
        for _ in tf.range(k):
            h = self._bernoulli_sample(h_prob)
            _, v_prob = self.propdown(h)
            v = self._bernoulli_sample(v_prob)
            _, h_prob = self.propup(v)
        return v, h_prob, v_prob

    # ----- Recon pass used by 'mse' mode -----
    @tf.function(jit_compile=False)
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward reconstruction v -> h -> v̂ (probabilities)."""
        _, h_prob = self.propup(inputs)
        _, v_prob = self.propdown(h_prob)
        return v_prob

    # ----- Energy / Free energy (optional diagnostics) -----
    @tf.function(jit_compile=False)
    def free_energy(self, v: tf.Tensor) -> tf.Tensor:
        vbias_term = tf.reduce_sum(v * self.v_bias, axis=1)
        hidden_lin = tf.linalg.matmul(v, self.W) + self.h_bias
        hidden_term = tf.reduce_sum(tf.math.softplus(hidden_lin), axis=1)
        # F(v) = -v·b - sum log(1+exp(Wv+c))
        return -(vbias_term + hidden_term)

    # ----- Training steps -----
    @tf.function(jit_compile=False)
    def train_step_cd(
        self, v0: tf.Tensor, *, k: int = 1, lr: float = 1e-3, weight_decay: float = 0.0
    ) -> tf.Tensor:
        """
        Classical CD-k parameter update:
            ΔW ∝ <v0 h0>_data - <vk hk>_model
            Δb ∝ <v0 - vk>
            Δc ∝ <h0 - hk>
        Returns reconstruction MSE for monitoring.
        """
        # Positive phase
        _, h0_prob = self.propup(v0)

        # Negative phase via k-step Gibbs
        vk, hk_prob, v_prob = self.gibbs_k(v0, k=k)

        # Expectations (averages over batch)
        batch = tf.cast(tf.shape(v0)[0], v0.dtype)
        pos_grad = tf.linalg.matmul(tf.transpose(v0), h0_prob) / batch
        neg_grad = tf.linalg.matmul(tf.transpose(vk), hk_prob) / batch

        dW = pos_grad - neg_grad - weight_decay * self.W
        dvb = tf.reduce_mean(v0 - vk, axis=0)
        dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)

        # Parameter updates
        self.W.assign_add(lr * dW)
        self.v_bias.assign_add(lr * dvb)
        self.h_bias.assign_add(lr * dhb)

        # Diagnostic: reconstruction error on this batch
        recon_mse = tf.reduce_mean(tf.square(v0 - v_prob))
        return recon_mse

    @tf.function(jit_compile=False)
    def train_step_mse(self, v0: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer, k: int = 1) -> tf.Tensor:
        """
        Autograd training: minimize ||v0 - v̂||^2 with v̂ from a single up/down pass.
        (k is accepted for API symmetry; it’s ignored here.)
        """
        with tf.GradientTape() as tape:
            v_hat = self(v0, training=True)
            loss = tf.reduce_mean(tf.square(v0 - v_hat))
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


# =============================================================================
# Builders & Sampling
# =============================================================================
def build_rbm(cfg: RBMConfig) -> RBM:
    """Construct and initialize an RBM from `RBMConfig` (variables built)."""
    _set_seed(cfg.seed)
    model = RBM(cfg.visible_units, cfg.hidden_units)
    # Build variables by a dummy forward pass (Keras 3 friendly).
    _ = model(tf.zeros((1, cfg.visible_units), dtype=tf.float32))
    return model


def sample_gibbs(
    rbm: RBM,
    *,
    num_samples: int,
    k: int,
    init: Optional[np.ndarray] = None,
    img_shape: Optional[Tuple[int, int, int]] = None,
    binarize_init: bool = True,
) -> np.ndarray:
    """
    Draw `num_samples` samples via k-step block Gibbs.

    Returns float32 in {0,1} shaped (N,D) or (N,H,W,C) if `img_shape` provided.
    """
    V = rbm.visible_units
    if init is None:
        v0 = tf.cast(tf.random.uniform([num_samples, V]) < 0.5, tf.float32)
    else:
        v0 = tf.convert_to_tensor(to_float01(init).reshape(-1, V), dtype=tf.float32)
        if binarize_init:
            v0 = tf.cast(v0 >= 0.5, tf.float32)

    vk, _, _ = rbm.gibbs_k(v0, k=k)
    out = vk.numpy().astype("float32", copy=False)
    if img_shape is not None:
        out = reshape_to_images(out, img_shape)
    return out


# =============================================================================
# Wrapper to match expected external API
# =============================================================================
class BernoulliRBM(RBM):
    """
    Thin wrapper over `RBM` that:
      - Accepts `visible_dim` / `hidden_dim` (as used by train/sample code)
      - Builds variables immediately so `load_weights(...)` works
      - Provides `sample_h_given_v` / `sample_v_given_h` methods expected by samplers
    """

    def __init__(
        self,
        *,
        visible_dim: Optional[int] = None,
        hidden_dim: int = 256,
        name: str = "rbm",
        **kwargs,
    ) -> None:
        # Accept legacy aliases if passed
        if visible_dim is None:
            if "visible_units" in kwargs:
                visible_dim = int(kwargs.pop("visible_units"))
            else:
                raise TypeError("BernoulliRBM requires `visible_dim=` integer.")
        if "hidden_units" in kwargs:
            hidden_dim = int(kwargs.pop("hidden_units"))

        super().__init__(visible_units=int(visible_dim), hidden_units=int(hidden_dim), name=name)
        # Build weights right away so load_weights(...) succeeds reliably
        _ = self(tf.zeros((1, int(visible_dim)), dtype=tf.float32))

    @tf.function(reduce_retracing=True)
    def sample_h_given_v(self, v: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """v -> h sample using RBM prop-up. Returns (h_sample in {0,1}, h_prob in [0,1])."""
        _, h_prob = self.propup(v)
        rnd = tf.random.uniform(tf.shape(h_prob), dtype=tf.float32)
        h_sample = tf.cast(rnd < h_prob, tf.float32)
        return h_sample, h_prob

    @tf.function(reduce_retracing=True)
    def sample_v_given_h(self, h: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """h -> v sample using RBM prop-down. Returns (v_sample in {0,1}, v_prob in [0,1])."""
        _, v_prob = self.propdown(h)
        rnd = tf.random.uniform(tf.shape(v_prob), dtype=tf.float32)
        v_sample = tf.cast(rnd < v_prob, tf.float32)
        return v_sample, v_prob


__all__ = [
    "RBMConfig",
    "RBM",
    "build_rbm",
    "to_float01",
    "binarize01",
    "flatten_images",
    "reshape_to_images",
    "sample_gibbs",
    "BernoulliRBM",
]
0