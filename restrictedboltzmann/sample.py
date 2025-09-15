# restrictedboltzmann/sample.py

"""
Sampling utilities for Restricted Boltzmann Machines (RBMs).

What this module provides
-------------------------
- `sample_gibbs(...)`: Vectorized block-Gibbs sampler that draws N visible
  samples from a trained Bernoulli–Bernoulli RBM and returns them as images
  shaped (N, H, W, C) with values in {0,1} (float32).
- `gibbs_step(...)`: One blocked Gibbs update v -> h -> v.

Assumptions
-----------
- Visible and hidden units are Bernoulli.
- The RBM exposes:
    * `sample_h_given_v(v)` -> (h_sample, h_prob)
    * `sample_v_given_h(h)` -> (v_sample, v_prob)
  where v/h tensors are float32 in {0,1} (samples) or [0,1] (probs).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# ------------------------- Core Gibbs ops ------------------------- #
@tf.function(reduce_retracing=True)
def gibbs_step(rbm, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Perform one blocked Gibbs step: v_t -> h_t -> v_{t+1}.

    Args
    ----
    rbm : RBM-like model exposing sample_h_given_v / sample_v_given_h
    v   : (B, V) float32 visible samples in {0,1}

    Returns
    -------
    (v_next_sample, v_next_prob)
      v_next_sample : (B, V) float32 in {0,1}
      v_next_prob   : (B, V) float32 in [0,1]
    """
    h_sample, _ = rbm.sample_h_given_v(v)     # (B, H) samples in {0,1}
    v_sample, v_prob = rbm.sample_v_given_h(h_sample)  # (B, V) {0,1} and [0,1]
    return tf.cast(v_sample, tf.float32), tf.cast(v_prob, tf.float32)


def _rand_bernoulli(shape, seed: Optional[int] = None) -> tf.Tensor:
    """Draw Bernoulli(0.5) samples with given shape, float32 in {0,1}."""
    rnd = tf.random.uniform(shape, seed=seed, dtype=tf.float32)
    return tf.cast(rnd > 0.5, tf.float32)


def _ensure_init_visible(
    init: Optional[np.ndarray | tf.Tensor],
    *,
    batch: int,
    visible_dim: int,
    binarize_init: bool,
    seed: Optional[int],
) -> tf.Tensor:
    """
    Prepare initial visible state for Gibbs chains.
    """
    if init is None:
        return _rand_bernoulli((batch, visible_dim), seed=seed)

    v0 = tf.convert_to_tensor(init, dtype=tf.float32)
    v0 = tf.reshape(v0, (batch, visible_dim))

    if binarize_init:
        # If init isn't already {0,1}, threshold it.
        v0 = tf.cast(v0 > 0.5, tf.float32)
    else:
        # Clip to valid probability range if using soft init.
        v0 = tf.clip_by_value(v0, 0.0, 1.0)

    # If soft init, sample once to start at a valid state on the hypercube.
    if not binarize_init:
        rnd = tf.random.uniform(tf.shape(v0), seed=seed, dtype=tf.float32)
        v0 = tf.cast(rnd < v0, tf.float32)
    return v0


# ------------------------- Public API ------------------------- #
def sample_gibbs(
    rbm,
    num_samples: int,
    k: int = 1,
    *,
    init: Optional[np.ndarray | tf.Tensor] = None,
    img_shape: Optional[Tuple[int, int, int]] = None,
    binarize_init: bool = False,
    burn_in: int = 0,
    thin: int = 1,
    return_probs: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from a trained Bernoulli–Bernoulli RBM via block Gibbs.

    Args
    ----
    rbm : Trained RBM exposing sample_h_given_v / sample_v_given_h
    num_samples : Number of visible samples to draw (batch size of chains)
    k : Number of Gibbs steps to run after burn-in
    init : Optional initial visible state. If None, starts from Bernoulli(0.5).
           If provided, must be shape (num_samples, V) or broadcastable; values
           in {0,1} or [0,1]. See `binarize_init`.
    img_shape : Optional (H, W, C) to reshape output; if None, returns (N, V).
    binarize_init : If True, thresholds `init > 0.5`. If False, treats `init`
                    as probabilities and samples a starting state.
    burn_in : Extra Gibbs steps to run before collecting the final state.
    thin : Keep every `thin`-th state after burn-in (>=1). When >1, we still
           return exactly `num_samples` states by running longer internally.
    return_probs : If True, also return the final visible probabilities.
    seed : Optional RNG seed for initial state / soft init sampling.

    Returns
    -------
    samples : np.ndarray of shape (N, H, W, C) with values in {0,1} (float32)
              if `img_shape` is provided; otherwise shape is (N, V).
    probs   : (optional) np.ndarray of same shape with values in [0,1].
    """
    # Visible dimension inferred from RBM
    try:
        visible_dim = int(rbm.W.shape[0])
    except Exception as e:
        raise ValueError("RBM must expose a weight matrix W of shape (V, H).") from e

    # Prepare initial state
    v = _ensure_init_visible(
        init,
        batch=int(num_samples),
        visible_dim=visible_dim,
        binarize_init=binarize_init,
        seed=seed,
    )

    total_steps = int(burn_in + k * thin)
    v_prob = None

    # Run the chain(s)
    for t in range(total_steps):
        v, v_prob = gibbs_step(rbm, v)

    # If thinning > 1, run extra steps to approximate thinning effect while
    # still returning exactly `num_samples` samples (vectorized chains).
    if thin > 1:
        extra = (thin - 1)
        for _ in range(extra):
            v, v_prob = gibbs_step(rbm, v)

    # Convert to numpy and reshape to images if requested
    out = v.numpy().astype("float32", copy=False)
    out = np.clip(out, 0.0, 1.0)

    if img_shape is not None:
        H, W, C = img_shape
        out = out.reshape((-1, H, W, C))

    if return_probs:
        p = v_prob.numpy().astype("float32", copy=False)
        p = np.clip(p, 0.0, 1.0)
        if img_shape is not None:
            H, W, C = img_shape
            p = p.reshape((-1, H, W, C))
        return out, p

    return out


__all__ = ["gibbs_step", "sample_gibbs"]
