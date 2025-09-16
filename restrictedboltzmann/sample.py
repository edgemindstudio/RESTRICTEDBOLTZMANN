# restrictedboltzmann/sample.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image

from .model import BernoulliRBM


# ------------------------- Gibbs sampling ------------------------- #
@tf.function(reduce_retracing=True)
def _gibbs_step(rbm: BernoulliRBM, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """One blocked Gibbs update v -> h -> v."""
    h_sample, _ = rbm.sample_h_given_v(v)                 # (B,H) {0,1}
    v_sample, v_prob = rbm.sample_v_given_h(h_sample)     # (B,V) {0,1}, [0,1]
    return tf.cast(v_sample, tf.float32), tf.cast(v_prob, tf.float32)


def _rand_bernoulli(shape, seed: Optional[int] = None) -> tf.Tensor:
    rnd = tf.random.uniform(shape, dtype=tf.float32, seed=seed)
    return tf.cast(rnd > 0.5, tf.float32)


def sample_gibbs(
    rbm: BernoulliRBM,
    num_samples: int,
    k: int = 1,
    *,
    init: Optional[np.ndarray | tf.Tensor] = None,
    img_shape: Optional[Tuple[int, int, int]] = None,
    binarize_init: bool = False,
    burn_in: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw visible samples via block Gibbs; returns float32 in {0,1}.
    """
    V = int(rbm.W.shape[0])

    if seed is not None:
        tf.random.set_seed(int(seed))

    # initial visible state
    if init is None:
        v = _rand_bernoulli((num_samples, V), seed=seed)
    else:
        v = tf.convert_to_tensor(init, dtype=tf.float32)
        v = tf.reshape(v, (num_samples, V))
        if binarize_init:
            v = tf.cast(v > 0.5, tf.float32)
        else:
            v = tf.clip_by_value(v, 0.0, 1.0)
            rnd = tf.random.uniform(tf.shape(v), dtype=tf.float32, seed=seed)
            v = tf.cast(rnd < v, tf.float32)

    # burn-in
    for _ in range(int(burn_in)):
        v, _ = _gibbs_step(rbm, v)

    # k steps
    for _ in range(int(k)):
        v, _ = _gibbs_step(rbm, v)

    out = v.numpy().astype("float32", copy=False)
    if img_shape is not None:
        H, W, C = img_shape
        out = out.reshape((-1, H, W, C))
    return out


# ------------------------- Checkpoint loading ------------------------- #
def _find_ckpt_for_class(ckpt_root: Path, k: int) -> Optional[Path]:
    """
    Try multiple layouts:
      a) {root}/class_{k}/RBM_best.weights.h5
      b) {root}/class_{k}/RBM_last.weights.h5
      c) newest {root}/class_{k}/RBM_epoch_*.weights.h5
      d) {root}/RBM_class_{k}.weights.h5           (flat — your screenshot)
      e) {root}/RBM_class_{k}.h5 or RBM_LAST_OK    (extra fallbacks)
    """
    per_class = ckpt_root / f"class_{k}"
    candidates: List[Path] = [
        per_class / "RBM_best.weights.h5",
        per_class / "RBM_last.weights.h5",
    ]
    if per_class.exists():
        epochs = sorted(per_class.glob("RBM_epoch_*.weights.h5"))
        if epochs:
            candidates.append(epochs[-1])

    # flat names
    candidates += [
        ckpt_root / f"RBM_class_{k}.weights.h5",
        ckpt_root / f"RBM_class_{k}.h5",
        ckpt_root / "RBM_LAST_OK",  # if someone dropped a single file
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_rbm(ckpt_path: Path, visible_dim: int, hidden_dim: int) -> BernoulliRBM:
    rbm = BernoulliRBM(visible_dim=visible_dim, hidden_dim=hidden_dim)
    rbm.load_weights(str(ckpt_path))
    return rbm


# ------------------------- PNG helpers ------------------------- #
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


# ------------------------- Config helper ------------------------- #
def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ------------------------- Preview grid (optional) ------------------------- #
def save_grid_from_checkpoints(
    *,
    ckpt_root: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    path: Path,
    per_class: int = 1,
    hidden_dim: int = 256,
    gibbs_k: int = 1,
    seed: int = 42,
) -> None:
    """Sample `per_class` images per class and save a compact PNG grid."""
    import matplotlib.pyplot as plt

    H, W, C = img_shape
    V = H * W * C

    tiles: List[np.ndarray] = []
    for k in range(int(num_classes)):
        ckpt = _find_ckpt_for_class(ckpt_root, k)
        if ckpt is None:
            tiles.append(np.zeros((per_class, H, W, C), dtype=np.float32))
            continue
        rbm = _load_rbm(ckpt, V, hidden_dim)
        xk = sample_gibbs(rbm, per_class, k=gibbs_k, img_shape=img_shape, seed=seed + k)
        tiles.append(xk)

    x = np.stack(tiles, axis=0)  # (K, per_class, H, W, C)
    rows, cols = per_class, int(num_classes)

    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for j in range(cols):
        for i in range(rows):
            ax = axes[i, j]
            img = x[j, i]
            if C == 1:
                ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"C{j}", fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------- Unified-CLI synth() ------------------------- #
def synth(cfg: dict, output_root: str, seed: int = 42) -> dict:
    """
    Unified CLI entrypoint.
    Loads RBM checkpoints, draws S PNGs per class, writes a manifest dict.
    """
    # Shapes & counts (support a couple of key aliases)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))
    hidden = int(_cfg_get(cfg, "RBM_HIDDEN", 256))
    gibbs_k = int(_cfg_get(cfg, "RBM_GIBBS_K", _cfg_get(cfg, "CD_K", 1)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_root = Path(_cfg_get(
        cfg,
        "ARTIFACTS.restrictedboltzmann_checkpoints",
        artifacts_root / "restrictedboltzmann" / "checkpoints",
    ))

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    V = H * W * C
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[dict] = []

    tf.random.set_seed(int(seed))

    for k in range(K):
        ckpt = _find_ckpt_for_class(ckpt_root, k)
        if ckpt is None:
            print(f"[synth][rbm] missing checkpoint for class {k}; skipping.")
            continue

        rbm = _load_rbm(ckpt, V, hidden)
        xk = sample_gibbs(rbm, S, k=gibbs_k, img_shape=(H, W, C), seed=seed + k)

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(xk.shape[0]):
            out_path = cls_dir / f"rbm_{j:05d}.png"
            _save_png(xk[j], out_path)
            paths.append({"path": str(out_path), "label": int(k)})

        per_class_counts[str(k)] = int(xk.shape[0])

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


__all__ = [
    "sample_gibbs",
    "_gibbs_step",
    "save_grid_from_checkpoints",
    "synth",
]
