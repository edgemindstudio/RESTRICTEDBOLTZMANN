# app/main.py
# =============================================================================
# Restricted Boltzmann Machine (per-class) — production pipeline entry point
#
# Commands
# --------
# python -m app.main train         # fit per-class RBMs, save checkpoints
# python -m app.main synth         # load checkpoints and synthesize per-class samples
# python -m app.main eval          # standardized evaluation with/without synthetic
# python -m app.main all           # train -> synth -> eval
#
# Phase-2 evaluation outputs
# --------------------------
# - runs/console.txt               (clean, human-readable block; via gcs_core or local fallback)
# - runs/summary.jsonl             (one JSON line per run; via gcs_core or local fallback)
# - artifacts/.../summaries/*.json (pretty JSON in Phase-2 schema for the aggregator)
#
# Notes
# -----
# • Images are float32 NHWC in [0, 1]; labels are one-hot.
# • This file mirrors the structure of the other repos (GAN/MAF/VAE/AR/GMM)
#   so aggregation and CI stay consistent.
# • Includes a robust writer shim that supports multiple gcs_core signatures and
#   falls back to a local writer to avoid API drift failures.
# =============================================================================

from __future__ import annotations

# --- Make repo-local packages importable (restrictedboltzmann/, etc.) ---------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import argparse
import json
from typing import Dict, Tuple, Optional, Any, Mapping, Union

import numpy as np
import tensorflow as tf
import yaml

# ------------------------------------------------------------------------------
# gcs_core (the “frozen core”) for synth discovery + evaluation/writing
# ------------------------------------------------------------------------------
from gcs_core.synth_loader import resolve_synth_dir, load_synth_any
from gcs_core.val_common import compute_all_metrics, write_summary_with_gcs_core

# ------------------------------------------------------------------------------
# RBM pipeline (prefer package; fallback kept for compatibility)
# ------------------------------------------------------------------------------
try:
    from restrictedboltzmann.pipeline import RBMPipeline  # type: ignore
except Exception:  # pragma: no cover
    from cRestrictedBoltzmann import RBMPipeline  # type: ignore


# =============================================================================
# GPU niceties (safe on CPU-only machines)
# =============================================================================
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Set NumPy + TF seeds for reproducibility."""
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_yaml(path: Path) -> Dict:
    """Parse YAML at `path`."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict) -> None:
    """Create artifact directories present in cfg (idempotent)."""
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot (N, K) float32; pass-through when already one-hot."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32")
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype("float32")


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load four .npy files from DATA_DIR:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns:
      x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h
    Each x_* is float32 in [0,1], shaped (N,H,W,C); labels are one-hot.
    Splits provided test -> (val, test) using `val_fraction`.
    """
    H, W, C = img_shape

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    def to_01_hwc(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if x.max() > 1.5:  # handle 0..255
            x = x / 255.0
        x = x.reshape((-1, H, W, C))
        return np.clip(x, 0.0, 1.0)

    x_train01 = to_01_hwc(x_train)
    x_test01  = to_01_hwc(x_test)

    y_train1h = one_hot(y_train, num_classes)
    y_test1h  = one_hot(y_test,  num_classes)

    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]
    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


def _save_preview_grid_png(arr: np.ndarray, path: Path) -> None:
    """Save a horizontal grayscale grid (one image per class) as PNG."""
    import matplotlib.pyplot as plt  # lazy import
    x = arr.copy()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    k = x.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(1.4 * k, 1.6))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].imshow(x[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i].set_axis_off()
        axes[i].set_title(f"C{i}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sanitize_synth(
    x_s: Optional[np.ndarray],
    y_s: Optional[np.ndarray],
    img_shape: Tuple[int, int, int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Drop non-finite samples and clamp to [0,1]. Leave (None, None) untouched.
    Ensures shape (N,H,W,C).
    """
    if x_s is None or y_s is None:
        return None, None
    if not isinstance(x_s, np.ndarray) or x_s.size == 0:
        return None, None
    H, W, C = img_shape
    x_s = np.asarray(x_s, dtype=np.float32)
    y_s = np.asarray(y_s, dtype=np.float32)
    if x_s.ndim == 3:  # (N,H,W) -> (N,H,W,C)
        x_s = x_s.reshape((-1, H, W, C))
    mask = np.isfinite(x_s).all(axis=(1, 2, 3))
    if not mask.any():
        print("[warn] All synthetic samples were non-finite; evaluating REAL-only.")
        return None, None
    if not mask.all():
        print(f"[warn] Dropping {(~mask).sum()} non-finite synthetic samples.")
    x_s = np.clip(x_s[mask], 0.0, 1.0)
    y_s = y_s[mask]
    return x_s, y_s


# =============================================================================
# Phase-2 writer shim (supports multiple gcs_core signatures + local fallback)
# =============================================================================
PathLike = Union[str, Path]

def _build_real_dirs(data_dir: Path) -> Dict[str, str]:
    """Stable 'real_dirs' mapping used by newer writers."""
    return {
        "train": str(data_dir / "train_data.npy"),
        "val":   f"{data_dir}/(split of test_data.npy)",
        "test":  f"{data_dir}/(split of test_data.npy)",
    }


def _map_util_names(util_block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize utility metric names to a stable schema."""
    if not util_block:
        return {}
    bal = util_block.get("balanced_accuracy", util_block.get("bal_acc"))
    return {
        "accuracy":               util_block.get("accuracy"),
        "macro_f1":               util_block.get("macro_f1"),
        "balanced_accuracy":      bal,
        "macro_auprc":            util_block.get("macro_auprc"),
        "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
        "ece":                    util_block.get("ece"),
        "brier":                  util_block.get("brier"),
        "per_class":              util_block.get("per_class"),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None) else float(a - b)


def _build_phase2_record(
    *,
    model_name: str,
    seed: int,
    images_counts: Mapping[str, Optional[int]],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct Phase-2 aggregator record from metrics and counts."""
    util_R  = _map_util_names(metrics.get("real_only"))
    util_RS = _map_util_names(metrics.get("real_plus_synth"))
    deltas = {
        "accuracy":           _delta(util_RS.get("accuracy"),          util_R.get("accuracy")),
        "macro_f1":           _delta(util_RS.get("macro_f1"),          util_R.get("macro_f1")),
        "balanced_accuracy":  _delta(util_RS.get("balanced_accuracy"), util_R.get("balanced_accuracy")),
        "macro_auprc":        _delta(util_RS.get("macro_auprc"),       util_R.get("macro_auprc")),
        "recall_at_1pct_fpr": _delta(util_RS.get("recall_at_1pct_fpr"),util_R.get("recall_at_1pct_fpr")),
        "ece":                _delta(util_RS.get("ece"),               util_R.get("ece")),
        "brier":              _delta(util_RS.get("brier"),             util_R.get("brier")),
    }
    generative = {
        "fid":          metrics.get("fid_macro"),
        "fid_macro":    metrics.get("fid_macro"),
        "cfid_macro":   metrics.get("cfid_macro"),
        "js":           metrics.get("js"),
        "kl":           metrics.get("kl"),
        "diversity":    metrics.get("diversity"),
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }
    return {
        "model": model_name,
        "seed":  int(seed),
        "images": {
            "train_real": int(images_counts.get("train_real") or 0),
            "val_real":   int(images_counts.get("val_real") or 0),
            "test_real":  int(images_counts.get("test_real") or 0),
            "synthetic":  (int(images_counts["synthetic"]) if images_counts.get("synthetic") is not None else None),
        },
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
    }


def _write_console_block(record: Dict[str, Any]) -> str:
    """Format a concise console block and return it."""
    gen = record.get("generative", {})
    util_R  = record.get("utility_real_only", {})
    util_RS = record.get("utility_real_plus_synth", {})
    counts  = record.get("images", {})
    lines = [
        f"Model: {record.get('model')}   Seed: {record.get('seed')}",
        f"Counts → train:{counts.get('train_real')}  "
        f"val:{counts.get('val_real')}  "
        f"test:{counts.get('test_real')}  "
        f"synth:{counts.get('synthetic')}",
        f"Generative → FID(macro): {gen.get('fid_macro')}  cFID(macro): {gen.get('cfid_macro')}  "
        f"JS: {gen.get('js')}  KL: {gen.get('kl')}  Div: {gen.get('diversity')}",
        f"Utility R   → acc: {util_R.get('accuracy')}  bal_acc: {util_R.get('balanced_accuracy')}  "
        f"macro_f1: {util_R.get('macro_f1')}",
        f"Utility R+S → acc: {util_RS.get('accuracy')}  bal_acc: {util_RS.get('balanced_accuracy')}  "
        f"macro_f1: {util_RS.get('macro_f1')}",
    ]
    return "\n".join(lines) + "\n"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def _save_console(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(block)


def _ensure_images_block(record: Dict[str, Any], images_counts: Mapping[str, Optional[int]]) -> None:
    """Ensure 'images' counts exist even if the core writer omitted them."""
    record.setdefault("images", {})
    for k, v in images_counts.items():
        if record["images"].get(k) is None:
            record["images"][k] = v


def _try_core_writer(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: PathLike,
    output_console_path: PathLike,
) -> Optional[Dict[str, Any]]:
    """
    Try multiple signatures of gcs_core.write_summary_with_gcs_core.
    Returns the record on success; None if every attempt fails.
    """
    base_kwargs = dict(
        model_name=model_name,
        seed=seed,
        fid_cap_per_class=fid_cap_per_class,
        output_json=str(output_json_path),
        output_console=str(output_console_path),
        metrics=dict(metrics),
        notes=notes,
    )
    real_dirs = _build_real_dirs(data_dir)

    attempts = [
        # Newest: real_dirs + images_counts + synth_dir
        dict(base_kwargs, real_dirs=real_dirs, images_counts=dict(images_counts), synth_dir=synth_dir),
        # Mid: real_dirs + synth_dir (no images_counts)
        dict(base_kwargs, real_dirs=real_dirs, synth_dir=synth_dir),
        # Old: synth_dir only
        dict(base_kwargs, synth_dir=synth_dir),
    ]

    for kw in attempts:
        try:
            rec = write_summary_with_gcs_core(**kw)
            _ensure_images_block(rec, images_counts)
            return rec
        except TypeError:
            continue  # signature mismatch – try next layout
        except Exception:
            continue  # internal failure – try next layout
    return None


def _local_write_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: PathLike,
    output_console_path: PathLike,
) -> Dict[str, Any]:
    """
    Build the Phase-2 record locally and write console + JSONL outputs.
    Used when gcs_core writer signatures don't match (API drift).
    """
    record = _build_phase2_record(
        model_name=model_name,
        seed=seed,
        images_counts=images_counts,
        metrics=metrics,
    )
    # Include a few extras for traceability:
    record.setdefault("meta", {})
    record["meta"].update({
        "notes": notes,
        "fid_cap_per_class": int(fid_cap_per_class),
        "synth_dir": synth_dir,
        "real_dirs": _build_real_dirs(data_dir),
    })

    console_block = _write_console_block(record)
    _save_console(Path(output_console_path), console_block)
    _append_jsonl(Path(output_json_path), record)
    return record


def _write_phase2_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: Optional[str],
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str = "",
    output_json_path: PathLike = "runs/summary.jsonl",
    output_console_path: PathLike = "runs/console.txt",
) -> Dict[str, Any]:
    """
    Version-agnostic writer:
      1) Try multiple gcs_core signatures.
      2) If all fail, write locally with a schema-compatible record.
    """
    sdir = synth_dir or ""
    rec = _try_core_writer(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )
    if rec is not None:
        return rec

    return _local_write_summary(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )


# =============================================================================
# Orchestration
# =============================================================================
def run_train(cfg: Dict) -> None:
    """Fit per-class RBMs and save a small 1×K preview grid."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    pipe = RBMPipeline(cfg)
    # Accept either .fit(...) or .train(...) method names
    if hasattr(pipe, "fit"):
        pipe.fit(x_train=x_train01, y_train=y_train)
    else:
        pipe.train(x_train=x_train01, y_train=y_train)

    # Optional preview: one sample per class (best effort, non-fatal)
    preview_path = Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png"
    try:
        if hasattr(pipe, "_sample_batch"):
            grid, _ = pipe._sample_batch(n_per_class=1)  # type: ignore[attr-defined]
        else:
            old = getattr(pipe, "samples_per_class", 1)
            setattr(pipe, "samples_per_class", 1)
            grid, _ = pipe.synthesize()
            setattr(pipe, "samples_per_class", old)
        if isinstance(grid, np.ndarray) and grid.size:
            _save_preview_grid_png(grid[:num_classes], preview_path)
            print(f"Saved preview grid to {preview_path}")
    except Exception:
        pass  # keep training output usable even if preview fails


def run_synth(cfg: Dict) -> None:
    """Load latest RBM checkpoints and synthesize per-class datasets to disk."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    pipe = RBMPipeline(cfg)
    x_s, y_s = pipe.synthesize()  # pipeline handles checkpoint selection & saving
    synth_dir = Path(cfg["ARTIFACTS"]["synthetic"])
    print(f"Synthesized: {x_s.shape[0]} samples (saved under {synth_dir}).")


def run_eval(cfg: Dict, include_synth: bool) -> None:
    """
    Phase-2 standardized evaluation:
      • Generative quality (FID/cFID/JS/KL/Diversity) on VAL vs SYNTH
      • Downstream utility on REAL test with the fixed small CNN
      • Writes:
          - runs/console.txt
          - runs/summary.jsonl
          - ARTIFACTS/summaries/RestrictedBoltzmann_eval_summary_seed{SEED}.json
    """
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)
    Path("runs").mkdir(exist_ok=True)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])
    seed        = int(cfg.get("SEED", 42))
    fid_cap     = int(cfg.get("FID_CAP", 200))
    eval_epochs = int(cfg.get("EVAL_EPOCHS", 20))

    # --- REAL data ---
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # --- Optional SYNTH ---
    x_s, y_s = (None, None)
    synth_dir_str = ""
    if include_synth:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            synth_dir = resolve_synth_dir(cfg, repo_root)          # supports absolute/relative layouts
            x_s, y_s  = load_synth_any(synth_dir, num_classes)     # monolithic or per-class dumps
            x_s, y_s  = _sanitize_synth(x_s, y_s, img_shape)
            synth_dir_str = str(synth_dir)
            if x_s is not None:
                print(f"[eval] Using synthetic from {synth_dir} (N={len(x_s)})")
            else:
                print(f"[eval] WARN: no usable synthetic under {synth_dir}; evaluating REAL-only.")
        except Exception as e:
            print(f"[eval] WARN: could not load synthetic -> {e}. Evaluating REAL-only.")
            x_s, y_s = None, None
            synth_dir_str = str(Path(cfg.get("ARTIFACTS", {}).get("synthetic", "artifacts/synthetic")))
    else:
        # Keep a stable string for writers even when synth is skipped
        synth_dir_str = str(Path(cfg.get("ARTIFACTS", {}).get("synthetic", "")))

    # --- Metrics (compute_all_metrics standardizes ranges/shapes) ---
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_tr, y_train_real=y_tr,
        x_val_real=x_va,   y_val_real=y_va,
        x_test_real=x_te,  y_test_real=y_te,
        x_synth=x_s,       y_synth=y_s,
        fid_cap_per_class=fid_cap,
        seed=seed,
        domain_embed_fn=None,
        epochs=eval_epochs,
    )

    # --- Writer shim: core (multiple signatures) -> local fallback -----------
    images_counts = {
        "train_real": int(x_tr.shape[0]),
        "val_real":   int(x_va.shape[0]),
        "test_real":  int(x_te.shape[0]),
        "synthetic":  (int(x_s.shape[0]) if isinstance(x_s, np.ndarray) else None),
    }

    record = _write_phase2_summary(
        model_name="RestrictedBoltzmann",
        seed=seed,
        data_dir=data_dir,
        synth_dir=synth_dir_str,
        fid_cap_per_class=fid_cap,
        metrics=metrics,
        images_counts=images_counts,
        notes="phase2-real",
        output_json_path="runs/summary.jsonl",
        output_console_path="runs/console.txt",
    )

    # --- Pretty JSON copy under ARTIFACTS/summaries (authoritative) ----------
    out_path = Path(cfg["ARTIFACTS"]["summaries"]) / f"RestrictedBoltzmann_eval_summary_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved evaluation summary to {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restricted Boltzmann (per-class) pipeline runner")
    p.add_argument("command", choices=["train", "synth", "eval", "all"], help="Which step to run")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    p.add_argument("--no-synth", action="store_true", help="(for eval/all) skip synthetic data in evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Non-destructive defaults (parity with other repos)
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("FID_CAP", 200)
    cfg.setdefault("EVAL_EPOCHS", 20)         # evaluator CNN epochs
    cfg.setdefault("IMG_SHAPE", [40, 40, 1])
    cfg.setdefault("NUM_CLASSES", 9)
    cfg.setdefault("SAMPLES_PER_CLASS", 25)   # RBMPipeline.synthesize
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/restrictedboltzmann/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/restrictedboltzmann/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/restrictedboltzmann/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    print(f"[config] Using {Path(args.config).resolve()}")
    print(f"Synth outputs -> {Path(cfg['ARTIFACTS']['synthetic']).resolve()}")

    if args.command == "train":
        run_train(cfg)
    elif args.command == "synth":
        run_synth(cfg)
    elif args.command == "eval":
        run_eval(cfg, include_synth=not args.no_synth)
    elif args.command == "all":
        run_train(cfg)
        run_synth(cfg)
        run_eval(cfg, include_synth=not args.no_synth)


if __name__ == "__main__":
    main()
