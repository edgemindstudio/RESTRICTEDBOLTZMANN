#!/usr/bin/env python3
# app/main.py
# -----------------------------------------------------------------------------
# GenCyberSynth — Unified CLI for all 7 model repositories
#
# Drop this file (unchanged) into each repo. It expects exactly one of the
# following top-level module folders to exist in the repo root:
#   gan/ , vaes/ , diffusion/ , autoregressive/ ,
#   maskedautoflow/ , restrictedboltzmann/ , gaussianmixture/
#
# Commands
#   python -m app.main train  --config configs/config.yaml
#   python -m app.main synth  --config configs/config.yaml --seed 42
#   python -m app.main eval   --config configs/config.yaml [--no-synth]
#
# Contract expected from each repo:
#   - <module>/train.py    provides one of: train(cfg) | fit(cfg) | main(cfg)
#   - <module>/sample.py   provides one of:
#         synth(cfg, output_root=..., seed=...) |
#         generate(cfg, output_root=..., seed=...) |
#         sample(cfg, output_root=..., seed=...)
#     This should write per-class PNGs under:
#         artifacts/<module>/synthetic/<class>/<seed>/*.png
#     and ideally write a manifest at:
#         artifacts/<module>/synthetic/manifest.json
#     (If not provided, we will scan the directory and build it.)
#
# Evaluation:
#   If 'gcs-core' is importable and a manifest exists, we compute:
#     - cFID, KID, generative precision/recall, MS-SSIM
#   A JSON summary is always written to:
#     artifacts/<module>/summaries/summary_<timestamp>.json
#
# Keep this file identical across repos for a consistent UX and reproducibility.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ------------------------------- Optional deps -------------------------------

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # We’ll error with guidance if/when a config is requested.

# gcs-core (optional but recommended for metrics)
try:
    from gcs_core import val_common as _val  # type: ignore
    from gcs_core import synth_loader as _synth  # type: ignore

    _HAS_GCS = True
except Exception:
    _HAS_GCS = False

# ------------------------------ Repo detection -------------------------------

KNOWN_MODULES = [
    "gan",
    "vaes",
    "diffusion",
    "autoregressive",
    "maskedautoflow",
    "restrictedboltzmann",
    "gaussianmixture",
]

# ------------------------------- Small logger --------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def info(msg: str) -> None:
    print(f"[{_ts()}] [info] {msg}")

def warn(msg: str) -> None:
    print(f"[{_ts()}] [warn] {msg}")

def error(msg: str) -> None:
    print(f"[{_ts()}] [error] {msg}", file=sys.stderr)

# --------------------------------- Paths -------------------------------------

@dataclass
class Paths:
    model: str
    artifacts_root: str

    @property
    def model_root(self) -> str:
        return os.path.join(self.artifacts_root, self.model)

    @property
    def synthetic_root(self) -> str:
        return os.path.join(self.model_root, "synthetic")

    @property
    def summaries_root(self) -> str:
        return os.path.join(self.model_root, "summaries")

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------------- Config I/O ----------------------------------

def _load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if yaml is None:
        raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
    if not os.path.exists(path):
        raise SystemExit(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

# ------------------------------- Repo module ---------------------------------

def _detect_repo_module(explicit: Optional[str] = None) -> str:
    """
    Detect which model module this repo provides (folder present in root).
    If multiple are present, prefer the one whose name is in the repo basename.
    """
    if explicit:
        return explicit.lower()
    here = os.getcwd()
    present = [m for m in KNOWN_MODULES if os.path.isdir(os.path.join(here, m))]
    if len(present) == 1:
        return present[0]
    if len(present) > 1:
        repo_basename = os.path.basename(here).lower()
        for m in present:
            if m in repo_basename:
                return m
        warn(f"Multiple candidate modules found {present}; defaulting to '{present[0]}'")
        return present[0]
    raise SystemExit(
        f"Could not detect repo module. Expected one of {KNOWN_MODULES} "
        f"to exist in {here}."
    )

# ---------------------------- Dynamic import utils ---------------------------

def _maybe_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        debug = os.environ.get("GENCS_DEBUG_IMPORTS", "0") == "1"
        if debug:
            warn(f"Import failed for {module_path}: {e}")
        return None

def _call_first(module, candidates: Tuple[str, ...], *args, **kwargs):
    """
    Try calling the first available function in 'candidates' on 'module'.
    Returns (called: bool, result: Any|None).
    """
    if module is None:
        return False, None
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return True, fn(*args, **kwargs)
    return False, None

# ------------------------------- Manifest I/O --------------------------------

def _scan_to_manifest(synth_root: str, dataset: str, seed: int) -> Dict:
    """
    Fallback: build a manifest by scanning
      artifacts/<model>/synthetic/<class>/<seed>/*.png
    """
    paths: List[Dict] = []
    per_class: Dict[str, int] = {}
    if not os.path.isdir(synth_root):
        return {
            "dataset": dataset,
            "seed": seed,
            "per_class_counts": {},
            "paths": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "synthetic directory not found",
        }

    for cls in sorted(os.listdir(synth_root)):
        cls_dir = os.path.join(synth_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        seed_dir = os.path.join(cls_dir, str(seed))
        search_dir = seed_dir if os.path.isdir(seed_dir) else cls_dir
        count = 0
        for fn in sorted(os.listdir(search_dir)):
            if fn.lower().endswith(".png"):
                paths.append({"path": os.path.join(search_dir, fn), "label": int(cls)})
                count += 1
        per_class[str(cls)] = count

    return {
        "dataset": dataset,
        "seed": seed,
        "per_class_counts": per_class,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

def _write_manifest(p: Paths, manifest: Dict) -> str:
    _ensure_dir(p.synthetic_root)
    out = os.path.join(p.synthetic_root, "manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    info(
        f"Manifest → {out}  "
        f"(classes={len(manifest.get('per_class_counts', {}))}, "
        f"paths={len(manifest.get('paths', []))})"
    )
    return out

# ------------------------------- Subprocess shim -----------------------------

def _maybe_run_script(script_path: str, *args: str) -> bool:
    """
    If a shell script exists (e.g., scripts/train.sh or scripts/generate.sh),
    run it and return True. Otherwise return False.
    """
    if os.path.exists(script_path) and os.access(script_path, os.X_OK):
        info(f"Running script: {script_path} {' '.join(args)}")
        try:
            subprocess.run([script_path, *args], check=True)
            return True
        except subprocess.CalledProcessError as e:
            error(f"Script failed ({script_path}): {e}")
            raise
    return False

# --------------------------------- Commands ----------------------------------

def cmd_train(cfg: Dict, p: Paths, repo_mod: str) -> None:
    """
    Train the model.
    First tries Python entrypoints in <module>/train.py.
    If not present, falls back to scripts/train.sh if available.
    """
    mod = _maybe_import(f"{repo_mod}.train")
    called, _ = _call_first(mod, ("train", "fit", "main"), cfg)
    if called:
        info("Training completed.")
        return

    # Shell fallback
    if _maybe_run_script(os.path.join("scripts", "train.sh"), cfg.get("__path__", "")):
        info("Training (script) completed.")
        return

    raise SystemExit(
        f"No train entrypoint found in {repo_mod}/train.py "
        f"(expected train(cfg) | fit(cfg) | main(cfg)) "
        f"and scripts/train.sh not present."
    )

def cmd_synth(cfg: Dict, p: Paths, repo_mod: str, seed: int) -> None:
    """
    Generate synthetic samples and ensure a manifest exists.
    Looks for <module>/sample.py: synth|generate|sample, or scripts/generate.sh
    """
    _ensure_dir(p.synthetic_root)

    sampler = _maybe_import(f"{repo_mod}.sample")
    called, manifest = _call_first(
        sampler,
        ("synth", "generate", "sample"),
        cfg,
        output_root=p.synthetic_root,
        seed=seed,
    )

    if not called:
        # Shell fallback
        if _maybe_run_script(os.path.join("scripts", "generate.sh"), str(seed)):
            manifest = None  # script may or may not produce manifest; we’ll scan.
        else:
            warn(
                f"No sampler found in {repo_mod}/sample.py and no scripts/generate.sh. "
                f"Will attempt to scan outputs and build a manifest."
            )
            manifest = None

    dataset_name = cfg.get("data", {}).get("root", "USTC-TFC2016_40x40_gray")
    if not manifest:
        manifest = _scan_to_manifest(p.synthetic_root, dataset=dataset_name, seed=seed)

    _write_manifest(p, manifest)
    info("Synthesis completed.")

def cmd_eval(
    cfg: Dict,
    p: Paths,
    repo_mod: str,
    no_synth: bool,
    per_class_cap: int,
    encoder_name: Optional[str],
) -> None:
    """
    Evaluate synthetic data quality (if available) and write a summary JSON.
    """
    _ensure_dir(p.summaries_root)
    manifest_path = os.path.join(p.synthetic_root, "manifest.json")

    metrics: Dict[str, object] = {}

    if no_synth:
        metrics["note"] = "No-synth mode: skipping sample-based metrics."

    do_samples = (not no_synth) and os.path.exists(manifest_path) and _HAS_GCS
    if do_samples:
        try:
            man = _synth.load_manifest(manifest_path)
            imgs, labels = _synth.load_images(man, per_class_cap=per_class_cap)

            # encoder_name is reserved for a future encoder switch inside gcs-core
            _ = encoder_name

            metrics["cfid"] = float(_val.compute_cfid(imgs, labels))
            metrics["kid"] = float(_val.compute_kid(imgs, labels))
            prec, rec = _val.generative_precision_recall(imgs, labels)
            metrics["gen_precision"] = float(prec)
            metrics["gen_recall"] = float(rec)
            metrics["ms_ssim"] = float(_val.ms_ssim_intra_class(imgs, labels))
        except Exception as e:
            metrics["sample_metrics_error"] = f"{type(e).__name__}: {e}"
    else:
        if not _HAS_GCS:
            metrics.setdefault("sample_metrics_error", "gcs-core not importable")
        if not os.path.exists(manifest_path) and not no_synth:
            metrics.setdefault("sample_metrics_error", "manifest.json not found")

    # Downstream utility placeholder (wire in your classifier when ready)
    metrics["downstream"] = {
        "macro_f1": None,
        "macro_auprc": None,
        "balanced_acc": None,
        "note": "Hook your Real vs Real+Synth evaluation here.",
    }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": p.model,
        "config_used": True,
        "metrics": metrics,
    }

    out = os.path.join(p.summaries_root, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    info(f"Evaluation summary → {out}")

# ----------------------------------- CLI -------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gencs",
        description="GenCyberSynth unified CLI (train | synth | eval)",
    )
    parser.add_argument(
        "--model",
        help="Override module detection (gan, vaes, diffusion, autoregressive, maskedautoflow, restrictedboltzmann, gaussianmixture).",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--artifacts",
        default=None,
        help="Override artifacts root (default: from config.paths.artifacts or ./artifacts)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.set_defaults(cmd="train")

    p_synth = sub.add_parser("synth", help="Generate class-conditional samples")
    p_synth.add_argument("--seed", type=int, default=42, help="Seed to record in manifest.")
    p_synth.set_defaults(cmd="synth")

    p_eval = sub.add_parser("eval", help="Evaluate samples and downstream utility")
    p_eval.add_argument("--no-synth", action="store_true", help="Skip sample-based metrics.")
    p_eval.add_argument("--per-class-cap", type=int, default=200, help="Cap per-class images for metrics.")
    p_eval.add_argument("--encoder", default=None, help="(Optional) domain encoder name.")
    p_eval.set_defaults(cmd="eval")

    args = parser.parse_args(argv)

    # Detect repo module and load config
    repo_mod = _detect_repo_module(args.model)
    cfg = _load_yaml(args.config)

    # Provide config path to downstream (scripts may want it)
    cfg["__path__"] = os.path.abspath(args.config)

    # Resolve artifacts root
    artifacts_root = args.artifacts or cfg.get("paths", {}).get("artifacts", "artifacts")
    p = Paths(model=repo_mod, artifacts_root=artifacts_root)

    # Create standard dirs up-front
    _ensure_dir(p.synthetic_root)
    _ensure_dir(p.summaries_root)

    info(f"repo_module={repo_mod} | artifacts_root={artifacts_root} | command={args.cmd}")

    if args.cmd == "train":
        cmd_train(cfg, p, repo_mod)
    elif args.cmd == "synth":
        # Prefer CLI seed; else first from config.random_seeds; else 42
        seed = int(args.seed)
        if seed is None:
            seed = int((cfg.get("random_seeds") or [42])[0])
        cmd_synth(cfg, p, repo_mod, seed=seed)
    elif args.cmd == "eval":
        cmd_eval(
            cfg,
            p,
            repo_mod,
            no_synth=args.no_synth,
            per_class_cap=int(args.per_class_cap),
            encoder_name=args.encoder,
        )
    else:
        parser.error(f"Unknown command: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
