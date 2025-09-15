# restrictedboltzmann/__init__.py

"""
Restricted Boltzmann Machine (RBM) baseline package.

What this package provides
--------------------------
- `RestrictedBoltzmannPipeline`: a small, production-friendly wrapper that:
    • trains an RBM (optionally class-conditional) on flattened/binarized images
    • synthesizes samples (e.g., via Gibbs sampling / CD-k)
    • saves artifacts in a layout compatible with the shared eval suite
- `make_pipeline(cfg)`: convenience constructor returning a pipeline instance
  from a simple config dict.

Design notes
------------
- We *lazily import* the actual implementation from `restrictedboltzmann.pipeline`
  to keep import latency and dependency surface small (TensorFlow, etc.).
- The pipeline is designed to duck-type `common.interfaces.GenerativePipeline`
  but does not hard-depend on it.

Typical usage
-------------
>>> from restrictedboltzmann import make_pipeline
>>> pipe = make_pipeline(cfg)
>>> pipe.train(x_train, y_train)
>>> x_s, y_s = pipe.synthesize()
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = ["RestrictedBoltzmannPipeline", "make_pipeline", "__version__"]
__version__ = "0.1.0"

# Hold a deferred import error (if any) so we can re-raise with context later.
_IMPORT_ERROR: Exception | None = None


def _load_pipeline_class():
    """
    Import and return `RestrictedBoltzmannPipeline` from
    `restrictedboltzmann.pipeline`. This is done lazily to avoid importing
    heavy deps unless actually needed.
    """
    global _IMPORT_ERROR
    try:
        mod = import_module(".pipeline", __name__)
        return getattr(mod, "RestrictedBoltzmannPipeline")
    except Exception as e:
        _IMPORT_ERROR = e
        raise


def make_pipeline(cfg: Dict[str, Any]):
    """
    Convenience constructor.

    Args
    ----
    cfg : dict
        Configuration mapping (paths, shapes, CD-k steps, samples_per_class, etc.)

    Returns
    -------
    RestrictedBoltzmannPipeline
        A ready-to-use pipeline instance.
    """
    cls = _load_pipeline_class()
    return cls(cfg)


# Try to expose the symbol eagerly. If it fails, keep a lazy fallback via __getattr__.
try:
    RestrictedBoltzmannPipeline = _load_pipeline_class()  # type: ignore[assignment]
except Exception:
    RestrictedBoltzmannPipeline = None  # type: ignore[assignment]


def __getattr__(name: str):
    """
    Provide `RestrictedBoltzmannPipeline` on demand if eager import above failed.
    """
    if name == "RestrictedBoltzmannPipeline":
        if RestrictedBoltzmannPipeline is None:
            # Re-raise the original import error with clearer context if available.
            if _IMPORT_ERROR is not None:
                raise ImportError(
                    "Failed to import 'RestrictedBoltzmannPipeline' from "
                    "'restrictedboltzmann.pipeline'. Ensure 'restrictedboltzmann/pipeline.py' "
                    "exists and its dependencies (e.g., TensorFlow) are installed."
                ) from _IMPORT_ERROR
            raise ImportError(
                "RestrictedBoltzmannPipeline is unavailable. "
                "Check that 'restrictedboltzmann/pipeline.py' is present."
            )
        return RestrictedBoltzmannPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
