# common/interfaces.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# -----------------------------------------------------------------------------
# Callback type used by some pipelines (epoch, train_loss, val_loss)
# For non-epoch logging, you can ignore or repurpose the arguments.
# -----------------------------------------------------------------------------
LogCallback = Callable[[int, Optional[float], Optional[float]], None]


# -----------------------------------------------------------------------------
# Minimal interface for a generative pipeline:
#   - train(...)  -> fit the model(s), optionally saving checkpoints
#   - synthesize(...) -> return (x_synth, y_synth) in shapes compatible with evaluator
# Attributes are optional but commonly present across our pipelines.
# -----------------------------------------------------------------------------
@runtime_checkable
class GenerativePipeline(Protocol):
    # Common attributes (not strictly required by all implementations)
    img_shape: Tuple[int, int, int]
    num_classes: int
    ckpt_dir: Path
    synth_dir: Path

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *args,
        **kwargs,
    ) -> Any:
        """
        Fit the model(s) on training data.
        Implementations may save checkpoints to self.ckpt_dir.
        Return value is implementation-specific (e.g., fitted model or bundle).
        """
        ...

    def synthesize(
        self,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples.

        Returns
        -------
        x_synth : float32, shape (N, H, W, C), values in [0,1]
        y_synth : float32, shape (N, K), one-hot labels
        """
        ...
