"""
Orbax Checkpoint Manager for Model Zoo

Handles versioned, atomic saving/restoring of model artifacts
using Orbax's CheckpointManager with composite saves:
  - model: parameter PyTrees (ndarray dicts or Flax TrainStates)
  - metadata: architecture config, noise model, training info (JSON)

Section 7 §3.2: Composite checkpoints with retention policies.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ocp = None
    ORBAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


def _check_orbax():
    if not ORBAX_AVAILABLE:
        raise ImportError(
            "orbax-checkpoint is required for Model Zoo checkpointing. "
            "Install with: pip install unified-qec[zoo]"
        )


class ModelCheckpoint:
    """Versioned model checkpoint manager using Orbax.

    Saves and restores composite checkpoints containing model weights
    and JSON metadata. Supports retention policies, atomic writes,
    and metadata-only restoration for efficient model inspection.

    Parameters
    ----------
    path : str or Path
        Directory for checkpoints.
    max_to_keep : int
        Maximum number of checkpoints to retain.
    save_interval_steps : int
        Minimum step interval between saves (to avoid excessive I/O).

    Examples
    --------
    >>> ckpt = ModelCheckpoint("./checkpoints", max_to_keep=5)
    >>> ckpt.save(step=100, model_state=params, metadata={"d": 5, "p": 0.001})
    >>> state, meta = ckpt.restore()  # latest
    >>> meta_only = ckpt.restore(metadata_only=True)  # no weight loading
    """

    def __init__(
        self,
        path: str,
        max_to_keep: int = 5,
        save_interval_steps: int = 1,
    ):
        _check_orbax()

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

        # Metadata sidecar directory
        self._meta_dir = self.path / "_metadata"
        self._meta_dir.mkdir(exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        self._manager = ocp.CheckpointManager(
            self.path,
            options=options,
        )

    def save(
        self,
        step: int,
        model_state: Any,
        metadata: Dict[str, Any],
    ) -> None:
        """Save a composite checkpoint.

        Parameters
        ----------
        step : int
            Training step or epoch number.
        model_state : Any
            Model parameters — JAX PyTree, dict of ndarrays,
            or Flax TrainState.
        metadata : dict
            Architecture config, noise model, training info.
            Must be JSON-serializable.
        """
        # Save model state via Orbax
        self._manager.save(
            step,
            args=ocp.args.StandardSave(model_state),
        )

        # Save metadata as JSON sidecar
        meta_path = self._meta_dir / f"step_{step:08d}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

    def restore(
        self,
        step: Optional[int] = None,
        metadata_only: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Restore a checkpoint.

        Parameters
        ----------
        step : int, optional
            Specific step to restore. Default: latest.
        metadata_only : bool
            If True, only load JSON metadata without weight tensors.

        Returns
        -------
        tuple of (model_state, metadata)
            model_state is None if metadata_only=True.
        """
        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.path}"
                )

        # Load metadata
        meta_path = self._meta_dir / f"step_{step:08d}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if metadata_only:
            return None, metadata

        # Load model state
        model_state = self._manager.restore(step)

        return model_state, metadata

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpoint step, or None if empty."""
        steps = self._manager.all_steps()
        return max(steps) if steps else None

    def all_steps(self):
        """Return all available checkpoint steps."""
        return sorted(self._manager.all_steps())

    def best_step(self, metric_key: str = "loss", minimize: bool = True):
        """Find the best checkpoint by a metadata metric.

        Parameters
        ----------
        metric_key : str
            Key in metadata dict to compare.
        minimize : bool
            If True, find minimum; otherwise maximum.

        Returns
        -------
        int or None
            Step number of the best checkpoint.
        """
        best = None
        best_val = float("inf") if minimize else float("-inf")

        for step in self.all_steps():
            _, meta = self.restore(step, metadata_only=True)
            val = meta.get(metric_key)
            if val is None:
                continue
            if minimize and val < best_val:
                best_val = val
                best = step
            elif not minimize and val > best_val:
                best_val = val
                best = step

        return best


class NumpyCheckpoint:
    """Lightweight checkpoint manager using NumPy .npz files.

    For users who don't need JAX/Orbax but want versioned model saving.
    Saves parameter dicts as compressed .npz with JSON metadata sidecars.

    Parameters
    ----------
    path : str or Path
        Directory for checkpoints.
    max_to_keep : int
        Maximum checkpoints to retain.
    """

    def __init__(self, path: str, max_to_keep: int = 5):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._meta_dir = self.path / "_metadata"
        self._meta_dir.mkdir(exist_ok=True)
        self.max_to_keep = max_to_keep

    def save(
        self,
        step: int,
        model_state: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """Save parameters as .npz and metadata as JSON."""
        npz_path = self.path / f"step_{step:08d}.npz"
        np.savez_compressed(npz_path, **model_state)

        meta_path = self._meta_dir / f"step_{step:08d}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        self._enforce_retention()

    def restore(
        self,
        step: Optional[int] = None,
        metadata_only: bool = False,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Restore .npz weights and/or JSON metadata."""
        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.path}"
                )

        meta_path = self._meta_dir / f"step_{step:08d}.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        if metadata_only:
            return None, metadata

        npz_path = self.path / f"step_{step:08d}.npz"
        data = dict(np.load(npz_path))
        return data, metadata

    def latest_step(self) -> Optional[int]:
        """Return latest checkpoint step."""
        steps = self._list_steps()
        return max(steps) if steps else None

    def all_steps(self):
        """Return all available steps."""
        return sorted(self._list_steps())

    def _list_steps(self):
        steps = []
        for p in self.path.glob("step_*.npz"):
            try:
                steps.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return steps

    def _enforce_retention(self):
        """Delete oldest checkpoints beyond max_to_keep."""
        steps = sorted(self._list_steps())
        while len(steps) > self.max_to_keep:
            old = steps.pop(0)
            npz = self.path / f"step_{old:08d}.npz"
            meta = self._meta_dir / f"step_{old:08d}.json"
            npz.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
