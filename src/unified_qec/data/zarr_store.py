"""
Zarr-Based Syndrome Data Store

Cloud-friendly storage for QEC syndrome datasets using Zarr arrays.
Supports parallel I/O, selective slice reading, and works with
local paths or S3/GCS URIs via fsspec.

Section 7 §4.2.2: Zarr for analytical random access.
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    zarr = None
    ZARR_AVAILABLE = False


def _check_zarr():
    if not ZARR_AVAILABLE:
        raise ImportError(
            "zarr is required for ZarrStore. "
            "Install with: pip install unified-qec[data]"
        )


class ZarrStore:
    """Cloud-friendly syndrome data store using Zarr.

    Stores syndrome vectors and observable labels as compressed,
    chunked N-dimensional arrays. Supports parallel writes,
    selective slice reading, and metadata attachment.

    Parameters
    ----------
    path : str or Path
        Path to the Zarr store (directory or cloud URI).
    mode : str
        File mode: ``'w'`` (create/overwrite), ``'a'`` (append),
        ``'r'`` (read-only).
    chunk_shots : int
        Number of shots per chunk along the shot dimension.
    compressor : str
        Compression codec: ``'blosc'``, ``'zlib'``, or ``'none'``.

    Examples
    --------
    Write:
    >>> store = ZarrStore("./syndromes.zarr", mode="w")
    >>> store.write_syndromes(syndromes, observables, metadata={"d": 5})
    >>> store.close()

    Read:
    >>> store = ZarrStore("./syndromes.zarr", mode="r")
    >>> chunk = store.read_slice(shot_range=(0, 1000))
    >>> meta = store.get_metadata()
    """

    def __init__(
        self,
        path: str,
        mode: str = "w",
        chunk_shots: int = 10000,
        compressor: str = "blosc",
    ):
        _check_zarr()

        self.path = str(path)
        self.mode = mode
        self.chunk_shots = chunk_shots

        # Select compressor
        if compressor == "blosc":
            self._compressor = zarr.Blosc(cname="zstd", clevel=3)
        elif compressor == "zlib":
            self._compressor = zarr.Zlib(level=3)
        else:
            self._compressor = None

        # Open or create store
        if mode == "r":
            self._root = zarr.open_group(self.path, mode="r")
        elif mode == "a":
            self._root = zarr.open_group(self.path, mode="a")
        else:
            self._root = zarr.open_group(self.path, mode="w")

        self._syndromes = None
        self._observables = None
        self._shot_count = 0

    def write_syndromes(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Write syndrome data to the Zarr store.

        Parameters
        ----------
        syndromes : ndarray, shape (num_shots, num_detectors)
            Detection event vectors (uint8 or bool).
        observables : ndarray, shape (num_shots, num_observables)
            Observable flip labels (uint8 or bool).
        metadata : dict, optional
            Descriptive metadata (code distance, noise model, etc.).

        Returns
        -------
        int
            Total number of shots written so far.
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        observables = np.asarray(observables, dtype=np.uint8)

        num_shots = syndromes.shape[0]
        num_det = syndromes.shape[1]
        num_obs = observables.shape[1]

        if self._syndromes is None and "syndromes" not in self._root:
            # Create arrays
            self._syndromes = self._root.create_dataset(
                "syndromes",
                shape=(num_shots, num_det),
                chunks=(min(self.chunk_shots, num_shots), num_det),
                dtype=np.uint8,
                compressor=self._compressor,
            )
            self._observables = self._root.create_dataset(
                "observables",
                shape=(num_shots, num_obs),
                chunks=(min(self.chunk_shots, num_shots), num_obs),
                dtype=np.uint8,
                compressor=self._compressor,
            )
            self._syndromes[:] = syndromes
            self._observables[:] = observables
            self._shot_count = num_shots
        else:
            # Append
            if self._syndromes is None:
                self._syndromes = self._root["syndromes"]
                self._observables = self._root["observables"]
                self._shot_count = self._syndromes.shape[0]

            old_len = self._shot_count
            new_len = old_len + num_shots
            self._syndromes.resize(new_len, self._syndromes.shape[1])
            self._observables.resize(new_len, self._observables.shape[1])
            self._syndromes[old_len:new_len] = syndromes
            self._observables[old_len:new_len] = observables
            self._shot_count = new_len

        # Store metadata
        if metadata:
            for key, val in metadata.items():
                if isinstance(val, (dict, list)):
                    self._root.attrs[key] = json.dumps(val)
                else:
                    self._root.attrs[key] = val
        self._root.attrs["total_shots"] = self._shot_count

        return self._shot_count

    def append(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
    ) -> int:
        """Append additional syndrome data.

        Parameters
        ----------
        syndromes : ndarray, shape (num_shots, num_detectors)
        observables : ndarray, shape (num_shots, num_observables)

        Returns
        -------
        int
            Total shots after append.
        """
        return self.write_syndromes(syndromes, observables)

    def read_slice(
        self,
        shot_range: Optional[Tuple[int, int]] = None,
        detector_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a slice of the dataset.

        Zarr fetches only the chunks covering the requested range,
        minimizing data transfer for cloud-backed stores.

        Parameters
        ----------
        shot_range : tuple (start, end), optional
            Range of shots to read. Default: all.
        detector_range : tuple (start, end), optional
            Range of detector columns. Default: all.

        Returns
        -------
        tuple of (syndromes, observables)
            Sliced arrays as numpy arrays.
        """
        syn = self._root["syndromes"]
        obs = self._root["observables"]

        s_slice = slice(*(shot_range or (None, None)))
        d_slice = slice(*(detector_range or (None, None)))

        return np.array(syn[s_slice, d_slice]), np.array(obs[s_slice])

    def read_failures(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read only shots where at least one observable flipped.

        Returns
        -------
        tuple of (syndromes, observables)
            Filtered to failure events only.
        """
        obs = np.array(self._root["observables"])
        syn = np.array(self._root["syndromes"])
        mask = obs.any(axis=1)
        return syn[mask], obs[mask]

    def get_metadata(self) -> Dict[str, Any]:
        """Return all stored metadata.

        Returns
        -------
        dict
            Metadata including total_shots, code distance, etc.
        """
        meta = dict(self._root.attrs)
        # Deserialize JSON-encoded complex types
        for key, val in meta.items():
            if isinstance(val, str) and val.startswith(("{", "[")):
                try:
                    meta[key] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        return meta

    def get_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Return shapes of stored arrays."""
        return {
            "syndromes": tuple(self._root["syndromes"].shape),
            "observables": tuple(self._root["observables"].shape),
        }

    def close(self):
        """Flush and close the store."""
        if hasattr(self._root.store, "close"):
            self._root.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
