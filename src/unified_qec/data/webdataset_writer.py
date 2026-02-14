"""
WebDataset Writer for Streaming ML Training

Writes syndrome data as sharded POSIX tar archives for efficient
streaming to ML training pipelines. Each shard contains thousands
of samples, avoiding inode exhaustion and enabling line-rate
streaming from object storage.

Section 7 §4.2.1: WebDataset for massive training pipelines.
"""

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class WebDatasetWriter:
    """Write syndrome data as sharded tar archives.

    Each shard is a standard .tar file containing numbered samples,
    each with a .syndrome.npy, .observable.npy, and optional .json
    metadata file. Compatible with the ``webdataset`` library for
    streaming reads in JAX/PyTorch data loaders.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output shards.
    shard_size : int
        Maximum number of samples per shard.
    prefix : str
        Filename prefix for shards.
    compress : bool
        If True, write .tar.gz shards (slightly slower but smaller).

    Examples
    --------
    >>> writer = WebDatasetWriter("./shards", shard_size=10000)
    >>> for syndrome, obs in data:
    ...     writer.add_sample(syndrome, obs, {"p": 0.001})
    >>> writer.close()
    >>> # Produces: shards/surface_000000.tar, shards/surface_000001.tar, ...
    """

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 10000,
        prefix: str = "surface",
        compress: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self.compress = compress

        self._shard_idx = 0
        self._sample_idx = 0
        self._total_samples = 0
        self._current_tar = None
        self._open_shard()

    def _shard_path(self, idx: int) -> Path:
        ext = ".tar.gz" if self.compress else ".tar"
        return self.output_dir / f"{self.prefix}_{idx:06d}{ext}"

    def _open_shard(self):
        """Open a new shard for writing."""
        if self._current_tar is not None:
            self._current_tar.close()

        path = self._shard_path(self._shard_idx)
        mode = "w:gz" if self.compress else "w"
        self._current_tar = tarfile.open(path, mode)
        self._sample_idx = 0

    def add_sample(
        self,
        syndrome: np.ndarray,
        observable: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single sample to the current shard.

        Parameters
        ----------
        syndrome : ndarray, shape (num_detectors,)
            Detection event vector.
        observable : ndarray, shape (num_observables,)
            Observable flip labels.
        metadata : dict, optional
            Sample-level metadata (error rate, distance, etc.).
        """
        sample_key = f"{self._total_samples:010d}"

        # Syndrome
        self._add_npy(f"{sample_key}.syndrome.npy", syndrome)

        # Observable
        self._add_npy(f"{sample_key}.observable.npy", observable)

        # Metadata
        if metadata:
            self._add_json(f"{sample_key}.json", metadata)

        self._sample_idx += 1
        self._total_samples += 1

        # Rotate shard if full
        if self._sample_idx >= self.shard_size:
            self._shard_idx += 1
            self._open_shard()

    def add_batch(
        self,
        syndromes: np.ndarray,
        observables: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a batch of samples.

        Parameters
        ----------
        syndromes : ndarray, shape (num_shots, num_detectors)
        observables : ndarray, shape (num_shots, num_observables)
        metadata : dict, optional
            Shared metadata for all samples in the batch.
        """
        for i in range(syndromes.shape[0]):
            self.add_sample(syndromes[i], observables[i], metadata)

    def _add_npy(self, name: str, array: np.ndarray):
        """Add a numpy array to the current tar archive."""
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        info = tarfile.TarInfo(name=name)
        info.size = buf.getbuffer().nbytes
        self._current_tar.addfile(info, buf)

    def _add_json(self, name: str, data: dict):
        """Add a JSON metadata file to the current tar archive."""
        content = json.dumps(data, default=_json_default).encode("utf-8")
        buf = io.BytesIO(content)
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        self._current_tar.addfile(info, buf)

    def close(self):
        """Close the current shard and finalize."""
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None

        # Write manifest
        manifest = {
            "total_samples": self._total_samples,
            "num_shards": self._shard_idx + 1,
            "shard_size": self.shard_size,
            "prefix": self.prefix,
            "compressed": self.compress,
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def num_shards(self) -> int:
        return self._shard_idx + 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class WebDatasetReader:
    """Read syndrome data from WebDataset shards.

    Streams samples from tar archives without extracting to disk.
    Supports optional shuffling via in-memory buffer.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .tar or .tar.gz shards.
    shuffle_buffer : int
        Size of in-memory shuffle buffer. 0 = no shuffling.
    """

    def __init__(self, input_dir: str, shuffle_buffer: int = 0):
        self.input_dir = Path(input_dir)
        self.shuffle_buffer = shuffle_buffer

        # Read manifest
        manifest_path = self.input_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {}

        # Find shards
        self._shards = sorted(
            list(self.input_dir.glob("*.tar"))
            + list(self.input_dir.glob("*.tar.gz"))
        )

    def __iter__(self):
        """Iterate over samples, yielding (syndrome, observable, metadata)."""
        buffer = []
        rng = np.random.default_rng()

        for shard_path in self._shards:
            mode = "r:gz" if str(shard_path).endswith(".gz") else "r"
            with tarfile.open(shard_path, mode) as tar:
                members = tar.getmembers()

                # Group by sample key
                samples = {}
                for m in members:
                    key = m.name.split(".")[0]
                    if key not in samples:
                        samples[key] = {}
                    if m.name.endswith(".syndrome.npy"):
                        f = tar.extractfile(m)
                        samples[key]["syndrome"] = np.load(io.BytesIO(f.read()))
                    elif m.name.endswith(".observable.npy"):
                        f = tar.extractfile(m)
                        samples[key]["observable"] = np.load(io.BytesIO(f.read()))
                    elif m.name.endswith(".json"):
                        f = tar.extractfile(m)
                        samples[key]["metadata"] = json.loads(f.read())

                for key in sorted(samples.keys()):
                    s = samples[key]
                    item = (
                        s.get("syndrome", np.array([])),
                        s.get("observable", np.array([])),
                        s.get("metadata", {}),
                    )

                    if self.shuffle_buffer > 0:
                        buffer.append(item)
                        if len(buffer) >= self.shuffle_buffer:
                            rng.shuffle(buffer)
                            yield from buffer
                            buffer.clear()
                    else:
                        yield item

        # Flush remaining buffer
        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def __len__(self):
        return self._manifest.get("total_samples", 0)


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
