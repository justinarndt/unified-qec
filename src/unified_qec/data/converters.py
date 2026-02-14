"""
Sinter-to-Data Converters

Bridges Sinter sampling output to Zarr and WebDataset storage formats.
Converts raw detection events and observable flips from sinter.collect()
or stim circuit sampling into cloud-native storage.

Section 7 §4: Open Data compliance.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import stim

try:
    import sinter
    SINTER_AVAILABLE = True
except ImportError:
    sinter = None
    SINTER_AVAILABLE = False

from unified_qec.data.zarr_store import ZarrStore
from unified_qec.data.webdataset_writer import WebDatasetWriter


class SinterToData:
    """Convert Sinter/Stim sampling output to Zarr and/or WebDataset.

    Bridges the gap between Sinter's Monte Carlo sampling and the
    Open Data storage formats required by Section 7.

    Examples
    --------
    From a Stim circuit:
    >>> converter = SinterToData()
    >>> converter.from_stim_samples(
    ...     circuit=circuit,
    ...     num_shots=100000,
    ...     output_path="./datasets/d5_p001",
    ...     formats=["zarr", "webdataset"],
    ... )

    From Sinter stats:
    >>> converter.from_sinter_stats(stats, "./datasets/benchmark")
    """

    def from_stim_samples(
        self,
        circuit: stim.Circuit,
        num_shots: int,
        output_path: str,
        formats: List[str] = None,
        batch_size: int = 10000,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Sample a Stim circuit and write to storage.

        Parameters
        ----------
        circuit : stim.Circuit
            Stim circuit with noise.
        num_shots : int
            Total number of shots to sample.
        output_path : str
            Base output directory.
        formats : list of str
            Storage formats: ``'zarr'``, ``'webdataset'``, or both.
        batch_size : int
            Number of shots per sampling batch.
        metadata : dict, optional
            Additional metadata to store.

        Returns
        -------
        dict
            Summary with paths and shot counts per format.
        """
        if formats is None:
            formats = ["zarr"]

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Auto-extract metadata from circuit
        if metadata is None:
            metadata = {}
        metadata.update({
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "num_qubits": circuit.num_qubits,
        })

        sampler = circuit.compile_detector_sampler()

        # Initialize writers
        zarr_store = None
        wds_writer = None

        if "zarr" in formats:
            zarr_store = ZarrStore(str(output_path / "syndromes.zarr"), mode="w")

        if "webdataset" in formats:
            wds_writer = WebDatasetWriter(
                str(output_path / "shards"),
                shard_size=batch_size,
            )

        # Sample in batches
        total_written = 0
        remaining = num_shots

        while remaining > 0:
            batch = min(batch_size, remaining)
            det_events, obs_flips = sampler.sample(
                batch, separate_observables=True
            )

            det_events = det_events.astype(np.uint8)
            obs_flips = obs_flips.astype(np.uint8)

            if zarr_store is not None:
                zarr_store.write_syndromes(
                    det_events, obs_flips,
                    metadata=metadata if total_written == 0 else None,
                )

            if wds_writer is not None:
                wds_writer.add_batch(det_events, obs_flips, metadata)

            total_written += batch
            remaining -= batch

        # Finalize
        summary = {"total_shots": total_written, "paths": {}}

        if zarr_store is not None:
            zarr_store.close()
            summary["paths"]["zarr"] = str(output_path / "syndromes.zarr")

        if wds_writer is not None:
            wds_writer.close()
            summary["paths"]["webdataset"] = str(output_path / "shards")
            summary["num_shards"] = wds_writer.num_shards

        return summary

    def from_sinter_stats(
        self,
        stats: list,
        output_path: str,
        formats: List[str] = None,
    ) -> dict:
        """Convert Sinter task stats to storage formats.

        Re-samples the circuits from the stats to generate the actual
        detection event data (since sinter.TaskStats only stores
        aggregate counts, not raw shots).

        Parameters
        ----------
        stats : list of sinter.TaskStats
            Output from sinter.collect().
        output_path : str
            Base output directory.
        formats : list of str
            Storage formats.

        Returns
        -------
        dict
            Summary with paths and shot counts.
        """
        if not SINTER_AVAILABLE:
            raise ImportError(
                "sinter is required for from_sinter_stats. "
                "Install with: pip install unified-qec[bposd]"
            )

        if formats is None:
            formats = ["zarr"]

        output_path = Path(output_path)
        results = {}

        for i, stat in enumerate(stats):
            meta = stat.json_metadata or {}
            task_dir = output_path / f"task_{i:04d}"
            task_dir.mkdir(parents=True, exist_ok=True)

            # Re-sample the circuit
            circuit = stat.circuit
            if circuit is None:
                continue

            num_shots = min(stat.shots, 100000)  # cap for storage

            result = self.from_stim_samples(
                circuit=circuit,
                num_shots=num_shots,
                output_path=str(task_dir),
                formats=formats,
                metadata={
                    "sinter_task_index": i,
                    "original_shots": stat.shots,
                    "original_errors": stat.errors,
                    **meta,
                },
            )
            results[f"task_{i:04d}"] = result

        return results

    def from_detection_events(
        self,
        det_events: np.ndarray,
        obs_flips: np.ndarray,
        output_path: str,
        formats: List[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Write pre-sampled detection events to storage.

        Parameters
        ----------
        det_events : ndarray, shape (num_shots, num_detectors)
            Detection event data.
        obs_flips : ndarray, shape (num_shots, num_observables)
            Observable flip data.
        output_path : str
            Base output directory.
        formats : list of str
            Storage formats.
        metadata : dict, optional
            Metadata to store.

        Returns
        -------
        dict
            Summary with paths.
        """
        if formats is None:
            formats = ["zarr"]

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        det_events = np.asarray(det_events, dtype=np.uint8)
        obs_flips = np.asarray(obs_flips, dtype=np.uint8)

        if metadata is None:
            metadata = {}
        metadata.update({
            "num_detectors": det_events.shape[1],
            "num_observables": obs_flips.shape[1],
            "num_shots": det_events.shape[0],
        })

        summary = {"total_shots": det_events.shape[0], "paths": {}}

        if "zarr" in formats:
            zarr_path = str(output_path / "syndromes.zarr")
            with ZarrStore(zarr_path, mode="w") as store:
                store.write_syndromes(det_events, obs_flips, metadata)
            summary["paths"]["zarr"] = zarr_path

        if "webdataset" in formats:
            shard_path = str(output_path / "shards")
            with WebDatasetWriter(shard_path) as writer:
                writer.add_batch(det_events, obs_flips, metadata)
            summary["paths"]["webdataset"] = shard_path
            summary["num_shards"] = 1

        return summary
