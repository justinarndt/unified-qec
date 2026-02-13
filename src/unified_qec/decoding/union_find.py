"""
Union-Find Decoder (Local Clustering)

Baseline decoder using the ``fusion-blossom`` library for comparison
benchmarks against BP+OSD. Currently implements the sinter interface
with a placeholder decoding backend.

Requires: pip install unified-qec[uf]

Migrated from: qec/src/asr_mp/union_find_decoder.py
"""

import numpy as np
import stim

try:
    import fusion_blossom
    FUSION_AVAILABLE = True
except ImportError:
    fusion_blossom = None
    FUSION_AVAILABLE = False

try:
    import sinter
    SINTER_AVAILABLE = True
except ImportError:
    sinter = None
    SINTER_AVAILABLE = False


class UnionFindDecoder:
    """
    Union-Find / Local Clustering decoder.

    Serves as a baseline decoder with O(n·α(n)) complexity per shot
    for comparison against the BP+OSD decoder.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model from a Stim circuit.

    Raises
    ------
    ImportError
        If fusion-blossom is not installed.

    Notes
    -----
    The actual decoding logic is currently a placeholder that returns
    zero predictions. The sinter integration structure is complete
    and ready for a full implementation.
    """

    def __init__(self, dem: stim.DetectorErrorModel):
        if not FUSION_AVAILABLE:
            raise ImportError(
                "fusion-blossom is required for UnionFindDecoder. "
                "Install with: pip install unified-qec[uf]"
            )
        self.dem = dem
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode a single syndrome vector.

        Parameters
        ----------
        syndrome : ndarray, shape (num_detectors,)
            Binary syndrome vector.

        Returns
        -------
        ndarray, shape (num_observables,)
            Predicted observable flips.
        """
        # Placeholder: full implementation would use fusion-blossom
        return np.zeros(self.num_observables, dtype=np.uint8)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndrome vectors."""
        results = []
        for syndrome in syndromes:
            results.append(self.decode(syndrome))
        return np.array(results)


class UnionFindCompiledDecoder:
    """Compiled decoder for sinter integration."""

    def __init__(self, dem: stim.DetectorErrorModel):
        self._inner = UnionFindDecoder(dem)

    def decode_shots_bit_packed(
        self,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        """Decode bit-packed shots for sinter compatibility."""
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det = self._inner.num_detectors
        num_obs = self._inner.num_observables

        det_bytes = (num_det + 7) // 8
        det_data = bit_packed_detection_event_data[:, :det_bytes]

        predictions = np.zeros((num_shots, num_obs), dtype=np.uint8)
        for i in range(num_shots):
            syndrome = np.unpackbits(det_data[i], bitorder="little")[:num_det]
            predictions[i] = self._inner.decode(syndrome)

        obs_bytes = (num_obs + 7) // 8
        packed = np.zeros((num_shots, obs_bytes), dtype=np.uint8)
        for i in range(num_shots):
            packed[i] = np.packbits(predictions[i], bitorder="little")[:obs_bytes]

        return packed


class UnionFindSinterDecoder:
    """Sinter-compatible decoder factory for Union-Find."""

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ) -> UnionFindCompiledDecoder:
        """Create a compiled decoder for the given DEM."""
        return UnionFindCompiledDecoder(dem)
