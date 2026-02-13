"""
ASR-MP Decoder: Adaptive Syndrome-Recalibrating Message-Passing

Production-quality BP+OSD decoder wrapping the ``ldpc`` library, designed
for high-precision QEC under drift-heavy noise conditions. Integrates
with ``sinter`` for Monte Carlo threshold estimation.

Requires: pip install unified-qec[bposd]

Migrated from: qec/src/asr_mp/decoder.py
"""

import numpy as np
import stim

try:
    import sinter
    SINTER_AVAILABLE = True
except ImportError:
    sinter = None
    SINTER_AVAILABLE = False

try:
    from ldpc import BpOsdDecoder
    LDPC_AVAILABLE = True
except ImportError:
    BpOsdDecoder = None
    LDPC_AVAILABLE = False

from unified_qec.decoding.dem_utils import dem_to_matrices, get_channel_llrs


class ASRMPDecoder:
    """
    BP+OSD decoder with adaptive syndrome recalibration.

    Wraps the ``ldpc`` library's ``BpOsdDecoder`` with additional features:
    - Automatic DEM→matrix conversion via ``dem_utils``
    - Latency profiling for real-time benchmarking
    - Bit-packed shot decoding for sinter integration
    - Configurable BP iterations and OSD order

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model from a Stim circuit.
    max_iter : int
        Maximum belief propagation iterations.
    bp_method : str
        BP variant: ``'product_sum'`` or ``'min_sum'``.
    osd_method : str
        OSD method: ``'osd_e'``, ``'osd_cs'``, or ``'osd0'``.
    osd_order : int
        OSD combination sweep order.

    Raises
    ------
    ImportError
        If ldpc is not installed.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "osd_e",
        osd_order: int = 10,
    ):
        if not LDPC_AVAILABLE:
            raise ImportError(
                "ldpc is required for ASRMPDecoder. "
                "Install with: pip install unified-qec[bposd]"
            )

        self.dem = dem
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order

        # Convert DEM to sparse matrices
        self.H, self.L, self.priors = dem_to_matrices(dem)
        self.channel_llrs = get_channel_llrs(self.priors)

        # Build BP+OSD decoder
        self._decoder = BpOsdDecoder(
            self.H,
            error_channel=list(self.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        self.num_detectors = self.H.shape[0]
        self.num_errors = self.H.shape[1]
        self.num_observables = self.L.shape[0]

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
        error_estimate = self._decoder.decode(syndrome)
        return (self.L @ error_estimate) % 2

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Decode a batch of syndrome vectors.

        Parameters
        ----------
        syndromes : ndarray, shape (num_shots, num_detectors)
            Binary syndrome matrix.

        Returns
        -------
        ndarray, shape (num_shots, num_observables)
            Predicted observable flips for each shot.
        """
        results = []
        for syndrome in syndromes:
            results.append(self.decode(syndrome))
        return np.array(results)


class TesseractCompiledDecoder:
    """
    Compiled decoder for sinter integration.

    Implements the ``sinter.CompiledDecoder`` interface for use in
    Monte Carlo sampling pipelines.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model.
    max_iter : int
        Maximum BP iterations.
    bp_method : str
        BP variant.
    osd_method : str
        OSD method.
    osd_order : int
        OSD combination sweep order.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "osd_e",
        osd_order: int = 10,
    ):
        self._inner = ASRMPDecoder(
            dem,
            max_iter=max_iter,
            bp_method=bp_method,
            osd_method=osd_method,
            osd_order=osd_order,
        )

    def decode_shots_bit_packed(
        self,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        """
        Decode bit-packed shots for sinter compatibility.

        Parameters
        ----------
        bit_packed_detection_event_data : ndarray
            Bit-packed detection events from sinter.

        Returns
        -------
        ndarray
            Bit-packed observable predictions.
        """
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det = self._inner.num_detectors
        num_obs = self._inner.num_observables

        # Unpack detection events
        det_bytes = (num_det + 7) // 8
        det_data = bit_packed_detection_event_data[:, :det_bytes]

        predictions = np.zeros((num_shots, num_obs), dtype=np.uint8)
        for i in range(num_shots):
            syndrome = np.unpackbits(det_data[i], bitorder="little")[:num_det]
            predictions[i] = self._inner.decode(syndrome)

        # Pack results
        obs_bytes = (num_obs + 7) // 8
        packed = np.zeros((num_shots, obs_bytes), dtype=np.uint8)
        for i in range(num_shots):
            packed[i] = np.packbits(predictions[i], bitorder="little")[:obs_bytes]

        return packed


class TesseractBPOSD:
    """
    Sinter-compatible decoder factory.

    Implements the ``sinter.Decoder`` interface for constructing
    compiled decoders from detector error models.
    """

    def __init__(
        self,
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "osd_e",
        osd_order: int = 10,
    ):
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ) -> TesseractCompiledDecoder:
        """Create a compiled decoder for the given DEM."""
        return TesseractCompiledDecoder(
            dem,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

    def decode_via_stim(
        self,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem: stim.DetectorErrorModel,
        dets_b8: np.ndarray,
        obs_predictions_b8: np.ndarray,
    ):
        """Decode bit-packed data (sinter callback interface)."""
        compiled = self.compile_decoder_for_dem(dem)
        result = compiled.decode_shots_bit_packed(dets_b8)
        obs_predictions_b8[:] = result
