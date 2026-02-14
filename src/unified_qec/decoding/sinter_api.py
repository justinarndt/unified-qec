"""
Unified Sinter Decoder API

Pickle-safe sinter.Decoder factory that routes to the correct backend
(BP+OSD, Union-Find, PyMatching, or neural) based on a config string.
All heavy initialization (JAX, model weights, XLA compilation) is
deferred to compile_decoder_for_dem so the factory can be pickled
across Sinter worker processes.

Section 7 §2.2: compile_decoder_for_dem amortizes compilation cost.
Section 7 §2.2.3: decode_shots_bit_packed uses uint8 bit-packed I/O.
"""

import numpy as np
import stim

try:
    import sinter
    SINTER_AVAILABLE = True
except ImportError:
    sinter = None
    SINTER_AVAILABLE = False


class _PyMatchingCompiledDecoder:
    """Compiled decoder wrapping PyMatching for sinter compatibility."""

    def __init__(self, dem: stim.DetectorErrorModel):
        import pymatching
        self._matcher = pymatching.Matching.from_detector_error_model(dem)
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

    def decode_shots_bit_packed(
        self,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        """Decode bit-packed shots using PyMatching MWPM."""
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det = self.num_detectors
        num_obs = self.num_observables
        det_bytes = (num_det + 7) // 8
        det_data = bit_packed_detection_event_data[:, :det_bytes]

        predictions = np.zeros((num_shots, num_obs), dtype=np.uint8)
        for i in range(num_shots):
            syndrome = np.unpackbits(det_data[i], bitorder="little")[:num_det]
            predictions[i] = self._matcher.decode(syndrome)

        obs_bytes = (num_obs + 7) // 8
        packed = np.zeros((num_shots, obs_bytes), dtype=np.uint8)
        for i in range(num_shots):
            packed[i] = np.packbits(predictions[i], bitorder="little")[:obs_bytes]
        return packed


class _NeuralCompiledDecoder:
    """Compiled decoder using a JAX/Flax neural network.

    Loads weights from the Model Zoo (Orbax checkpoint or HuggingFace)
    and JIT-compiles the inference graph for the specific DEM shape.
    """

    def __init__(self, dem: stim.DetectorErrorModel, model_path: str, **kwargs):
        from unified_qec.zoo.neural_decoder import NeuralSyndromeDecoder
        from unified_qec.zoo.checkpoint import ModelCheckpoint

        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        # Load model from checkpoint
        ckpt = ModelCheckpoint(model_path)
        state, metadata = ckpt.restore()

        # Build model from metadata and load weights
        self._model = NeuralSyndromeDecoder(
            num_detectors=self.num_detectors,
            num_observables=self.num_observables,
            hidden_dim=metadata.get("hidden_dim", 256),
            num_layers=metadata.get("num_layers", 4),
        )
        self._params = state

        # JIT compile inference
        try:
            import jax
            import jax.numpy as jnp
            self._jit_infer = jax.jit(self._model.apply)
            self._jnp = jnp
            self._np_to_device = jax.device_put
        except ImportError:
            raise ImportError(
                "JAX is required for neural decoder. "
                "Install with: pip install unified-qec[jax]"
            )

    def decode_shots_bit_packed(
        self,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        """Decode bit-packed shots via JIT-compiled neural network."""
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det = self.num_detectors
        num_obs = self.num_observables
        det_bytes = (num_det + 7) // 8
        det_data = bit_packed_detection_event_data[:, :det_bytes]

        # Unpack to float32 on device
        syndromes_np = np.zeros((num_shots, num_det), dtype=np.float32)
        for i in range(num_shots):
            syndromes_np[i] = np.unpackbits(
                det_data[i], bitorder="little"
            )[:num_det].astype(np.float32)

        syndromes_device = self._np_to_device(syndromes_np)

        # JIT inference
        logits = self._jit_infer(self._params, syndromes_device)
        predictions = np.array(logits > 0.0, dtype=np.uint8)

        # Pack results
        obs_bytes = (num_obs + 7) // 8
        packed = np.zeros((num_shots, obs_bytes), dtype=np.uint8)
        for i in range(num_shots):
            packed[i] = np.packbits(predictions[i], bitorder="little")[:obs_bytes]
        return packed


class UnifiedQECDecoder:
    """Pickle-safe unified decoder factory for Sinter.

    Implements the ``sinter.Decoder`` interface. The ``__init__`` method
    stores only primitive-typed configuration (strings, ints, dicts) so
    the object can be pickled across multiprocessing workers.

    Heavy initialization — loading model weights, JIT compilation,
    graph construction — is deferred to ``compile_decoder_for_dem``,
    which is called once per DEM inside the worker process.

    Parameters
    ----------
    backend : str
        Decoder backend: ``'bposd'``, ``'uf'``, ``'pymatching'``,
        or ``'neural'``.
    model_path : str, optional
        Path to Orbax checkpoint or HuggingFace repo for neural backend.
    max_iter : int
        Maximum BP iterations (bposd only).
    bp_method : str
        BP variant (bposd only).
    osd_method : str
        OSD method (bposd only).
    osd_order : int
        OSD combination sweep order (bposd only).

    Raises
    ------
    ImportError
        If sinter is not installed.
    ValueError
        If ``backend`` is not recognized.
    """

    def __init__(
        self,
        backend: str = "bposd",
        model_path: str = "",
        max_iter: int = 50,
        bp_method: str = "product_sum",
        osd_method: str = "osd_e",
        osd_order: int = 10,
    ):
        if not SINTER_AVAILABLE:
            raise ImportError(
                "sinter is required for UnifiedQECDecoder. "
                "Install with: pip install unified-qec[bposd]"
            )

        valid_backends = {"bposd", "uf", "pymatching", "neural"}
        if backend not in valid_backends:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: {sorted(valid_backends)}"
            )

        # Only primitives — pickle safe
        self.backend = backend
        self.model_path = model_path
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.osd_method = osd_method
        self.osd_order = osd_order

    def compile_decoder_for_dem(
        self, dem: stim.DetectorErrorModel
    ):
        """Compile a decoder for the given detector error model.

        This is called once per DEM inside the Sinter worker process.
        All heavy initialization happens here, amortized over millions
        of subsequent decode calls.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            Detector error model from a Stim circuit.

        Returns
        -------
        CompiledDecoder
            A compiled decoder implementing ``decode_shots_bit_packed``.
        """
        if self.backend == "bposd":
            from unified_qec.decoding.asr_mp import TesseractCompiledDecoder
            return TesseractCompiledDecoder(
                dem,
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                osd_method=self.osd_method,
                osd_order=self.osd_order,
            )

        elif self.backend == "uf":
            from unified_qec.decoding.union_find import UnionFindCompiledDecoder
            return UnionFindCompiledDecoder(dem)

        elif self.backend == "pymatching":
            return _PyMatchingCompiledDecoder(dem)

        elif self.backend == "neural":
            if not self.model_path:
                raise ValueError(
                    "model_path is required for neural backend. "
                    "Provide a local Orbax checkpoint path or "
                    "HuggingFace repo ID."
                )
            return _NeuralCompiledDecoder(
                dem,
                model_path=self.model_path,
                max_iter=self.max_iter,
            )

        raise ValueError(f"Unhandled backend: {self.backend}")

    def __getstate__(self):
        """Return pickle-safe state (primitives only)."""
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "max_iter": self.max_iter,
            "bp_method": self.bp_method,
            "osd_method": self.osd_method,
            "osd_order": self.osd_order,
        }

    def __setstate__(self, state):
        """Restore from pickled state."""
        self.backend = state["backend"]
        self.model_path = state["model_path"]
        self.max_iter = state["max_iter"]
        self.bp_method = state["bp_method"]
        self.osd_method = state["osd_method"]
        self.osd_order = state["osd_order"]
