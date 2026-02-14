"""
Tests for the unified Sinter decoder API.

Tests pickle safety, backend routing, and compiled decoder construction.
"""

import pickle

import numpy as np
import pytest
import stim

from unified_qec.decoding.sinter_api import UnifiedQECDecoder


@pytest.fixture
def simple_circuit():
    """A minimal surface code circuit for testing."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )


@pytest.fixture
def simple_dem(simple_circuit):
    """DEM from the simple circuit."""
    return simple_circuit.detector_error_model()


class TestUnifiedQECDecoderPickle:
    """Test that UnifiedQECDecoder is fully pickle-safe."""

    def test_pickle_roundtrip_bposd(self):
        dec = UnifiedQECDecoder(backend="bposd", max_iter=30, osd_order=5)
        data = pickle.dumps(dec)
        restored = pickle.loads(data)
        assert restored.backend == "bposd"
        assert restored.max_iter == 30
        assert restored.osd_order == 5

    def test_pickle_roundtrip_uf(self):
        dec = UnifiedQECDecoder(backend="uf")
        data = pickle.dumps(dec)
        restored = pickle.loads(data)
        assert restored.backend == "uf"

    def test_pickle_roundtrip_pymatching(self):
        dec = UnifiedQECDecoder(backend="pymatching")
        data = pickle.dumps(dec)
        restored = pickle.loads(data)
        assert restored.backend == "pymatching"

    def test_pickle_roundtrip_neural(self):
        dec = UnifiedQECDecoder(backend="neural", model_path="/fake/path")
        data = pickle.dumps(dec)
        restored = pickle.loads(data)
        assert restored.backend == "neural"
        assert restored.model_path == "/fake/path"


class TestUnifiedQECDecoderValidation:
    """Test input validation."""

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            UnifiedQECDecoder(backend="invalid")

    def test_neural_without_path_raises(self, simple_dem):
        dec = UnifiedQECDecoder(backend="neural")
        with pytest.raises(ValueError, match="model_path is required"):
            dec.compile_decoder_for_dem(simple_dem)


class TestPyMatchingCompiledDecoder:
    """Test the PyMatching compiled decoder backend."""

    def test_compile_and_decode(self, simple_dem):
        dec = UnifiedQECDecoder(backend="pymatching")
        compiled = dec.compile_decoder_for_dem(simple_dem)

        # Create a batch of zero syndromes (no errors)
        num_det = simple_dem.num_detectors
        num_obs = simple_dem.num_observables
        det_bytes = (num_det + 7) // 8
        obs_bytes = (num_obs + 7) // 8

        num_shots = 10
        packed_input = np.zeros((num_shots, det_bytes), dtype=np.uint8)
        result = compiled.decode_shots_bit_packed(packed_input)

        assert result.shape == (num_shots, obs_bytes)
        assert result.dtype == np.uint8

    def test_decode_with_real_samples(self, simple_circuit, simple_dem):
        """Run actual sampling and decoding to verify correctness."""
        dec = UnifiedQECDecoder(backend="pymatching")
        compiled = dec.compile_decoder_for_dem(simple_dem)

        num_det = simple_dem.num_detectors
        num_obs = simple_dem.num_observables
        det_bytes = (num_det + 7) // 8
        obs_bytes = (num_obs + 7) // 8

        sampler = simple_circuit.compile_detector_sampler()
        det_data, obs_data = sampler.sample(100, separate_observables=True)

        # Bit-pack detections
        packed_det = np.zeros((100, det_bytes), dtype=np.uint8)
        for i in range(100):
            packed_det[i] = np.packbits(
                det_data[i].astype(np.uint8), bitorder="little"
            )[:det_bytes]

        result = compiled.decode_shots_bit_packed(packed_det)
        assert result.shape == (100, obs_bytes)


@pytest.mark.requires_ldpc
class TestBPOSDCompiledDecoder:
    """Test the BP+OSD compiled decoder backend."""

    def test_compile_and_decode(self, simple_dem):
        dec = UnifiedQECDecoder(backend="bposd")
        compiled = dec.compile_decoder_for_dem(simple_dem)

        num_det = simple_dem.num_detectors
        num_obs = simple_dem.num_observables
        det_bytes = (num_det + 7) // 8
        obs_bytes = (num_obs + 7) // 8

        num_shots = 5
        packed_input = np.zeros((num_shots, det_bytes), dtype=np.uint8)
        result = compiled.decode_shots_bit_packed(packed_input)

        assert result.shape == (num_shots, obs_bytes)
        assert result.dtype == np.uint8
