"""
Tests for the Model Zoo (checkpoint + hub) subsystem.

Tests NumpyCheckpoint save/restore, metadata-only restore,
and retention policies. Orbax tests are conditional on availability.
"""


import numpy as np
import pytest

from unified_qec.zoo.checkpoint import NumpyCheckpoint


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    return str(tmp_path / "checkpoints")


class TestNumpyCheckpoint:
    """Test the lightweight NumPy-based checkpoint manager."""

    def test_save_and_restore_latest(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=5)

        weights = {"w1": np.random.randn(10, 10), "b1": np.random.randn(10)}
        metadata = {"hidden_dim": 256, "num_layers": 4, "loss": 0.05}

        ckpt.save(step=100, model_state=weights, metadata=metadata)

        state, meta = ckpt.restore()  # latest
        assert meta["hidden_dim"] == 256
        assert meta["loss"] == 0.05
        np.testing.assert_array_almost_equal(state["w1"], weights["w1"])
        np.testing.assert_array_almost_equal(state["b1"], weights["b1"])

    def test_restore_specific_step(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=5)

        for step in [100, 200, 300]:
            weights = {"w": np.full((3, 3), step)}
            ckpt.save(step=step, model_state=weights, metadata={"step": step})

        state, meta = ckpt.restore(step=200)
        assert meta["step"] == 200
        np.testing.assert_array_equal(state["w"], np.full((3, 3), 200))

    def test_metadata_only_restore(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=5)

        weights = {"w": np.random.randn(5, 5)}
        ckpt.save(step=1, model_state=weights, metadata={"arch": "transformer"})

        state, meta = ckpt.restore(metadata_only=True)
        assert state is None
        assert meta["arch"] == "transformer"

    def test_latest_step(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=10)

        for step in [10, 50, 30]:
            ckpt.save(step=step, model_state={"x": np.array([1.0])}, metadata={})

        assert ckpt.latest_step() == 50

    def test_all_steps(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=10)

        for step in [10, 50, 30]:
            ckpt.save(step=step, model_state={"x": np.array([1.0])}, metadata={})

        assert ckpt.all_steps() == [10, 30, 50]

    def test_retention_policy(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir, max_to_keep=2)

        for step in [1, 2, 3, 4, 5]:
            ckpt.save(step=step, model_state={"x": np.array([step])}, metadata={})

        steps = ckpt.all_steps()
        assert len(steps) == 2
        assert steps == [4, 5]

    def test_restore_empty_raises(self, tmp_ckpt_dir):
        ckpt = NumpyCheckpoint(tmp_ckpt_dir)
        with pytest.raises(FileNotFoundError):
            ckpt.restore()

    def test_numpy_types_in_metadata(self, tmp_ckpt_dir):
        """Ensure numpy scalars are serializable as metadata."""
        ckpt = NumpyCheckpoint(tmp_ckpt_dir)

        metadata = {
            "loss": np.float32(0.05),
            "epoch": np.int64(100),
            "params_shape": np.array([10, 20]),
        }
        ckpt.save(step=1, model_state={"x": np.array([1.0])}, metadata=metadata)

        _, meta = ckpt.restore(metadata_only=True)
        assert isinstance(meta["loss"], float)
        assert isinstance(meta["epoch"], int)
