"""
Tests for the Open Data subsystem (Zarr + WebDataset).

Tests ZarrStore write/read roundtrips, slicing, metadata,
WebDatasetWriter shard creation, and WebDatasetReader streaming.
"""

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest


class TestWebDatasetWriter:
    """Test WebDataset shard creation and reading."""

    def test_write_single_sample(self, tmp_path):
        from unified_qec.data.webdataset_writer import WebDatasetWriter

        output_dir = str(tmp_path / "shards")
        writer = WebDatasetWriter(output_dir, shard_size=100)

        syndrome = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        observable = np.array([1], dtype=np.uint8)
        writer.add_sample(syndrome, observable, {"p": 0.01})
        writer.close()

        assert writer.total_samples == 1
        assert writer.num_shards == 1

        # Verify tar contents
        tar_files = list(Path(output_dir).glob("*.tar"))
        assert len(tar_files) == 1

        with tarfile.open(tar_files[0], "r") as tar:
            members = tar.getnames()
            assert any(".syndrome.npy" in m for m in members)
            assert any(".observable.npy" in m for m in members)
            assert any(".json" in m for m in members)

    def test_write_batch(self, tmp_path):
        from unified_qec.data.webdataset_writer import WebDatasetWriter

        output_dir = str(tmp_path / "shards")
        writer = WebDatasetWriter(output_dir, shard_size=50)

        syndromes = np.random.randint(0, 2, (100, 10), dtype=np.uint8)
        observables = np.random.randint(0, 2, (100, 1), dtype=np.uint8)
        writer.add_batch(syndromes, observables, {"d": 3})
        writer.close()

        assert writer.total_samples == 100
        assert writer.num_shards >= 2  # writer may pre-open next shard

        # Verify manifest
        manifest_path = Path(output_dir) / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["total_samples"] == 100
        assert manifest["num_shards"] >= 2

    def test_context_manager(self, tmp_path):
        from unified_qec.data.webdataset_writer import WebDatasetWriter

        output_dir = str(tmp_path / "shards")
        with WebDatasetWriter(output_dir, shard_size=10) as writer:
            for i in range(25):
                writer.add_sample(
                    np.array([i % 2], dtype=np.uint8),
                    np.array([0], dtype=np.uint8),
                )

        assert Path(output_dir).exists()
        tar_files = list(Path(output_dir).glob("*.tar"))
        assert len(tar_files) == 3  # 25 samples / 10 shard_size = 3 shards

    def test_reader_roundtrip(self, tmp_path):
        from unified_qec.data.webdataset_writer import (
            WebDatasetWriter,
            WebDatasetReader,
        )

        output_dir = str(tmp_path / "shards")
        syndromes = np.random.randint(0, 2, (20, 5), dtype=np.uint8)
        observables = np.random.randint(0, 2, (20, 1), dtype=np.uint8)

        with WebDatasetWriter(output_dir, shard_size=10) as writer:
            writer.add_batch(syndromes, observables)

        reader = WebDatasetReader(output_dir)
        read_samples = list(reader)
        assert len(read_samples) == 20

        # Verify first sample content
        syn_out, obs_out, _ = read_samples[0]
        np.testing.assert_array_equal(syn_out, syndromes[0])
        np.testing.assert_array_equal(obs_out, observables[0])


class TestZarrStore:
    """Test Zarr-based syndrome storage."""

    @pytest.fixture(autouse=True)
    def _check_zarr(self):
        pytest.importorskip("zarr")

    def test_write_and_read(self, tmp_path):
        from unified_qec.data.zarr_store import ZarrStore

        zarr_path = str(tmp_path / "test.zarr")
        syndromes = np.random.randint(0, 2, (100, 10), dtype=np.uint8)
        observables = np.random.randint(0, 2, (100, 1), dtype=np.uint8)

        with ZarrStore(zarr_path, mode="w") as store:
            store.write_syndromes(
                syndromes, observables,
                metadata={"d": 5, "p": 0.001},
            )

        with ZarrStore(zarr_path, mode="r") as store:
            syn_out, obs_out = store.read_slice()
            meta = store.get_metadata()

        np.testing.assert_array_equal(syn_out, syndromes)
        np.testing.assert_array_equal(obs_out, observables)
        assert meta["d"] == 5
        assert meta["total_shots"] == 100

    def test_slice_reading(self, tmp_path):
        from unified_qec.data.zarr_store import ZarrStore

        zarr_path = str(tmp_path / "test.zarr")
        syndromes = np.arange(50 * 10, dtype=np.uint8).reshape(50, 10)
        observables = np.zeros((50, 1), dtype=np.uint8)

        with ZarrStore(zarr_path, mode="w") as store:
            store.write_syndromes(syndromes, observables)

        with ZarrStore(zarr_path, mode="r") as store:
            syn_slice, _ = store.read_slice(shot_range=(10, 20))

        expected = syndromes[10:20]
        np.testing.assert_array_equal(syn_slice, expected)

    def test_shape_info(self, tmp_path):
        from unified_qec.data.zarr_store import ZarrStore

        zarr_path = str(tmp_path / "test.zarr")
        syndromes = np.zeros((42, 17), dtype=np.uint8)
        observables = np.zeros((42, 3), dtype=np.uint8)

        with ZarrStore(zarr_path, mode="w") as store:
            store.write_syndromes(syndromes, observables)
            shapes = store.get_shape()

        assert shapes["syndromes"] == (42, 17)
        assert shapes["observables"] == (42, 3)

    def test_read_failures(self, tmp_path):
        from unified_qec.data.zarr_store import ZarrStore

        zarr_path = str(tmp_path / "test.zarr")
        syndromes = np.random.randint(0, 2, (100, 10), dtype=np.uint8)
        observables = np.zeros((100, 1), dtype=np.uint8)
        observables[5] = 1
        observables[42] = 1
        observables[99] = 1

        with ZarrStore(zarr_path, mode="w") as store:
            store.write_syndromes(syndromes, observables)

        with ZarrStore(zarr_path, mode="r") as store:
            fail_syn, fail_obs = store.read_failures()

        assert fail_syn.shape[0] == 3
        assert fail_obs.shape[0] == 3
        assert np.all(fail_obs == 1)


class TestSinterToData:
    """Test the Sinter-to-storage converter."""

    def test_from_stim_samples_zarr(self, tmp_path):
        import stim
        from unified_qec.data.converters import SinterToData

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )

        converter = SinterToData()
        result = converter.from_stim_samples(
            circuit=circuit,
            num_shots=500,
            output_path=str(tmp_path / "output"),
            formats=["zarr"],
            metadata={"d": 3},
        )

        assert result["total_shots"] == 500
        assert "zarr" in result["paths"]
        assert Path(result["paths"]["zarr"]).exists()

    def test_from_stim_samples_webdataset(self, tmp_path):
        import stim
        from unified_qec.data.converters import SinterToData

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )

        converter = SinterToData()
        result = converter.from_stim_samples(
            circuit=circuit,
            num_shots=200,
            output_path=str(tmp_path / "output"),
            formats=["webdataset"],
            batch_size=100,
        )

        assert result["total_shots"] == 200
        assert "webdataset" in result["paths"]

    def test_from_detection_events(self, tmp_path):
        from unified_qec.data.converters import SinterToData

        det = np.random.randint(0, 2, (50, 10), dtype=np.uint8)
        obs = np.random.randint(0, 2, (50, 1), dtype=np.uint8)

        converter = SinterToData()
        result = converter.from_detection_events(
            det, obs,
            output_path=str(tmp_path / "output"),
            formats=["zarr", "webdataset"],
        )

        assert result["total_shots"] == 50
        assert "zarr" in result["paths"]
        assert "webdataset" in result["paths"]
