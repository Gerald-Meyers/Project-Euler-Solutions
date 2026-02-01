import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from lib.storage.manager import ShardManager
from lib.storage.metadata_io import MetadataFile
from lib.storage.pathing import HashFile, LockFile
from lib.storage.shard_io import ShardFile
from lib.types import NumpyIntegerArray, String


class MockShardStorage:
    """
    In-memory implementation of ShardStorage protocol for testing.
    """

    _storage: dict[str, dict[str, NumpyIntegerArray]] = {}

    def __init__(self, path: Path | String):
        self.path = str(path)

    def write(
        self, chunk_dict: dict[String, NumpyIntegerArray], overwrite: bool = False
    ) -> None:
        if self.path in self._storage and not overwrite:
            raise FileExistsError(f"Mock file {self.path} exists")
        self._storage[self.path] = chunk_dict

    def read(self) -> NumpyIntegerArray:
        if self.path not in self._storage:
            raise FileNotFoundError(f"Mock file {self.path} not found")

        # Simulate reading chunks and concatenating, similar to ShardFile.read
        chunks = list(self._storage[self.path].values())
        if not chunks:
            return np.array([], dtype=int)
        return np.unique(np.concatenate(chunks))

    def delete(self) -> None:
        if self.path in self._storage:
            del self._storage[self.path]

    @classmethod
    def clear(cls):
        cls._storage.clear()


class TestShardManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the metadata file (which still writes to disk)
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_path = Path(self.temp_dir) / "test_metadata.json"

        # Clear mock storage state
        MockShardStorage.clear()

        # Inject the Mock factory
        self.manager = ShardManager(shard_factory=MockShardStorage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        MockShardStorage.clear()

    def test_save_and_load_integrity(self):
        # Generate random sorted unique integers
        primes = np.unique(np.random.randint(1, 100000, 1000))

        # Save using the manager (writes metadata to temp_dir, shards to Mock memory)
        self.manager.save(
            primes_array=primes,
            metadata_path=self.metadata_path,
            target_total_shard_count=4,
            overwrite_shards=True,
            overwrite_metadata=True,
        )

        # Load back the data
        loaded_primes = self.manager.load(metadata_path=self.metadata_path)

        # Verify the data round-tripped correctly
        np.testing.assert_array_equal(primes, loaded_primes)

    def test_save_overwrite_false_exists(self):
        # Generate random sorted unique integers
        primes = np.unique(np.random.randint(1, 100000, 1000))

        # First save
        self.manager.save(
            primes_array=primes,
            metadata_path=self.metadata_path,
            target_total_shard_count=4,
            overwrite_shards=True,
            overwrite_metadata=True,
        )

        # Second save with overwrite_shards=False should fail
        with self.assertRaises(FileExistsError):
            self.manager.save(
                primes_array=primes,
                metadata_path=self.metadata_path,
                target_total_shard_count=4,
                overwrite_shards=False,
                overwrite_metadata=True,
            )


class TestPathing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "test_file.txt"
        self.file_path.write_text("Hello World")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_hash_file(self):
        hf = HashFile(self.file_path)
        # Compute hash
        h = hf.compute_hash()
        self.assertIsInstance(h, str)

        # Write hash
        hf.write_hash_to_file(h)
        self.assertTrue(hf.verify_file())

        # Modify file and verify failure (using new instance to bypass cache)
        self.file_path.write_text("Modified")
        hf2 = HashFile(self.file_path)
        self.assertFalse(hf2.verify_file())

    def test_lock_file(self):
        lf = LockFile(self.file_path)
        self.assertFalse(lf.is_locked)

        lf.acquire_lock()
        self.assertTrue(lf.is_locked)

        # Test re-acquire by another instance (simulated)
        lf2 = LockFile(self.file_path)
        with self.assertRaises(TimeoutError):
            lf2.acquire_lock()

        lf.release_lock()
        self.assertFalse(lf.is_locked)


class TestMetadataIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir) / "meta.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_write(self):
        mf = MetadataFile(self.path)
        data = {"key": "value", "num": 123}
        mf.write(data)

        read_data = mf.read()
        self.assertEqual(data, read_data)

        # Test merge (overwrite=False)
        merge_data = {"other": "data"}
        mf.write(merge_data, overwrite=False)
        expected = {"key": "value", "num": 123, "other": "data"}
        self.assertEqual(mf.read(), expected)


class TestShardIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir) / "shard.npz"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_write(self):
        sf = ShardFile(self.path)
        chunk1 = np.array([1, 2, 3], dtype=int)
        data = {"c1": chunk1}

        sf.write(data)
        loaded = sf.read()
        np.testing.assert_array_equal(loaded, chunk1)


if __name__ == "__main__":
    unittest.main()
