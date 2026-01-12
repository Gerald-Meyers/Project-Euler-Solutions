"""
Save and load sharded primes to and from npz files.
"""

import time
from datetime import datetime
from json import JSONDecodeError, dump
from json import load as load_json_from_file
from pathlib import Path, PurePath

from lib.types import (
    Any,
    Callable,
    Integer,
    IntegerArray,
    NumpyIntegerArray,
    Optional,
    Sequence,
)
from numpy import array, concatenate
from numpy import load as load_array_from_file
from numpy import save, savez_compressed, unique
from numpy.lib.npyio import NpzFile

MAX_CHUNK_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_SHARD_SIZE_BYTES = 10 * MAX_CHUNK_SIZE_BYTES  # 250 MB


class MetadataIOHandler:
    """
    A Manager class to handle to IO for the metadata file.
    """

    def __init__(
        self,
        storage_dir: Path,
        metadata_filename: str = "metadata.json",
    ) -> None:

        self.metadata_file = storage_dir / metadata_filename
        self.lock_file = self.metadata_file.with_suffix(".lock")

    def _acquire_lock(
        self,
    ) -> None:

        attempts = 0
        while self.lock_file.exists():
            # Since I don't expect many of these instances of the saving
            # and loading of primes processes from ever existing, I
            # don't think there is any cause to implement an algorithm to
            # implement Binary Exponential Backoff. There should be no
            # herd of processes attempting to save and load the
            # metadata file.

            if attempts > 50:
                raise TimeoutError(f"Could not acquire lock on " f"{self.lock_file}")
            time.sleep(0.1)
            attempts += 1

        self.lock_file.touch()

        return

    def _release_lock(
        self,
    ) -> None:
        if self.lock_file.exists():
            self.lock_file.unlink()

        return

    def save(
        self,
        metadata: dict[str, Any],
    ) -> None:
        self._acquire_lock()

        try:
            if not self.metadata_file.exists():
                self.metadata_file.write_text("{}")

            with self.metadata_file.open("r+") as metadata_file:
                try:
                    existing_metadata: dict[str, Any] = load_json_from_file(
                        metadata_file
                    )

                except (JSONDecodeError, ValueError):
                    existing_metadata: dict[str, Any] = dict()

                metadata_file.seek(0)
                dump(
                    existing_metadata | metadata,
                    metadata_file,
                    indent=4,
                    sort_keys=True,
                )
                metadata_file.truncate()
        finally:
            self._release_lock()

    def load(
        self,
    ) -> dict[str, dict[str, Any]]:

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Could not locate metadata file at {self.metadata_file}"
            )

        with self.metadata_file.open("r") as metadata_file:

            metadata: dict[str, dict[str, Any]] = load_json_from_file(metadata_file)

        return metadata

    def _flush_metadata(
        self,
    ) -> None:
        if self.metadata_file.with_suffix(".lock").exists():
            raise FileExistsError(
                f"Cannot flush meta data while the metadata is being saved."
            )

        try:
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Cannot flush metadata that doesn't exist.")

            self.metadata_file.unlink()
            self.save({})

        except Exception as e:
            print(
                "An unexpected error has occured while "
                "flushing the metadata:\n"
                f"{e}"
            )


class ShardIOHandler:

    def __init__(
        self,
        shard_path: Path,
        chunk_size: Integer = MAX_CHUNK_SIZE_BYTES,
        shard_size: Integer = MAX_SHARD_SIZE_BYTES,
    ) -> None:

        self.chunk_size = chunk_size
        self.shard_size = shard_size

        self.shard_path = shard_path

    def recalculate_sizes(
        self,
        total_elements: Integer,
        primes_itemsize: Integer,
        target_chunk_count: Integer,
        target_shard_count: Integer,
    ) -> None:
        if target_shard_count < 1:
            raise ValueError("Target shard count must be greater than 0.")

        self.recalculate_chunk_size(total_elements, primes_itemsize, target_chunk_count)
        self.recalculate_shard_size()

    def recalculate_chunk_size(
        self,
        total_elements: Integer,
        primes_itemsize: Integer,
        target_chunk_count: Integer,
    ) -> None:
        if target_chunk_count < 1:
            raise ValueError("Target chunk count must be greater than 0.")

        elements_per_chunk = max(1, total_elements // target_chunk_count)

        # Calculate size and clip it to the default maximum size
        calculated_size = elements_per_chunk * primes_itemsize
        self.chunk_size = calculated_size

    def recalculate_shard_size(
        self, total_elements: Integer, primes_itemsize: Integer, target
    ) -> None: ...

    def load_shard(
        self,
    ):

        if not self.shard_path.exists():
            raise FileNotFoundError(
                f"Shard {self.shard_path.name} is not stored on disk."
            )

        # Collect all primes loaded from a shard into a list
        collected_primes: list[NumpyIntegerArray] = list()

        # np.load (load_array_from_file) returns a dictionary-like object.
        with load_array_from_file(self.shard_path) as sharded_primes:

            # Iterate through the stored arrays using the
            # NPZfile.files attribute.
            for chunk_name in sharded_primes.files:
                collected_primes.append(sharded_primes[chunk_name])

        # union1d returns the unique, sorted array of values that are in
        # either of the two input arrays
        return unique(concatenate(collected_primes))

    def save_shard(
        self,
        chunk_dict: dict[str, NumpyIntegerArray],
    ) -> None:
        """
        Write a single shard to a single NPZ file.

        :param self: Description
        :param chunk_dict: Description
        :type chunk_dict: dict[str, NumpyIntegerArray]
        """
        savez_compressed(self.shard_path, **chunk_dict)
