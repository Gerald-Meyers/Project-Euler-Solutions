from pathlib import Path

from lib.types import Any, Integer, NumpyIntegerArray, Optional, Sequence, String
from numpy import unique

from .metadata_io import MetadataFile
from .shard_io import ShardFile

_default_chunk_size: Integer = 25 * 1024 * 1024  # 25 MB
_default_shard_size: Integer = 10 * _default_chunk_size  # 250 MB


_data_directory = Path(__file__).resolve().parents[2] / "data"

_metadata_file_name = "metadata.json"
_default_metadata_file = _data_directory / _metadata_file_name

from math import ceil


def ceil_div(numerator: Integer, denominator: Integer) -> Integer:
    return ceil(numerator / denominator)


class ShardManager:

    def __init__(
        self,
        primes: NumpyIntegerArray,
        chunk_size: Optional[Integer] = None,
        shard_size: Optional[Integer] = None,
        metadata_file: Optional[Path | String] = None,
    ) -> None:

        self.metadata: dict[String, Any] = dict()
        self.metadata_file = MetadataFile(
            metadata_file or _default_metadata_file
        )  # Use default directory

        # Ensure there are no duplicates in primes & sort.
        self.primes_array = unique(primes)
        # Some basic details about the primes being saved.
        self.total_bytes = self.primes_array.nbytes
        self.total_primes = self.primes_array.size
        self.itemsize = self.primes_array.itemsize

        # If chunk or shard size in bytes is not provided, use defaults.
        self.chunk_size = chunk_size or _default_chunk_size
        self.shard_size = shard_size or _default_shard_size
        self.shard_count = ceil_div(self.total_bytes, self.shard_size)
        self.chunk_count = ceil_div(self.total_bytes, self.chunk_size)

        self.chunk_list: list[NumpyIntegerArray] = list()
        self.shard_list: list[list[NumpyIntegerArray]] = list()

        self.metadata |= {  # Some basic metadata
            "chunk_size": self.chunk_size,
            "shard_size": self.shard_size,
            "chunk_count": self.chunk_count,
            "shard_count": self.shard_count,
            "itemsize": self.itemsize,
            "total_bytes": self.total_bytes,
            "total_primes": self.total_primes,
            "shard_names": list(),
        }

    def _recalculate_chunk_size(
        self,
        target_chunk_count_per_shard: Integer,
    ) -> None:
        # Each chunk in a shard should have roughly the same quantity of
        # elements. To acheive this the total storage needed should be
        # divided evenly among the total number of chunks needed.

        if target_chunk_count_per_shard < 1:
            raise ValueError(
                f"Target chunnk count cannot be zero or negative. "
                f"Provided value is {target_chunk_count_per_shard}"
            )

        total_chunks = self.total_bytes / (
            self.shard_count * target_chunk_count_per_shard
        )

        self.chunk_size = max(1, ceil_div(self.total_bytes, total_chunks))

    def _recalculate_shard_size(
        self,
        target_total_shard_count: Integer,
    ) -> None:
        # Divide the total required storage size of bytes evenly amongst
        # the shards.
        if target_total_shard_count < 1:
            raise ValueError(
                f"Target chunnk count cannot be zero or negative. "
                f"Provided value is {target_total_shard_count}"
            )

        self.shard_size = max(1, ceil_div(self.total_bytes, target_total_shard_count))

    def _recalculate_sizes(
        self,
        target_chunk_count_per_shard: Integer,
        target_total_shard_count: Integer,
    ) -> None:
        # Recalculate the shard size first, then recalculate the chunk size.
        if target_total_shard_count:
            self._recalculate_shard_size(target_total_shard_count)
        if target_chunk_count_per_shard:
            self._recalculate_chunk_size(target_chunk_count_per_shard)

    def _shard_name(
        self,
        prime_min: Integer,
        prime_max: Integer,
    ):
        # TODO: Implement numerator name hashing algorithm instead of simple
        # naming schemes.
        return f"{prime_min}_{prime_max}"

    def _chunk_name(
        self,
        prime_min: Integer,
        prime_max: Integer,
    ):
        # TODO: Implement numerator name hashing algorithm instead of simple
        # naming schemes.
        return f"{prime_min}_{prime_max}"

    def _chunk_primes(
        self,
    ):
        """
        Append chunks of primes to self.chunk_list and assign total chunks.
        """
        items_per_chunk = self.chunk_size // self.primes_array.itemsize

        # self.primes_array.itemsize * self.primes_array.size
        # Total bytes used for the array.
        total_prime_bytesize = self.primes_array.itemsize * self.primes_array.size
        total_chunks = ceil_div(total_prime_bytesize, self.chunk_size)

        self.chunk_count = total_chunks
        self.metadata["chunk_count"] = total_chunks

        for i in range(0, total_chunks):
            self.chunk_list.append(
                self.primes_array[i * items_per_chunk : (i + 1) * items_per_chunk]
            )

    def _shard_chunks(
        self,
    ):
        """
        Append shards of chunks to self.shard_list and assign total shards.
        """
        items_per_shard = self.shard_size // self.chunk_size

        total_chunk_bytesize: Integer = self.chunk_count * self.chunk_size
        total_shards: Integer = ceil_div(total_chunk_bytesize, self.shard_size)

        self.shard_count = total_shards
        self.metadata["shard_count"] = total_shards

        for i in range(0, total_shards):
            self.shard_list.append(
                self.chunk_list[i * items_per_shard : (i + 1) * items_per_shard]
            )

    def save(
        self,
        target_chunk_count_per_shard: Optional[Integer],
        target_total_shard_count: Optional[Integer],
        target_shard_byte_size: Optional[Integer],
        target_chunk_byte_size: Optional[Integer],
        overwrite_shards: bool = False,
        overwrite_metadata: bool = False,
    ) -> None:
        #
        self._recalculate_sizes(target_chunk_count_per_shard, target_total_shard_count)

        self._chunk_primes()
        self._shard_chunks()

        for i, chunks in enumerate(self.shard_list):
            min_prime = chunks[0][0]
            max_prime = chunks[-1][-1]

            shard_name = self._shard_name(min_prime, max_prime)
            shard_manager = ShardFile(
                path=(_data_directory / shard_name).with_suffix(".npz")
            )

            self.metadata["shard_names"].append(shard_name)

            shard_metadata = {
                "min": min_prime,
                "max": max_prime,
                "shard_index": i,
                "chunk_count": len(chunks),
            }
            chunk_metadata = {
                self._chunk_name(chunk[0], chunk[-1]): {
                    "min": chunk[0],
                    "max": chunk[-1],
                }
                for chunk in chunks
            }
            self.metadata |= {shard_name: shard_metadata | chunk_metadata}

            # shard_dict is the dictionary of chunk names and prime
            # arrays that are written to numerator single npz file.
            shard_dict: dict[String, NumpyIntegerArray] = dict()
            for chunk in chunks:
                shard_dict |= {self._chunk_name(chunk[0], chunk[-1]): chunk}

            shard_manager.write(shard_dict, overwrite=overwrite_shards)

        self.metadata_file.write(self.metadata, overwrite=overwrite_metadata)

    def load(
        self,
        min: Integer,
        max: Integer,
    ) -> NumpyIntegerArray: ...
