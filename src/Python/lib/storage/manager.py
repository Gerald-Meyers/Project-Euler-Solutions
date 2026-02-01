"""
Loading primes from storage is faster than generating them on the fly,
for large quantities, so long as the storage medium is not too cold.

Generating 1 billion primes takes a bit over two minutes, while loading
the same quantity from storage takes less than 5 seconds.
"""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from lib.types import Integer, NumpyIntegerArray, String
from numpy import concatenate, unique

from .metadata_io import MetadataFile
from .shard_io import ShardFile, ShardStorage

_default_chunk_size: Integer = 25 * 1024 * 1024  # 25 MB
_default_shard_size: Integer = 10 * _default_chunk_size  # 250 MB

_data_directory = Path(__file__).resolve().parents[2] / "data"
_metadata_file_name = "metadata.json"
_default_metadata_file = _data_directory / _metadata_file_name


@dataclass(frozen=True)
class PartitionPlan:
    """
    Simple data object to hold the calculated results.
    """

    items_per_shard: Integer
    chunks_per_shard: Integer
    items_per_chunk: Integer
    total_shards: Integer
    total_chunks: Integer


class PartitionStrategy:
    """
    Determines how an array should be sliced.
    Stateless: uses @staticmethod and pure inputs.
    """

    @classmethod
    def calculate_plan(
        self,
        total_items: Integer,
        item_byte_size: Integer,
        # The following are user constraints passed from ShardManager.save()
        target_shard_count: Optional[Integer] = None,
        target_chunks_per_shard: Optional[Integer] = None,
        max_shard_bytes: Optional[Integer] = None,
        max_chunk_bytes: Optional[Integer] = None,
        # Defaults
    ) -> PartitionPlan:
        """
        Calculates the partitioning strategy based on user constraints or defaults.

        :param total_items: The total number of elements in the array to be partitioned.
        :type total_items: Integer
        :param target_shard_count: The specific number of shards requested by the user.
        :type target_shard_count: Optional[Integer]
        :param max_shard_bytes: The maximum file size in bytes for each shard.
        :type max_shard_bytes: Optional[Integer]
        :param chunks_per_shard: The specific number of chunks requested per shard.
        :type chunks_per_shard: Optional[Integer]
        :param max_chunk_bytes: The maximum size in bytes for each chunk.
        :type max_chunk_bytes: Optional[Integer]
        :return: A plan object containing the calculated item counts and totals.
        :rtype: PartitionPlan
        """

        # 1. Resolve Shard Size (Items per Shard)
        items_per_shard = PartitionStrategy._resolve_limit(
            total_items=total_items,
            item_bytes=item_byte_size,
            target_count=target_shard_count,
            target_bytes=max_shard_bytes,
        )

        # 2. Resolve Chunk Size (Items per Chunk)
        items_per_chunk = PartitionStrategy._resolve_limit(
            total_items=items_per_shard,
            item_bytes=item_byte_size,
            target_count=target_chunks_per_shard,
            target_bytes=max_chunk_bytes,
        )

        # Ensure a chunk can't be larger than the shard it lives in.
        if items_per_chunk > items_per_shard:
            items_per_chunk = items_per_shard

        return PartitionPlan(
            items_per_shard=items_per_shard,
            chunks_per_shard=ceil(items_per_shard / items_per_chunk),
            items_per_chunk=items_per_chunk,
            total_shards=ceil(total_items / items_per_shard),  # total_shards
            total_chunks=ceil(total_items / items_per_chunk),  # total_chunks
        )

    @staticmethod
    def _resolve_limit(
        total_items: Integer,
        item_bytes: Integer,
        target_count: Optional[Integer] = None,
        target_bytes: Optional[Integer] = None,
        default_bytes: Optional[Integer] = None,
    ) -> Integer:
        """
        Calculates the maximum number of items allowed in a single partition.

        This method resolves the partition size by prioritizing an explicit
        partition count over a byte-size limit.

        Resolution Logic:
        1. Target Count Priority: If 'target_count' is provided, the method
           calculates the slice size required to produce exactly that many
           partitions (ceil(total / count)). Byte limits are ignored in this case.
        2. Byte Limit Fallback: If 'target_count' is None, the method calculates
           how many items fit within 'byte_limit' (byte_limit // item_size).
        3. Safety: The result is clamped to a minimum of 1 item to prevent
           zero-size partitions or infinite loops.

        :param total_items: The total number of elements available to be partitioned.
        :type total_items: Integer
        :param target_count: The specific number of partitions requested by the user.
        :type target_count: Optional[Integer]
        :param target_bytes: The maximum file size in bytes requested by the user.
        :type target_bytes: Optional[Integer]
        :param default_bytes: The fallback byte limit if no user constraints are provided.
        :type default_bytes: Integer
        :return: The calculated number of items per partition (slice size).
        :rtype: Integer
        """
        # The same logic to decide chunk_size and shard_size in bytes
        # is repeated, so this is a helper function

        # Fail fast guard clause
        if item_bytes <= 0:
            raise ValueError(f"item_bytes must be > 0, somehow got {item_bytes}")

        # Priority 1: Target Count Priority
        if target_count is not None:
            if target_count <= 0:
                raise ValueError(
                    f"target_count must be > 0, somehow got {target_count}"
                )
            # Simple ratio between item counts.
            return ceil(total_items / target_count)

        # Prority 2: Byte Limits
        limit_bytes = target_bytes or default_bytes

        # Fail fast guard clauses
        if limit_bytes is None:
            raise ValueError(
                f"Unknown partition strategy.\n"
                f"None of target_count, target_bytes, or default_bytes were passed."
            )

        if limit_bytes <= 0:
            raise ValueError(
                f"Byte limit must be an integer greater than 0. Somehow received {limit_bytes} instead."
            )

        # Convert byte limit to item count (min 1 to prevent infinite loops).
        return max(1, limit_bytes // item_bytes)


class ShardManager:
    """
    ShardManager manages the saving and loading of shard files,
    verification of shard files against the stored metadata file, and
    merges potentially inhomogeneous shards into homogeneous shards.
    """

    def __init__(
        self,
        shard_factory: Callable[[Path | String], ShardStorage] = ShardFile,
    ) -> None:
        self._shard_factory = shard_factory

    def save(
        self,
        primes_array: NumpyIntegerArray | None = None,
        metadata_path: Path | String = _default_metadata_file,
        # Shard & Chunk size parameters
        target_total_shard_count: Optional[Integer] = None,
        target_chunk_count_per_shard: Optional[Integer] = 10,
        shard_byte_size: Integer = _default_shard_size,
        chunk_byte_size: Integer = _default_chunk_size,
        overwrite_shards: bool = False,
        overwrite_metadata: bool = False,
    ) -> None:

        if primes_array is None:
            raise ValueError(f"")

        metadata_file = MetadataFile(Path(metadata_path))

        # Ensure there are no duplicates in primes & sort.
        primes = unique(primes_array)
        # Basic details about the prime
        total_primes = primes.size

        # Some basic metadata
        metadata = {
            "chunk_size": chunk_byte_size,
            "shard_size": shard_byte_size,
            "itemsize": primes.itemsize,
            "total_bytes": primes.nbytes,
            "total_primes": total_primes,
            "shard_paths": list(),
        }

        plan = PartitionStrategy.calculate_plan(
            total_items=total_primes,
            item_byte_size=primes.itemsize,
            #
            target_shard_count=target_total_shard_count,
            target_chunks_per_shard=target_chunk_count_per_shard,
            #
            max_shard_bytes=shard_byte_size,
            max_chunk_bytes=chunk_byte_size,
        )
        metadata["config"] = plan.__dict__

        # possibly redundant, remove if so.
        metadata["total_chunks"] = plan.total_chunks
        metadata["total_shards"] = plan.total_shards

        for shard_idx, chunks in self._partitions(primes, plan):

            shard_path = (
                _data_directory
                / f"prime_shard_{shard_idx+1}_of_{plan.total_shards}.npz"
            )
            metadata["shard_paths"].append(str(shard_path))
            shard_file = self._shard_factory(shard_path)

            metadata |= self._prepare_shard_metadata(shard_idx, chunks, shard_path)

            # shard_dict is the dictionary of chunk names and prime
            # arrays that are written to numerator single npz file.
            shard_dict: dict[String, NumpyIntegerArray] = dict()
            for chunk in chunks:
                shard_dict |= {self._generate_name(chunk[0], chunk[-1]): chunk}
                # basic_name is a placeholder for a better alternative.

            shard_file.write(shard_dict, overwrite=overwrite_shards)

        metadata_file.write(metadata, overwrite=overwrite_metadata)

    def load(
        self,
        min: Optional[Integer] = None,  # Minimum prime to load, inclusively.
        max: Optional[Integer] = None,  # Maximum prime to load, inclusively.
        metadata_path: Path | str = _default_metadata_file,
    ) -> NumpyIntegerArray:

        def interval_intersection(
            requested_minmax: tuple[Integer, Integer | float],
            partition_minmax: tuple[Integer, Integer],
        ):
            """Test to see if the partition intersects with the requested interval."""
            rmin, rmax = requested_minmax
            pmin, pmax = partition_minmax
            # False if either is true.
            # Upperbound of first interval is lower than lowerbound of second
            # Lowerbound of second interval is lower than upperbound of first
            if not (rmin < rmax):
                raise ValueError(
                    f"requested_minmax is {'zero length' if rmin==rmax else 'reversed'}, "
                    f"recieved {rmin}, and {rmax}."
                )
            if not (pmin < pmax):
                raise ValueError(
                    f"partition_minmax is {'zero length' if pmin==pmax else 'reversed'}, "
                    f"recieved {pmin}, and {pmax}."
                )

            return (rmin < pmax) and (pmin < rmax)

        interval: tuple[Integer, Integer | float] = (min or 0, max or float("inf"))

        # Load metadata
        metadata = MetadataFile(metadata_path).read()
        if not metadata:
            raise ValueError(f"No metadata file at path {metadata_path}")

        collected_primes: list[NumpyIntegerArray] = list()
        for shard_path in metadata["shard_paths"]:
            shard_interval = metadata[shard_path]["prime_interval"]

            if interval_intersection(interval, shard_interval):
                shard_file = self._shard_factory(Path(shard_path))
                collected_primes.append(shard_file.read())

        return unique(concatenate(collected_primes))

    def repartition_shards(
        self,
        target_chunk_count_per_shard: Optional[Integer],
        target_total_shard_count: Optional[Integer],
        target_shard_byte_size: Optional[Integer],
        target_chunk_byte_size: Optional[Integer],
        overwrite_shards: bool = True,
        overwrite_metadata: bool = True,
    ) -> None:
        if not self.verify_shard_integrity():
            # TODO: Implement this section
            raise
        # Load the saved primes.
        saved_primes = self.load()

        # Save the loaded primes using the new information provided for
        # the repartitioning. PartitionStrategy inside ShardManager.save
        # will take the new parameters and resave the new primes.
        self.save(
            primes_array=saved_primes,
            target_total_shard_count=target_total_shard_count,
            target_chunk_count_per_shard=target_chunk_count_per_shard,
            target_shard_byte_size=target_shard_byte_size,
            target_chunk_byte_size=target_chunk_byte_size,
            overwrite_shards=overwrite_shards,
            overwrite_metadata=overwrite_metadata,
        )

    def verify_shard_integrity(
        self, metadata_path: Path | str = _default_metadata_file
    ) -> bool:
        # TODO: Check existing npz shards against files listed in metadata.
        # TODO: Check existing npz shards have right file sizes.
        # TODO: Check existing npz shards have the data metadata says
        # they should have.
        ...

    def _partitions(
        self,
        primes: NumpyIntegerArray,
        plan: PartitionPlan,
    ) -> Iterator[tuple[Integer, list[NumpyIntegerArray]]]:
        """
        Generate the list of chunks to be saved into a single shard.

        :param plan: Description
        :type plan: PartitionPlan
        """
        for shard_idx in range(plan.total_shards):

            shard_data = primes[
                # Using np.slice here for readability and to avoid unneccessary
                # and verbose variables.
                slice(
                    shard_idx * plan.items_per_shard,
                    (shard_idx + 1) * plan.items_per_shard,
                )
            ]

            # Generate the list of NumpyIntegerArray views to minimize memory usage.
            chunks = [
                shard_data[idx : idx + plan.items_per_chunk]
                for idx in range(0, shard_data.size, plan.items_per_chunk)
            ]

            yield shard_idx, chunks

    def _prepare_shard_metadata(
        self,
        shard_idx: Integer,
        chunks: list[NumpyIntegerArray],
        shard_path: Path | str,
    ) -> dict[String, Any]:
        # Cast numpy values to python integers.
        min_prime, max_prime = int(chunks[0][0]), int(chunks[-1][-1])

        shard_metadata = {
            "prime_interval": (min_prime, max_prime),
            "shard_index": shard_idx,
            "chunk_count": len(chunks),
        }

        chunk_metadata = {
            self._generate_name([chunk[0], chunk[-1]]): {
                "min": int(chunk[0]),
                "max": int(chunk[-1]),
            }
            for chunk in chunks
        }

        return {str(shard_path): shard_metadata | chunk_metadata}

    @staticmethod
    def _generate_name(
        *args: Any,
    ) -> String:
        """A basic placeholder function"""
        # TODO: Implement name hashing algorithm instead of simple
        # naming schemes.
        return "_".join(map(str, args))
