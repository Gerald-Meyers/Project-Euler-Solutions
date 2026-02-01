from pathlib import Path
from typing import Protocol, runtime_checkable

from lib.types import NumpyIntegerArray, String
from numpy import concatenate, load, savez_compressed, unique

from .pathing import HashFile


@runtime_checkable
class ShardStorage(Protocol):
    def read(self) -> NumpyIntegerArray: ...
    def write(
        self, chunk_dict: dict[String, NumpyIntegerArray], overwrite: bool = False
    ) -> None: ...
    def delete(self) -> None: ...


class ShardFile:

    def __init__(
        self,
        path: Path | String,
    ) -> None:
        self.path = Path(path)

    def read(
        self,
    ) -> NumpyIntegerArray:
        if not self.path.exists():
            raise FileNotFoundError(f"Shard file {self.path} does not exist.")

        with load(self.path) as shard:
            collected_chunks: list[NumpyIntegerArray] = list()
            for chunk in shard.files:
                collected_chunks.append(shard[chunk])

        return unique(concatenate(collected_chunks))

    def write(
        self,
        chunk_dict: dict[String, NumpyIntegerArray],
        overwrite: bool = False,
    ):
        self.validate(chunk_dict)

        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Shard file {self.path} already exists.")
        elif overwrite:
            self.delete()

        savez_compressed(self.path, **chunk_dict)

    def delete(
        self,
    ) -> None:
        self.path.unlink(missing_ok=True)

    def validate(
        self,
        chunk_dict: dict[String, NumpyIntegerArray],
    ) -> None:
        if not chunk_dict:
            raise ValueError(f"Chunk dictionary is empty.")

        for chunk_name, chunk_array in chunk_dict.items():
            if chunk_array.size == 0:
                raise ValueError(f"Chunk {chunk_name} is empty.")
            if not chunk_name:
                raise ValueError(f"Chunk name is empty.")
