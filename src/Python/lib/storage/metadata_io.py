from json import JSONDecodeError, dump, load
from pathlib import Path

from lib.types import Any, Optional, String

from .pathing import ManagedPath


class MetadataLock(ManagedPath):

    def __init__(
        self,
        metadata_path: Path | str,
    ):
        # TODO: In the event that using an XML or other format file
        # instead of JSON, append the .lock instead of replacing the
        # extension with the lock extension.
        super().__init__(Path(metadata_path).with_suffix(".lock"))

    @property
    def is_locked(
        self,
    ) -> bool:
        return self.path.exists()

    def acquire_lock(
        self,
    ) -> None:
        # TODO: Implement stale lock prevention by writing the PID to the
        # lock file, and checking the if the current PID matches the
        # written PID. If not, it is probably safe to delete the old
        # lock and create a new one.
        try:
            self.path.touch(exist_ok=False)
        except FileExistsError:
            raise TimeoutError(f"Resource locked: {self.path.name}")

    def release_lock(
        self,
    ) -> None:
        # Atomically delete the metadata lock file.
        # No need to overengineer this.
        self.path.unlink(missing_ok=True)


class MetadataFile(ManagedPath):
    """
    MetadataFile represents the file itself, and handles the path resolution.
    """

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        super().__init__(Path(path))
        self.lock = MetadataLock(self.path)

    def read(
        self,
        ignore_lock: bool = False,
    ) -> dict[str, Any]:
        try:
            if self.lock.is_locked and not ignore_lock:
                raise FileExistsError(f"Resource locked: {self.lock.path.name}")

            with self.path.open("r") as file:
                return dict(load(file))

        except (JSONDecodeError, ValueError, FileNotFoundError) as e:
            print(f"Error occurred while reading {self.path.name}: {e}")
            return dict()
        except FileExistsError as e:
            print(f"{e}")
            return dict()

    def write(
        self,
        metadata: dict[str, Any],
        overwrite: Optional[bool] = False,
    ) -> None:

        try:
            # 0. Acquire the lock
            self.lock.acquire_lock()

            # 1. Read existing data (if any) to preserve unrelated keys.
            current_metadata: dict[String, Any] = (
                self.read(ignore_lock=True) if not overwrite else dict()
            )

            # 2. Merge new metadata with the existing metadata.
            if metadata is None:
                raise ValueError(f"No metadata provided for {self.path.name}")

            updated_data = current_metadata | metadata

            # 3. Write fresh metadata.
            with self.path.open("w") as file:
                dump(
                    updated_data,
                    file,
                    indent=4,
                    sort_keys=True,
                )

        finally:
            # 4. Always release the lock
            self.lock.release_lock()
            self.lock.release_lock()
