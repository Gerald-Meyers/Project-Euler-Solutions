from hashlib import _Hash, sha256
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from typing import Any, Generator, Optional

from .pathing import HashFile, LockFile, ManagedPath


class MetadataFile:
    """
    MetadataFile represents the metadata_file itself, and handles the path resolution.
    """

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        # MetadataFile uses two features, a lock file to prevent
        # concurrent writes, and a hash file to verify that the contents
        # of the hash file between writes.
        self.path = Path(path)
        self.lock = LockFile(self.path)
        self.hash = HashFile(self.path)

    def read(
        self,
        ignore_lock: bool = False,
    ) -> dict[str, Any]:
        """
        Return a Dictionary containing the contents of the metadata JSON file.

        :param self: Description
        :param ignore_lock: Description
        :type ignore_lock: bool
        :return: Description
        :rtype: dict[str, Any]
        """
        try:
            # If the lockfile exists, then raise FileExistsError, unless
            # the ignore_lock flag has been set to True.
            if self.lock.is_locked and not ignore_lock:
                raise FileExistsError(f"Resource locked: {self.lock.path.name}")

            # Load the metadata file in 8kB chunks, while updating the
            # hash object.

            # If the stored hash does not match the evaluated hash of the
            # stored metadata file, then the file may be corrupted or
            # tampered with.

            # A valid security improvement here would be to either store,
            # or entirely evaluate the hash of the metadata file off site.

            # If a file corruption takes place after the metadata is
            # written, it is unlikely to totally ruin the contents, and
            # may be recovered via manual or automated intervention.

            # If the metadata file and hash are both stored locally, a
            # malicious actor could modify both to cause system breaking
            # modifications.
            self.hash.verify_file()

            # Assuming that the metadata file is unmodified since the
            # hash was written, then load the contents as a dictionary.
            with self.path.open("r") as metadata_file:
                metadata = loads(metadata_file.read())

            return dict(metadata)

        # There has been an error.
        # Probably while reading the file itself, or while verifying the
        # metadata file is unmodified.
        except (JSONDecodeError, ValueError, FileNotFoundError) as e:
            # Ingore the file contents and simply return an empty dictionary.
            # Logging and alternative recovery mechanisms may be placed here.
            print(f"Error occurred while reading {self.path.name}: {e}")
            return dict()

        # There has been an error.
        # Probably from the first check.
        except FileExistsError as e:
            # Ingore the file contents and simply return an empty dictionary.
            # Logging and alternative recovery mechanisms may be placed here.
            print(f"{e}")
            return dict()

    def write(
        self,
        metadata: dict[str, Any],
        overwrite: Optional[bool] = False,
    ) -> None:

        try:
            # Acquire the lock
            self.lock.acquire_lock()

            # Read existing data (if any) to preserve unrelated keys.
            current_metadata: dict[str, Any] = (
                self.read(ignore_lock=True) if not overwrite else dict()
            )

            # Merge new metadata with the existing metadata.
            if metadata is None:
                raise ValueError(f"No metadata provided for {self.path.name}")

            updated_data = current_metadata | metadata

            json_bytes = dumps(
                updated_data,
                indent=4,
                sort_keys=True,
            ).encode("utf-8")

            # Write fresh metadata.
            with self.path.open("wb") as metadata_file:
                metadata_file.write(json_bytes)

            self.hash.verify_file()

        finally:
            # Always release the lock
            self.lock.release_lock()
