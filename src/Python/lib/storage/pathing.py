from datetime import datetime
from hashlib import sha256
from os import PathLike, getpid, kill
from pathlib import Path
from typing import Any, Iterator, Self


class HashFile:
    """
    Manage a hash-file with file extension .sha256 based for a given file path.
    """

    # Size in bytes to load the ground truth file.
    byte_chunk_size: int = 8192  # 8 kB

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        # Add the "base_path" to the class attributes so the file itself can be hashed.
        self._path = Path(path)
        self.path = self._path.with_suffix(".sha256")

    def read_hash_from_file(
        self,
    ) -> str:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Hash metadata_file for metadata not located at {self.path}"
            )

        with self.path.open("r") as hash_file:
            return hash_file.read()

    def write_hash_to_file(
        self,
        hash_value: str | int,
        overwrite_hashfile: bool = False,
    ) -> None:

        if self.path.exists() and not overwrite_hashfile:
            raise FileExistsError(
                f"Hash metadata_file for metadata already exists at {self.path}"
            )

        if overwrite_hashfile:
            self.delete()

        with self.path.open("w") as hash_file:
            hash_file.writelines(str(hash_value))

    def delete(
        self,
    ) -> None:

        self.path.unlink(missing_ok=True)

    def verify_file(
        self,
    ) -> bool:
        # The stored hash file must exist AND the file being hashed must exist.
        if not self.path.exists() or not self._path.exists():
            return False

        try:

            return self == self.read_hash_from_file().strip()

        except (OSError, ValueError) as e:
            # Like most of the common exceptions which need to be
            # properly logged, for now simply print to the console.
            print(e)
            return False

    def __eq__(
        self,
        hash_value: str,
    ) -> bool:
        # Compare the hash values of the file to a provided reference
        # hash value.
        return self.compute_hash() == hash_value

    def compute_hash(
        self,
    ) -> str:
        """
        Hash the contents of the file.

        :param self: Description
        :return: Description
        :rtype: str
        """

        try:
            # If the hash_value has been cached, return it.
            return self.hash_value

        except NameError:
            # If the hash_value has not been cached, calculate it,
            # then return it.
            self._update_hash_value()
            return self.hash_value

    def _update_hash_value(self) -> None:
        content = sha256()
        for chunk in self._read_bytes():
            content.update(chunk)
        self.hash_value = content.hexdigest()

    def _read_bytes(self) -> Iterator[bytes]:
        with self._path.open("rb") as _file:
            while chunk := _file.read(self.byte_chunk_size):
                yield chunk


class LockFile:
    """
    Manage a lock file with a .lock extension.
    """

    _lock_timeout_seconds = 60

    def __init__(
        self,
        metadata_path: Path | str,
    ):
        # TODO: In the event that using an XML or other format metadata_file
        # instead of JSON, append the .lock instead of replacing the
        # extension with the lock extension.
        self._path = Path(metadata_path)
        self.path = self._path.with_suffix(".lock")

        self._pid = getpid()  # Grab the Process ID and save it to a local variable.

    @property
    def is_locked(
        self,
    ) -> bool:
        return self.path.exists()

    def acquire_lock(
        self,
    ) -> None:
        # TODO: Implement stale lock prevention by writing the PID to the
        # lock metadata_file, and checking the if the current PID matches the
        # written PID. If not, it is probably safe to delete the old
        # lock and create a new one.
        try:
            self.path.touch(exist_ok=False)
            self._write_lock_info()

        except FileExistsError:
            if self._is_stale():
                # lock is stale, assuming safe to remove.
                self.release_lock(ignore_lock=True)

            raise TimeoutError(f"Resource locked: {self.path.name}")

    def release_lock(
        self,
        ignore_lock=False,
    ) -> None:
        # If this PID owns the lock file, or if we can ignore which PID owns it, then remove it.
        if self.verify_lock_file() or ignore_lock:
            self.path.unlink(missing_ok=True)
            return

        # TODO: Unknown error should go here!
        raise RuntimeError(f"Could not release lock: {self.path}")

    def _write_lock_info(
        self,
    ) -> None:
        # PID
        # floating-point number representing the current local time as seconds since the Unix epoch
        with self.path.open("w") as lock_file:
            lock_file.write(f"{self._pid}\n{datetime.now().timestamp()}")

    def _read_lock_info(
        self,
    ) -> tuple[int, float]:
        with self.path.open("r") as lock_file:
            lock_file_lines = lock_file.read().strip().splitlines()

            if len(lock_file_lines) != 2:
                raise ValueError(
                    f"Lock file is corrupted and has {len(lock_file_lines)} lines; expected 2 lines"
                )

            pid, timestamp = lock_file_lines
            return (int(pid), float(timestamp))

    def _is_stale(
        self,
    ) -> bool | None:

        if not self.path.exists():
            return False  # File does not exist, therefore not stale.

        try:
            timestamp = self._read_lock_info()[1]
            return (datetime.now().timestamp() - timestamp) > self._lock_timeout_seconds

        except (ValueError, FileNotFoundError, OSError):
            # ignore the error and return True (stale)
            # common errors mean that the file is stale.
            return True
        except Exception as e:
            print(f"Unexpected error has occured.\n{e}")

    def _check_pid(
        self,
    ) -> bool:

        return self._read_lock_info()[0] == self._pid

    def verify_lock_file(
        self,
    ) -> bool:
        if not self.path.exists():
            return False

        try:
            pid, timestamp = self._read_lock_info()
            return (pid == self._pid) and (
                (datetime.now().timestamp() - timestamp) < self._lock_timeout_seconds
            )  # Both the PID must be the same and the age of the lock must be young enough.

        except (ValueError, FileNotFoundError, OSError):
            # ignore the error and return False (bad lock)
            # common errors mean that the file is not usable.
            return False

        except Exception as e:
            print(f"Unexpected error has occured.\n{e}")
            return False  # File is not valid for unexpected reason.

    def refresh(
        self,
    ) -> None:
        # Mostly a prototype function to be implemented in greater
        # detail when larger writes are necessary.
        if not self.path.exists():
            raise FileNotFoundError(
                f"Cannot refresh lock; path not found at {self.path}"
            )

        # Verify the current PID owns the lock.
        if not self._check_pid():
            raise FileExistsError(f"Lock file belongs to a different process.")

        # Update timestamp
        self._write_lock_info()
