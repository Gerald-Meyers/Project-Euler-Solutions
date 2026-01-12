from os import PathLike
from pathlib import Path


class ManagedPath(PathLike):
    """
    Base class to handle path resolution.
    """

    def __init__(
        self,
        path: Path | str,
    ):
        # Centralized resolve logic so that the code is more robust.
        self.path = Path(path)

    def __fspath__(self) -> str:
        # REQUIRED: Return the string representation of the path
        return str(self.path)

    def __str__(self) -> str:
        # Human-readable string (delegates to the internal Path object)
        return str(self.path)

    def __repr__(self) -> str:
        # Debugging string (shows class name and current path)
        return f"{self.__class__.__name__}({self.path!r})"
