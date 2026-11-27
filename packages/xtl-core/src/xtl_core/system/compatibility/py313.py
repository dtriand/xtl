"""
Stand-ins for what Python 3.13 lacks, imported from here on every version. Delete this module when dropping support
for Python 3.13, and use the :class:`~pathlib.Path` methods instead.

Before Python 3.14, ``Path.exists()``, ``is_file()`` and ``is_dir()`` raise :class:`OSError` for some paths instead of
returning ``False``, *e.g.* names that are too long on Linux and macOS (`Querying file type and status
<https://docs.python.org/3.14/library/pathlib.html#querying-file-type-and-status>`_). Use these for paths from users;
paths that xtl builds can keep the methods.
"""

import os

__all__ = ['path_exists', 'path_is_file', 'path_is_dir']


# NB: `os.path` returns False for any `OSError` on every version, and Python 3.14's `Path` methods call it


def path_exists(path: str | os.PathLike) -> bool:
    """:meth:`pathlib.Path.exists` that returns ``False`` instead of raising."""
    return os.path.exists(path)


def path_is_file(path: str | os.PathLike) -> bool:
    """:meth:`pathlib.Path.is_file` that returns ``False`` instead of raising."""
    return os.path.isfile(path)


def path_is_dir(path: str | os.PathLike) -> bool:
    """:meth:`pathlib.Path.is_dir` that returns ``False`` instead of raising."""
    return os.path.isdir(path)
