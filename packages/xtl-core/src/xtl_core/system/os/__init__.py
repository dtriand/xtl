"""
Information about the operating system and the user, and file permissions (from
:mod:`xtl_core.system.os.permissions`).
"""

import os
from pathlib import Path
import platform
import tempfile

from xtl_core.system.os.permissions import (
    FileType,
    FilePermissionsBit,
    FilePermissions,
    get_permissions_in_decimal,
    chmod_recursively,
)


__all__ = [
    'CPU_CORES',
    'get_os_name_and_version',
    'get_username',
    'get_temp_location',
    'is_temp',
    'FileType',
    'FilePermissionsBit',
    'FilePermissions',
    'get_permissions_in_decimal',
    'chmod_recursively',
]


CPU_CORES: int | None = os.cpu_count()
"""Number of CPU cores available on the system, used for parallel job execution."""


def get_os_name_and_version() -> str:
    if platform.system() == 'Linux':
        try:
            import distro

            return f'{distro.name()} {distro.version()}'
        except ModuleNotFoundError:
            return f'{platform.system()} {platform.version()}'
    else:
        return f'{platform.system()} {platform.version()}'


def get_username() -> str:
    try:
        return os.getlogin()
    except OSError:
        # Fallback for CI/CD environments where `os.getlogin()` fails
        import getpass

        return getpass.getuser()


def get_temp_location() -> Path:
    """
    Get the system's temporary directory as a `Path` object.
    """
    temp_dir = tempfile.gettempdir()
    return Path(temp_dir)


def is_temp(f: str | Path) -> bool:
    """
    Check if the given file path is located within the system's temporary directory.
    Both are resolved first, so links and short names (*e.g.* macOS's ``/var`` for
    ``/private/var``, or ``RUNNER~1`` on Windows) don't matter.

    :param f: The path to check.
    """
    temp_location = get_temp_location().resolve()
    f = Path(f).resolve()
    return temp_location in f.parents
