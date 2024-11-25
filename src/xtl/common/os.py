import os
from pathlib import Path
import platform
from typing import Optional


def get_os_name_and_version() -> str:
    if platform.system() == 'Linux':
        try:
            import distro
            return f'{distro.name()} {distro.version()}'
        except ModuleNotFoundError:
            return f'{platform.system()} {platform.version()}'
    else:
        return f'{platform.system()} {platform.version()}'


def get_permissions_in_decimal(value: int | str) -> int:
    """
    Convert an octal permission value to its decimal representation.
    """
    if not isinstance(value, (int, str)):
        raise TypeError(f'\'value\' must be an int or str, not {type(value)}')

    # Convert to string and check if it is a number
    value = str(value)
    if value.startswith('0o'):  # Remove the '0o' octal representation prefix if present
        value = value[2:]
    if not value.isdigit() or len(value) != 3:
        raise ValueError(f'\'value\' must be a 3-digit integer')

    # Check for a valid permission value
    for digit in value:
        if int(digit) not in range(8):
            raise ValueError(f'\'value\' must be a 3-digit integer with each digit in the range 0-7')
    return int(f'0o{value}', 8)  # Return octal in the decimal representation


def chmod_recursively(path: str | Path, files_permissions: Optional[int | str] = None,
                      directories_permissions: Optional[int | str] = None):
    """
    Change the permissions of all files and subdirectories within a directory. If symbolic links are encountered, they
    are skipped. Permissions are provided as 3-digit decimal integers.
    :param path: The path to the directory
    :param files_permissions: The desired permissions for files
    :param directories_permissions: The desired permissions for directories
    """
    if not files_permissions and not directories_permissions:
        raise ValueError('At least one of \'file_permissions\' or \'directories_permissions\' must be provided')
    files_permissions = get_permissions_in_decimal(value=files_permissions)
    directories_permissions = get_permissions_in_decimal(value=directories_permissions)

    # Check if path exists and skip if it is a symlink
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'\'{path}\' does not exist')
    if path.is_symlink():  # Skip symbolic links
        return

    # Check if the desired permissions are more permissive than the current ones
    more_permissive = directories_permissions >= int(path.stat().st_mode & 0o777)
    if more_permissive:   # update the root first
        if path.is_file():
            path.chmod(mode=files_permissions) if files_permissions else None
            return
        path.chmod(mode=directories_permissions) if directories_permissions else None
    # Walk through the directory top-down when increasing the permissions, bottom-up otherwise
    for root, dirs, files in os.walk(path, topdown=more_permissive):
        if files_permissions:
            for file in files:
                file = Path(root) / file
                if file.is_symlink():
                    continue
                file.chmod(mode=files_permissions)
        if directories_permissions:
            for directory in dirs:
                directory = Path(root) / directory
                if directory.is_symlink():
                    continue
                directory.chmod(mode=directories_permissions)
    if not more_permissive:  # update the root last
        if path.is_file():
            path.chmod(mode=files_permissions) if files_permissions else None
            return
        path.chmod(mode=directories_permissions) if directories_permissions else None
