from pathlib import Path
import platform
import os


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


def chmod_recursively(path: str | Path, permissions: int | str, directories_only: bool = False):
    """
    Change the permissions of all files and subdirectories within a directory.
    :param path: The path to the directory
    :param permissions: The desired permissions as a decimal
    :param directories_only: Whether to change the permissions of directories only
    """
    permissions = get_permissions_in_decimal(value=permissions)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'\'{path}\' does not exist')

    # Check if the desired permissions are more permissive than the current ones
    more_permissive = permissions >= int(path.stat().st_mode & 0o777)
    if more_permissive:
        # Change permissions from top to bottom of the tree
        path.lchmod(mode=permissions)
        if path.is_file():
            return
        for root, dirs, files in os.walk(path, topdown=True):
            if not directories_only:
                for file in files:
                    file = Path(root) / file
                    file.lchmod(mode=permissions)
            for directory in dirs:
                directory = Path(root) / directory
                directory.lchmod(mode=permissions)
    else:
        # Change permissions from bottom to top of the tree
        if path.is_file():
            path.lchmod(mode=permissions)
            return
        for root, dirs, files in os.walk(path, topdown=False):
            if not directories_only:
                for file in files:
                    file = Path(root) / file
                    file.lchmod(mode=permissions)
            for directory in dirs:
                directory = Path(root) / directory
                directory.lchmod(mode=permissions)
        path.lchmod(mode=permissions)
