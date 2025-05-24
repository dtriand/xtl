from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import platform
from types import NoneType
from typing import Optional

from .compatibility import PY311_OR_LESS
if PY311_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


def get_os_name_and_version() -> str:
    if platform.system() == 'Linux':
        try:
            import distro
            return f'{distro.name()} {distro.version()}'
        except ModuleNotFoundError:
            return f'{platform.system()} {platform.version()}'
    else:
        return f'{platform.system()} {platform.version()}'


class FileType(StrEnum):
    """
    POSIX file type representation.
    """
    FILE = '-'
    DIRECTORY = 'd'
    SYMLINK = 'l'
    CHARACTER_DEVICE = 'c'
    BLOCK_DEVICE = 'b'
    SOCKET = 's'
    FIFO = 'p'

_file_type_mappings: dict[str, FileType] = {
    '-': FileType.FILE,
    'd': FileType.DIRECTORY,
    'l': FileType.SYMLINK,
    'c': FileType.CHARACTER_DEVICE,
    'b': FileType.BLOCK_DEVICE,
    's': FileType.SOCKET,
    'p': FileType.FIFO
}
"""Mappings of POSIX file types to the respective `FileType` enum"""

_permission_mappings: dict[int, str] = {
    0o0: '---', 0o1: '--x', 0o2: '-w-', 0o3: '-wx',
    0o4: 'r--', 0o5: 'r-x', 0o6: 'rw-', 0o7: 'rwx'
}
""""Mappings of POSIX permissions from octal to string representation"""

_permission_mappings_short: dict[int, str] = {
    0o0: '-', 0o1: 'x', 0o2: 'w', 0o3: 'wx',
    0o4: 'r', 0o5: 'rx', 0o6: 'rw', 0o7: 'rwx'
}
"""Mappings of POSIX permissions from octal to shorthand string representation"""

_valid_permission_strings: dict[str | None, int] = {
    '': 0o0, '-': 0o0, '---': 0o0, None: 0o0,
    'x': 0o1, '--x': 0o1,
    'w': 0o2, '-w-': 0o2,
    'wx': 0o3, '-wx': 0o3,
    'r': 0o4, 'r--': 0o4,
    'rx': 0o5, 'r-x': 0o5,
    'rw': 0o6, 'rw-': 0o6,
    'rwx': 0o7
}
"""Mapping of valid POSIX permission strings to their octal representation"""


@dataclass(eq=True, order=True)
class FilePermissionsBit:
    """
    Representation of POSIX file permissions for a single group, e.g., permissions for
    owner.
    """

    value: int | str | None
    """Permission value in octal format"""

    def __post_init__(self):
        self._parse_value()

    def _parse_value(self):
        """
        Parse the value to ensure it is a valid octal representation.
        """
        value = self.value
        if isinstance(value, int):
            if value < 0o0:
                raise ValueError(f'\'value\' must be a non-negative integer')
            if value > 0o7:
                raise ValueError(f'\'value\' must be a less than 8')
        elif isinstance(value, str) or value is None:
            if value in _valid_permission_strings:
                self.value = _valid_permission_strings[value]
            elif value.startswith('0o'):
                self.value = int(value, 8)
                self._parse_value()
            else:
                raise ValueError(f'\'value\' must be a valid permission string, '
                                 f'not {value!r}')
        else:
            raise TypeError(f'\'value\' must be an int or str, not {type(value)}')

    @property
    def string(self) -> str:
        """
        Get the short string representation of the permission bit, e.g., `'rw'`.

        :return: The shorthand string representation of the permission bit.
        """
        return _permission_mappings_short[self.value]

    @property
    def string_canonical(self) -> str:
        """
        Get the canonical string representation of the permission bit, e.g., `'rw-'`.

        :return: The canonical string representation of the permission bit.
        """
        return _permission_mappings[self.value]

    @property
    def octal(self) -> str:
        """
        Get the octal representation of the permission bit, e.g., `'0o6'`.

        :return: The octal representation of the permission bit.
        """
        return oct(self.value)

    @property
    def decimal(self) -> int:
        """
        Get the decimal representation of the permission bit, e.g., `6`.

        :return: The decimal representation of the permission bit.
        """
        return self.value

    @property
    def can_read(self) -> bool:
        """
        Check if the permission bit allows reading.

        :return: True if it has read permission, False otherwise.
        """
        return bool(self.value & 0o4)

    @can_read.setter
    def can_read(self, value: bool):
        """
        Set the permission bit to allow or disallow reading.
        """
        if not isinstance(value, bool):
            raise TypeError(f'\'value\' must be a bool, not {type(value)}')
        if value != self.can_read:
            self.value ^= 0o4  # Toggle the read permission bit

    @property
    def can_write(self) -> bool:
        """
        Check if the permission bit allows writing.

        :return: True if it has write permission, False otherwise.
        """
        return bool(self.value & 0o2)

    @can_write.setter
    def can_write(self, value: bool):
        """
        Set the permission bit to allow or disallow writing.
        """
        if not isinstance(value, bool):
            raise TypeError(f'\'value\' must be a bool, not {type(value)}')
        if value != self.can_write:
            self.value ^= 0o2  # Toggle the write permission bit

    @property
    def can_execute(self) -> bool:
        """
        Check if the permission bit allows executing.

        :return: True if it has execute permission, False otherwise.
        """
        return bool(self.value & 0o1)

    @can_execute.setter
    def can_execute(self, value: bool):
        """
        Set the permission bit to allow or disallow executing.
        """
        if not isinstance(value, bool):
            raise TypeError(f'\'value\' must be a bool, not {type(value)}')
        if value != self.can_execute:
            self.value ^= 0o1  # Toggle the execute permission bit

    @property
    def tuple(self) -> tuple[bool, bool, bool]:
        """
        Get the permission bit as a tuple of booleans (read, write, execute).

        :return: A tuple containing three boolean values representing read, write, and
            execute permissions.
        """
        return self.can_read, self.can_write, self.can_execute

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.string_canonical!r})'


@dataclass(eq=True, order=True)
class FilePermissions:
    """
    Representation of POSIX file permissions for owner, group, and other.
    """

    owner: int | str | FilePermissionsBit = field(
        default_factory=lambda: FilePermissionsBit(0o0))
    group: int | str | FilePermissionsBit = field(
        default_factory=lambda: FilePermissionsBit(0o0))
    other: int | str | FilePermissionsBit = field(
        default_factory=lambda: FilePermissionsBit(0o0))
    file_type: Optional[str | FileType] = field(default=None, compare=False)

    def __post_init__(self):
        # Check if a single value is provided that contains all permissions
        value = getattr(self, 'owner')
        if isinstance(value, int):
            if value <= 0o7:
                # Single digit octal
                pass
            elif 0o7 < value < 0o777:
                self.owner, self.group, self.other = self._split_octal(value)
            else:
                raise ValueError(f'Permissions must be a 3-digit octal, not {value:#o}')
        elif isinstance(value, str):
            if (len(value) == 3 and value.isnumeric()) or \
                    (len(value) == 5 and value.startswith('0o')):
                # e.g. '760' or '0o760'
                self.owner, self.group, self.other = self._split_octal(int(value, 8))
            elif len(value) == 9:
                # e.g. 'rwxrw----'
                self.owner = value[0:3]
                self.group = value[3:6]
                self.other = value[6:9]
            elif len(value) == 10:
                # e.g. 'drwxrw----' (includes file type)
                self.file_type = value[0]
                self.owner = value[1:4]
                self.group = value[4:7]
                self.other = value[7:10]

        # Validate and cast the values
        for name in ['owner', 'group', 'other']:
            value = getattr(self, name)
            if isinstance(value, (int, str, NoneType)):
                try:
                    value = FilePermissionsBit(value)
                    setattr(self, name, value)
                except ValueError as e:
                    raise ValueError(f'\'{name}\' must be a valid permission string, '
                                     f'not {value!r}') from e
            elif not isinstance(value, FilePermissionsBit):
                raise TypeError(f'\'{name}\' must be an int, str or '
                                f'{FilePermissionsBit.__class__.__name__}, not '
                                f'{type(value)}')
        if self.file_type is not None:
            if isinstance(self.file_type, str) and self.file_type in _file_type_mappings:
                self.file_type = _file_type_mappings[self.file_type]
            elif not isinstance(self.file_type, FileType):
                raise TypeError(f'\'file_type\' must be a str or '
                                f'{FileType.__class__.__name__}, not '
                                f'{type(self.file_type)}')

    @staticmethod
    def _split_octal(value: int) -> tuple[int, int, int]:
        """
        Split a 3-digit octal value into owner, group, and other permission digits.

        :param value: The octal value in integer representation (e.g., `0o760` or `496`
            for
        :return: A tuple containing the owner, group, and other permission digits.
        """
        owner = (value >> 6) & 0o7
        group = (value >> 3) & 0o7
        other = value & 0o7
        return owner, group, other

    @property
    def octal(self) -> str:
        """
        Get the octal representation of the file permissions, e.g., `'0o644'`.

        :return: The octal representation of the file permissions.
        """
        return f'0o{self.owner.octal[2:]}{self.group.octal[2:]}{self.other.octal[2:]}'

    @property
    def decimal(self) -> int:
        """
        Get the decimal representation of the file permissions, e.g., `420` for `0o640`.

        :return: The decimal representation of the file permissions.
        """
        return int(self.octal, 8)

    @property
    def string(self) -> str:
        """
        Get the string representation of the file permissions, e.g., `'rwxr-xr--'`. If
        `file_type` is set, then it is included in the string representation (length 10),
        otherwise it is omitted (length 9).

        :return: The string representation of the file permissions.
        """
        s = (f'{self.owner.string_canonical}{self.group.string_canonical}'
             f'{self.other.string_canonical}')
        if self.file_type is not None:
            s = f'{self.file_type.value}{s}'
        return s

    @property
    def tuple(self) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool],
        tuple[bool, bool, bool]]:
        """
        Get the permissions as three (read, write, execute) tuples for owner, group and
        other.

        :return: A tuple containing three tuples, each representing the permissions for
            owner, group, and other.
        """
        return self.owner.tuple, self.group.tuple, self.other.tuple

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.string!r})'

    @classmethod
    def from_path(cls, p: str | Path) -> 'FilePermissions':
        """
        Create a `FilePermissions` object from a file or directory path.

        :param p: The path to the file or directory.
        :return: A `FilePermissions` object.
        """
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f'File does not exist: {p}')

        file_type = None
        if p.is_file():
            file_type = FileType.FILE
        elif p.is_dir():
            file_type = FileType.DIRECTORY
        elif p.is_symlink():
            file_type = FileType.SYMLINK
        elif p.is_char_device():
            file_type = FileType.CHARACTER_DEVICE
        elif p.is_block_device():
            file_type = FileType.BLOCK_DEVICE
        elif p.is_socket():
            file_type = FileType.SOCKET
        elif p.is_fifo():
            file_type = FileType.FIFO

        permissions = p.stat().st_mode & 0o777  # integer representation
        permissions = oct(permissions)[2:]  # octal string
        owner = int(f'0o{permissions[0]}', 8)
        group = int(f'0o{permissions[1]}', 8)
        other = int(f'0o{permissions[2]}', 8)
        return cls(
            owner=_permission_mappings[owner],
            group=_permission_mappings[group],
            other=_permission_mappings[other],
            file_type=file_type
        )


def get_permissions_in_decimal(value: int | str | FilePermissions) -> int:
    """
    Convert an octal permission value to its decimal representation.
    """
    if not isinstance(value, (int, str, FilePermissions)):
        raise TypeError(f'\'value\' must be an int, str or '
                        f'{FilePermissions.__class__.__name__}, not {type(value)}')

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


def chmod_recursively(path: str | Path,
                      files_permissions: Optional[int | str | FilePermissions] = None,
                      directories_permissions: Optional[int | str | FilePermissions] = None):
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
