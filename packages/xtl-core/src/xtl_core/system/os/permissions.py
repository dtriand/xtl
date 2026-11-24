"""
POSIX file permissions: parsing, conversion between their forms, and changing them
recursively.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from types import NoneType
from typing import Optional

from xtl_core.system.compatibility.py310 import StrEnum


__all__ = ['FileType', 'FilePermissionsBit', 'FilePermissions', 'get_permissions_in_decimal', 'chmod_recursively']


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
    'p': FileType.FIFO,
}
"""Mappings of POSIX file types to the respective `FileType` enum"""

_BITS: tuple[tuple[str, int], ...] = (('r', 0o4), ('w', 0o2), ('x', 0o1))
"""Letter and value of each permission bit, in the order they are written"""

_BIT_MAX_LENGTH = 3
"""Longest permission string for a single group, e.g. `'r-x'` or `'0o5'`"""

_BIT_OCTAL = re.compile(r'0o([0-7])')
"""A single octal digit with its prefix, e.g. `'0o5'`"""

_BIT_SYMBOLIC = re.compile(r'[r-][w-][x-]|r?w?x?|-')
"""Canonical (`'r-x'`) or short (`'rx'`, `''`, `'-'`) permission letters"""

# NB: The patterns are only used with `fullmatch()`, on strings no longer than
#  `_BIT_MAX_LENGTH`: no trailing newline slips through (as with `$`), and the
#  matching time is bounded whatever the input. They are also stricter than
#  `int(s, 8)`, which accepts whitespace, underscores and non-ASCII digits.


def _shorten(value: object, limit: int = 30) -> str:
    """
    The ``repr()`` of a value for error messages, cut to ``limit`` characters, so that a
    huge input doesn't end up in full in messages and logs.
    """
    r = repr(value)
    return r if len(r) <= limit else f'{r[: limit - 3]}...'


@dataclass(eq=True, order=True)
class FilePermissionsBit:
    """
    Representation of POSIX file permissions for a single group, e.g., permissions for
    owner.
    """

    value: int | str | None
    """Permission value in octal format: an int from 0 to 7, once initialized. Created
    from such an int, an octal string (`'0o5'`), canonical (`'r-x'`) or short (`'rx'`)
    permission letters, or `''`, `'-'` or `None` for no permissions."""

    def __post_init__(self):
        self.value = self._parse(self.value)

    @staticmethod
    def _parse(value: int | str | None) -> int:
        """
        Convert any of the accepted forms to an int from 0 to 7.

        :raises TypeError: If the value is not an int, str or None (bools included).
        :raises ValueError: If the value is not a valid permission.
        """
        if value is None:
            return 0o0
        if isinstance(value, str):
            if len(value) > _BIT_MAX_LENGTH:
                raise ValueError(f"'value' must be a valid permission string, not one of {len(value)} characters")
            if match := _BIT_OCTAL.fullmatch(value):
                return int(match.group(1), 8)
            if _BIT_SYMBOLIC.fullmatch(value):
                return sum(bit for letter, bit in _BITS if letter in value)
            raise ValueError(f"'value' must be a valid permission string, not {_shorten(value)}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"'value' must be an int or str, not {type(value)}")
        if not 0o0 <= value <= 0o7:
            raise ValueError(f"'value' must be an integer from 0 to 7, not {value}")
        return value

    def _set(self, bit: int, allowed: bool):
        """
        Set or clear one permission bit.
        """
        if not isinstance(allowed, bool):
            raise TypeError(f"'value' must be a bool, not {type(allowed)}")
        if allowed:
            self.value |= bit
        else:
            self.value &= ~bit

    @property
    def string(self) -> str:
        """
        Get the short string representation of the permission bit, e.g., `'rw'`.

        :return: The shorthand string representation of the permission bit.
        """
        return ''.join(letter for letter, bit in _BITS if self.value & bit) or '-'

    @property
    def string_canonical(self) -> str:
        """
        Get the canonical string representation of the permission bit, e.g., `'rw-'`.

        :return: The canonical string representation of the permission bit.
        """
        return ''.join(letter if self.value & bit else '-' for letter, bit in _BITS)

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
        self._set(0o4, value)

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
        self._set(0o2, value)

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
        self._set(0o1, value)

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

    owner: int | str | FilePermissionsBit = field(default_factory=lambda: FilePermissionsBit(0o0))
    group: int | str | FilePermissionsBit = field(default_factory=lambda: FilePermissionsBit(0o0))
    other: int | str | FilePermissionsBit = field(default_factory=lambda: FilePermissionsBit(0o0))
    file_type: Optional[str | FileType] = field(default=None, compare=False)

    def __post_init__(self):
        # Check if a single value is provided that contains all permissions
        value = getattr(self, 'owner')
        if isinstance(value, FilePermissions):
            self.owner = value.owner
            self.group = value.group
            self.other = value.other
            self.file_type = value.file_type
        elif isinstance(value, int):
            if value <= 0o7:
                # Single digit octal
                pass
            elif 0o7 < value < 0o777:
                self.owner, self.group, self.other = self._split_octal(value)
            else:
                raise ValueError(f'Permissions must be a 3-digit octal, not {value:#o}')
        elif isinstance(value, str):
            if (len(value) == 3 and value.isnumeric()) or (len(value) == 5 and value.startswith('0o')):
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
                    raise ValueError(f"'{name}' must be a valid permission string, not {_shorten(value)}") from e
            elif not isinstance(value, FilePermissionsBit):
                raise TypeError(
                    f"'{name}' must be an int, str or {FilePermissionsBit.__name__}, not {type(value).__name__}"
                )
        if self.file_type is not None:
            if isinstance(self.file_type, str) and self.file_type in _file_type_mappings:
                self.file_type = _file_type_mappings[self.file_type]
            elif not isinstance(self.file_type, FileType):
                raise TypeError(
                    f"'file_type' must be a str or {FileType.__class__.__name__}, not {type(self.file_type)}"
                )

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
        s = f'{self.owner.string_canonical}{self.group.string_canonical}{self.other.string_canonical}'
        if self.file_type is not None:
            s = f'{self.file_type.value}{s}'
        return s

    @property
    def tuple(self) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool], tuple[bool, bool, bool]]:
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
            owner=owner,
            group=group,
            other=other,
            file_type=file_type,
        )

    @classmethod
    def from_string(cls, s: str) -> 'FilePermissions':
        """
        Create a `FilePermissions` object from a string representation of permissions.

        :param s: The string representation of permissions, e.g., `'rwxr-xr--'` or
            `'700'`.
        :return: A `FilePermissions` object.
        """
        return cls(s)


def get_permissions_in_decimal(value: int | str | FilePermissions) -> int:
    """
    Convert an octal permission value to its decimal representation.
    """
    if not isinstance(value, (int, str, FilePermissions)):
        raise TypeError(f"'value' must be an int, str or {FilePermissions.__class__.__name__}, not {type(value)}")

    value = FilePermissions(value)
    return value.decimal


def chmod_recursively(
    path: str | Path,
    files_permissions: Optional[int | str | FilePermissions] = None,
    directories_permissions: Optional[int | str | FilePermissions] = None,
):
    """
    Change the permissions of all files and subdirectories within a directory. If symbolic links are encountered, they
    are skipped. Permissions are provided as 3-digit decimal integers.
    :param path: The path to the directory
    :param files_permissions: The desired permissions for files
    :param directories_permissions: The desired permissions for directories
    """
    if not files_permissions and not directories_permissions:
        raise ValueError("At least one of 'file_permissions' or 'directories_permissions' must be provided")
    files_permissions = get_permissions_in_decimal(value=files_permissions)
    directories_permissions = get_permissions_in_decimal(value=directories_permissions)

    # Check if path exists and skip if it is a symlink
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist")
    if path.is_symlink():  # Skip symbolic links
        return

    # Check if the desired permissions are more permissive than the current ones
    more_permissive = directories_permissions >= int(path.stat().st_mode & 0o777)
    if more_permissive:  # update the root first
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
