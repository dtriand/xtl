"""
POSIX file permissions: parsing, conversion between their forms, and changing them
recursively.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import stat
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


_BITS: tuple[tuple[str, int], ...] = (('r', 0o4), ('w', 0o2), ('x', 0o1))
"""Letter and value of each permission bit, in the order they are written"""

_BIT_MAX_LENGTH = 3
"""Longest permission string for a single group, e.g. `'r-x'` or `'0o5'`"""

_BIT_OCTAL = re.compile(r'0o([0-7])')
"""A single octal digit with its prefix, e.g. `'0o5'`"""

_BIT_SYMBOLIC = re.compile(r'[r-][w-][x-]|r?w?x?|-')
"""Canonical (`'r-x'`) or short (`'rx'`, `''`, `'-'`) permission letters"""

_MODE_MAX_LENGTH = 10
"""Longest permission string for owner, group and other, e.g. `'drwxr-x---'`"""

_MODE_OCTAL = re.compile(r'(?:0o|0)?([0-7]{3})')
"""Three octal digits, optionally with a prefix, e.g. `'750'`, `'0750'` or `'0o750'`"""

_MODE_SYMBOLIC = re.compile(
    rf'([{re.escape("".join(t.value for t in FileType))}])?([r-][w-][x-])([r-][w-][x-])([r-][w-][x-])'
)
"""Canonical permission letters, optionally with the file type, e.g. `'drwxr-x---'`"""

# NB: The patterns are only used with `fullmatch()`, on strings no longer than
#  `_BIT_MAX_LENGTH` or `_MODE_MAX_LENGTH`: no trailing newline slips through (as
#  with `$`), and the matching time is bounded whatever the input. They are also
#  stricter than `int(s, 8)` and `str.isnumeric()`, which accept whitespace,
#  underscores and non-ASCII digits.


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

    Created from the three parts, each as for :class:`FilePermissionsBit` (e.g.
    ``FilePermissions(7, 'r-x', 0)``), or from a single value for all of them: an int
    (``0o750``), three octal digits (``'750'``, ``'0750'`` or ``'0o750'``), canonical
    letters with or without the file type (``'rwxr-x---'``, ``'drwxr-x---'``) or another
    ``FilePermissions``. A single value that is none of these is the owner's part
    (``'rwx'`` is ``0o700``). The parts are always copies: changing a
    :class:`FilePermissionsBit` passed in doesn't change the permissions.
    """

    owner: int | str | FilePermissionsBit | None = None
    group: int | str | FilePermissionsBit | None = None
    other: int | str | FilePermissionsBit | None = None
    file_type: Optional[str | FileType] = field(default=None, compare=False)

    def __post_init__(self):
        if self.group is None and self.other is None:
            parts, file_type = self._parse_whole(self.owner)
        else:
            parts, file_type = (self.owner, self.group, self.other), None

        explicit_type = self._parse_file_type(self.file_type)
        if explicit_type is not None and file_type is not None and explicit_type is not file_type:
            raise ValueError(f"'file_type' is {explicit_type.value!r}, but the permissions give {file_type.value!r}")
        self.file_type = explicit_type if explicit_type is not None else file_type

        for name, part in zip(('owner', 'group', 'other'), parts):
            setattr(self, name, self._parse_part(name, part))

    @classmethod
    def _parse_whole(cls, value: object) -> tuple[tuple[object, object, object], FileType | None]:
        """
        Split a single value into the owner, group and other parts, and the file type if
        it gives one. A value that isn't permissions as a whole is the owner's part.

        :raises ValueError: If the value is an int out of range.
        """
        if isinstance(value, FilePermissions):
            return (value.owner, value.group, value.other), value.file_type
        if isinstance(value, int) and not isinstance(value, bool):
            if not 0o000 <= value <= 0o777:
                raise ValueError(f'Permissions must be from 0o000 to 0o777, not {value:#o}')
            return cls._split_octal(value), None
        if isinstance(value, str) and len(value) <= _MODE_MAX_LENGTH:
            if match := _MODE_OCTAL.fullmatch(value):
                return cls._split_octal(int(match.group(1), 8)), None
            if match := _MODE_SYMBOLIC.fullmatch(value):
                file_type = match.group(1)
                return match.group(2, 3, 4), FileType(file_type) if file_type else None
        return (value, None, None), None

    @staticmethod
    def _parse_part(name: str, part: object) -> FilePermissionsBit:
        """
        Cast one part to a new :class:`FilePermissionsBit`.

        :raises TypeError: If the part has the wrong type.
        :raises ValueError: If the part isn't valid permissions.
        """
        if isinstance(part, FilePermissionsBit):
            return FilePermissionsBit(part.value)
        if part is None or isinstance(part, (int, str)):
            try:
                return FilePermissionsBit(part)
            except ValueError as e:
                raise ValueError(f"'{name}' must be a valid permission string, not {_shorten(part)}") from e
        raise TypeError(f"'{name}' must be an int, str or {FilePermissionsBit.__name__}, not {type(part).__name__}")

    @staticmethod
    def _parse_file_type(value: object) -> FileType | None:
        """
        Cast the file type to a :class:`FileType`.

        :raises TypeError: If it isn't a str or :class:`FileType`.
        :raises ValueError: If it isn't one of the file type letters.
        """
        if value is None or isinstance(value, FileType):
            return value
        if isinstance(value, str):
            try:
                return FileType(value)
            except ValueError:
                letters = ', '.join(repr(t.value) for t in FileType)
                raise ValueError(f"'file_type' must be one of {letters}, not {_shorten(value)}") from None
        raise TypeError(f"'file_type' must be a str or {FileType.__name__}, not {type(value).__name__}")

    @staticmethod
    def _split_octal(value: int) -> tuple[int, int, int]:
        """
        Split a 3-digit octal value into owner, group, and other permission digits.

        :param value: The octal value in integer representation (e.g., `0o760`, or `496`
            for the same).
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
        return f'0o{self.decimal:03o}'

    @property
    def decimal(self) -> int:
        """
        Get the decimal representation of the file permissions, e.g., `420` for `0o640`.

        :return: The decimal representation of the file permissions.
        """
        return self.owner.value << 6 | self.group.value << 3 | self.other.value

    @property
    def string(self) -> str:
        """
        Get the string representation of the file permissions, e.g., `'rwxr-xr--'`. If
        `file_type` is set, then it is included in the string representation (length 10),
        otherwise it is omitted (length 9).

        :return: The string representation of the file permissions.
        """
        s = ''.join(part.string_canonical for part in (self.owner, self.group, self.other))
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

        # Follows symbolic links, so their target's type and permissions
        mode = p.stat().st_mode
        letter = stat.filemode(mode)[0]  # '?' for an unknown type
        file_type = next((t for t in FileType if t.value == letter), None)
        return cls(mode & 0o777, file_type=file_type)

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
        raise TypeError(f"'value' must be an int, str or {FilePermissions.__name__}, not {type(value)}")

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
