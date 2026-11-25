# ruff: noqa: F403, F405 (star imports)
from pathlib import *
from types import NoneType

import toml

from xtl_core.system.os.permissions import FilePermissions


def _basic_string(text: str) -> str:
    """
    A TOML basic string: ``"`` and ``\\`` escaped, control characters as ``\\uXXXX``.
    """
    escaped = []
    for char in text:
        if char in '"\\':
            escaped.append(f'\\{char}')
        elif char < ' ' or char == '\x7f':
            escaped.append(f'\\u{ord(char):04x}')
        else:
            escaped.append(char)
    return f'"{"".join(escaped)}"'


class ExtendedTomlEncoder(toml.TomlPreserveCommentEncoder):
    def __init__(self, _dict=dict, preserve: bool = False):
        super().__init__(_dict=_dict, preserve=preserve)

        # Nones are serialized as empty strings because the toml lib
        #  does not support missing values
        self.dump_funcs[NoneType] = lambda v: '""'

        # Path objects are serialized as strings in POSIX format
        for type_ in (Path, PurePath, WindowsPath, PosixPath, PureWindowsPath, PurePosixPath):
            self.dump_funcs[type_] = lambda v: _basic_string(v.as_posix())

        # FilePermissions objects are serialized as octal strings
        self.dump_funcs[FilePermissions] = lambda v: f'"{v.octal[2:]}"'
