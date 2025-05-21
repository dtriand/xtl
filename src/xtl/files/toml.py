from pathlib import *
from types import NoneType

import toml

from xtl.common.os import FilePermissions


class ExtendedTomlEncoder(toml.TomlPreserveCommentEncoder):

    def __init__(self, _dict=dict, preserve: bool = False):
        super().__init__(_dict=_dict, preserve=preserve)

        # Nones are serialized as empty strings because the toml lib
        #  does not support missing values
        self.dump_funcs[NoneType] = lambda v: '""'

        # Path objects are serialized as strings in POSIX format
        self.dump_funcs[Path] = lambda v: f'"{v.as_posix()}"'
        self.dump_funcs[PurePath] = lambda v: f'"{v.as_posix()}"'
        self.dump_funcs[WindowsPath] = lambda v: f'"{v.as_posix()}"'
        self.dump_funcs[PosixPath] = lambda v: f'"{v.as_posix()}"'
        self.dump_funcs[PureWindowsPath] = lambda v: f'"{v.as_posix()}"'
        self.dump_funcs[PurePosixPath] = lambda v: f'"{v.as_posix()}"'

        # FilePermissions objects are serialized as octal strings
        self.dump_funcs[FilePermissions] = lambda v: f'"{v.octal[2:]}"'
