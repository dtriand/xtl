from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xtl.common.lazy_exports import lazy_getattr, lazy_dir

__all__ = [
    'FileContainer',
    'FileContainerType',
    'FileTypeEnum',
    'DataType',
    'FileReaderMeta',
    'mmCIF',
    'NpxFile',
    'ExtendedTomlEncoder',
]

if TYPE_CHECKING:
    from .meta import FileContainer, FileContainerType, FileTypeEnum, DataType, FileReaderMeta
    from .mmcif import mmCIF
    from .npx import NpxFile
    from .toml import ExtendedTomlEncoder


# symbol -> (module, attr)
_EXPORTS = {
    'FileContainer': ('.meta', 'FileContainer'),
    'FileContainerType': ('.meta', 'FileContainerType'),
    'FileTypeEnum': ('.meta', 'FileTypeEnum'),
    'DataType': ('.meta', 'DataType'),
    'FileReaderMeta': ('.meta', 'FileReaderMeta'),
    'mmCIF': ('.mmcif', 'mmCIF'),
    'NpxFile': ('.npx', 'NpxFile'),
    'ExtendedTomlEncoder': ('.toml', 'ExtendedTomlEncoder'),
}


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, _EXPORTS, globals(), name)


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
