"""
|Options| models whose data layout carries a version of its own.

.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from typing_extensions import Self

from xtl_core.options.base import Options
from xtl_core.version import VersionInfo

logger = logging.getLogger(__name__)


class VersionedOptions(Options):
    _version_path: ClassVar[str] = None
    _current_version: ClassVar[VersionInfo | None] = None
    """The current version of this model's data layout, a tag of its own (not the version
    of xtl). Data from a newer version triggers a warning when loaded. ``None`` (the
    default) skips the check."""

    @classmethod
    def _resolve_version(cls, data: dict) -> VersionInfo:
        node = data
        for part in cls._version_path.split('.'):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f'Could not find key `{part}` in the data at path `{cls._version_path}`')
            node = node[part]
        return VersionInfo.from_value(node)

    @classmethod
    def _check_version(cls, version: VersionInfo) -> None:
        current = cls._current_version
        if current is not None and version.tuple_safe > current.tuple_safe:
            logger.warning(
                "%(model_cls)s data is from version %(version)s, which is newer than the model's current version "
                '%(current)s.',
                {'model_cls': cls.__name__, 'version': version.pep440, 'current': current.pep440},
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        cls._check_version(cls._resolve_version(data))
        return super().from_dict(data)

    @classmethod
    def from_json(cls, s: str | Path) -> Self:
        if isinstance(s, str):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                # Not a valid JSON string, treat as a file path
                path = Path(s)
                if not path.exists():
                    raise FileNotFoundError(f'File not found: {path}')
                data = json.loads(path.read_text())
        else:
            path = Path(s)
            if not path.exists():
                raise FileNotFoundError(f'File not found: {path}')
            data = json.loads(path.read_text())

        cls._check_version(cls._resolve_version(data))
        return super().from_json(s)
