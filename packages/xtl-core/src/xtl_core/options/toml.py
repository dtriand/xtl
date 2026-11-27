"""
TOML items for |Options|, written with ``tomlkit``.

.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
"""

from pathlib import PurePath
from typing import Any

import tomlkit
from tomlkit.exceptions import ConvertError
from tomlkit.items import Item

from xtl_core.system.os.permissions import FilePermissions


def toml_scalar(value: Any) -> Any:
    """
    A value that ``tomlkit`` can write: paths as strings in POSIX format, and
    :class:`FilePermissions` as octal strings (e.g. ``'750'``). Other values are
    returned as they are.
    """
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, FilePermissions):
        return value.octal[2:]
    return value


def comment_text(description: str) -> str:
    """
    A description as the text of an inline comment: its lines joined with spaces, since
    an inline comment can't span lines.
    """
    return ' '.join(line.strip() for line in description.splitlines() if line.strip())


def toml_item(value: Any, comment: str | None = None) -> Item:
    """
    A TOML item for a value (see :func:`toml_scalar`), with an optional inline comment.
    Values of types that ``tomlkit`` doesn't know are written as their ``str()``.

    :param value: The value. ``None`` has no TOML form; callers leave it out.
    :param comment: The text of the comment, without ``#``. Multi-line text is joined
        into one line, see :func:`comment_text`.
    """
    value = toml_scalar(value)
    try:
        item = tomlkit.item(value)
    except ConvertError:
        item = tomlkit.item(str(value))
    return with_comment(item, comment)


def with_comment(item: Item, comment: str | None) -> Item:
    """
    Add an inline comment to a TOML item (on the header line for a table), if there is
    any text.
    """
    text = comment_text(comment) if comment else ''
    if text:
        item.comment(text)
    return item
