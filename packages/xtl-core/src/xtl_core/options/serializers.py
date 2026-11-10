"""
This module defines custom serializers for use with `Pydantic` models.

.. |FilePermissions| replace:: :class:`FilePermissions <xtl_core.system.os.FilePermissions>`
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xtl_core.system.os import FilePermissions


def PermissionOctal(x: 'FilePermissions') -> str:
    """
    Serializes an |FilePermissions| instance to an octal string.

    :param x: |FilePermissions| instance
    :return: Octal string representation of the permissions
    """
    return x.octal[2:]
