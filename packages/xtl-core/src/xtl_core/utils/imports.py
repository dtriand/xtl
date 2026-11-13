"""
Helpers to refer to classes and functions by their import path, e.g. to send them to another process.
"""

import importlib
from typing import Any


__all__ = ['import_path', 'import_object']


def import_path(obj: Any) -> str:
    """
    The import path of a class or function, e.g. ``'logging.FileHandler'``. The object must be defined at module level
    for :func:`import_object` to find it again.

    :param obj: A class or function.
    """
    return f'{obj.__module__}.{obj.__qualname__}'


def import_object(path: str) -> Any:
    """
    Import a class, function or other module attribute from its import path, e.g. ``'logging.FileHandler'``.

    :param path: The module's dotted path, a dot, and the attribute's name.
    :raises ImportError: If the path is malformed, or the module or attribute can't be found.
    """
    module_path, _, attr = str(path).rpartition('.')
    if not module_path or not attr:
        raise ImportError(f'Invalid import path: {path!r}')
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f'Could not resolve {path!r}') from e
