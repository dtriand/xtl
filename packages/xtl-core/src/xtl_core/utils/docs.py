"""
Helpers for documentation.
"""

import inspect
from typing import Any, Callable, TypeVar


_F = TypeVar('_F', bound=Callable[..., Any])


def extend_docstring(parent: Callable, *, note: str, params: dict[str, str]) -> Callable[[_F], _F]:
    """
    Copy ``parent``'s docstring, add ``note`` after its description and ``params`` before
    its first ``:return:``/``:raises:`` line.
    """

    def decorator(func: _F) -> _F:
        lines = inspect.cleandoc(parent.__doc__ or '').splitlines()
        fields = next((i for i, line in enumerate(lines) if line.startswith(':')), len(lines))
        returns = next((i for i, line in enumerate(lines) if line.startswith((':return', ':raises'))), len(lines))
        new_params = [f':param {name}: {text}' for name, text in params.items()]
        # Keep the note a paragraph of its own, with or without a blank line after the description
        description = lines[:fields]
        while description and not description[-1].strip():
            description.pop()
        doc = [*description, '', note, '', *lines[fields:returns], *new_params, *lines[returns:]]
        func.__doc__ = '\n'.join(doc).strip()
        return func

    return decorator
