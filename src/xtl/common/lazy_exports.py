"""
Helpers for implementing lazy re-exports in package ``__init__.py`` files.

Typical usage:

    def __getattr__(name: str) -> Any:
        return lazy_getattr(__name__, _EXPORTS, globals(), name)

    def __dir__() -> list[str]:
        return lazy_dir(globals(), __all__)
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, MutableMapping, Sequence, TypeAlias

ExportTarget: TypeAlias = tuple[str, str]
ExportMap: TypeAlias = Mapping[str, ExportTarget]


def lazy_getattr(package_name: str, exports: ExportMap, namespace: MutableMapping[str, Any],
                 name: str) -> Any:
    """
    Resolve and cache a lazily exported symbol.

    :param package_name: Caller package/module name (usually ``__name__``).
    :param exports: Mapping of public symbol -> (relative module path, attribute name).
    :param namespace: Caller module namespace (usually ``globals()``).
    :param name: Requested symbol name.
    :raises AttributeError: If ``name`` is not declared in ``exports``.
    :return: The resolved symbol.
    """
    try:
        module_name, attr_name = exports[name]
    except KeyError as exc:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}") from exc

    # Fast-path, use cache
    cached = namespace.get(name)
    if cached is not None:
        return cached

    module = import_module(module_name, package_name)
    value = getattr(module, attr_name)
    namespace[name] = value  # cache in caller module globals
    return value


def lazy_dir(namespace: Mapping[str, Any], public_names: Sequence[str]) -> list[str]:
    """
    Build a ``__dir__`` result that includes lazily exported public names.

    :param namespace: Caller module namespace (usually ``globals()``).
    :param public_names: Public API names (usually ``__all__``).
    :return: Sorted list of visible names.
    """
    return sorted(set(namespace) | set(public_names))


def validate_exports(package_name: str, exports: ExportMap) -> None:
    """
    Optional helper for tests/CI to fail fast on broken export maps.
    """
    for public_name, (module_name, attr_name) in exports.items():
        module = import_module(module_name, package_name)
        if not hasattr(module, attr_name):
            raise AttributeError(
                f'Invalid lazy export {public_name!r} -> ({module_name!r}, {attr_name!r}) '
                f'in {package_name!r}'
            )
