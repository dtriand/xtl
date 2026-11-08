"""
Make a package importable under a second name, without loading any module twice.

``alias_package('xtl.core', 'xtl_core')`` makes ``import xtl.core.options`` return the
very module object of ``xtl_core.options``. Modules keep their canonical names
(``__name__``, ``__spec__``, the ``__module__`` of their classes), so pickles and
docs always refer to the target package. Nothing is imported before it's used.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType

__all__ = ['alias_package']


class _AliasLoader(importlib.abc.Loader):
    """
    Loads an aliased module by importing its canonical name and returning that same
    module object.
    """

    def __init__(self, target: str):
        self._target = target
        self._spec: importlib.machinery.ModuleSpec | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The import system sets `__spec__` to the alias' spec before calling this;
        #  put back the canonical one, so that e.g. `importlib.reload()` keeps working
        module.__spec__ = self._spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """
    Maps ``<alias>.X`` to ``<target>.X``. Has to come before the regular finders on
    ``sys.meta_path``, which would otherwise load ``<target>/X.py`` a second time, as
    ``<alias>.X``.
    """

    def __init__(self, alias: str, target: str):
        self.alias = alias
        self.target = target

    def find_spec(self, fullname: str, path=None, target=None) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(f'{self.alias}.'):
            return None
        canonical = f'{self.target}{fullname[len(self.alias) :]}'
        # Missing modules are left to the other finders, so that the error names the alias
        if importlib.util.find_spec(canonical) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(canonical))


def alias_package(alias: str, target: str) -> ModuleType:
    """
    Make package ``target`` importable as ``alias`` too: ``alias`` becomes the very
    module object of ``target``, and ``alias.X`` that of ``target.X``, for every
    submodule. Call it from the ``__init__.py`` of the alias package, as
    ``alias_package(__name__, target)``; the import system then returns ``target``
    in place of that module. Calling it again for the same alias does nothing.

    :param alias: The second name, e.g. ``'xtl.core'``
    :param target: The canonical name of the package, e.g. ``'xtl_core'``
    :return: The ``target`` package
    """
    if not any(isinstance(f, _AliasFinder) and f.alias == alias for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder(alias, target))
    module = importlib.import_module(target)
    sys.modules[alias] = module
    return module
