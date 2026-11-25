"""
Stand-ins for what Python 3.14 lacks, imported from here on every version: on Python 3.15+, the built-ins.
Delete this module when dropping support for Python 3.14, and use the built-ins instead.

.. code-block:: python

   from xtl_core.system.compatibility.py314 import sentinel

   MISSING = sentinel('MISSING')
"""

from xtl_core.system.compatibility import PY310_OR_LESS, PY314_OR_LESS

__all__ = ['sentinel']


if PY314_OR_LESS:
    import sys
    import typing

    class sentinel:
        """
        Stand-in for the built-in :class:`sentinel` (Python 3.15+, :pep:`661`): a unique object, compared by
        identity, whose ``repr()`` is its name (or ``repr``). It stays the same object when copied or pickled (by
        its name in the module that created it, so assign it to a variable of that name), and ``|`` makes a union
        with it for type annotations. It can't be subclassed or changed.

        :param name: Name of the sentinel, *i.e.* of the variable it is assigned to.
        :param repr: Text for ``repr()``, if not the name.
        """

        def __init__(self, name: str, /, *, repr: str | None = None) -> None:
            object.__setattr__(self, '__name__', name)
            object.__setattr__(self, '_repr', name if repr is None else repr)
            # Pickle looks the sentinel up by its name in this module
            object.__setattr__(self, '__module__', sys._getframe(1).f_globals.get('__name__', '__main__'))

        def __init_subclass__(cls, **kwargs) -> None:
            raise TypeError("type 'sentinel' is not an acceptable base type")

        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError(f"'sentinel' object attribute {name!r} is read-only")

        def __delattr__(self, name: str) -> None:
            raise AttributeError(f"'sentinel' object attribute {name!r} is read-only")

        def __repr__(self) -> str:
            return self._repr

        def __copy__(self) -> 'sentinel':
            return self

        def __deepcopy__(self, memo: dict) -> 'sentinel':
            return self

        def __reduce__(self) -> str:
            return self.__name__

        def __or__(self, other: object) -> object:
            return typing.Union[self, other]

        def __ror__(self, other: object) -> object:
            return typing.Union[other, self]

        if PY310_OR_LESS:
            # NB: Python 3.10's `typing.Union` takes only callables
            def __call__(self, *args, **kwargs):
                raise TypeError("'sentinel' object is not callable")

else:
    from builtins import sentinel
