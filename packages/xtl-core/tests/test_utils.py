import logging
import threading

import pytest

from xtl_core.utils.docs import extend_docstring
from xtl_core.utils.imports import import_object, import_path
from xtl_core.utils.mappings import deepmerge
from xtl_core.utils.pickling import find_unpicklable, is_picklable


class TestImports:
    def test_round_trip(self):
        assert import_path(logging.FileHandler) == 'logging.FileHandler'
        assert import_object('logging.FileHandler') is logging.FileHandler
        assert import_object(import_path(deepmerge)) is deepmerge

    @pytest.mark.parametrize('path', ['logging', '', 'logging.NoSuchHandler', 'no_such_module.X'])
    def test_errors(self, path):
        with pytest.raises(ImportError):
            import_object(path)


class TestDeepmerge:
    def test_nested(self):
        base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        override = {'b': {'c': 20}, 'e': 5}
        assert deepmerge(base, override) == {'a': 1, 'b': {'c': 20, 'd': 3}, 'e': 5}

    def test_override_replaces_non_dicts(self):
        assert deepmerge({'a': {'b': 1}}, {'a': 2}) == {'a': 2}
        assert deepmerge({'a': 1}, {'a': {'b': 2}}) == {'a': {'b': 2}}

    def test_inputs_unchanged(self):
        base, override = {'a': {'b': 1}}, {'a': {'c': 2}}
        deepmerge(base, override)
        assert base == {'a': {'b': 1}}
        assert override == {'a': {'c': 2}}


class TestPickling:
    def test_picklable(self):
        assert is_picklable({'a': [1, 2, (3, 'x')]}) == (True, None)
        assert find_unpicklable({'a': 1}) is None

    def test_finds_the_unpicklable_object(self):
        lock = threading.Lock()
        ok, (path, type_, _) = is_picklable({'a': [1, lock]}, path='data')
        assert not ok
        assert path == "data['a'][1]"
        assert type_ is type(lock)


def _parent():
    """
    Do something.

    :param a: The a.
    :return: Something.
    :raises ValueError: Sometimes.
    """


def _description_only():
    """
    Only a description.
    """


def _no_return():
    """
    Do something.

    :param a: The a.
    """


def _no_docstring():
    pass


class TestExtendDocstring:
    def extend(self, parent):
        @extend_docstring(parent, note='The note.', params={'b': 'The b.'})
        def func():
            pass

        return func

    def test_returns_func(self):
        def func():
            pass

        assert extend_docstring(_parent, note='n', params={})(func) is func

    def test_full(self):
        assert self.extend(_parent).__doc__ == (
            'Do something.\n'
            '\n'
            'The note.\n'
            '\n'
            ':param a: The a.\n'
            ':param b: The b.\n'
            ':return: Something.\n'
            ':raises ValueError: Sometimes.'
        )

    def test_description_only(self):
        """The note stays a paragraph of its own."""
        assert self.extend(_description_only).__doc__ == 'Only a description.\n\nThe note.\n\n:param b: The b.'

    def test_no_return(self):
        assert self.extend(_no_return).__doc__.endswith(':param a: The a.\n:param b: The b.')

    def test_no_docstring(self):
        assert self.extend(_no_docstring).__doc__ == 'The note.\n\n:param b: The b.'

    def test_parent_unchanged(self):
        before = _parent.__doc__
        self.extend(_parent)
        assert _parent.__doc__ == before
