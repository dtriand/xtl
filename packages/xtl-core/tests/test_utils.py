import threading

from xtl_core.utils.mappings import deepmerge
from xtl_core.utils.pickling import find_unpicklable, is_picklable


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
