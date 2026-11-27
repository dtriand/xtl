import threading

from xtl_core.utils.pickling import find_unpicklable, is_picklable


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
