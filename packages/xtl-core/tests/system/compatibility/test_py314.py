import copy
import pickle
import typing

import pytest

from xtl_core.system.compatibility.py314 import sentinel


MISSING = sentinel('MISSING')


class TestSentinel:
    # The built-in from Python 3.15, a stand-in before: both must behave alike

    def test_repr_is_the_name(self):
        assert repr(MISSING) == 'MISSING'
        assert str(MISSING) == 'MISSING'
        assert MISSING.__name__ == 'MISSING'

    def test_repr_can_be_given(self):
        assert repr(sentinel('MISSING', repr='<missing>')) == '<missing>'

    def test_module_is_the_creator(self):
        assert MISSING.__module__ == __name__

    def test_compared_by_identity(self):
        other = sentinel('MISSING')
        assert MISSING is MISSING
        assert MISSING == MISSING
        assert MISSING != other
        assert MISSING != 'MISSING'
        assert len({MISSING, MISSING, other}) == 2

    def test_truthy(self):
        assert MISSING

    @pytest.mark.parametrize('duplicate', [copy.copy, copy.deepcopy], ids=['copy', 'deepcopy'])
    def test_copies_are_the_same(self, duplicate):
        assert duplicate(MISSING) is MISSING
        assert duplicate({'value': MISSING})['value'] is MISSING

    @pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickles_as_the_same(self, protocol):
        assert pickle.loads(pickle.dumps(MISSING, protocol=protocol)) is MISSING

    def test_union(self):
        assert typing.get_args(int | MISSING) == (int, MISSING)
        assert typing.get_args(MISSING | None) == (MISSING, type(None))

    def test_not_callable(self):
        with pytest.raises(TypeError, match='not callable'):
            MISSING()

    def test_cannot_be_subclassed(self):
        with pytest.raises(TypeError, match='not an acceptable base type'):

            class Subclass(sentinel):
                pass

    @pytest.mark.parametrize('name', ['__name__', 'other'])
    def test_cannot_be_changed(self, name):
        with pytest.raises(AttributeError):
            setattr(MISSING, name, 'value')
        with pytest.raises(AttributeError):
            delattr(MISSING, name)
        assert MISSING.__name__ == 'MISSING'
