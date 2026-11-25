from types import MappingProxyType

from hypothesis import given, strategies as st
import pytest

from xtl_core.options import Option, Options, OptionsValidationError
from xtl_core.options.typed_vars import TypedDict, TypedList


class Point(Options):
    x: int = Option(default=0)


class Lists(Options):
    numbers: TypedList[int] = Option(default_factory=lambda: TypedList(int))
    points: TypedList[Point] = Option(default_factory=lambda: TypedList(Point))


class Model(Lists):
    mapping: TypedDict[str, int] = Option(default_factory=lambda: TypedDict(str, int))


class TestTypedList:
    class TestInit:
        def test_empty(self):
            assert list(TypedList(int)) == []

        @given(values=st.lists(st.integers()))
        def test_initial(self, values):
            assert list(TypedList(int, values)) == values

        def test_initial_cast(self):
            assert list(TypedList(int, ['1', 2])) == [1, 2]
            assert list(TypedList(Point, [{'x': 1}])) == [Point(x=1)]

        def test_initial_invalid(self):
            with pytest.raises(OptionsValidationError, match='expected int got str instead'):
                TypedList(int, ['a'])

    class TestGetitem:
        def test_index(self):
            assert TypedList(int, [1, 2])[-1] == 2

        def test_slice(self):
            assert TypedList(int, [1, 2, 3])[1:] == [2, 3]

    class TestSetitem:
        def test_index(self):
            t = TypedList(int, [1, 2])
            t[0] = '5'
            assert t == [5, 2]

        def test_slice(self):
            t = TypedList(int, [1, 2, 3])
            t[1:] = ['7']
            assert t == [1, 7]

        def test_invalid_keeps_list(self):
            t = TypedList(int, [1, 2])
            with pytest.raises(OptionsValidationError):
                t[0] = 'a'
            with pytest.raises(OptionsValidationError):
                t[:] = [3, 'a']
            assert t == [1, 2]

        def test_slice_not_iterable(self):
            with pytest.raises(TypeError, match='Expected an iterable for slice assignment'):
                TypedList(int, [1])[:] = 5

    class TestDelitem:
        def test_delete(self):
            t = TypedList(int, [1, 2, 3])
            del t[0]
            del t[-1:]
            assert t == [2]

    class TestInsert:
        def test_insert(self):
            t = TypedList(int, [1, 3])
            t.insert(1, '2')
            assert t == [1, 2, 3]

        def test_invalid(self):
            with pytest.raises(OptionsValidationError):
                TypedList(int).insert(0, 'a')

    class TestAppend:
        def test_append(self):
            t = TypedList(int)
            t.append('1')
            assert t == [1]

        def test_invalid(self):
            with pytest.raises(OptionsValidationError, match=r'TypedList<int>|TypedList\.item'):
                TypedList(int).append('a')

    class TestExtend:
        def test_extend(self):
            t = TypedList(int, [1])
            t.extend(['2', 3])
            t += [4]
            assert t == [1, 2, 3, 4]

        def test_invalid_keeps_list(self):
            t = TypedList(int, [1])
            with pytest.raises(OptionsValidationError):
                t.extend([2, 'a'])
            assert t == [1]

    class TestContains:
        def test_contains(self):
            assert 2 in TypedList(int, [1, 2])
            assert 3 not in TypedList(int, [1, 2])

    class TestEq:
        def test_typed_list(self):
            assert TypedList(int, [1]) == TypedList(int, [1])
            assert TypedList(int, [1]) != TypedList(int, [2])

        def test_list(self):
            assert TypedList(int, [1]) == [1]
            assert TypedList(int) == []

        @pytest.mark.xfail(strict=True, reason='Comparing with other types raises NotImplementedError')
        def test_other_types(self):
            assert TypedList(int, [1]) != 1
            assert 1 not in [TypedList(int, [1])]

    class TestRepr:
        def test_repr(self):
            assert repr(TypedList(int, [1])) == 'TypedList<int>([1])'

    class TestInOptions:
        def test_validated(self):
            m = Model(numbers=['1'], points=[{'x': 2}])
            assert isinstance(m.numbers, TypedList)
            assert (m.numbers, m.points) == ([1], [Point(x=2)])

        def test_invalid(self):
            with pytest.raises(OptionsValidationError, match='Model.numbers'):
                Model(numbers=['a'])

        def test_changes_validated(self):
            m = Model()
            with pytest.raises(OptionsValidationError):
                m.numbers.append('a')

        def test_serialized_as_list(self):
            assert Model(numbers=[1], points=[{'x': 2}]).to_dict()['numbers'] == [1]
            assert Model(points=[{'x': 2}]).to_dict()['points'] == [{'x': 2}]

        @pytest.mark.parametrize('fmt', ['dict', 'json', 'toml'])
        @given(numbers=st.lists(st.integers(-(2**63), 2**63 - 1)), xs=st.lists(st.integers(0, 10), max_size=3))
        def test_round_trip(self, fmt, numbers, xs):
            m = Lists(numbers=numbers, points=[Point(x=x) for x in xs])
            assert getattr(Lists, f'from_{fmt}')(getattr(m, f'to_{fmt}')()) == m


class TestTypedDict:
    class TestInit:
        def test_empty(self):
            assert dict(TypedDict(str, int)) == {}

        @given(values=st.dictionaries(st.text(), st.integers()))
        def test_initial(self, values):
            assert dict(TypedDict(str, int, values)) == values

        def test_initial_pairs(self):
            assert dict(TypedDict(str, int, [('a', '1')])) == {'a': 1}

        def test_invalid_key(self):
            with pytest.raises(OptionsValidationError, match='expected int got str instead'):
                TypedDict(int, int, {'a': 1})

        def test_invalid_value(self):
            with pytest.raises(OptionsValidationError, match='expected int got str instead'):
                TypedDict(str, int, {'a': 'b'})

        @pytest.mark.xfail(strict=True, reason='The check for hashable key types never fails')
        def test_unhashable_key_type(self):
            with pytest.raises(TypeError, match='is not hashable'):
                TypedDict(list, int)

    class TestSetitem:
        def test_cast(self):
            t = TypedDict(str, int)
            t['a'] = '1'
            assert t == {'a': 1}

        def test_invalid_keeps_dict(self):
            t = TypedDict(str, int, {'a': 1})
            with pytest.raises(OptionsValidationError):
                t['b'] = 'x'
            assert t == {'a': 1}

    class TestDelitem:
        def test_delete(self):
            t = TypedDict(str, int, {'a': 1, 'b': 2})
            del t['a']
            assert t == {'b': 2}

    class TestUpdate:
        def test_mapping(self):
            t = TypedDict(str, int, {'a': 1})
            t.update({'a': '2', 'b': 3})
            assert t == {'a': 2, 'b': 3}

        def test_typed_dict(self):
            assert TypedDict(str, int, TypedDict(str, int, {'a': 1})) == {'a': 1}

        @pytest.mark.xfail(strict=True, reason='A read-only mapping is iterated as key-value pairs')
        def test_read_only_mapping(self):
            assert TypedDict(str, int, MappingProxyType({'a': 1})) == {'a': 1}

    class TestEq:
        def test_mappings(self):
            assert TypedDict(str, int, {'a': 1}) == TypedDict(str, int, {'a': 1})
            assert TypedDict(str, int, {'a': 1}) == {'a': 1}
            assert TypedDict(str, int, {'a': 1}) != {'a': 2}

        @pytest.mark.xfail(strict=True, reason='Comparing with other types raises NotImplementedError')
        def test_other_types(self):
            assert TypedDict(str, int) != 1

    class TestRepr:
        def test_repr(self):
            assert repr(TypedDict(str, int, {'a': 1})) == "TypedDict<str, int>({'a': 1})"

    class TestInOptions:
        def test_validated(self):
            m = Model(mapping={'a': '1'})
            assert isinstance(m.mapping, TypedDict)
            assert m.mapping == {'a': 1}

        def test_serialized_as_dict(self):
            assert Model(mapping={'a': 1}).to_dict()['mapping'] == {'a': 1}

        @pytest.mark.parametrize('fmt', ['dict', 'json'])
        @given(mapping=st.dictionaries(st.text(), st.integers()))
        def test_round_trip(self, fmt, mapping):
            m = Model(mapping=mapping)
            assert getattr(Model, f'from_{fmt}')(getattr(m, f'to_{fmt}')()) == m

        def test_toml(self):
            """Regression: the field was written to TOML as a list of its keys."""
            m = Model(mapping={'a': 1})
            assert m.to_toml().endswith('[mapping]\na = 1\n')
            assert Model.from_toml(m.to_toml()) == m
