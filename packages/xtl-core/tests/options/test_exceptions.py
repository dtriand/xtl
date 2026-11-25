import pickle

from pydantic import BeforeValidator, ValidationError
import pytest

from xtl_core.errors import XTLCoreError
from xtl_core.options import Option, Options, OptionsValidationError
from xtl_core.options.typed_vars import TypedList


class Point(Options):
    x: int = Option(default=0)


class Model(Options):
    count: int = Option(default=1)
    points: list[Point] = Option(default_factory=list)
    numbers: TypedList[int] = Option(default_factory=lambda: TypedList(int))


def error(**data) -> OptionsValidationError:
    with pytest.raises(OptionsValidationError) as e:
        Model(**data)
    return e.value


class TestOptionsValidationError:
    @pytest.mark.parametrize('base', [XTLCoreError, ValueError])
    def test_bases(self, base):
        """Callers can catch it as the root, as a ValueError, or by name."""
        with pytest.raises(base) as e:
            Model(count='abc')
        assert isinstance(e.value, OptionsValidationError)
        assert not isinstance(e.value, ValidationError)


class TestExtractErrors:
    def test_fields(self):
        assert error(count='abc').errors == [
            {'loc': 'Model.count', 'msg': 'Input should be a valid integer, unable to parse string as an integer',
             'type': 'int_parsing', 'input': 'abc'}
        ]  # fmt: skip

    def test_nested_models(self):
        assert [e['loc'] for e in error(points=[{'x': 1}, {'x': 'a'}]).errors] == ['Model.points[1].x']

    def test_typed_list_items(self):
        assert [e['loc'] for e in error(numbers=[1, 'a']).errors] == ['Model.numbers[1]']

    def test_inner_options_error(self):
        # A validator that raises an OptionsValidationError of its own: its errors get the field's location
        class Outer(Options):
            point: Point = Option(default_factory=Point, validator=BeforeValidator(lambda v: Point(x=v)))

        with pytest.raises(OptionsValidationError) as e:
            Outer(point='a')
        (inner,) = e.value.errors
        assert inner['loc'] == 'Outer.point.x'
        assert inner['type'] == 'int_parsing'


class TestFormatLocation:
    @pytest.mark.parametrize(
        'loc, expected',
        [
            ((), 'Model'),
            (('a',), 'Model.a'),
            (('tables', 0, 'data'), 'Model.tables[0].data'),
            (('a', 1, 2), 'Model.a[1][2]'),
        ],
    )
    def test_format(self, loc, expected):
        assert error(count='abc')._format_location(loc) == expected


class TestStr:
    def test_one_error(self):
        assert str(error(count='abc')).splitlines() == [
            '1 validation error for Model',
            '  Model.count',
            '    Input should be a valid integer, unable to parse string as an integer',
            '    [type=int_parsing, input_value=abc, input_type=str]',
        ]

    def test_errors_counted(self):
        assert str(error(count='abc', points=[{'x': 'a'}])).startswith('2 validation errors for Model')

    @pytest.mark.xfail(strict=True, reason='The message repeats the whole input')
    def test_huge_input(self):
        assert len(str(error(count='x' * 10**6))) < 1000


class TestRepr:
    def test_repr(self):
        e = error(count='abc')
        assert repr(e) == f'OptionsValidationError({e.errors!r})'


class TestReduce:
    def test_pickle(self):
        # E.g. a job that fails in a worker process, whose result travels back pickled
        e = error(count='abc')
        restored = pickle.loads(pickle.dumps(e))
        assert type(restored) is OptionsValidationError
        assert (restored.model_name, restored.errors) == (e.model_name, e.errors)
        assert str(restored) == str(e)
