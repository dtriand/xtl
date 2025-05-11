import pytest

from pydantic import BeforeValidator, AfterValidator, Field, ValidationError

from xtl.config.options import Option, Options
from xtl.config.validators import cast_as, validate_length


class TestOption:

    def test_no_defaults(self):
        with pytest.raises(ValueError, match='Either \'default\' or \'default_factory\' '
                                             'must be provided'):
            o = Option()

    def test_validators(self):
        o = Option(default=1)
        assert not o.json_schema_extra

        o = Option(default=1, cast_as=str, length=3)
        assert o.json_schema_extra
        assert o.json_schema_extra['validators']
        # Check the signature of the validator functions
        #  validator.func -> partial function
        #  partial.func -> original function
        assert o.json_schema_extra['validators'][0].func.func == validate_length
        assert o.json_schema_extra['validators'][1].func.func == cast_as


class TestOptions:

    class MyModel(Options):
        name: str = Option(default=None, choices=('Alice', 'Bob', 'Charlie'))
        age: int = Option(default=None, gt=0)
        double_this: int = Option(default=1, cast_as=lambda x: 2 * int(x))
        # pydantic.Field can also be used along with Option
        field: float = Field()

    def test_init(self):
        m = self.MyModel(name='Alice', age=2, double_this='3', field=1.5)
        assert m.model_dump() == {
            'name': 'Alice',
            'age': 2,
            'double_this': 6,
            'field': 1.5
        }

    def test_assignment(self):
        m = self.MyModel(name='Alice', age=2, double_this=3, field=1.5)
        m.name = 'Bob'
        m.age = 3
        m.double_this = 4
        m.field = 2.5
        assert m.model_dump() == {
            'name': 'Bob',
            'age': 3,
            'double_this': 8,
            'field': 2.5
        }

        # Test immutability when validation errors occur
        with pytest.raises(ValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {
            'name': 'Bob',
            'age': 3,
            'double_this': 8,
            'field': 2.5
        }


