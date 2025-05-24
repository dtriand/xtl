import pytest

import os
from pathlib import Path

from pydantic import Field, ValidationError

from xtl.common.options import Option, Options
from xtl.common.validators import cast_as, validate_length


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
        assert o.json_schema_extra['validators'][0].func.func == cast_as
        assert o.json_schema_extra['validators'][1].func.func == validate_length

    def test_formatters(self):
        o = Option(default=1)
        assert not o.json_schema_extra

        o = Option(default=1, formatter=str)
        assert o.json_schema_extra
        assert o.json_schema_extra['serializer']
        assert o.json_schema_extra['serializer'] == str


class TestOptions:

    class MyModel(Options):
        name: str = Option(default=None, choices=('Alice', 'Bob', 'Charlie'))
        age: int = Option(default=None, gt=0)
        double_this: int = Option(default=1, cast_as=lambda x: 2 * int(x))
        # pydantic.Field can also be used along with Option
        field: float = Field()

    class ComplexModel(Options):
        name: str = Option(default=None, choices=('Alice', 'Bob', 'Charlie'),
                           alias='first_name', desc='First name of the person')
        age: int = Option(default=None, gt=0, desc='Age in years')
        env: str = Option(default='${XTLMAGIC}')
        envf: str = Option(default='${XTLMAGIC}', formatter=lambda x: x.upper(),
                           alias='formatted_env', desc='Formatted environment variable')

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

    def test_immutability(self):
        m = self.MyModel(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(ValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {
            'name': 'Alice',
            'age': 2,
            'double_this': 6,
            'field': 1.5
        }

    def test_formatter(self):
        class MyFormatModel(self.MyModel):
            age: int = Option(default=None, gt=0, cast_as=int, formatter=float)
            aliased: str = Option(default=None, alias='is_aliased')

        # Input a string
        m = MyFormatModel(name='Alice', age='2', double_this=3, field=1.5, aliased='a')
        assert m.age == 2  # that is cast as an int
        assert isinstance(m.age, int)
        assert m.to_dict() == {
            'name': 'Alice',
            'age': 2.,  # but is dumped as a float
            'double_this': 6,
            'field': 1.5,
            'is_aliased': 'a'
        }
        assert isinstance(m.model_dump()['age'], float)

    def test_validation(self):
        class ComplexModel(self.MyModel):
            path: Path = Option(default='.', cast_as=Path, path_exists=True)

        m = ComplexModel(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(ValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        with pytest.raises(ValidationError, match='Input should be greater than'):
            m.age = -1
        with pytest.raises(ValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {
            'name': 'Alice',
            'age': 2,
            'double_this': 6,
            'field': 1.5,
            'path': Path('.')
        }

    def test_envvar(self):
        class EnvModel(Options):
            user: str = Option(default=None, formatter=lambda x: x.title())

        os.environ['XTLUSER'] = 'user1'
        e = EnvModel(user='${XTLUSER}', _parse_env=True)
        assert e._parse_env == True
        assert e.model_dump() == {'user': 'User1'}

        os.environ['XTLUSER'] = 'user2'
        assert e.user == 'user1'  # value is not updated
        e.user = '${XTLUSER}'
        assert e._parse_env == True
        assert e.model_dump() == {'user': 'User2'}

        e = EnvModel(user='${XTLUSER}', _parse_env=False)
        assert e._parse_env == False
        assert e.model_dump() == {'user': '${Xtluser}'}

        os.environ.pop('XTLUSER')

    def test_envvar_from_file(self):
        # Test with simple models
        class EnvModel(Options):
            user: str = Option(default='${XTLUSER}', formatter=lambda x: x.title())

        os.environ['XTLUSER'] = 'user1'

        e = EnvModel.from_dict({'user': '${XTLUSER}', '_parse_env': True})
        assert e._parse_env == True
        assert e.user == 'user1'

        e = EnvModel.from_json('{"user": "${XTLUSER}", "_parse_env": true}')
        assert e._parse_env == True
        assert e.user == 'user1'

        e = EnvModel.from_toml('user = "${XTLUSER}"\n_parse_env = true')
        assert e._parse_env == True
        assert e.user == 'user1'

        # Test with nested models
        class ParentModel(Options):
            child: EnvModel = Option(default=EnvModel(user='${XTLUSER}'))

        p = ParentModel.from_dict({'child': {'user': '${XTLUSER}', '_parse_env': True}})
        assert p.child._parse_env == True
        assert p.child.user == 'user1'

        p = ParentModel.from_json('{"child": {"user": "${XTLUSER}", "_parse_env": true}}')
        assert p.child._parse_env == True
        assert p.child.user == 'user1'

        p = ParentModel.from_toml('[child]\nuser = "${XTLUSER}"\n_parse_env = true')
        assert p.child._parse_env == True
        assert p.child.user == 'user1'

        os.environ.pop('XTLUSER')

    def test_dict(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_dict() == {
            'first_name': 'Alice',
            'age': 2,
            'env': '${XTLMAGIC}',
            'formatted_env': '${XTLMAGIC}'
        }

        c1 = self.ComplexModel.from_dict(c0.to_dict())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_dict() == {
            'first_name': 'Alice',
            'age': 2,
            'env': 'magic',
            'formatted_env': 'MAGIC'
        }

        c1 = self.ComplexModel.from_dict(c0.to_dict())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')

    def test_json(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_json(indent=None) == ('{"first_name":"Alice",'
                                           '"age":2,'
                                           '"env":"${XTLMAGIC}",'
                                           '"formatted_env":"${XTLMAGIC}"}')

        c1 = self.ComplexModel.from_json(c0.to_json())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_json(indent=None) == ('{"first_name":"Alice",'
                                           '"age":2,'
                                           '"env":"magic",'
                                           '"formatted_env":"MAGIC"}')

        c1 = self.ComplexModel.from_json(c0.to_json())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')

    def test_toml(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_toml() == ('first_name = "Alice" \n'
                                'age = 2 \n'
                                'env = "${XTLMAGIC}" \n'
                                'formatted_env = "${XTLMAGIC}" \n')

        c1 = self.ComplexModel.from_toml(c0.to_toml())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_toml() == ('first_name = "Alice" \n'
                                'age = 2 \n'
                                'env = "magic" \n'
                                'formatted_env = "MAGIC" \n')

        c1 = self.ComplexModel.from_toml(c0.to_toml())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')
