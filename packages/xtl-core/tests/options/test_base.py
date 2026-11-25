import json
import math
from pathlib import Path

from hypothesis import given, strategies as st
from pydantic import Field, model_validator
import pytest
import toml

from xtl_core.options import Option, Options
from xtl_core.options.exceptions import OptionsValidationError
from xtl_core.options.typed_vars import TypedList
from xtl_core.options.validators import cast_as, validate_length


class Person(Options):
    name: str = Option(default=None, choices=('Alice', 'Bob', 'Charlie'))
    age: int = Option(default=None, gt=0)
    double_this: int = Option(default=1, cast_as=lambda x: 2 * int(x))
    # pydantic.Field can also be used along with Option
    field: float = Field()


class AliasedPerson(Options):
    name: str = Option(
        default=None, choices=('Alice', 'Bob', 'Charlie'), alias='first_name', desc='First name of the person'
    )
    age: int = Option(default=None, gt=0, desc='Age in years')
    env: str = Option(default='${XTLMAGIC}')
    envf: str = Option(
        default='${XTLMAGIC}',
        formatter=lambda x: x.upper(),
        alias='formatted_env',
        desc='Formatted environment variable',
    )


class Inner(Options):
    name: str = Option(default='')
    values: list[int] = Option(default_factory=list)


class Everything(Options):
    """One field of every kind that round-trips through dicts, JSON and TOML."""

    text: str = Option(default='', desc='Some text')
    number: int = Option(default=0)
    real: float = Option(default=0.0)
    flag: bool = Option(default=False)
    maybe: int | None = Option(default=None)
    items: list[int] = Option(default_factory=list)
    mapping: dict[str, int] = Option(default_factory=dict)
    inner: Inner = Option(default_factory=Inner)
    inners: list[Inner] = Option(default_factory=list)
    path: Path = Option(default=Path('.'))
    typed: TypedList[int] = Option(default_factory=lambda: TypedList(int))


class Cast(Options):
    n: int = Option(default=1, cast_as=int)


_INT64 = st.integers(-(2**63), 2**63 - 1)  # TOML's integers
_SEGMENT = st.from_regex(r'[A-Za-z0-9_-]{1,8}', fullmatch=True)

# What the `toml` library writes and reads back unchanged (see `TestFromToml.test_strings_the_toml_library_breaks`):
#  values without control characters, no-break space or soft hyphen, and not only double quotes; keys only of assigned,
#  printable characters, with no space other than ' '
_TOML_TEXT = st.text(st.characters(exclude_categories=('Cc', 'Cs'), exclude_characters='\xa0\xad')).filter(
    lambda s: not s or s.strip('"')
)
_TOML_KEYS = st.text(
    st.characters(exclude_categories=('Cc', 'Cs', 'Cf', 'Cn', 'Co', 'Zs', 'Zl', 'Zp'), include_characters=' ')
)


def everything(
    text: st.SearchStrategy[str] = st.text(), keys: st.SearchStrategy[str] = st.text()
) -> st.SearchStrategy[Everything]:
    inner = st.builds(Inner, name=text, values=st.lists(_INT64, max_size=3))
    return st.builds(
        Everything,
        text=text,
        number=_INT64,
        real=st.floats(allow_nan=False, allow_infinity=False),
        flag=st.booleans(),
        maybe=st.none() | _INT64,
        items=st.lists(_INT64, max_size=5),
        mapping=st.dictionaries(keys, _INT64, max_size=3),
        inner=inner,
        inners=st.lists(inner, max_size=2),
        path=st.lists(_SEGMENT, min_size=1, max_size=3).map(lambda parts: Path(*parts)),
        typed=st.lists(_INT64, max_size=3),
    )


class TestOption:
    def test_no_defaults(self):
        Option()
        Option(...)

    @pytest.mark.parametrize(
        'option, field',
        [
            (Option(), Field(json_schema_extra={})),
            (Option(...), Field(..., json_schema_extra={})),
            (Option(default=1), Field(default=1, json_schema_extra={})),
            (Option(default_factory=list), Field(default_factory=list, json_schema_extra={})),
        ],
        ids=['nothing', 'required', 'default', 'default_factory'],
    )
    def test_arguments_not_given_are_pydantic_defaults(self, option, field):
        assert repr(option) == repr(field)
        for name in field.__slots__:
            assert getattr(option, name, None) == getattr(field, name, None), name

    def test_arguments_given_reach_pydantic(self):
        o = Option(default=1, name='Number', desc='A number', alias='num', repr=False, exclude=True, gt=0)
        assert o.title == 'Number'
        assert o.description == 'A number'
        assert o.serialization_alias == 'num'
        assert o.repr is False
        assert o.exclude is True
        assert [m.gt for m in o.metadata] == [0]

    def test_in_a_model(self):
        class Model(Options):
            shown: int = Option(default=1)
            hidden: int = Option(default=2, repr=False)
            optional: str | None = Option(default=None, desc='Not given')

        assert repr(Model()) == 'Model(shown=1, optional=None)'
        assert Model().to_dict() == {'shown': 1, 'hidden': 2, 'optional': None}

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
        assert o.json_schema_extra['serializer'] is str

    def test_extra_kept(self):
        o = Option(default=1, extra={'unit': 'nm'}, length=1)
        assert o.json_schema_extra['unit'] == 'nm'
        assert o.json_schema_extra['validators']

    @pytest.mark.parametrize('validator', [str, [str]], ids=['one', 'list'])
    def test_validator_not_a_validator(self, validator):
        with pytest.raises(TypeError, match='Validator 0 is not a valid validator'):
            Option(default=1, validator=validator)

    def test_validator_takes_one_argument(self):
        from pydantic import AfterValidator

        with pytest.raises(TypeError, match='must take exactly 1 argument, but it takes 2'):
            Option(default=1, validator=AfterValidator(lambda value, other: value))

    def test_formatter_not_callable(self):
        with pytest.raises(TypeError, match="'formatter' must be a callable"):
            Option(default=1, formatter='upper')


class TestInit:
    def test_values(self):
        m = Person(name='Alice', age=2, double_this='3', field=1.5)
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5}

    def test_alias_as_input(self):
        assert AliasedPerson(first_name='Bob', age=1).name == 'Bob'

    def test_name_wins_over_alias(self):
        assert AliasedPerson(name='Alice', first_name='Bob', age=1).name == 'Alice'

    def test_invalid(self):
        with pytest.raises(OptionsValidationError, match='Person.age') as e:
            Person(name='Alice', age=-1, field=1.0)
        assert e.value.model_name == 'Person'
        assert [error['loc'] for error in e.value.errors] == ['Person.age']

    def test_errors_of_all_fields(self):
        with pytest.raises(OptionsValidationError) as e:
            Person(name='Alice', age=-1, field='x')
        assert sorted(error['loc'] for error in e.value.errors) == ['Person.age', 'Person.field']

    def test_custom_validators_after_pydantic(self):
        # Custom after-validators run once pydantic's own checks pass, so their errors come in a second round
        with pytest.raises(OptionsValidationError) as e:
            Person(name='Dave', age=-1, field=1.0)
        assert [error['loc'] for error in e.value.errors] == ['Person.age']

    @pytest.mark.parametrize('value', [None, [1]], ids=['None', 'list'])
    def test_cast_type_error(self, value):
        """Regression: ``cast_as`` let a TypeError through, e.g. ``int(None)``."""
        with pytest.raises(OptionsValidationError, match='Cast.n'):
            Cast(n=value)

    @pytest.mark.xfail(strict=True, reason="`cast_as`'s error repeats the whole input")
    def test_huge_input_message(self):
        with pytest.raises(OptionsValidationError) as e:
            Cast(n='x' * 10**6)
        assert len(str(e.value)) < 1000


class TestSetattr:
    def test_assignment(self):
        m = Person(name='Alice', age=2, double_this=3, field=1.5)
        m.name = 'Bob'
        m.age = 3
        m.double_this = 4
        m.field = 2.5
        assert m.model_dump() == {'name': 'Bob', 'age': 3, 'double_this': 8, 'field': 2.5}

    def test_invalid_keeps_value(self):
        m = Person(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5}

    def test_validators(self, tmp_path):
        class WithPath(Person):
            path: Path = Option(default='.', cast_as=Path, path_exists=True)

        m = WithPath(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        with pytest.raises(OptionsValidationError, match='Input should be greater than'):
            m.age = -1
        with pytest.raises(OptionsValidationError, match='Path does not exist'):
            m.path = tmp_path / 'missing'
        m.path = str(tmp_path)
        assert m.path == tmp_path
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5, 'path': tmp_path}

    def test_model_validator(self):
        class Range(Options):
            lo: int = Option(default=0)
            hi: int = Option(default=1)

            @model_validator(mode='after')
            def ordered(self):
                if self.lo > self.hi:
                    raise ValueError('lo > hi')
                return self

        m = Range()
        with pytest.raises(OptionsValidationError, match='lo > hi'):
            m.lo = 5
        assert (m.lo, m.hi) == (0, 1)

    def test_nested(self):
        class Outer(Options):
            inner: Inner = Option(default_factory=Inner)

        m = Outer()
        m.inner = {'name': 'a'}
        assert m.inner == Inner(name='a')
        with pytest.raises(OptionsValidationError, match=r'Outer\.inner\.values'):
            m.inner = {'values': 'abc'}
        assert m.inner == Inner(name='a')

    def test_fields_set(self):
        m = Person(name='Alice', age=2, field=1.0)
        m.double_this = 2
        assert m.model_fields_set == {'name', 'age', 'field', 'double_this'}

    def test_not_a_field(self):
        m = Person(name='Alice', age=2, field=1.0)
        with pytest.raises(ValueError):
            m.unknown = 1

    def test_other_fields_formatter(self):
        """Regression: assignment validated the model's formatted output, not its values."""

        class Joined(Options):
            tags: list[str] = Option(default_factory=lambda: ['a', 'b'], formatter=' '.join)
            n: int = Option(default=1)

        m = Joined()
        m.n = 2
        assert (m.n, m.tags) == (2, ['a', 'b'])

    def test_cast_error(self):
        """Regression: a failing before validator raised pydantic's ValidationError."""
        m = Cast()
        with pytest.raises(OptionsValidationError, match='Cast.n'):
            m.n = 'abc'
        assert m.n == Cast().n


class TestParseEnv:
    @pytest.fixture(autouse=True)
    def env(self, monkeypatch):
        monkeypatch.setenv('XTLUSER', 'user1')
        monkeypatch.delenv('XTLMISSING', raising=False)
        return monkeypatch

    class EnvModel(Options):
        user: str = Option(default='${XTLUSER}', formatter=lambda x: x.title())

    def test_init(self, env):
        class EnvModel(Options):
            user: str = Option(default=None, formatter=lambda x: x.title())

        e = EnvModel(user='${XTLUSER}', _parse_env=True)
        assert e._parse_env is True
        assert e.model_dump() == {'user': 'User1'}

        env.setenv('XTLUSER', 'user2')
        assert e.user == 'user1'  # value is not updated
        e.user = '${XTLUSER}'
        assert e._parse_env is True
        assert e.model_dump() == {'user': 'User2'}

    def test_off_by_default(self):
        e = self.EnvModel(user='${XTLUSER}')
        assert e._parse_env is False
        assert e.model_dump() == {'user': '${Xtluser}'}

    def test_per_instance(self):
        """Regression: the setting was kept on the class, so every instance had that of
        the instance created last."""
        on = self.EnvModel(_parse_env=True)
        off = self.EnvModel()
        on.user = off.user = '${XTLUSER}'
        assert (on._parse_env, on.user) == (True, 'user1')
        assert (off._parse_env, off.user) == (False, '${XTLUSER}')

    def test_after_validators(self):
        """Regression: with environment variables parsed, after validators were skipped for
        values without a variable."""

        class Choice(Options):
            name: str = Option(default='${XTLUSER}', choices=('user1', 'a'))

        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            Choice(name='b', _parse_env=True)
        m = Choice(_parse_env=True)
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'b'
        m.name = 'a'
        assert m.name == 'a'

    def test_default_values(self):
        assert self.EnvModel(_parse_env=True).user == 'user1'

    @pytest.mark.parametrize(
        'loader, source',
        [
            ('from_dict', {'user': '${XTLUSER}', '_parse_env': True}),
            ('from_json', '{"user": "${XTLUSER}", "_parse_env": true}'),
            ('from_toml', 'user = "${XTLUSER}"\n_parse_env = true'),
        ],
        ids=['dict', 'json', 'toml'],
    )
    def test_from_source(self, loader, source):
        e = getattr(self.EnvModel, loader)(source)
        assert e._parse_env is True
        assert e.user == 'user1'

    @pytest.mark.parametrize(
        'loader, source',
        [
            ('from_dict', {'child': {'user': '${XTLUSER}', '_parse_env': True}}),
            ('from_json', '{"child": {"user": "${XTLUSER}", "_parse_env": true}}'),
            ('from_toml', '[child]\nuser = "${XTLUSER}"\n_parse_env = true'),
        ],
        ids=['dict', 'json', 'toml'],
    )
    def test_nested(self, loader, source):
        class ParentModel(Options):
            child: TestParseEnv.EnvModel = Option(default_factory=TestParseEnv.EnvModel)

        p = getattr(ParentModel, loader)(source)
        assert p.child._parse_env is True
        assert p.child.user == 'user1'

    @pytest.mark.parametrize(
        'value, expected',
        [
            ('${XTLUSER}', 'user1'),
            ('a-${XTLUSER}-b', 'a-user1-b'),
            ('${XTLUSER}${XTLUSER}', 'user1user1'),
            ('${XTLMISSING}', ''),
            ('$XTLUSER', '$XTLUSER'),
            ('no variables', 'no variables'),
            (1, 1),
            (None, None),
        ],
    )
    def test_get_envvar(self, value, expected):
        assert Options._get_envvar(value) == expected


class TestCustomSerialization:
    class Formatted(Person):
        age: int = Option(default=None, gt=0, cast_as=int, formatter=float)
        aliased: str = Option(default=None, alias='is_aliased', formatter=str.upper)

    def test_formatter(self):
        m = self.Formatted(name='Alice', age='2', double_this=3, field=1.5, aliased='a')
        assert m.age == 2  # that is cast as an int
        assert isinstance(m.age, int)
        assert m.to_dict() == {
            'name': 'Alice',
            'age': 2.0,  # but is dumped as a float
            'double_this': 6,
            'field': 1.5,
            'is_aliased': 'A',
        }
        assert isinstance(m.model_dump()['age'], float)

    def test_formatter_without_alias(self):
        m = self.Formatted(name='Alice', age=2, field=1.5, aliased='a')
        assert m.to_dict(by_alias=False)['aliased'] == 'A'

    def test_private_attributes_left_out(self):
        assert '_parse_env' not in Person(name='Alice', age=2, field=1.0, _parse_env=True).to_dict()


class TestModelValidate:
    def test_valid(self):
        data = {'name': 'Alice', 'age': 2, 'field': 1.0}
        assert Person.model_validate(data) == Person(**data)

    def test_invalid(self):
        with pytest.raises(OptionsValidationError, match='Person.field'):
            Person.model_validate({'name': 'Alice', 'age': 2})


class TestToDict:
    def test_aliases(self, monkeypatch):
        monkeypatch.delenv('XTLMAGIC', raising=False)
        c0 = AliasedPerson(name='Alice', age=2)
        assert c0.to_dict() == {'first_name': 'Alice', 'age': 2, 'env': '${XTLMAGIC}', 'formatted_env': '${XTLMAGIC}'}
        assert c0.to_dict(by_alias=False)['name'] == 'Alice'

        monkeypatch.setenv('XTLMAGIC', 'magic')
        c0 = AliasedPerson(name='Alice', age=2, _parse_env=True)
        assert c0.to_dict() == {'first_name': 'Alice', 'age': 2, 'env': 'magic', 'formatted_env': 'MAGIC'}

    def test_nested(self):
        assert Everything(inner=Inner(name='a')).to_dict()['inner'] == {'name': 'a', 'values': []}


class TestFromDict:
    def test_aliases(self):
        c0 = AliasedPerson(name='Alice', age=2)
        assert AliasedPerson.from_dict(c0.to_dict()).to_dict() == c0.to_dict()

    def test_nested(self):
        assert Everything.from_dict({'inner': {'name': 'a'}}).inner == Inner(name='a')

    def test_input_not_mutated(self):
        data = {'first_name': 'Alice', 'age': 2}
        AliasedPerson.from_dict(data)
        assert data == {'first_name': 'Alice', 'age': 2}

    @given(m=everything())
    def test_round_trip(self, m):
        assert Everything.from_dict(m.to_dict()) == m


class TestToJson:
    def test_aliases(self, monkeypatch):
        monkeypatch.delenv('XTLMAGIC', raising=False)
        c0 = AliasedPerson(name='Alice', age=2)
        assert c0.to_json(indent=None) == (
            '{"first_name":"Alice","age":2,"env":"${XTLMAGIC}","formatted_env":"${XTLMAGIC}"}'
        )
        assert '"name":"Alice"' in c0.to_json(indent=None, by_alias=False)

    def test_indent(self):
        assert Inner().to_json() == '{\n    "name": "",\n    "values": []\n}'


class TestFromJson:
    def test_aliases(self, monkeypatch):
        monkeypatch.setenv('XTLMAGIC', 'magic')
        c0 = AliasedPerson(name='Alice', age=2, _parse_env=True)
        assert AliasedPerson.from_json(c0.to_json()).to_dict() == c0.to_dict()

    @given(m=everything())
    def test_round_trip(self, m):
        assert Everything.from_json(m.to_json()) == m

    @pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', '-inf'])
    def test_non_finite_floats(self, value):
        """Regression: NaN and infinity were written as null, which then failed to load."""
        loaded = Everything.from_json(Everything(real=value).to_json()).real
        assert loaded == value or (math.isnan(loaded) and math.isnan(value))


class TestToToml:
    def test_aliases(self, monkeypatch):
        monkeypatch.delenv('XTLMAGIC', raising=False)
        c0 = AliasedPerson(name='Alice', age=2)
        assert c0.to_toml() == ('first_name = "Alice"\nage = 2\nenv = "${XTLMAGIC}"\nformatted_env = "${XTLMAGIC}"\n')

        monkeypatch.setenv('XTLMAGIC', 'magic')
        c0 = AliasedPerson(name='Alice', age=2, _parse_env=True)
        assert c0.to_toml() == ('first_name = "Alice"\nage = 2\nenv = "magic"\nformatted_env = "MAGIC"\n')

    def test_comments(self):
        assert AliasedPerson(name='Alice', age=2).to_toml(comments=True).splitlines()[:2] == [
            'first_name = "Alice" # First name of the person',
            'age = 2 # Age in years',
        ]

    class Item(Options):
        a: int = 1
        tags: list[str] = Option(default_factory=lambda: ['x', 'y'])

    class Container(Options):
        items: list['TestToToml.Item'] = Option(default_factory=list)
        mapping: dict[str, 'TestToToml.Item'] = Option(default_factory=dict)
        scalars: dict[str, int] = Option(default_factory=dict)
        records: list[dict[str, int]] = Option(default_factory=list)
        nested: list[list[int]] = Option(default_factory=list)

    class Deep(Options):
        outers: list['TestToToml.Container'] = Option(default_factory=list)

    def test_nested(self):
        o0 = self.Container(
            items=[self.Item(), self.Item(a=2, tags=[])],
            mapping={'k': self.Item(a=3)},
            scalars={'x': 1, 'y': 2},
            records=[{'p': 1}, {'p': 2}],
            nested=[[1, 2], [3]],
        )
        s = o0.to_toml()
        # Lists of Options/dicts as arrays of tables, dicts as tables
        assert '[[items]]' in s
        assert '[[records]]' in s
        assert '[mapping.k]' in s
        assert '[scalars]' in s
        assert self.Container.from_toml(s).to_dict() == o0.to_dict()

        # Nested arrays of tables
        d0 = self.Deep(outers=[o0, self.Container(items=[self.Item(a=5)])])
        s = d0.to_toml()
        assert '[[outers]]' in s
        assert '[[outers.items]]' in s
        assert self.Deep.from_toml(s).to_dict() == d0.to_dict()

    def test_formatter(self):
        class FormatModel(Options):
            joined: list[str] = Option(default_factory=lambda: ['a', 'b'], formatter=lambda x: ' '.join(x))
            spec: tuple[int, ...] = Option(default=(1, 2), formatter=lambda x: [str(i) for i in x])
            dropped: int = Option(default=0, formatter=lambda x: x or None)
            none_ok: int | None = Option(default=None, formatter=lambda x: 'auto' if x is None else x)

        m = FormatModel()
        # Formatters apply to iterables too, and their output is converted recursively
        assert m.to_toml() == ('joined = "a b"\nspec = [ "1", "2",]\nnone_ok = "auto"\n')

        # Fields formatted as None are left out
        assert 'dropped' not in FormatModel(dropped=0).to_toml()
        assert 'dropped = 3' in FormatModel(dropped=3).to_toml()

    def test_exclude(self):
        class ExcludeModel(Options):
            kept: int = Option(default=1)
            hidden: int = Option(default=2, exclude=True)

        # Excluded fields are left out of TOML, as they are from dicts
        s = ExcludeModel(hidden=3).to_toml()
        assert s == 'kept = 1\n'
        assert ExcludeModel.from_toml(s).hidden == 2

    class NoneModel(Options):
        name: str | None = Option(default='default', desc='A name')
        value: int | None = Option(default=None)
        values: list[int | None] = Option(default_factory=list)
        mapping: dict[str, int | None] = Option(default_factory=dict)
        nested: list[list[int | None]] = Option(default_factory=list)

    def test_none_left_out(self):
        # None fields are left out and load as the field's default
        m0 = self.NoneModel(name=None, value=None, values=[1, 2], mapping={'a': 1})
        s = m0.to_toml(comments=True)
        assert 'name' not in s
        assert 'value =' not in s
        m1 = self.NoneModel.from_toml(s)
        assert m1.name == 'default'
        assert m1.value is None
        assert m1.values == [1, 2]
        assert m1.mapping == {'a': 1}

    @pytest.mark.parametrize(
        'values, location',
        [
            ({'values': [1, None, 3]}, r'values\[1\]'),
            ({'mapping': {'a': None}}, r'mapping\.a'),
            ({'nested': [[1, None]]}, r'nested\[0\]\[1\]'),
        ],
        ids=['list', 'dict', 'nested'],
    )
    def test_none_inside(self, values, location):
        # None inside iterables or dicts cannot be left out without changing the data
        with pytest.raises(ValueError, match=rf'Cannot serialize None at `{location}`'):
            self.NoneModel(**values).to_toml()

    def test_none_location_nested(self):
        # The location includes nested Options and serialization aliases
        class ParentModel(Options):
            child: TestToToml.NoneModel = Option(default_factory=TestToToml.NoneModel, alias='kid')
            children: list[TestToToml.NoneModel] = Option(default_factory=list)

        with pytest.raises(ValueError, match=r'Cannot serialize None at `kid\.values\[0\]`'):
            ParentModel(kid=self.NoneModel(values=[None])).to_toml()
        with pytest.raises(ValueError, match=r'Cannot serialize None at `children\[1\]\.mapping\.a`'):
            ParentModel(children=[self.NoneModel(), self.NoneModel(mapping={'a': None})]).to_toml()

    def test_paths_posix(self):
        assert Everything(path=Path('a', 'b')).to_toml().count('path = "a/b"') == 1


class TestFromToml:
    def test_aliases(self, monkeypatch):
        monkeypatch.setenv('XTLMAGIC', 'magic')
        c0 = AliasedPerson(name='Alice', age=2, _parse_env=True)
        assert AliasedPerson.from_toml(c0.to_toml()).to_dict() == c0.to_dict()

    @given(m=everything(text=_TOML_TEXT, keys=_TOML_KEYS))
    def test_round_trip(self, m):
        assert Everything.from_toml(m.to_toml()) == m

    @given(m=everything(text=_TOML_TEXT, keys=_TOML_KEYS))
    def test_round_trip_with_comments(self, m):
        assert Everything.from_toml(m.to_toml(comments=True)) == m

    @pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', '-inf'])
    def test_non_finite_floats(self, value):
        loaded = Everything.from_toml(Everything(real=value).to_toml()).real
        assert loaded == value or (math.isnan(loaded) and math.isnan(value))

    @pytest.mark.xfail(strict=True, reason='The `toml` library drops, changes or fails on these strings')
    @pytest.mark.parametrize('text', ['"', '""', '\x00', '\x1f', '\x7f', '\x85', '\xa0', '\xad', 'a\x00b'])
    def test_strings_the_toml_library_breaks(self, text):
        assert Everything.from_toml(Everything(text=text).to_toml()).text == text

    @pytest.mark.xfail(strict=True, reason='The `toml` library drops, changes or fails on these keys')
    @pytest.mark.parametrize('key', ['a\x00', 'a\xa0b', 'a b', '', '\U000e0001'])
    def test_keys_the_toml_library_breaks(self, key):
        assert Everything.from_toml(Everything(mapping={key: 1}).to_toml()).mapping == {key: 1}

    @pytest.mark.xfail(strict=True, reason='The `toml` library writes -0.0 as 0.0')
    def test_negative_zero(self):
        assert math.copysign(1, Everything.from_toml(Everything(real=-0.0).to_toml()).real) == -1


class TestWriteText:
    """Writing JSON and TOML files, through ``to_json()`` and ``to_toml()``."""

    @pytest.fixture(params=['json', 'toml'])
    def fmt(self, request):
        """The format under test, with its writer, reader and file extension."""
        if request.param == 'json':
            return Inner.to_json, Inner.from_json, '.json'
        return Inner.to_toml, Inner.from_toml, '.toml'

    @pytest.mark.parametrize('as_str', [False, True], ids=['Path', 'str'])
    def test_round_trip(self, tmp_path, fmt, as_str):
        to_, from_, suffix = fmt
        m = Inner(name='b', values=[2])
        path = to_(m, tmp_path / f'out{suffix}')
        assert path == tmp_path / f'out{suffix}'
        assert isinstance(path, Path)
        assert from_(str(path) if as_str else path) == m

    def test_returns_string_without_file(self, fmt):
        to_, from_, _ = fmt
        text = to_(Inner(name='b'))
        assert isinstance(text, str)
        assert from_(text) == Inner(name='b')

    def test_suffix_added(self, tmp_path, fmt):
        to_, _, suffix = fmt
        assert to_(Inner(), tmp_path / 'out') == tmp_path / f'out{suffix}'
        assert to_(Inner(), tmp_path / 'other.txt') == tmp_path / f'other{suffix}'

    def test_keep_file_ext(self, tmp_path, fmt):
        to_, from_, _ = fmt
        path = to_(Inner(name='c'), tmp_path / 'out.txt', keep_file_ext=True)
        assert path == tmp_path / 'out.txt'
        assert from_(path).name == 'c'

    def test_no_overwrite_after_suffix(self, tmp_path, fmt):
        """Regression: the existence check ran before the suffix was added, so writing
        to ``out`` overwrote an existing ``out.json``/``out.toml``."""
        to_, _, suffix = fmt
        existing = tmp_path / f'out{suffix}'
        existing.write_text('KEEP ME')
        with pytest.raises(FileExistsError, match='already exists'):
            to_(Inner(), tmp_path / 'out')
        assert existing.read_text() == 'KEEP ME'

    def test_overwrite(self, tmp_path, fmt):
        to_, from_, suffix = fmt
        existing = tmp_path / f'out{suffix}'
        existing.write_text('OLD')
        to_(Inner(name='new'), existing, overwrite=True)
        assert from_(existing).name == 'new'

    def test_creates_parents(self, tmp_path, fmt):
        to_, _, suffix = fmt
        path = to_(Inner(), tmp_path / 'a' / 'b' / 'out')
        assert path == tmp_path / 'a' / 'b' / f'out{suffix}'
        assert path.exists()

    @pytest.mark.parametrize('writer', ['to_json', 'to_toml'])
    def test_signatures(self, tmp_path, writer):
        """``filename`` is positional-only and the flags are keyword-only."""
        write = getattr(Inner(), writer)
        with pytest.raises(TypeError, match='positional-only'):
            write(filename=tmp_path / 'out')
        with pytest.raises(TypeError, match='positional argument'):
            write(tmp_path / 'out', True)

    def test_utf8(self, tmp_path, fmt):
        """Regression: files were read and written in the locale's encoding."""
        to_, from_, suffix = fmt
        path = to_(Inner(name='α → β'), tmp_path / f'out{suffix}')
        assert 'α → β' in path.read_text(encoding='utf-8')
        assert from_(path).name == 'α → β'

    def test_encoding(self, tmp_path, fmt):
        to_, from_, suffix = fmt
        path = to_(Inner(name='α → β'), tmp_path / f'out{suffix}', encoding='utf-16')
        assert 'α → β' in path.read_text(encoding='utf-16')
        assert from_(path, encoding='utf-16').name == 'α → β'


class TestReadContents:
    """Reading JSON and TOML strings and files, through ``from_json()`` and ``from_toml()``."""

    @pytest.mark.parametrize('fmt_name, text', [('json', '{"values": "abc"}'), ('toml', 'values = "abc"')])
    @pytest.mark.parametrize('source', ['string', 'file'])
    def test_invalid_value(self, tmp_path, fmt_name, text, source):
        """Regression: a bad value in a JSON string raised FileNotFoundError, and in a JSON
        file pydantic's own ValidationError."""
        from_ = Inner.from_json if fmt_name == 'json' else Inner.from_toml
        if source == 'file':
            path = tmp_path / f'bad.{fmt_name}'
            path.write_text(text)
            text = path
        with pytest.raises(OptionsValidationError, match='values'):
            from_(text)

    @pytest.mark.parametrize(
        'fmt_name, text, error',
        [('json', '{"values": ', json.JSONDecodeError), ('toml', 'values = ', toml.TomlDecodeError)],
    )
    def test_invalid_string(self, fmt_name, text, error):
        from_ = Inner.from_json if fmt_name == 'json' else Inner.from_toml
        with pytest.raises(error):
            from_(text)

    @pytest.mark.parametrize('fmt_name, error', [('json', json.JSONDecodeError), ('toml', toml.TomlDecodeError)])
    def test_invalid_file_contents(self, tmp_path, fmt_name, error):
        path = tmp_path / f'bad.{fmt_name}'
        path.write_text('{"values": ' if fmt_name == 'json' else 'values = ')
        from_ = Inner.from_json if fmt_name == 'json' else Inner.from_toml
        with pytest.raises(error):
            from_(path)

    @pytest.mark.parametrize('loader', ['from_json', 'from_toml'])
    def test_missing_file(self, tmp_path, loader):
        with pytest.raises(FileNotFoundError, match='File not found'):
            getattr(Inner, loader)(tmp_path / 'missing')

    def test_json_not_an_object(self):
        with pytest.raises(OptionsValidationError):
            Inner.from_json('[1, 2]')

    @pytest.mark.parametrize('text', ['', '# only a comment\n'], ids=['empty', 'comment'])
    def test_empty_toml_loads_defaults(self, text):
        assert Inner.from_toml(text) == Inner()

    def test_mode(self):
        with pytest.raises(ValueError, match="Invalid read mode: 'yaml'"):
            Options._read_contents('a: 1', mode='yaml')


def test_package_exports():
    import xtl_core.options
    from xtl_core.options import base, exceptions, versioned

    sources = {'OptionsValidationError': exceptions, 'VersionedOptions': versioned}
    for name in xtl_core.options.__all__:
        source = sources.get(name, base)
        assert getattr(xtl_core.options, name) is getattr(source, name)
