import pytest

import json
import os
from pathlib import Path

from pydantic import Field
import toml

from xtl_core.options import Option, Options
from xtl_core.options.validators import cast_as, validate_length
from xtl_core.options.exceptions import OptionsValidationError


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


class TestOptions:
    class MyModel(Options):
        name: str = Option(default=None, choices=('Alice', 'Bob', 'Charlie'))
        age: int = Option(default=None, gt=0)
        double_this: int = Option(default=1, cast_as=lambda x: 2 * int(x))
        # pydantic.Field can also be used along with Option
        field: float = Field()

    class ComplexModel(Options):
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

    def test_init(self):
        m = self.MyModel(name='Alice', age=2, double_this='3', field=1.5)
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5}

    def test_assignment(self):
        m = self.MyModel(name='Alice', age=2, double_this=3, field=1.5)
        m.name = 'Bob'
        m.age = 3
        m.double_this = 4
        m.field = 2.5
        assert m.model_dump() == {'name': 'Bob', 'age': 3, 'double_this': 8, 'field': 2.5}

    def test_immutability(self):
        m = self.MyModel(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5}

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
            'age': 2.0,  # but is dumped as a float
            'double_this': 6,
            'field': 1.5,
            'is_aliased': 'a',
        }
        assert isinstance(m.model_dump()['age'], float)

    def test_validation(self):
        class ComplexModel(self.MyModel):
            path: Path = Option(default='.', cast_as=Path, path_exists=True)

        m = ComplexModel(name='Alice', age=2, double_this=3, field=1.5)
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        with pytest.raises(OptionsValidationError, match='Input should be greater than'):
            m.age = -1
        with pytest.raises(OptionsValidationError, match='Value is not in choices'):
            m.name = 'Dave'
        assert m.model_dump() == {'name': 'Alice', 'age': 2, 'double_this': 6, 'field': 1.5, 'path': Path('.')}

    def test_envvar(self):
        class EnvModel(Options):
            user: str = Option(default=None, formatter=lambda x: x.title())

        os.environ['XTLUSER'] = 'user1'
        e = EnvModel(user='${XTLUSER}', _parse_env=True)
        assert e._parse_env is True
        assert e.model_dump() == {'user': 'User1'}

        os.environ['XTLUSER'] = 'user2'
        assert e.user == 'user1'  # value is not updated
        e.user = '${XTLUSER}'
        assert e._parse_env is True
        assert e.model_dump() == {'user': 'User2'}

        e = EnvModel(user='${XTLUSER}', _parse_env=False)
        assert e._parse_env is False
        assert e.model_dump() == {'user': '${Xtluser}'}

        os.environ.pop('XTLUSER')

    def test_envvar_from_file(self):
        # Test with simple models
        class EnvModel(Options):
            user: str = Option(default='${XTLUSER}', formatter=lambda x: x.title())

        os.environ['XTLUSER'] = 'user1'

        e = EnvModel.from_dict({'user': '${XTLUSER}', '_parse_env': True})
        assert e._parse_env is True
        assert e.user == 'user1'

        e = EnvModel.from_json('{"user": "${XTLUSER}", "_parse_env": true}')
        assert e._parse_env is True
        assert e.user == 'user1'

        e = EnvModel.from_toml('user = "${XTLUSER}"\n_parse_env = true')
        assert e._parse_env is True
        assert e.user == 'user1'

        # Test with nested models
        class ParentModel(Options):
            child: EnvModel = Option(default=EnvModel(user='${XTLUSER}'))

        p = ParentModel.from_dict({'child': {'user': '${XTLUSER}', '_parse_env': True}})
        assert p.child._parse_env is True
        assert p.child.user == 'user1'

        p = ParentModel.from_json('{"child": {"user": "${XTLUSER}", "_parse_env": true}}')
        assert p.child._parse_env is True
        assert p.child.user == 'user1'

        p = ParentModel.from_toml('[child]\nuser = "${XTLUSER}"\n_parse_env = true')
        assert p.child._parse_env is True
        assert p.child.user == 'user1'

        os.environ.pop('XTLUSER')

    def test_dict(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_dict() == {'first_name': 'Alice', 'age': 2, 'env': '${XTLMAGIC}', 'formatted_env': '${XTLMAGIC}'}

        c1 = self.ComplexModel.from_dict(c0.to_dict())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_dict() == {'first_name': 'Alice', 'age': 2, 'env': 'magic', 'formatted_env': 'MAGIC'}

        c1 = self.ComplexModel.from_dict(c0.to_dict())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')

    def test_json(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_json(indent=None) == (
            '{"first_name":"Alice","age":2,"env":"${XTLMAGIC}","formatted_env":"${XTLMAGIC}"}'
        )

        c1 = self.ComplexModel.from_json(c0.to_json())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_json(indent=None) == ('{"first_name":"Alice","age":2,"env":"magic","formatted_env":"MAGIC"}')

        c1 = self.ComplexModel.from_json(c0.to_json())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')

    def test_toml(self):
        c0 = self.ComplexModel(name='Alice', age=2)
        assert c0.to_toml() == ('first_name = "Alice"\nage = 2\nenv = "${XTLMAGIC}"\nformatted_env = "${XTLMAGIC}"\n')

        c1 = self.ComplexModel.from_toml(c0.to_toml())
        assert c0.to_dict() == c1.to_dict()

        os.environ['XTLMAGIC'] = 'magic'
        c0 = self.ComplexModel(name='Alice', age=2, _parse_env=True)
        assert c0.to_toml() == ('first_name = "Alice"\nage = 2\nenv = "magic"\nformatted_env = "MAGIC"\n')

        c1 = self.ComplexModel.from_toml(c0.to_toml())
        assert c0.to_dict() == c1.to_dict()

        os.environ.pop('XTLMAGIC')

    class Inner(Options):
        a: int = 1
        tags: list[str] = Option(default_factory=lambda: ['x', 'y'])

    class Outer(Options):
        items: list['TestOptions.Inner'] = Option(default_factory=list)
        mapping: dict[str, 'TestOptions.Inner'] = Option(default_factory=dict)
        scalars: dict[str, int] = Option(default_factory=dict)
        records: list[dict[str, int]] = Option(default_factory=list)
        nested: list[list[int]] = Option(default_factory=list)

    class Deep(Options):
        outers: list['TestOptions.Outer'] = Option(default_factory=list)

    def test_toml_nested(self):
        o0 = self.Outer(
            items=[self.Inner(), self.Inner(a=2, tags=[])],
            mapping={'k': self.Inner(a=3)},
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

        o1 = self.Outer.from_toml(s)
        assert o0.to_dict() == o1.to_dict()

        # Nested arrays of tables
        d0 = self.Deep(outers=[o0, self.Outer(items=[self.Inner(a=5)])])
        s = d0.to_toml()
        assert '[[outers]]' in s
        assert '[[outers.items]]' in s

        d1 = self.Deep.from_toml(s)
        assert d0.to_dict() == d1.to_dict()

    def test_toml_formatter(self):
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

    def test_toml_exclude(self):
        class ExcludeModel(Options):
            kept: int = Option(default=1)
            hidden: int = Option(default=2, exclude=True)

        # Excluded fields are left out of TOML, as they are from dicts
        s = ExcludeModel(hidden=3).to_toml()
        assert s == 'kept = 1\n'
        assert ExcludeModel.from_toml(s).hidden == 2

    def test_toml_none(self):
        class NoneModel(Options):
            name: str | None = Option(default='default', desc='A name')
            value: int | None = Option(default=None)
            values: list[int | None] = Option(default_factory=list)
            mapping: dict[str, int | None] = Option(default_factory=dict)
            nested: list[list[int | None]] = Option(default_factory=list)

        # None fields are left out and load as the field's default
        m0 = NoneModel(name=None, value=None, values=[1, 2], mapping={'a': 1})
        s = m0.to_toml(comments=True)
        assert 'name' not in s
        assert 'value =' not in s
        m1 = NoneModel.from_toml(s)
        assert m1.name == 'default'
        assert m1.value is None
        assert m1.values == [1, 2]
        assert m1.mapping == {'a': 1}

        # None inside iterables or dicts cannot be left out without changing the data
        with pytest.raises(ValueError, match=r'Cannot serialize None at `values\[1\]`'):
            NoneModel(values=[1, None, 3]).to_toml()
        with pytest.raises(ValueError, match=r'Cannot serialize None at `mapping\.a`'):
            NoneModel(mapping={'a': None}).to_toml()
        with pytest.raises(ValueError, match=r'Cannot serialize None at `nested\[0\]\[1\]`'):
            NoneModel(nested=[[1, None]]).to_toml()

        # The location includes nested Options and serialization aliases
        class ParentModel(Options):
            child: NoneModel = Option(default_factory=NoneModel, alias='kid')
            children: list[NoneModel] = Option(default_factory=list)

        with pytest.raises(ValueError, match=r'Cannot serialize None at `kid\.values\[0\]`'):
            ParentModel(kid=NoneModel(values=[None])).to_toml()
        with pytest.raises(ValueError, match=r'Cannot serialize None at `children\[1\]\.mapping\.a`'):
            ParentModel(children=[NoneModel(), NoneModel(mapping={'a': None})]).to_toml()


class TestOptionsIO:
    """Reading and writing JSON and TOML, from strings and files."""

    class Model(Options):
        name: str = Option(default='a')
        count: int = Option(default=1)

    @pytest.fixture(params=['json', 'toml'])
    def fmt(self, request):
        """The format under test, with its writer, reader and file extension."""
        model = self.Model
        if request.param == 'json':
            return 'json', model.to_json, model.from_json, '.json'
        return 'toml', model.to_toml, model.from_toml, '.toml'

    def test_to_string(self, fmt):
        _, to_, from_, _ = fmt
        text = to_(self.Model(name='b', count=2))
        assert isinstance(text, str)
        assert from_(text) == self.Model(name='b', count=2)

    @pytest.mark.parametrize('as_str', [False, True], ids=['Path', 'str'])
    def test_file_round_trip(self, tmp_path, fmt, as_str):
        _, to_, from_, suffix = fmt
        m = self.Model(name='b', count=2)
        path = to_(m, tmp_path / f'out{suffix}')
        assert path == tmp_path / f'out{suffix}'
        assert isinstance(path, Path)
        assert from_(str(path) if as_str else path) == m

    def test_suffix_added(self, tmp_path, fmt):
        _, to_, _, suffix = fmt
        assert to_(self.Model(), tmp_path / 'out') == tmp_path / f'out{suffix}'
        assert to_(self.Model(), tmp_path / 'other.txt') == tmp_path / f'other{suffix}'

    def test_keep_file_ext(self, tmp_path, fmt):
        _, to_, from_, _ = fmt
        path = to_(self.Model(count=3), tmp_path / 'out.txt', keep_file_ext=True)
        assert path == tmp_path / 'out.txt'
        assert from_(path).count == 3

    def test_no_overwrite_after_suffix(self, tmp_path, fmt):
        """Regression: the existence check ran before the suffix was added, so writing
        to ``out`` overwrote an existing ``out.json``/``out.toml``."""
        _, to_, _, suffix = fmt
        existing = tmp_path / f'out{suffix}'
        existing.write_text('KEEP ME')
        with pytest.raises(FileExistsError, match='already exists'):
            to_(self.Model(), tmp_path / 'out')
        assert existing.read_text() == 'KEEP ME'

    def test_overwrite(self, tmp_path, fmt):
        _, to_, from_, suffix = fmt
        existing = tmp_path / f'out{suffix}'
        existing.write_text('OLD')
        to_(self.Model(count=5), existing, overwrite=True)
        assert from_(existing).count == 5

    def test_creates_parents(self, tmp_path, fmt):
        _, to_, _, suffix = fmt
        path = to_(self.Model(), tmp_path / 'a' / 'b' / 'out')
        assert path == tmp_path / 'a' / 'b' / f'out{suffix}'
        assert path.exists()

    @pytest.mark.parametrize('fmt_name, text', [('json', '{"count": "abc"}'), ('toml', 'count = "abc"')])
    @pytest.mark.parametrize('source', ['string', 'file'])
    def test_invalid_value(self, tmp_path, fmt_name, text, source):
        """Regression: a bad value in a JSON string raised FileNotFoundError, and in a JSON
        file pydantic's own ValidationError."""
        from_ = self.Model.from_json if fmt_name == 'json' else self.Model.from_toml
        if source == 'file':
            path = tmp_path / f'bad.{fmt_name}'
            path.write_text(text)
            with pytest.raises(OptionsValidationError, match='count'):
                from_(path)
        else:
            with pytest.raises(OptionsValidationError, match='count'):
                from_(text)

    @pytest.mark.parametrize(
        'fmt_name, text, error',
        [('json', '{"count": ', json.JSONDecodeError), ('toml', 'count = ', toml.TomlDecodeError)],
    )
    def test_invalid_string(self, fmt_name, text, error):
        from_ = self.Model.from_json if fmt_name == 'json' else self.Model.from_toml
        with pytest.raises(error):
            from_(text)

    @pytest.mark.parametrize('fmt_name, error', [('json', json.JSONDecodeError), ('toml', toml.TomlDecodeError)])
    def test_invalid_file_contents(self, tmp_path, fmt_name, error):
        path = tmp_path / f'bad.{fmt_name}'
        path.write_text('{"count": ' if fmt_name == 'json' else 'count = ')
        from_ = self.Model.from_json if fmt_name == 'json' else self.Model.from_toml
        with pytest.raises(error):
            from_(path)

    def test_missing_file(self, tmp_path, fmt):
        _, _, from_, suffix = fmt
        with pytest.raises(FileNotFoundError, match='File not found'):
            from_(tmp_path / f'missing{suffix}')

    def test_json_not_an_object(self):
        with pytest.raises(OptionsValidationError):
            self.Model.from_json('[1, 2]')

    @pytest.mark.parametrize('text', ['', '# only a comment\n'], ids=['empty', 'comment'])
    def test_empty_toml_loads_defaults(self, text):
        assert self.Model.from_toml(text) == self.Model()

    def test_read_contents_mode(self):
        with pytest.raises(ValueError, match="Invalid read mode: 'yaml'"):
            Options._read_contents('a: 1', mode='yaml')

    @pytest.mark.parametrize('writer', ['to_json', 'to_toml'])
    def test_writer_signatures(self, tmp_path, writer):
        """``filename`` is positional-only and the flags are keyword-only."""
        write = getattr(self.Model(), writer)
        with pytest.raises(TypeError, match='positional-only'):
            write(filename=tmp_path / 'out')
        with pytest.raises(TypeError, match='positional argument'):
            write(tmp_path / 'out', True)


def test_package_exports():
    import xtl_core.options
    from xtl_core.options import base, exceptions, versioned

    sources = {'OptionsValidationError': exceptions, 'VersionedOptions': versioned}
    for name in xtl_core.options.__all__:
        source = sources.get(name, base)
        assert getattr(xtl_core.options, name) is getattr(source, name)
