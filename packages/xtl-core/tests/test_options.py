import pytest

import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import textwrap
import uuid

from pydantic import ConfigDict, Field
import toml

from xtl_core.options import Option, Options, VersionedOptions
from xtl_core.version import VersionInfo
from xtl_core.options.validators import cast_as, validate_length
from xtl_core.options.exceptions import OptionsValidationError


class TestOption:
    def test_no_defaults(self):
        Option()
        Option(...)

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


class TestVersionedOptions:
    class Model(VersionedOptions):
        _version_path = 'version'
        version: str = Option(default='1.2.0')

    class TaggedModel(Model):
        _current_version = VersionInfo.from_str('1.2.0')

    newer = {'version': '1.3.0'}
    message = "newer than the model's current version"

    def test_no_current_version_by_default(self, caplog):
        assert VersionedOptions._current_version is None
        assert self.Model.from_dict(self.newer).version == '1.3.0'
        assert self.message not in caplog.text

    @pytest.mark.parametrize('version', ['1.1.0', '1.2.0'])
    def test_not_newer(self, caplog, version):
        assert self.TaggedModel.from_dict({'version': version}, upgrade=False).version == version
        assert self.message not in caplog.text

    def test_newer_from_dict(self, caplog):
        assert self.TaggedModel.from_dict(self.newer).version == '1.3.0'
        assert (
            "TaggedModel data is from version 1.3.0, which is newer than the model's current version 1.2.0"
            in caplog.text
        )

    def test_newer_from_json(self, caplog):
        assert self.TaggedModel.from_json('{"version": "1.3.0"}').version == '1.3.0'
        assert self.message in caplog.text


def versioned_model(current: str | None = '3.0.0', path: str = 'version') -> type[VersionedOptions]:
    """A fresh model per call, so steps registered in one test don't leak into another."""

    class Model(VersionedOptions):
        _version_path = path
        _current_version = current
        version: str = Option(default=current)

    return Model


def noop(data: dict) -> dict:
    return data


def never(data: dict) -> dict:
    raise AssertionError('this upgrade step should not run')


class TestVersionedOptionsClass:
    def test_current_version_from_str(self):
        Model = versioned_model('3.0.0')
        assert Model._current == VersionInfo.from_str('3.0.0')
        assert Model._current_version == '3.0.0'  # left as the model set it

    def test_current_version_inherited(self):
        class Child(versioned_model('3.0.0')):
            pass

        assert Child._current == VersionInfo.from_str('3.0.0')

    def test_no_current_version(self):
        assert versioned_model(current=None)._current is None

    @pytest.mark.parametrize('attr', ['_current_version', '_upgrades'])
    def test_needs_version_path(self, attr):
        with pytest.raises(TypeError, match='must define a `_version_path`'):
            type('Model', (VersionedOptions,), {attr: '1.0.0'})

    def test_default_holds_current_version(self):
        assert versioned_model('3.0.0')().to_dict()['version'] == '3.0.0'


class TestUpgradeStep:
    def test_registers_and_returns_func(self):
        Model = versioned_model()
        assert Model.upgrade_step(to='2.0.0')(noop) is noop
        (step,) = Model.upgrade_steps()
        assert step.to == VersionInfo.from_str('2.0.0')
        assert step.func is noop
        assert step.model is None

    def test_sorted_by_version(self):
        Model = versioned_model()

        @Model.upgrade_step(to='3.0.0')
        def b(data: dict) -> dict:
            return data

        @Model.upgrade_step(to='1.5.0')
        def a(data: dict) -> dict:
            return data

        assert [step.func for step in Model.upgrade_steps()] == [a, b]

    def test_registries_per_class(self):
        A, B = versioned_model(), versioned_model()
        A.upgrade_step(to='2.0.0')(noop)

        class C(A):
            pass

        assert len(A.upgrade_steps()) == 1
        assert B.upgrade_steps() == []
        # A subclass runs its parent's steps (D-065), but registers none of its own
        assert [step.func for step in C.upgrade_steps()] == [noop]
        assert C._upgrade_steps == {}

    def test_on_base_class(self):
        with pytest.raises(TypeError, match='only be registered on subclasses'):
            VersionedOptions.upgrade_step(to='1.0.0')

    def test_without_current_version(self):
        with pytest.raises(TypeError, match='does not define a `_current_version`'):
            versioned_model(current=None).upgrade_step(to='1.0.0')

    def test_newer_than_current(self):
        with pytest.raises(ValueError, match="4.0.0 is newer than the model's current version 3.0.0"):
            versioned_model('3.0.0').upgrade_step(to='4.0.0')

    def test_duplicate(self):
        Model = versioned_model()
        Model.upgrade_step(to='2.0.0')(noop)
        with pytest.raises(ValueError, match='already has an upgrade step to version 2.0.0: noop'):
            Model.upgrade_step(to='2.0.0')(never)

    @pytest.mark.parametrize('model', [42, dict])
    def test_model_not_options(self, model):
        with pytest.raises(TypeError, match='must be a subclass of Options'):
            versioned_model().upgrade_step(to='2.0.0', model=model)

    def test_model_versioned(self):
        with pytest.raises(TypeError, match='cannot be a subclass of VersionedOptions'):
            versioned_model().upgrade_step(to='2.0.0', model=versioned_model())

    @pytest.mark.parametrize('func', [lambda: {}, lambda data, extra: data], ids=['no args', 'two args'])
    def test_wrong_arity(self, func):
        with pytest.raises(TypeError, match='must accept a single positional argument'):
            versioned_model().upgrade_step(to='2.0.0')(func)


class TestUpgrade:
    @pytest.fixture
    def Sample(self):
        class Sample(VersionedOptions):
            """The current (2.0.0) layout for the rename/add/split/remove tests."""

            model_config = ConfigDict(extra='forbid')
            _version_path = 'version'
            _current_version = '2.0.0'
            version: str = Option(default='2.0.0')
            units: str = Option(default='nm')
            start: float = Option(default=0.0)
            stop: float = Option(default=1.0)
            label: str = Option(default='')

        return Sample

    old = {'version': '1.0.0', 'unit': 'A', 'range': [2.0, 5.0], 'comment': 'old'}

    def upgrade(self, Sample, step) -> dict:
        Sample.upgrade_step(to='2.0.0')(step)
        return Sample.upgrade(self.old)

    def test_rename(self, Sample):
        def rename(data: dict) -> dict:
            data['units'] = data.pop('unit')
            return data

        data = self.upgrade(Sample, rename)
        assert data['units'] == 'A'
        assert 'unit' not in data

    def test_add(self, Sample):
        def add(data: dict) -> dict:
            data['label'] = f'{data["unit"]} sample'
            return data

        assert self.upgrade(Sample, add)['label'] == 'A sample'

    def test_split(self, Sample):
        def split(data: dict) -> dict:
            data['start'], data['stop'] = data.pop('range')
            return data

        data = self.upgrade(Sample, split)
        assert (data['start'], data['stop']) == (2.0, 5.0)
        assert 'range' not in data

    def test_remove(self, Sample):
        def remove(data: dict) -> dict:
            del data['comment']
            return data

        assert 'comment' not in self.upgrade(Sample, remove)

    def test_result_validates(self, Sample):
        @Sample.upgrade_step(to='2.0.0')
        def all_changes(data: dict) -> dict:
            data['units'] = data.pop('unit')
            data['start'], data['stop'] = data.pop('range')
            data['label'] = 'added'
            del data['comment']
            return data

        with pytest.raises(OptionsValidationError):
            Sample.from_dict(self.old, upgrade=False)
        sample = Sample.from_dict(Sample.upgrade(self.old))
        assert sample.to_dict() == {
            'version': '2.0.0',
            'units': 'A',
            'start': 2.0,
            'stop': 5.0,
            'label': 'added',
        }

    @pytest.fixture
    def chain(self):
        """A 3.0.0 model with steps to 1.5.0 and 3.0.0 that record when they run and the
        version they see."""
        Model = versioned_model('3.0.0')
        calls = []

        @Model.upgrade_step(to='1.5.0')
        def a(data: dict) -> dict:
            calls.append(('a', data['version']))
            return data

        @Model.upgrade_step(to='3.0.0')
        def b(data: dict) -> dict:
            calls.append(('b', data['version']))
            return data

        return Model, calls

    @pytest.mark.parametrize(
        'version, expected',
        [
            ('1.0.0', [('a', '1.0.0'), ('b', '1.5.0')]),
            ('1.2.0', [('a', '1.2.0'), ('b', '1.5.0')]),
            ('1.5.0', [('b', '1.5.0')]),
            ('1.5.0.dev0', [('b', '1.5.0.dev0')]),
            ('2.0.0', [('b', '2.0.0')]),
        ],
    )
    def test_chain(self, chain, version, expected):
        Model, calls = chain
        assert Model.upgrade({'version': version}) == {'version': '3.0.0'}
        assert calls == expected

    @pytest.mark.parametrize('version', ['3.0.0', '4.0.0'])
    def test_current_or_newer(self, version):
        Model = versioned_model('3.0.0')
        Model.upgrade_step(to='2.0.0')(never)
        Model.upgrade_step(to='3.0.0')(never)
        assert Model.upgrade({'version': version, 'x': 1}) == {'version': version, 'x': 1}

    def test_newer_warns(self, caplog):
        versioned_model('3.0.0').upgrade({'version': '4.0.0'})
        assert "data is from version 4.0.0, which is newer than the model's current version 3.0.0" in caplog.text

    def test_missing_version(self):
        Model = versioned_model('3.0.0')
        Model.upgrade_step(to='2.0.0')(never)
        assert Model.upgrade({'x': 1}) == {'x': 1}

    def test_no_current_version(self):
        assert versioned_model(current=None).upgrade({'version': '1.0.0'}) == {'version': '1.0.0'}

    def test_no_steps_left(self):
        """Data newer than the last step but older than current is only stamped."""
        Model = versioned_model('3.0.0')
        Model.upgrade_step(to='1.5.0')(never)
        assert Model.upgrade({'version': '2.0.0'}) == {'version': '3.0.0'}

    def test_nested_version_path(self):
        Model = versioned_model('3.0.0', path='metadata.version')
        seen = []

        @Model.upgrade_step(to='2.0.0')
        def record(data: dict) -> dict:
            seen.append(data['metadata']['version'])
            return data

        data = Model.upgrade({'metadata': {'version': '1.0.0', 'name': 'x'}})
        assert data == {'metadata': {'version': '3.0.0', 'name': 'x'}}
        assert seen == ['1.0.0']

    def test_does_not_mutate(self):
        Model = versioned_model('2.0.0')

        @Model.upgrade_step(to='2.0.0')
        def mutate(data: dict) -> dict:
            data['nested']['items'].append(3)
            return data

        old = {'version': '1.0.0', 'nested': {'items': [1, 2]}}
        assert Model.upgrade(old)['nested']['items'] == [1, 2, 3]
        assert old == {'version': '1.0.0', 'nested': {'items': [1, 2]}}

    def test_non_dict_result(self):
        Model = versioned_model('2.0.0')

        @Model.upgrade_step(to='2.0.0')
        def forgot_return(data: dict) -> dict:
            data['x'] = 1

        with pytest.raises(TypeError, match=r'forgot_return \(to 2\.0\.0\) returned NoneType, expected a dict'):
            Model.upgrade({'version': '1.0.0'})

    def test_step_errors_propagate(self):
        Model = versioned_model('2.0.0')

        @Model.upgrade_step(to='2.0.0')
        def broken(data: dict) -> dict:
            return {'units': data['unit']}

        with pytest.raises(KeyError, match='unit'):
            Model.upgrade({'version': '1.0.0'})

    def test_logs(self, caplog):
        Model = versioned_model('3.0.0')
        Model.upgrade_step(to='2.0.0')(noop)
        with caplog.at_level(logging.DEBUG, logger='xtl_core.options.versioned'):
            Model.upgrade({'version': '1.0.0'})
        assert 'Running upgrade step from version 1.0.0 to 2.0.0: noop' in caplog.text
        assert 'Upgraded Model data from version 1.0.0 to 3.0.0.' in caplog.text


class SampleV1(Options):
    """A frozen copy of an old (1.0.0) layout, for typed upgrade steps."""

    version: str = Option(default='1.0.0')
    unit: str = Option(default='A')
    range: list[float] = Option(default_factory=lambda: [0.0, 1.0])


class TestTypedUpgradeStep:
    @pytest.fixture
    def model_and_seen(self):
        Model = versioned_model('2.0.0')
        seen = []

        @Model.upgrade_step(to='2.0.0', model=SampleV1)
        def typed(old: SampleV1) -> dict:
            seen.append(old)
            return {'version': old.version, 'span': old.range[1] - old.range[0]}

        return Model, seen

    def test_registered_with_model(self, model_and_seen):
        Model, _ = model_and_seen
        (step,) = Model.upgrade_steps()
        assert step.model is SampleV1

    def test_gets_model_instance(self, model_and_seen):
        Model, seen = model_and_seen
        data = Model.upgrade({'version': '1.0.0', 'range': ['2', 5]})
        (old,) = seen
        assert isinstance(old, SampleV1)
        assert old.range == [2.0, 5.0]
        assert data == {'version': '2.0.0', 'span': 3.0}

    def test_invalid_old_data(self, model_and_seen):
        Model, seen = model_and_seen
        with pytest.raises(OptionsValidationError):
            Model.upgrade({'version': '1.0.0', 'range': 'not a list'})
        assert seen == []


class TestLoad:
    """Upgrades on load: ``from_dict()``, ``from_json()`` and ``from_toml()``."""

    HOW = ['dict', 'json-string', 'json-file', 'toml-string', 'toml-file']
    old = {'version': '1.0.0', 'unit': 'A'}
    newer = {'version': '3.0.0', 'units': 'A'}
    newer_message = "Model data is from version 3.0.0, which is newer than the model's current version 2.0.0"

    @pytest.fixture
    def Model(self):
        """A 2.0.0 model that renamed ``unit`` to ``units``; the step fails on data without ``unit``."""

        class Model(VersionedOptions):
            model_config = ConfigDict(extra='forbid')
            _version_path = 'version'
            _current_version = '2.0.0'
            version: str = Option(default='2.0.0')
            units: str = Option(default='nm')

        @Model.upgrade_step(to='2.0.0')
        def rename(data: dict) -> dict:
            data['units'] = data.pop('unit')
            return data

        return Model

    @staticmethod
    def load(Model, data, how, tmp_path, **kwargs):
        if how == 'dict':
            return Model.from_dict(data, **kwargs)
        fmt, source = how.split('-')
        text = json.dumps(data) if fmt == 'json' else toml.dumps(data)
        loader = Model.from_json if fmt == 'json' else Model.from_toml
        if source == 'file':
            path = tmp_path / f'data.{fmt}'
            path.write_text(text)
            return loader(path, **kwargs)
        return loader(text, **kwargs)

    @pytest.mark.parametrize('how', HOW)
    def test_upgrades(self, Model, tmp_path, how):
        m = self.load(Model, self.old, how, tmp_path)
        assert (m.version, m.units) == ('2.0.0', 'A')

    def test_input_not_mutated(self, Model):
        old = dict(self.old)
        Model.from_dict(old)
        assert old == self.old

    @pytest.mark.parametrize('how', HOW)
    def test_upgrade_false(self, Model, tmp_path, how):
        """Old data is validated as it is, and fails on the old layout."""
        with pytest.raises(OptionsValidationError, match='unit'):
            self.load(Model, self.old, how, tmp_path, upgrade=False)

    def test_upgrade_false_runs_no_steps(self):
        Model = versioned_model('2.0.0')
        Model.upgrade_step(to='2.0.0')(never)
        assert Model.from_dict({'version': '1.0.0'}, upgrade=False).version == '1.0.0'

    @pytest.mark.parametrize('upgrade', [True, False])
    @pytest.mark.parametrize('how', HOW)
    def test_newer_warns_once(self, Model, tmp_path, caplog, how, upgrade):
        """Regression: ``from_json()`` checked the version, then called ``from_dict()``,
        which checked it again."""
        m = self.load(Model, self.newer, how, tmp_path, upgrade=upgrade)
        assert m.version == '3.0.0'
        assert caplog.text.count(self.newer_message) == 1

    @pytest.mark.parametrize('how', HOW)
    def test_missing_version(self, Model, tmp_path, how):
        """Data without a version loads as current: no steps run (``rename`` would fail),
        and the field default fills in the version."""
        m = self.load(Model, {'units': 'B'}, how, tmp_path)
        assert (m.version, m.units) == ('2.0.0', 'B')

    def test_upgrade_is_keyword_only(self, Model):
        with pytest.raises(TypeError, match='positional argument'):
            Model.from_dict(self.old, False)

    @pytest.mark.parametrize('loader', ['from_dict', 'from_json', 'from_toml'])
    def test_source_is_positional_only(self, Model, loader):
        name = 'data' if loader == 'from_dict' else 's'
        with pytest.raises(TypeError, match='positional-only'):
            getattr(Model, loader)(**{name: {} if loader == 'from_dict' else ''})

    @pytest.mark.parametrize('loader', ['from_dict', 'from_json', 'from_toml'])
    def test_docstrings(self, loader):
        """The loaders keep the docstrings of Options, plus the upgrade note and parameter."""
        doc = getattr(VersionedOptions, loader).__doc__
        parent = inspect.getdoc(getattr(Options, loader))
        assert doc.startswith(parent.split('\n\n')[0])
        assert 'Data from an older version is upgraded' in doc
        assert doc.index(':param upgrade:') < doc.index(':return:')
        for line in parent.splitlines():
            assert line in doc


class TestInheritedSteps:
    """Subclasses run their parents' steps as well as their own (D-065, D-066)."""

    @staticmethod
    def make_parent(calls: list | None = None) -> type[VersionedOptions]:
        """A 2.0.0 model that renamed ``directory`` to ``job_dir``."""

        class Parent(VersionedOptions):
            model_config = ConfigDict(extra='forbid')
            _version_path = 'version'
            _current_version = '2.0.0'
            version: str = Option(default='2.0.0')
            job_dir: str = Option(default='.')

        @Parent.upgrade_step(to='2.0.0')
        def rename(data: dict) -> dict:
            if calls is not None:
                calls.append(('parent', data['version']))
            data['job_dir'] = data.pop('directory')
            return data

        return Parent

    old = {'version': '1.0.0', 'directory': '/data/run1', 'nproc': 8}

    def test_child_runs_parent_steps(self):
        class Child(self.make_parent()):
            nproc: int = Option(default=4)

        child = Child.from_dict(self.old)
        assert (child.version, child.job_dir, child.nproc) == ('2.0.0', '/data/run1', 8)
        assert Child._upgrade_steps == {}

    def test_parent_first_for_same_version(self):
        calls = []

        class Child(self.make_parent(calls)):
            nproc: int = Option(default=4)

        @Child.upgrade_step(to='2.0.0')
        def double(data: dict) -> dict:
            calls.append(('child', data['version']))
            data['nproc'] *= 2
            return data

        assert [step.func.__name__ for step in Child.upgrade_steps()] == ['rename', 'double']
        assert Child.from_dict(self.old).nproc == 16
        assert calls == [('parent', '1.0.0'), ('child', '2.0.0')]

    @pytest.mark.parametrize(
        'version, expected',
        [('1.0.0', [('parent', '1.0.0'), ('child', '2.0.0')]), ('2.0.0', [('child', '2.0.0')])],
    )
    def test_child_raises_version(self, version, expected):
        calls = []
        Parent = self.make_parent(calls)

        class Child(Parent):
            _current_version = '2.1.0'
            nproc: int = Option(default=4)

        @Child.upgrade_step(to='2.1.0')
        def bump(data: dict) -> dict:
            calls.append(('child', data['version']))
            return data

        data = {'version': version, 'nproc': 8}
        data.update({'directory': '/x'} if version == '1.0.0' else {'job_dir': '/x'})
        child = Child.from_dict(data)
        assert (child.version, child.job_dir) == ('2.1.0', '/x')
        assert calls == expected
        assert [step.func for step in Parent.upgrade_steps()] != [step.func for step in Child.upgrade_steps()]
        assert len(Parent.upgrade_steps()) == 1

    def test_grandchild(self):
        calls = []

        class Child(self.make_parent(calls)):
            _current_version = '2.1.0'

        class Grandchild(Child):
            _current_version = '2.2.0'

        for model, to in [(Child, '2.1.0'), (Grandchild, '2.2.0')]:
            model.upgrade_step(to=to)(lambda data, name=model.__name__: calls.append((name, data['version'])) or data)

        assert Grandchild.upgrade({'version': '1.0.0', 'directory': '/x'})['version'] == '2.2.0'
        assert calls == [('parent', '1.0.0'), ('Child', '2.0.0'), ('Grandchild', '2.1.0')]

    @pytest.mark.parametrize('current', ['1.5.0', None])
    def test_cannot_lower_version(self, current):
        Parent = self.make_parent()
        with pytest.raises(TypeError, match=r'cannot lower `_current_version` below that of Parent \(2\.0\.0\)'):
            type('Child', (Parent,), {'_current_version': current})

    def test_cannot_change_version_path(self):
        Parent = self.make_parent()
        with pytest.raises(TypeError, match='cannot change `_version_path` of Parent'):
            type('Child', (Parent,), {'_version_path': 'metadata.version'})

    def test_sibling_steps_not_shared(self):
        Parent = self.make_parent()

        class A(Parent):
            pass

        class B(Parent):
            pass

        A.upgrade_step(to='2.0.0')(noop)
        assert len(A.upgrade_steps()) == 2
        assert len(B.upgrade_steps()) == 1


class JobConfigV1(Options):
    """The frozen 1.0.0 layout of the parent in ``TestTypedParentStep``."""

    model_config = ConfigDict(extra='forbid')
    version: str = Option(default='1.0.0')
    directory: str = Option(default='.', alias='dir')


class TestTypedParentStep:
    """A parent's typed step gets only the keys its frozen model knows; the rest are carried over (D-066)."""

    @pytest.fixture
    def Child(self):
        class Parent(VersionedOptions):
            model_config = ConfigDict(extra='forbid')
            _version_path = 'version'
            _current_version = '2.0.0'
            version: str = Option(default='2.0.0')
            job_dir: str = Option(default='.')

        @Parent.upgrade_step(to='2.0.0', model=JobConfigV1)
        def rename(old: JobConfigV1) -> dict:
            return {'version': old.version, 'job_dir': old.directory, 'nproc': 1}

        class Child(Parent):
            nproc: int = Option(default=4)
            label: str = Option(default='')

        return Child

    def test_child_fields_pass_through(self, Child):
        """``label`` isn't known to the frozen model and comes through; ``dir`` is its alias."""
        child = Child.from_dict({'version': '1.0.0', 'dir': '/x', 'label': 'run1'})
        assert (child.job_dir, child.label) == ('/x', 'run1')

    def test_step_wins_over_carried_keys(self, Child):
        """The step sets ``nproc`` itself, so the carried-over value is replaced."""
        assert Child.from_dict({'version': '1.0.0', 'dir': '/x', 'nproc': 8}).nproc == 1


class TestLazyUpgrades:
    """The ``_upgrades`` module is imported the first time its steps are needed."""

    @pytest.fixture
    def modules(self, tmp_path, monkeypatch):
        """Write a models module and an upgrades module with unique names; return their names."""
        monkeypatch.syspath_prepend(str(tmp_path))
        uid = uuid.uuid4().hex[:8]
        models, upgrades = f'xtl_test_models_{uid}', f'xtl_test_upgrades_{uid}'

        def write(upgrades_source: str | None = None) -> tuple[str, str]:
            (tmp_path / f'{models}.py').write_text(
                textwrap.dedent(f"""
                    from xtl_core.options import Option, VersionedOptions

                    loaded = []

                    class Parent(VersionedOptions):
                        _version_path = 'version'
                        _current_version = '2.0.0'
                        _upgrades = '{upgrades}'
                        version: str = Option(default='2.0.0')
                        job_dir: str = Option(default='.')

                    class Child(Parent):
                        nproc: int = Option(default=4)
                """)
            )
            default = f"""
                from {models} import Parent, loaded

                loaded.append('{upgrades}')

                @Parent.upgrade_step(to='2.0.0')
                def rename(data):
                    data['job_dir'] = data.pop('directory')
                    return data
            """
            (tmp_path / f'{upgrades}.py').write_text(textwrap.dedent(upgrades_source or default))
            importlib.invalidate_caches()
            return models, upgrades

        yield write
        for name in (models, upgrades):
            sys.modules.pop(name, None)

    old = {'version': '1.0.0', 'directory': '/x'}

    def test_imported_on_first_upgrade(self, modules):
        models, upgrades = modules()
        m = importlib.import_module(models)
        assert upgrades not in sys.modules
        assert m.Parent.from_dict({'version': '2.0.0'}).version == '2.0.0'  # current data
        assert upgrades not in sys.modules
        assert m.Parent.from_dict(self.old).job_dir == '/x'
        assert upgrades in sys.modules
        assert m.loaded == [upgrades]

    def test_imported_once(self, modules):
        models, upgrades = modules()
        m = importlib.import_module(models)
        m.Parent.from_dict(self.old)
        m.Parent.from_dict(self.old)
        m.Parent.upgrade_steps()
        assert m.loaded == [upgrades]

    def test_child_does_not_load_parents_module_as_own(self, modules):
        models, upgrades = modules()
        m = importlib.import_module(models)
        child = m.Child.from_dict({**self.old, 'nproc': 8})
        assert (child.job_dir, child.nproc) == ('/x', 8)
        assert m.loaded == [upgrades]
        assert (m.Parent._upgrades_loaded, m.Child._upgrades_loaded) == (True, False)
        assert m.Child._upgrade_steps == {}

    def test_imported_by_user_first(self, modules):
        models, upgrades = modules()
        m = importlib.import_module(models)
        importlib.import_module(upgrades)
        assert m.Parent.from_dict(self.old).job_dir == '/x'
        assert len(m.Parent.upgrade_steps()) == 1

    def test_no_steps_registered(self, modules, caplog):
        models, upgrades = modules('x = 1\n')
        m = importlib.import_module(models)
        m.Parent.upgrade({'version': '1.0.0'})
        assert f'Parent: importing {upgrades} registered no upgrade steps for it.' in caplog.text

    def test_bad_module(self):
        class Model(VersionedOptions):
            _version_path = 'version'
            _current_version = '2.0.0'
            _upgrades = 'xtl_test_no_such_module'

        for _ in range(2):
            with pytest.raises(ModuleNotFoundError):
                Model.upgrade({'version': '1.0.0'})
        assert Model._upgrades_loaded is False


def test_package_exports():
    import xtl_core.options
    from xtl_core.options import base, exceptions, versioned

    sources = {'OptionsValidationError': exceptions, 'VersionedOptions': versioned}
    for name in xtl_core.options.__all__:
        source = sources.get(name, base)
        assert getattr(xtl_core.options, name) is getattr(source, name)
