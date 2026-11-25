import importlib
import inspect
import json
import logging
import sys
import textwrap
import uuid

from hypothesis import given, strategies as st
from pydantic import ConfigDict
import pytest
import toml

from xtl_core.options import Option, Options, VersionedOptions
from xtl_core.options.exceptions import OptionsValidationError
from xtl_core.version import VersionInfo


def versioned_model(current: str | None = '3.0.0', path: str = 'version') -> type[VersionedOptions]:
    """A fresh model per call, so steps registered in one test don't leak into another."""

    class Model(VersionedOptions):
        _version_path = path
        _current_version = current
        version: str = Option(default=current)

    return Model


def make_parent(calls: list | None = None) -> type[VersionedOptions]:
    """A 2.0.0 model that renamed ``directory`` to ``job_dir``; its step records ``('parent', version)``."""

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


def noop(data: dict) -> dict:
    return data


def never(data: dict) -> dict:
    raise AssertionError('this upgrade step should not run')


class SampleV1(Options):
    """A frozen copy of an old (1.0.0) layout, for typed upgrade steps."""

    version: str = Option(default='1.0.0')
    unit: str = Option(default='A')
    range: list[float] = Option(default_factory=lambda: [0.0, 1.0])


class JobConfigV1(Options):
    """The frozen 1.0.0 layout of a parent with a typed step."""

    model_config = ConfigDict(extra='forbid')
    version: str = Option(default='1.0.0')
    directory: str = Option(default='.', alias='dir')


_VERSIONS = st.tuples(st.integers(0, 4), st.integers(0, 3), st.integers(0, 3)).map(lambda v: '.'.join(map(str, v)))


class TestInitSubclass:
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

    @pytest.mark.parametrize('current', ['1.5.0', None])
    def test_cannot_lower_version(self, current):
        Parent = make_parent()
        with pytest.raises(TypeError, match=r'cannot lower `_current_version` below that of Parent \(2\.0\.0\)'):
            type('Child', (Parent,), {'_current_version': current})

    def test_cannot_change_version_path(self):
        Parent = make_parent()
        with pytest.raises(TypeError, match='cannot change `_version_path` of Parent'):
            type('Child', (Parent,), {'_version_path': 'metadata.version'})


class TestCheckVersion:
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


class TestUpgradeStep:
    def test_registers_and_returns_func(self):
        Model = versioned_model()
        assert Model.upgrade_step(to='2.0.0')(noop) is noop
        (step,) = Model.upgrade_steps()
        assert step.to == VersionInfo.from_str('2.0.0')
        assert step.func is noop
        assert step.model is None

    def test_registered_with_model(self):
        Model = versioned_model('2.0.0')
        Model.upgrade_step(to='2.0.0', model=SampleV1)(noop)
        (step,) = Model.upgrade_steps()
        assert step.model is SampleV1

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


class TestUpgradeSteps:
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
        # A subclass runs its parent's steps, but registers none of its own
        assert [step.func for step in C.upgrade_steps()] == [noop]
        assert C._upgrade_steps == {}

    def test_parent_first_for_same_version(self):
        class Child(make_parent()):
            nproc: int = Option(default=4)

        @Child.upgrade_step(to='2.0.0')
        def double(data: dict) -> dict:
            return data

        assert [step.func.__name__ for step in Child.upgrade_steps()] == ['rename', 'double']

    def test_sibling_steps_not_shared(self):
        Parent = make_parent()

        class A(Parent):
            pass

        class B(Parent):
            pass

        A.upgrade_step(to='2.0.0')(noop)
        assert len(A.upgrade_steps()) == 2
        assert len(B.upgrade_steps()) == 1


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

    @given(steps=st.sets(_VERSIONS, max_size=4), version=_VERSIONS)
    def test_steps_newer_than_the_data_run_in_order(self, steps, version):
        Model = versioned_model('4.3.3')
        calls = []
        for to in steps:
            Model.upgrade_step(to=to)(lambda data, to=to: calls.append(to) or data)

        data = Model.upgrade({'version': version, 'x': 1})

        def key(v: str) -> tuple:
            return VersionInfo.from_str(v).tuple_safe

        assert calls == sorted((to for to in steps if key(to) > key(version)), key=key)
        assert data == {'version': '4.3.3' if key(version) < key('4.3.3') else version, 'x': 1}

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

    @pytest.fixture
    def typed(self):
        """A 2.0.0 model with a typed step from ``SampleV1``, and the old models the step got."""
        Model = versioned_model('2.0.0')
        seen = []

        @Model.upgrade_step(to='2.0.0', model=SampleV1)
        def typed(old: SampleV1) -> dict:
            seen.append(old)
            return {'version': old.version, 'span': old.range[1] - old.range[0]}

        return Model, seen

    def test_typed_step_gets_model_instance(self, typed):
        Model, seen = typed
        data = Model.upgrade({'version': '1.0.0', 'range': ['2', 5]})
        (old,) = seen
        assert isinstance(old, SampleV1)
        assert old.range == [2.0, 5.0]
        assert data == {'version': '2.0.0', 'span': 3.0}

    def test_typed_step_invalid_old_data(self, typed):
        Model, seen = typed
        with pytest.raises(OptionsValidationError):
            Model.upgrade({'version': '1.0.0', 'range': 'not a list'})
        assert seen == []

    def test_child_runs_parent_steps(self):
        class Child(make_parent()):
            nproc: int = Option(default=4)

        child = Child.from_dict({'version': '1.0.0', 'directory': '/data/run1', 'nproc': 8})
        assert (child.version, child.job_dir, child.nproc) == ('2.0.0', '/data/run1', 8)
        assert Child._upgrade_steps == {}

    def test_child_and_parent_steps_for_same_version(self):
        calls = []

        class Child(make_parent(calls)):
            nproc: int = Option(default=4)

        @Child.upgrade_step(to='2.0.0')
        def double(data: dict) -> dict:
            calls.append(('child', data['version']))
            data['nproc'] *= 2
            return data

        assert Child.from_dict({'version': '1.0.0', 'directory': '/x', 'nproc': 8}).nproc == 16
        assert calls == [('parent', '1.0.0'), ('child', '2.0.0')]

    @pytest.mark.parametrize(
        'version, expected',
        [('1.0.0', [('parent', '1.0.0'), ('child', '2.0.0')]), ('2.0.0', [('child', '2.0.0')])],
    )
    def test_child_raises_version(self, version, expected):
        calls = []
        Parent = make_parent(calls)

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

        class Child(make_parent(calls)):
            _current_version = '2.1.0'

        class Grandchild(Child):
            _current_version = '2.2.0'

        for model, to in [(Child, '2.1.0'), (Grandchild, '2.2.0')]:
            model.upgrade_step(to=to)(lambda data, name=model.__name__: calls.append((name, data['version'])) or data)

        assert Grandchild.upgrade({'version': '1.0.0', 'directory': '/x'})['version'] == '2.2.0'
        assert calls == [('parent', '1.0.0'), ('Child', '2.0.0'), ('Grandchild', '2.1.0')]

    @pytest.fixture
    def typed_parent(self):
        """A child of a parent whose step is typed: the step gets only the keys its frozen model knows."""

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

    def test_typed_parent_step_carries_child_fields(self, typed_parent):
        """``label`` isn't known to the frozen model and comes through; ``dir`` is its alias."""
        child = typed_parent.from_dict({'version': '1.0.0', 'dir': '/x', 'label': 'run1'})
        assert (child.job_dir, child.label) == ('/x', 'run1')

    def test_typed_parent_step_wins_over_carried_keys(self, typed_parent):
        """The step sets ``nproc`` itself, so the carried-over value is replaced."""
        assert typed_parent.from_dict({'version': '1.0.0', 'dir': '/x', 'nproc': 8}).nproc == 1


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

    @pytest.mark.parametrize('fmt', ['dict', 'json', 'toml'])
    @given(units=st.text(st.characters(codec='ascii', exclude_categories=('Cc',))).filter(lambda s: s.strip('"')))
    def test_current_data_round_trips(self, fmt, units):
        Model = make_parent()
        m = Model(job_dir=units)
        assert getattr(Model, f'from_{fmt}')(getattr(m, f'to_{fmt}')()) == m

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


class TestLoadUpgrades:
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
