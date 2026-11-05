import logging
from pathlib import Path

import pytest

from xtl.config.settings import (
    AtsasSettings,
    AutoprocSettings,
    DependencySettings,
    DependenciesSettings,
    EasyBraggSettings,
    XTLSettings,
)


class TestDependencySettings:
    def test_provides(self):
        assert AtsasSettings().provides == {'datcmp'}
        assert AutoprocSettings().provides == {'process', 'process_wf'}
        assert EasyBraggSettings().provides == {'easyBragg.python'}
        assert DependencySettings().provides == frozenset()

    def test_provides_not_serialized(self):
        s = XTLSettings()
        assert 'provides' not in s.dependencies.atsas.to_dict()
        assert 'provides' not in s.to_toml(comments=True)

    def test_provides_not_configurable(self):
        # A `provides` key in a config file is an unknown key, and doesn't change the
        #  executables of the dependency
        s = XTLSettings.from_toml('[dependencies.atsas]\nprovides = ["evil"]\n')
        assert s.dependencies.atsas.provides == {'datcmp'}
        assert ('Found and ignored unknown keys in the config file: dependencies.atsas.provides') in s.check()

    def test_partial_table(self):
        # A table with only some keys keeps the executables of the dependency
        s = XTLSettings.from_toml('[dependencies.atsas]\npath = "/opt/atsas/bin"\n')
        assert isinstance(s.dependencies.atsas, AtsasSettings)
        assert s.dependencies.atsas.path == Path('/opt/atsas/bin')
        assert s.dependencies.atsas.provides == {'datcmp'}

    def test_round_trip(self):
        s0 = XTLSettings()
        s0.dependencies.autoproc.modules = {'autoPROC/2024'}
        s1 = XTLSettings.from_toml(s0.to_toml())
        assert s1.dependencies.autoproc.modules == {'autoPROC/2024'}
        assert s1.dependencies.autoproc.provides == {'process', 'process_wf'}

    def test_defaults_are_not_shared(self):
        # Changing one instance leaves the defaults alone
        s = XTLSettings()
        s.dependencies.autoproc.modules = {'autoPROC/2024'}
        assert XTLSettings().dependencies.autoproc.modules == set()
        assert s.dependencies.autoproc is not XTLSettings().dependencies.autoproc

    def test_not_hashable(self):
        # Hashable settings would not be copied as defaults by pydantic
        with pytest.raises(TypeError):
            hash(AtsasSettings())


class TestDependenciesSettings:
    @pytest.fixture
    def deps(self) -> DependenciesSettings:
        # An unknown dependency table, as read from a config file
        return XTLSettings.from_toml('[dependencies.foo]\npath = "/x"\n').dependencies

    def test_names(self, deps):
        assert DependenciesSettings.names() == ['atsas', 'autoproc', 'easybragg']
        assert deps.names() == ['atsas', 'autoproc', 'easybragg']

    def test_contains(self, deps):
        assert 'atsas' in deps
        assert 'resolution' not in deps
        assert 'foo' not in deps

    def test_get(self, deps):
        assert deps.get('atsas') is deps.atsas
        assert deps.get('resolution') is None
        assert deps.get('foo') is None
        assert deps.get('missing') is None

    def test_items(self, deps):
        assert deps.items() == [('atsas', deps.atsas), ('autoproc', deps.autoproc), ('easybragg', deps.easybragg)]

    def test_unknown_table_is_reported(self):
        s = XTLSettings.from_toml('[dependencies.foo]\npath = "/x"\n')
        assert ('Found and ignored unknown keys in the config file: dependencies.foo.path') in s.check()

    @staticmethod
    def _ids(deps: list) -> list[int]:
        # Settings with equal values are equal, so compare by identity
        return [id(dep) for dep in deps]

    def test_resolve(self, deps):
        assert deps.resolve(None) == []
        assert self._ids(deps.resolve('atsas')) == self._ids([deps.atsas])
        assert self._ids(deps.resolve(deps.atsas)) == self._ids([deps.atsas])
        # In the given order
        assert self._ids(deps.resolve(['easybragg', 'atsas'])) == self._ids([deps.easybragg, deps.atsas])

        # Instances are passed through, even if they are not in the settings
        other = AtsasSettings()
        assert self._ids(deps.resolve([other, 'easybragg'])) == self._ids([other, deps.easybragg])

    def test_resolve_duplicates(self, deps):
        assert self._ids(deps.resolve(['atsas', 'autoproc', deps.atsas, 'atsas'])) == self._ids(
            [deps.atsas, deps.autoproc]
        )
        # An equal but different instance is kept
        other = AtsasSettings()
        assert other == deps.atsas
        assert self._ids(deps.resolve(['atsas', other])) == self._ids([deps.atsas, other])

    def test_resolve_loose(self, deps, caplog):
        assert deps.resolution == 'loose'
        with caplog.at_level(logging.WARNING):
            assert deps.resolve(['atsas', 'foo', 'resolution']) == [deps.atsas]
        assert 'Dependency `foo` not found in settings' in caplog.text
        assert 'Dependency `resolution` not found in settings' in caplog.text

    def test_resolve_strict(self, deps):
        deps.resolution = 'strict'
        assert deps.resolve('atsas') == [deps.atsas]
        with pytest.raises(ValueError, match='Missing dependency `foo`'):
            deps.resolve(['atsas', 'foo'])
