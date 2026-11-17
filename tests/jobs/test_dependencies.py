import pickle

import pytest

from xtl import settings
from xtl.config.settings import AtsasSettings, XTLSettings
from xtl.jobs.dependencies import DependencyCheck, dependency_name, site_key, unknown_dependencies
from xtl.jobs.exceptions import DependencyError
from xtl.jobs.shells import BashShell, CmdShell
from xtl.jobs.sites import LocalSite, ModulesSite


MISSING = DependencyCheck(dependency='atsas', found=False, missing=('datcmp',), details='datcmp: not found')
FOUND = DependencyCheck(dependency='atsas', found=True)


@pytest.fixture(autouse=True)
def no_checks():
    """Start and end each test without dependency checks."""
    settings.dependencies.clear_checks()
    yield
    settings.dependencies.clear_checks()


class TestNames:
    def test_dependency_name(self):
        assert dependency_name(settings.dependencies.atsas) == 'atsas'
        # Not in the settings: the class name
        assert dependency_name(AtsasSettings()) == 'AtsasSettings'

    def test_unknown_dependencies(self):
        assert unknown_dependencies([]) == []
        assert unknown_dependencies(['atsas', 'autoproc']) == []
        assert unknown_dependencies(['zeta', 'atsas', 'alpha', 'zeta']) == ['alpha', 'zeta']
        # Settings that aren't dependencies
        assert unknown_dependencies(['modules', 'check', 'resolution']) == ['check', 'modules', 'resolution']

    def test_site_key(self):
        assert site_key(LocalSite()) == site_key(LocalSite())
        assert site_key(LocalSite()) != site_key(ModulesSite())


class TestDependencyCheck:
    def test_round_trip(self):
        assert DependencyCheck.from_dict(MISSING.to_dict()) == MISSING
        assert DependencyCheck.from_dict({'dependency': 'atsas', 'found': True}) == FOUND


class TestChecksOnSettings:
    def test_remember_and_forget(self):
        dep = settings.dependencies.atsas
        assert dep.get_check(LocalSite(), BashShell()) is None
        dep.remember_check(LocalSite(), BashShell(), MISSING)
        # Another instance of the same site and shell
        assert dep.get_check(LocalSite(), BashShell()) is MISSING
        settings.dependencies.clear_checks()
        assert dep.get_check(LocalSite(), BashShell()) is None

    def test_by_site_and_shell(self):
        dep = settings.dependencies.atsas
        dep.remember_check(LocalSite(), BashShell(), FOUND)
        assert dep.get_check(ModulesSite(), BashShell()) is None
        assert dep.get_check(LocalSite(), CmdShell()) is None

    def test_per_dependency(self):
        settings.dependencies.atsas.remember_check(LocalSite(), BashShell(), FOUND)
        assert settings.dependencies.autoproc.get_check(LocalSite(), BashShell()) is None

    def test_stale_after_a_change(self, monkeypatch):
        dep = settings.dependencies.atsas
        dep.remember_check(LocalSite(), BashShell(), FOUND)
        monkeypatch.setattr(dep, 'modules', ['atsas/4'])
        assert dep.get_check(LocalSite(), BashShell()) is None

    def test_stale_after_a_change_in_place(self, monkeypatch):
        dep = settings.dependencies.atsas
        monkeypatch.setattr(dep, 'modules', [])
        dep.remember_check(LocalSite(), BashShell(), FOUND)
        dep.modules.append('atsas/4')
        assert dep.get_check(LocalSite(), BashShell()) is None

    def test_not_in_settings_files(self):
        settings.dependencies.atsas.remember_check(LocalSite(), BashShell(), FOUND)
        assert 'found' not in settings.to_toml()
        assert settings.dependencies.atsas.to_dict() == AtsasSettings().to_dict()

    def test_equality_ignores_checks(self):
        dep = AtsasSettings()
        dep.remember_check(LocalSite(), BashShell(), FOUND)
        assert dep == AtsasSettings()
        assert dep != AtsasSettings(modules=['atsas/4'])

    def test_transfer(self):
        # As a worker receives them: the checks are plain values next to the settings snapshot
        settings.dependencies.atsas.remember_check(LocalSite(), BashShell(), MISSING)
        data = pickle.loads(pickle.dumps(settings.dependencies.checks_to_dict()))
        assert list(data) == ['atsas']

        other = XTLSettings()
        other.dependencies.update_checks(data | {'unknown': []})
        assert other.dependencies.atsas.get_check(LocalSite(), BashShell()) == MISSING
        assert other.dependencies.autoproc.get_check(LocalSite(), BashShell()) is None

    def test_transfer_keeps_staleness(self):
        settings.dependencies.atsas.remember_check(LocalSite(), BashShell(), FOUND)
        other = XTLSettings()
        other.dependencies.atsas.modules = ['atsas/4']
        other.dependencies.update_checks(settings.dependencies.checks_to_dict())
        assert other.dependencies.atsas.get_check(LocalSite(), BashShell()) is None


class TestDependencyError:
    def test_fields(self):
        error = DependencyError('datcmp not found', dependencies=('atsas',))
        assert str(error) == 'datcmp not found'
        assert error.dependencies == ('atsas',)
        assert isinstance(error, RuntimeError)

    def test_survives_pickling(self):
        error = pickle.loads(pickle.dumps(DependencyError('datcmp not found', dependencies=('atsas',))))
        assert (str(error), error.dependencies) == ('datcmp not found', ('atsas',))
