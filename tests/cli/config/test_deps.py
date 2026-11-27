import subprocess

import pytest

from xtl import settings
from xtl.jobs.dependencies import DependencyCheck
from xtl.jobs.shells import DefaultShell
from xtl.jobs.sites import LocalSite


class Checks:
    """Stands in for the scripts that dependency checks run: the dependencies in ``missing`` aren't found."""

    def __init__(self, *missing: str):
        self.missing = set(missing)
        self.checked: list[str] = []

    # NB: Set on a site's class, an instance isn't bound to the site, so it gets only the script and the shell
    def __call__(self, script: str, shell) -> subprocess.CompletedProcess:
        # The dependency whose executables the script looks up
        name = next(
            name for name, dep in settings.dependencies.items() if all(exe in script for exe in sorted(dep.provides))
        )
        self.checked.append(name)
        code = 1 if name in self.missing else 0
        stdout = ''.join(
            f'xtl-check executable {exe} {code}\n' for exe in sorted(settings.dependencies.get(name).provides)
        )
        return subprocess.CompletedProcess(['script'], 0, stdout=stdout, stderr=f'{name} is not here' if code else '')


@pytest.fixture(autouse=True)
def plain_settings(monkeypatch):
    """No remembered checks; every dependency on the local site, without settings of its own; checks on."""
    settings.dependencies.clear_checks()
    monkeypatch.setattr(settings.jobs, 'compute_site', 'local')
    monkeypatch.setattr(settings.dependencies, 'check', True)
    monkeypatch.setattr(settings.dependencies, 'path', None)
    monkeypatch.setattr(settings.dependencies, 'modules', [])
    for _, dependency in settings.dependencies.items():
        for field in ('compute_site', 'shell', 'policy', 'path'):
            monkeypatch.setattr(dependency, field, None)
        monkeypatch.setattr(dependency, 'modules', [])
    yield
    settings.dependencies.clear_checks()


def checks(monkeypatch, *missing: str) -> Checks:
    checks = Checks(*missing)
    monkeypatch.setattr(LocalSite, '_run_script', checks)
    return checks


class TestDeps:
    def test_all_found(self, cli, monkeypatch):
        scripts = checks(monkeypatch)
        result = cli('deps')
        assert result.exit_code == 0, result.output
        assert scripts.checked == settings.dependencies.names()
        assert 'Shells' in result.stdout
        assert f'{DefaultShell().name} (default)' in result.stdout
        assert 'Dependencies' in result.stdout
        for name in settings.dependencies.names():
            assert name in result.stdout
        assert 'All dependencies found.' in result.stdout

    def test_missing(self, cli, monkeypatch):
        checks(monkeypatch, 'autoproc')
        result = cli('deps')
        assert result.exit_code == 1
        assert 'Missing dependencies:' in result.stdout
        assert '- executables aP_wf_process, process not found (required for autoproc)' in result.stdout
        assert 'atsas' not in result.stdout.split('Missing dependencies:')[1]
        assert 'autoproc is not here' not in result.stdout
        assert 'Run with -v to see what the checks printed.' in result.stdout

    def test_missing_with_details(self, cli, monkeypatch):
        checks(monkeypatch, 'autoproc')
        result = cli('deps', '-v')
        assert result.exit_code == 1
        assert f'What the check of autoproc (local, {DefaultShell().name}) printed:' in result.stdout
        assert 'autoproc is not here' in result.stdout

    def test_names(self, cli, monkeypatch):
        scripts = checks(monkeypatch, 'autoproc')
        result = cli('deps', 'atsas', 'atsas')
        assert result.exit_code == 0, result.output
        assert scripts.checked == ['atsas']
        assert 'autoproc' not in result.stdout

    def test_unknown_name(self, cli, monkeypatch):
        scripts = checks(monkeypatch)
        result = cli('deps', 'nope')
        assert result.exit_code == 1
        assert '- unknown dependency nope (not in the [dependencies] of xtl.toml)' in result.stdout
        assert scripts.checked == []

    def test_fresh(self, cli, monkeypatch):
        # A remembered outcome isn't trusted
        scripts = checks(monkeypatch, 'atsas')
        site, shell = LocalSite(), DefaultShell()
        settings.dependencies.atsas.remember_check(site, shell, DependencyCheck(dependency='atsas', found=True))
        result = cli('deps', 'atsas')
        assert result.exit_code == 1
        assert scripts.checked == ['atsas']

    def test_switched_off(self, cli, monkeypatch):
        # Checked anyway: the command was asked to
        scripts = checks(monkeypatch)
        monkeypatch.setattr(settings.dependencies, 'check', False)
        result = cli('deps', 'atsas')
        assert result.exit_code == 0, result.output
        assert scripts.checked == ['atsas']
        assert 'Checks are switched off for jobs' in result.stdout

    def test_site_and_shell(self, cli, monkeypatch):
        checks(monkeypatch)
        monkeypatch.setattr(settings.dependencies.atsas, 'modules', ['atsas/4'])
        result = cli('deps', 'atsas', '--compute-site', 'modules', '--shell', 'powershell')
        assert result.exit_code == 0, result.output
        row = next(line for line in result.stdout.splitlines() if line.startswith('atsas'))
        assert row.split()[:5] == ['atsas', 'modules', 'powershell', 'modules', 'atsas/4']

    def test_modules_on_a_local_site(self, cli, monkeypatch):
        checks(monkeypatch)
        monkeypatch.setattr(settings.dependencies.atsas, 'modules', ['atsas/4'])
        result = cli('deps', 'atsas')
        assert result.exit_code == 0, result.output
        assert 'PATH (modules not loaded on this site)' in result.stdout

    def test_settings_of_the_dependency(self, cli, monkeypatch, tmp_path):
        # The dependency's own site and shell, as a job would choose them
        checks(monkeypatch)
        monkeypatch.setattr(settings.dependencies.atsas, 'compute_site', 'modules')
        monkeypatch.setattr(settings.dependencies.atsas, 'shell', 'powershell')
        monkeypatch.setattr(settings.dependencies, 'path', tmp_path)
        result = cli('deps', 'atsas')
        assert result.exit_code == 0, result.output
        row = next(line for line in result.stdout.splitlines() if line.startswith('atsas'))
        assert row.split()[:4] == ['atsas', 'modules', 'powershell', 'PATH']
        assert 'Set up for every job: path' in result.stdout

    def test_could_not_choose(self, cli, monkeypatch):
        scripts = checks(monkeypatch)

        def refuse(*args, **kwargs):
            raise ValueError('Shell bash is not one that the compute site allows')

        monkeypatch.setattr('xtl.jobs.dependencies.select_site_and_shell', refuse)
        result = cli('deps', 'atsas')
        assert result.exit_code == 1
        assert '- atsas could not be checked: Shell bash is not one that the compute site allows' in result.stdout
        assert 'Not checked' in result.stdout
        assert scripts.checked == []

    def test_could_not_run(self, cli, monkeypatch):
        monkeypatch.setattr(LocalSite, '_run_script', lambda site, script, shell: None)
        result = cli('deps', 'atsas')
        assert result.exit_code == 1
        assert '- atsas could not be checked: the check in' in result.stdout
        assert 'Not checked' in result.stdout
        assert 'not found' not in result.stdout
