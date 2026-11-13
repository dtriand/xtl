"""
Tests for the modules that a modules compute site loads.
"""

import subprocess

import pytest

from xtl import settings
from xtl.cli.utilities import common
from xtl.cli.utilities.common import JobOptions, PermissionsOptions
from xtl.jobs.shells import BashShell
from xtl.jobs.sites import ComputeSite, ModulesSite
from xtl.tui.console import ConsoleIO


@pytest.fixture
def modules(monkeypatch):
    """Undo changes to the modules in the settings and the CLI."""
    monkeypatch.setattr(common, 'REQUIRED_DEPENDENCIES', {})
    monkeypatch.setattr(settings.dependencies, 'modules', [])
    monkeypatch.setattr(settings.dependencies.autoproc, 'modules', set())


class TestModulesSite:
    def test_common_modules_load_first(self, modules):
        settings.dependencies.modules = ['gcc/12', 'openmpi/4']
        settings.dependencies.autoproc.modules = {'autoPROC/2024'}
        lines = ModulesSite().prepare_preamble(dependencies=['autoproc'], shell=BashShell()).splitlines()
        assert lines[1:] == ['module purge', 'module load gcc/12 openmpi/4', 'module load autoPROC/2024']

    def test_no_common_modules(self, modules):
        lines = ModulesSite().prepare_preamble(dependencies=None, shell=BashShell()).splitlines()
        assert lines[1:] == ['module purge']

    @pytest.mark.parametrize(
        'available, found',
        [
            (['gcc/12', 'autoPROC/2024'], True),
            (['autoPROC/2024'], False),  # a common module is missing
            (['gcc/12'], False),  # a dependency's module is missing
        ],
    )
    def test_check_dependencies(self, modules, monkeypatch, available, found):
        def execute_command(command, timeout=20):
            script = command[-1]
            if 'module avail' in script:
                name = script.split()[-1]
                stdout = name if name in available else ''
            else:
                stdout = '0'  # the `module` command exists
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr='')

        monkeypatch.setattr(ModulesSite, 'execute_command', staticmethod(execute_command))
        settings.dependencies.modules = ['gcc/12']
        settings.dependencies.autoproc.modules = {'autoPROC/2024'}
        assert ModulesSite().check_dependencies(['autoproc'], shell=BashShell()) is found


def test_only_working_sites_are_offered():
    # The Slurm sites can't submit jobs yet, so `--compute-site` doesn't offer them
    assert {site.value for site in ComputeSite} - {'biotix'} == {'local', 'modules'}


class TestJobOptions:
    def test_cli_modules_go_to_the_settings(self, modules):
        settings.dependencies.modules = ['from/config']
        options = JobOptions(
            compute_site=ComputeSite(settings.jobs.compute_site),
            permissions=PermissionsOptions(
                update=settings.jobs.permissions.update,
                files=settings.jobs.permissions.files,
                directories=settings.jobs.permissions.directories,
            ),
            modules=JobOptions.parse_module_names('gcc/12,autoproc=autoPROC/2024,unknown=other/1'),
            keep_temp=settings.jobs.keep_temp,
        )
        ConsoleIO().apply_job_options(options)
        # Modules not of a known dependency replace the common ones; a dependency's replace its own
        assert settings.dependencies.modules == ['gcc/12', 'other/1']
        assert settings.dependencies.autoproc.modules == {'autoPROC/2024'}
