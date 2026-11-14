import asyncio
from dataclasses import dataclass
import logging
import subprocess
import sys

import pytest

from xtl import settings
from xtl.cli.utilities import common
from xtl.cli.utilities.common import JobOptions, PermissionsOptions
from xtl.config.settings import DependencySettings
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.exceptions import SubprocessError
from xtl.jobs.policies import BaseCommandPolicy
from xtl.jobs.shells import BashShell, CmdShell, DefaultShell, PowerShell, Shell
from xtl.jobs.sites import BaseComputeSite, ComputeSite, LocalSite, ModulesSite
from xtl.tui.console import ConsoleIO
from xtl_core.system.compatibility import OS_WINDOWS

SLEEP_2 = 'ping -n 3 127.0.0.1 > nul' if OS_WINDOWS else 'sleep 2'


@dataclass
class TaggingPolicy(BaseCommandPolicy):
    """Marks each part of a batch file it intercepts."""

    name: str = 'tagging'

    def intercept_preamble(self, preamble: str) -> str:
        return preamble + '[preamble]'

    def intercept_command(self, command: str) -> str:
        return f'[{command}]'

    def intercept_content(self, content: str) -> str:
        return f'<{content}>'

    def intercept_postamble(self, postamble: str) -> str:
        return postamble + '[postamble]'


class TaggingSite(LocalSite):
    _policy = TaggingPolicy()


class BashOnlySite(LocalSite):
    _default_shell = Shell.BASH
    _supported_shells = frozenset({Shell.BASH})


class Tool(DependencySettings):
    _provides = frozenset({'tool', 'helper'})


def completed(command, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr='')


@pytest.fixture
def modules(monkeypatch):
    """Undo changes to the modules in the settings and the CLI."""
    monkeypatch.setattr(common, 'REQUIRED_DEPENDENCIES', {})
    monkeypatch.setattr(settings.dependencies, 'modules', [])
    monkeypatch.setattr(settings.dependencies.autoproc, 'modules', set())


@pytest.fixture
def sites_log(caplog):
    """Records of the `xtl.jobs.sites` logger, which may not propagate to caplog's handler."""
    logger = logging.getLogger('xtl.jobs.sites')
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger='xtl.jobs.sites')
    yield caplog
    logger.removeHandler(caplog.handler)


class TestComputeSite:
    def test_only_working_sites_are_offered(self):
        # The Slurm sites can't submit jobs yet, so `--compute-site` doesn't offer them
        assert {site.value for site in ComputeSite} - {'biotix'} == {'local', 'modules'}

    @pytest.mark.parametrize('site, cls', [('local', LocalSite), ('modules', ModulesSite)])
    def test_get(self, site, cls):
        assert type(ComputeSite(site).get()) is cls

    def test_equality(self):
        assert ComputeSite.LOCAL == LocalSite()
        assert ComputeSite.LOCAL != ModulesSite()
        assert ComputeSite.MODULES == ModulesSite()
        assert ComputeSite.LOCAL == 'local'
        assert {ComputeSite.LOCAL: 1}['local'] == 1


class TestBaseComputeSite:
    def test_defaults(self):
        site = LocalSite()
        assert (site.default_shell, site.supported_shells, site.policy) == (None, frozenset(), None)

    def test_equality(self):
        assert LocalSite() == LocalSite()
        assert LocalSite() == ComputeSite.LOCAL
        # Sites of different types aren't equal, even if their shells and policies are
        assert LocalSite() != ModulesSite()
        assert ModulesSite() != LocalSite()
        assert LocalSite() != TaggingSite()
        # Other objects aren't sites
        assert LocalSite() != 'local'
        assert LocalSite() != None  # noqa: E711
        assert LocalSite() not in [None, 1]

    def test_is_valid_shell(self):
        assert all(LocalSite().is_valid_shell(shell) for shell in (Shell.BASH, Shell.CMD, Shell.POWERSHELL))
        assert BashOnlySite().is_valid_shell(Shell.BASH)
        assert not BashOnlySite().is_valid_shell(Shell.CMD)

    def test_resolve_dependencies(self):
        tool = Tool()
        resolved = BaseComputeSite.resolve_dependencies(['autoproc', tool, 'autoproc', tool])
        assert resolved == [settings.dependencies.autoproc, tool]
        assert BaseComputeSite.resolve_dependencies(None) == []

    def test_execute_command(self):
        result = BaseComputeSite.execute_command([sys.executable, '-c', 'print("hello")'])
        assert result.returncode == 0
        assert result.stdout.strip() == 'hello'

    def test_execute_command_timeout(self, sites_log):
        assert BaseComputeSite.execute_command([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=1) is None
        assert 'timed out' in sites_log.text

    def test_execute_command_missing_program(self, sites_log):
        assert BaseComputeSite.execute_command(['xtl-no-such-program']) is None
        assert 'could not be started' in sites_log.text

    @pytest.mark.parametrize('shell', [BashShell(), CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_prepare_shebang(self, shell):
        expected = f'{shell.shebang}{shell.new_line_char}' if shell.shebang else ''
        assert BaseComputeSite._prepare_shebang(shell) == expected


class TestLocalSite:
    def test_prepare_without_a_policy(self):
        site, shell = LocalSite(), BashShell()
        assert site.prepare_preamble(dependencies=['autoproc'], shell=shell) == f'{shell.shebang}\n'
        assert site.prepare_command('echo hi') == 'echo hi'
        assert site.prepare_content('echo hi') == 'echo hi'
        assert site.prepare_postamble() == ''

    def test_prepare_with_a_policy(self):
        site, shell = TaggingSite(), BashShell()
        assert site.prepare_preamble(dependencies=None, shell=shell) == f'{shell.shebang}\n[preamble]'
        assert site.prepare_command('echo hi') == '[echo hi]'
        assert site.prepare_content('echo hi') == '<echo hi>'
        assert site.prepare_postamble() == '[postamble]'

    @pytest.mark.parametrize(
        'stdout, found',
        [
            ({'tool': '0', 'helper': '0'}, True),
            ({'tool': '0', 'helper': '1'}, False),
            ({'tool': '0', 'helper': None}, False),  # timed out
        ],
    )
    def test_check_dependencies(self, monkeypatch, stdout, found):
        def execute_command(command, timeout=20):
            exe = next(exe for exe in stdout if exe in ' '.join(command))
            return None if stdout[exe] is None else completed(command, stdout[exe] + '\n')

        monkeypatch.setattr(LocalSite, 'execute_command', staticmethod(execute_command))
        assert LocalSite().check_dependencies(Tool(), shell=BashShell()) is found

    def test_check_dependencies_names_the_dependency(self, monkeypatch, sites_log):
        monkeypatch.setattr(
            LocalSite, 'execute_command', staticmethod(lambda command, timeout: completed(command, '1'))
        )
        assert LocalSite().check_dependencies(['autoproc'], shell=BashShell()) is False
        assert 'for dependency `autoproc`' in sites_log.text
        assert LocalSite().check_dependencies(Tool(), shell=BashShell()) is False
        assert 'for dependency `Tool`' in sites_log.text

    def test_check_dependencies_without_any(self):
        assert LocalSite().check_dependencies(None, shell=BashShell()) is True

    def test_check_dependencies_with_an_unsafe_name(self, monkeypatch, sites_log):
        class Unsafe(DependencySettings):
            _provides = frozenset({'tool; rm -rf x'})

        monkeypatch.setattr(LocalSite, 'execute_command', staticmethod(lambda *a, **k: pytest.fail('ran')))
        assert LocalSite().check_dependencies(Unsafe(), shell=BashShell()) is False

    def test_check_dependencies_runs_the_shell(self):
        # A program that exists everywhere: the shell itself
        shell = DefaultShell()
        exe = 'cmd' if OS_WINDOWS else 'sh'

        class Shell_(DependencySettings):
            _provides = frozenset({exe})

        assert LocalSite().check_dependencies(Shell_(), shell=shell) is True

    @pytest.mark.asyncio
    async def test_execute_batch_without_log_files(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line('echo discarded')
        batch.save()
        await LocalSite().execute_batch(batch)
        assert batch.status == BatchFileStatus.COMPLETED
        assert sorted(f.name for f in tmp_path.iterdir()) == [batch.file.name]

    @pytest.mark.asyncio
    async def test_execute_batch_with_a_log_file_that_cannot_be_opened(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line(SLEEP_2)
        batch.save()
        with pytest.raises(SubprocessError):
            await asyncio.wait_for(LocalSite().execute_batch(batch, stdout=tmp_path / 'missing' / 'out.log'), 30)
        # The program and the other stream's task are stopped
        assert batch.status == BatchFileStatus.FAILED
        assert batch.process.returncode is not None

    @pytest.mark.asyncio
    async def test_terminate_kills_a_process_that_does_not_stop(self, tmp_path, monkeypatch):
        class Stubborn:
            pid, returncode, calls = 1, None, []

            def terminate(self):
                self.calls.append('terminate')

            def kill(self):
                self.calls.append('kill')
                self.returncode = -9

            async def wait(self):
                while self.returncode is None:
                    await asyncio.sleep(0.01)
                return self.returncode

        async def no_patience(awaitable, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, 'wait_for', no_patience)
        batch = BatchFile(tmp_path / 'job')
        batch._process = Stubborn()
        await LocalSite._terminate_batch(batch)
        assert batch.process.calls == ['terminate', 'kill']
        assert batch.process.returncode == -9


class TestModulesSite:
    def test_common_modules_load_first(self, modules):
        settings.dependencies.modules = ['gcc/12', 'openmpi/4']
        settings.dependencies.autoproc.modules = {'autoPROC/2024'}
        lines = ModulesSite().prepare_preamble(dependencies=['autoproc'], shell=BashShell()).splitlines()
        assert lines[1:] == ['module purge', 'module load gcc/12 openmpi/4', 'module load autoPROC/2024']

    def test_no_common_modules(self, modules):
        lines = ModulesSite().prepare_preamble(dependencies=None, shell=BashShell()).splitlines()
        assert lines[1:] == ['module purge']

    def test_dependency_without_modules(self, modules):
        lines = ModulesSite().prepare_preamble(dependencies=[Tool()], shell=BashShell()).splitlines()
        assert lines[1:] == ['module purge']

    def test_cmd_calls_module(self, modules):
        settings.dependencies.modules = ['gcc/12']
        preamble = ModulesSite().prepare_preamble(dependencies=None, shell=CmdShell())
        assert preamble.split('\r\n')[1:] == ['call module purge', 'call module load gcc/12', '']

    def test_load_a_single_module(self):
        assert ModulesSite._load_modules('gcc/12', shell=BashShell()) == 'module load gcc/12\n'

    def test_preamble_with_a_policy(self, modules):
        class TaggingModulesSite(ModulesSite):
            _policy = TaggingPolicy()

        assert TaggingModulesSite().prepare_preamble(dependencies=None, shell=BashShell()).endswith('[preamble]')

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
            return completed(command, stdout)

        monkeypatch.setattr(ModulesSite, 'execute_command', staticmethod(execute_command))
        settings.dependencies.modules = ['gcc/12']
        settings.dependencies.autoproc.modules = {'autoPROC/2024'}
        assert ModulesSite().check_dependencies(['autoproc'], shell=BashShell()) is found

    @pytest.mark.parametrize('result', ['1', None], ids=['missing', 'timed out'])
    def test_check_dependencies_without_the_module_command(self, modules, monkeypatch, result):
        def execute_command(command, timeout=20):
            assert 'module avail' not in command[-1]
            return None if result is None else completed(command, result)

        monkeypatch.setattr(ModulesSite, 'execute_command', staticmethod(execute_command))
        assert ModulesSite().check_dependencies(None, shell=BashShell()) is False

    def test_check_module_timed_out(self, monkeypatch):
        monkeypatch.setattr(ModulesSite, 'execute_command', staticmethod(lambda *a, **k: None))
        assert ModulesSite()._check_module('gcc/12', shell=BashShell()) is False

    def test_check_module_with_an_unsafe_name(self, monkeypatch, sites_log):
        monkeypatch.setattr(ModulesSite, 'execute_command', staticmethod(lambda *a, **k: pytest.fail('ran')))
        assert ModulesSite()._check_module('gcc; rm -rf x', shell=BashShell()) is False
        assert 'Invalid module name' in sites_log.text

    @pytest.mark.parametrize(
        'name',
        ['gcc', 'gcc/12', 'gcc/12.2.0', 'python/3.11', 'openmpi/4.1.5-gcc_12', '.hidden/1', 'a/b/c', 'gcc\\12'],
    )
    def test_safe_module_names(self, name):
        assert ModulesSite._is_module_name_safe(name)

    @pytest.mark.parametrize(
        'name',
        ['', 'a' * 256, '-x', '--help', 'gcc/-x', 'gcc/', '/gcc', 'gcc 12', 'gcc;ls', 'gcc\n', '$(ls)', 'gcc//12'],
    )
    def test_unsafe_module_names(self, name):
        assert not ModulesSite._is_module_name_safe(name)


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
