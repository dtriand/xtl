import asyncio
from dataclasses import dataclass
import functools
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from tests.conftest import skipif_not_linux, skipif_not_windows
from xtl import settings
from xtl.config.settings import DependencySettings
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.dependencies import DependencyCheck
from xtl.jobs.exceptions import SubprocessError
from xtl.jobs.policies import BaseCommandPolicy, CommandPolicy
from xtl.jobs.shells import BashShell, CmdShell, PowerShell, Shell
from xtl.jobs.sites import (
    BaseComputeSite,
    ComputeSite,
    LocalSite,
    ModulesSite,
    SlurmLocalSite,
    SlurmModulesSite,
    SlurmSite,
)
from xtl_core.system.compatibility import OS_WINDOWS

SLEEP_2 = 'ping -n 3 127.0.0.1 > nul' if OS_WINDOWS else 'sleep 2'


CHILD = """
import os, signal, sys, time

pidfile, mode = sys.argv[1], sys.argv[2]
if mode == 'ignore-sigint':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
elif mode == 'note-sigint':
    def noted(*args):
        with open(pidfile + '.sigint', 'w') as f:
            f.write('SIGINT')
        sys.exit(0)

    signal.signal(signal.SIGINT, noted)
elif mode == 'ignore-break':
    signal.signal(signal.SIGBREAK, signal.SIG_IGN)
elif mode == 'note-break':
    def noted(*args):
        with open(pidfile + '.sigbreak', 'w') as f:
            f.write('SIGBREAK')
        sys.exit(0)

    signal.signal(signal.SIGBREAK, noted)
with open(pidfile, 'w') as f:
    f.write(str(os.getpid()))
for _ in range(300):
    time.sleep(0.1)  # short sleeps: on Windows, a handler only runs between them
"""


def process_alive(pid: int) -> bool:
    if OS_WINDOWS:
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    # A zombie has ended, it only waits to be reaped
    return not (stat.exists() and stat.read_text().rsplit(')', 1)[-1].split()[0] == 'Z')


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


def completed(command, stdout: str, stderr: str = '') -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)


class Scripts:
    """Stands in for the scripts that checks run: records them, and answers with recorded output."""

    def __init__(self, output=None):
        self.scripts: list[str] = []
        self.output = output if callable(output) else (lambda script: output)

    # NB: Set on a site's class, an instance isn't bound to the site, so it gets only the script and the shell
    def __call__(self, script: str, shell) -> subprocess.CompletedProcess | None:
        self.scripts.append(script)
        output = self.output(script)
        if output is None:
            return None
        stdout, stderr = output if isinstance(output, tuple) else (output, '')
        return completed(['script'], stdout, stderr)


def report(*found: str, missing: tuple[str, ...] = (), kind: str = 'executable') -> str:
    """Lines as a check prints them."""
    lines = [f'xtl-check {kind} {name} 0' for name in found] + [f'xtl-check {kind} {name} 1' for name in missing]
    return '\n'.join(lines) + '\n'


@pytest.fixture(autouse=True)
def no_checks():
    """Start and end each test without dependency checks remembered in the settings."""
    settings.dependencies.clear_checks()
    yield
    settings.dependencies.clear_checks()


@pytest.fixture
def modules(monkeypatch):
    """Undo changes to the modules and paths in the settings."""
    monkeypatch.setattr(settings.dependencies, 'modules', [])
    monkeypatch.setattr(settings.dependencies, 'path', None)
    monkeypatch.setattr(settings.dependencies.autoproc, 'modules', [])
    monkeypatch.setattr(settings.dependencies.autoproc, 'path', None)


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

    def test_shells_and_policy_as_arguments(self):
        site = LocalSite(default_shell=Shell.CMD, shells=[CmdShell(), Shell.POWERSHELL], policy=CommandPolicy.NICE)
        assert site.default_shell == Shell.CMD
        assert site.supported_shells == frozenset({Shell.CMD, Shell.POWERSHELL})
        assert site.is_valid_shell(PowerShell()) and not site.is_valid_shell(Shell.BASH)
        assert site.policy == CommandPolicy.NICE
        assert site != LocalSite()
        # Without arguments, the site's own
        assert BashOnlySite().supported_shells == frozenset({Shell.BASH})
        assert TaggingSite(default_shell=BashShell()).policy is TaggingSite._policy

    def test_policy_instance_as_argument(self):
        policy = TaggingPolicy()
        assert LocalSite(policy=policy).policy is policy

    def test_default_shell_must_be_supported(self):
        with pytest.raises(ValueError, match='not one of the shells'):
            BashOnlySite(default_shell=Shell.CMD)
        with pytest.raises(ValueError, match='not one of the shells'):
            LocalSite(default_shell=Shell.CMD, shells=[Shell.BASH])

    @pytest.mark.parametrize('shell', [BashShell(), CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_prepare_shebang(self, shell):
        expected = f'{shell.shebang}{shell.new_line_char}' if shell.shebang else ''
        assert BaseComputeSite._prepare_shebang(shell) == expected


class TestLocalSite:
    def test_prepare_without_a_policy(self, modules):
        site, shell = LocalSite(), BashShell()
        # A guard line per executable of the dependencies, sorted
        assert site.prepare_preamble(dependencies=['autoproc'], shell=shell).splitlines() == [
            shell.shebang,
            shell.get_guard_command('aP_wf_process'),
            shell.get_guard_command('process'),
        ]
        assert site.prepare_preamble(dependencies=None, shell=shell) == f'{shell.shebang}\n'
        assert site.prepare_command('echo hi') == 'echo hi'
        assert site.prepare_content('echo hi') == 'echo hi'
        assert site.prepare_postamble() == ''

    def test_prepare_with_a_policy(self, modules):
        site, shell = TaggingSite(), BashShell()
        assert site.prepare_preamble(dependencies=None, shell=shell) == f'{shell.shebang}\n[preamble]'
        assert site.prepare_command('echo hi') == '[echo hi]'
        assert site.prepare_content('echo hi') == '<echo hi>'
        assert site.prepare_postamble() == '[postamble]'

    def test_setup_puts_paths_first_on_path(self, modules):
        settings.dependencies.path = Path('/opt/common')
        shell = BashShell()
        # The common directory first, so that a dependency's comes before it on PATH
        assert LocalSite().prepare_setup([Tool(path='/opt/tool'), 'autoproc'], shell=shell).splitlines() == [
            shell.get_path_command(Path('/opt/common')),
            shell.get_path_command(Path('/opt/tool')),
        ]
        assert (
            LocalSite().prepare_setup(None, shell=CmdShell())
            == f'{CmdShell().get_path_command(Path("/opt/common"))}\r\n'
        )

    def test_preamble_sets_up_before_the_guards(self, modules):
        shell = CmdShell()
        lines = LocalSite().prepare_preamble([Tool(path='C:/tool')], shell=shell).split('\r\n')
        assert lines == [
            shell.shebang,
            shell.get_path_command(Path('C:/tool')),
            shell.get_guard_command('helper'),
            shell.get_guard_command('tool'),
            '',
        ]

    def test_check_runs_the_setup_then_looks_up_each_executable(self, modules, monkeypatch):
        scripts = Scripts(report('helper', 'tool'))
        monkeypatch.setattr(LocalSite, '_run_script', scripts)
        settings.dependencies.path = Path('/opt/common')
        shell = BashShell()

        (check,) = LocalSite().check_dependencies(Tool(path='/opt/tool'), shell=shell)
        assert check == DependencyCheck(dependency='Tool', found=True)
        assert scripts.scripts == [
            f'{shell.shebang}\n'
            f'{shell.get_path_command(Path("/opt/common"))}\n'
            f'{shell.get_path_command(Path("/opt/tool"))}\n'
            f'{shell.get_found_report_command("helper")}\n'
            f'{shell.get_found_report_command("tool")}\n'
        ]

    @pytest.mark.parametrize(
        'output, missing',
        [
            (report('tool', missing=('helper',)), ('helper',)),
            (report('tool'), ('helper',)),  # not reported: the script stopped
            ('', ('helper', 'tool')),
        ],
        ids=['not found', 'not reported', 'no output'],
    )
    def test_check_reports_what_is_missing(self, monkeypatch, output, missing):
        monkeypatch.setattr(LocalSite, '_run_script', Scripts(output))
        (check,) = LocalSite().check_dependencies(Tool(), shell=BashShell())
        assert (check.dependency, check.found, check.missing_executables) == ('Tool', False, missing)
        assert check.problems() == [
            f'executable{"s" if len(missing) > 1 else ""} {", ".join(missing)} not found (required for Tool)'
        ]

    def test_check_that_did_not_finish(self, monkeypatch):
        monkeypatch.setattr(LocalSite, '_run_script', Scripts(None))
        (check,) = LocalSite().check_dependencies(Tool(), shell=BashShell())
        # Not reported as missing: it couldn't be checked
        assert (check.found, check.missing) == (False, ())
        assert check.problems() == [
            'Tool could not be checked: the check in bash did not finish within 60 s, or could not start'
        ]

    def test_check_keeps_what_was_printed(self, monkeypatch):
        output = ('Loading tool\n' + report('helper', 'tool'), 'a warning\n')
        monkeypatch.setattr(LocalSite, '_run_script', Scripts(output))
        (check,) = LocalSite().check_dependencies(Tool(), shell=BashShell())
        assert check.found and check.details == 'Loading tool\na warning'

    def test_check_each_dependency_alone(self, modules, monkeypatch):
        scripts = Scripts(lambda script: report('process', 'aP_wf_process', 'helper', 'tool'))
        monkeypatch.setattr(LocalSite, '_run_script', scripts)
        checks = LocalSite().check_dependencies(['autoproc', Tool()], shell=BashShell())
        assert [(check.dependency, check.found) for check in checks] == [('autoproc', True), ('Tool', True)]
        assert len(scripts.scripts) == 2
        assert 'helper' not in scripts.scripts[0] and 'process' not in scripts.scripts[1]

    def test_check_is_remembered(self, modules, monkeypatch):
        scripts = Scripts(report(missing=('process', 'aP_wf_process')))
        monkeypatch.setattr(LocalSite, '_run_script', scripts)
        (first,) = LocalSite().check_dependencies('autoproc', shell=BashShell())
        # Once per site and shell, failures too
        (second,) = LocalSite().check_dependencies('autoproc', shell=BashShell())
        assert second is first and len(scripts.scripts) == 1
        assert settings.dependencies.autoproc.get_check(LocalSite(), BashShell()) is first
        LocalSite().check_dependencies('autoproc', shell=CmdShell())
        ModulesSite().check_dependencies('autoproc', shell=BashShell())
        assert len(scripts.scripts) == 3

    def test_check_again_after_a_change(self, modules, monkeypatch):
        scripts = Scripts(report('process', 'aP_wf_process'))
        monkeypatch.setattr(LocalSite, '_run_script', scripts)
        LocalSite().check_dependencies('autoproc', shell=BashShell())
        settings.dependencies.autoproc.path = Path('/opt/autoproc')
        LocalSite().check_dependencies('autoproc', shell=BashShell())
        assert len(scripts.scripts) == 2

    def test_check_without_any(self, monkeypatch):
        monkeypatch.setattr(LocalSite, '_run_script', Scripts(lambda script: pytest.fail('ran')))
        assert LocalSite().check_dependencies(None, shell=BashShell()) == []

    def test_check_with_an_unsafe_name(self, monkeypatch, sites_log):
        class Unsafe(DependencySettings):
            _provides = frozenset({'tool; rm -rf x'})

        monkeypatch.setattr(LocalSite, '_run_script', Scripts(lambda script: pytest.fail('ran')))
        (check,) = LocalSite().check_dependencies(Unsafe(), shell=BashShell())
        assert (check.found, check.missing) == (False, ())
        assert check.error.startswith('Invalid executable name')
        assert check.problems()[0].startswith('Unsafe could not be checked: Invalid executable name')
        assert 'Cannot check dependency `Unsafe`' in sites_log.text

    @pytest.mark.parametrize('shell', [BashShell(), CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_check_runs_in_the_shell(self, shell, tmp_path, modules):
        if not Path(shell.executable).exists():
            pytest.skip(f'{shell.executable} not found')
        folder = tmp_path / 'my bin'
        folder.mkdir()
        if OS_WINDOWS:
            (folder / 'xtl-tool.bat').write_text('@echo ran\r\n')
        else:
            (folder / 'xtl-tool').write_text('#!/bin/sh\necho ran\n')
            (folder / 'xtl-tool').chmod(0o755)

        class XTLTool(DependencySettings):
            _provides = frozenset({'xtl-tool'})

        (check,) = LocalSite().check_dependencies(XTLTool(), shell=shell)
        assert (check.found, check.missing_executables) == (False, ('xtl-tool',))
        # Only after its directory is put on PATH
        (check,) = LocalSite().check_dependencies(XTLTool(path=folder), shell=shell)
        assert check.found, check.details

    @pytest.mark.asyncio
    async def test_execute_batch_without_input(self, tmp_path, monkeypatch):
        # A program that reads its input gets none, instead of waiting for the keyboard (as cmd's "Terminate batch
        #  job (Y/N)?" does after Ctrl+C, competing with the application's own prompts)
        started = []
        create = asyncio.create_subprocess_exec

        async def spy(*args, **kwargs):
            started.append(kwargs)
            return await create(*args, **kwargs)

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spy)
        batch = BatchFile(tmp_path / 'job')
        quote = batch.shell.sanitize_value
        batch.add_line(f'{quote(Path(sys.executable))} -c "import sys; print(len(sys.stdin.read()))"')
        batch.save()
        await asyncio.wait_for(LocalSite().execute_batch(batch, stdout=tmp_path / 'stdout.log'), 30)
        assert started[0]['stdin'] == asyncio.subprocess.DEVNULL
        # In a process group of its own on Windows, so that the console's Ctrl+C doesn't reach it
        assert started[0]['creationflags'] == (subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        assert (tmp_path / 'stdout.log').read_text().strip() == '0'

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

    # Stopping a batch file stops the programs it started

    @staticmethod
    def program_batch(tmp_path: Path, mode: str = 'plain', background: bool = False) -> tuple[BatchFile, Path]:
        """A batch file that starts a Python program, which writes its pid to a file and sleeps."""
        child, pidfile = tmp_path / 'child.py', tmp_path / 'child.pid'
        child.write_text(CHILD, encoding='utf-8')
        batch = BatchFile(tmp_path / 'job')
        quote = batch.shell.sanitize_value
        line = f'{quote(Path(sys.executable))} {quote(child)} {quote(pidfile)} {mode}'
        batch.add_lines([f'{line} &', 'exit 0'] if background else [line])
        batch.save()
        return batch, pidfile

    @staticmethod
    async def started(pidfile: Path) -> int:
        for _ in range(500):
            if pidfile.exists() and pidfile.read_text():
                return int(pidfile.read_text())
            await asyncio.sleep(0.02)
        raise TimeoutError('the program did not start')

    @staticmethod
    async def gone(pid: int) -> bool:
        for _ in range(100):
            if not process_alive(pid):
                return True
            await asyncio.sleep(0.05)
        return False

    @pytest.mark.asyncio
    async def test_cancel_stops_the_programs_it_started(self, tmp_path):
        batch, pidfile = self.program_batch(tmp_path)
        start = time.monotonic()
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert batch.status == BatchFileStatus.CANCELLED
        assert await self.gone(pid)
        assert time.monotonic() - start < 15  # not the program's 30 s

    @pytest.mark.asyncio
    async def test_a_cancelled_task_stops_the_programs_it_started(self, tmp_path):
        batch, pidfile = self.program_batch(tmp_path)
        start = time.monotonic()
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 60)
        assert await self.gone(pid)
        assert time.monotonic() - start < 15  # not the program's 30 s

    @pytest.mark.asyncio
    async def test_a_second_cancellation_doesnt_stop_the_clean_up(self, tmp_path):
        # A pool's exit cancels its jobs again, while the first cancellation is still stopping their programs
        batch, pidfile = self.program_batch(tmp_path)
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        running.cancel()
        while batch.status != BatchFileStatus.CANCELLED:  # the clean-up has started
            await asyncio.sleep(0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 60)
        assert batch.process.returncode is not None
        assert await self.gone(pid)

    @skipif_not_linux
    @pytest.mark.asyncio
    async def test_programs_get_sigint_first(self, tmp_path):
        batch, pidfile = self.program_batch(tmp_path, mode='note-sigint')
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert await self.gone(pid)
        assert (tmp_path / 'child.pid.sigint').read_text() == 'SIGINT'

    @skipif_not_linux
    @pytest.mark.asyncio
    async def test_programs_that_ignore_sigint_get_sigterm(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LocalSite, '_TERMINATE_GRACE', 0.5)
        batch, pidfile = self.program_batch(tmp_path, mode='ignore-sigint')
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        start = time.monotonic()
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert await self.gone(pid)
        assert time.monotonic() - start < 5

    @skipif_not_linux
    @pytest.mark.asyncio
    async def test_a_program_left_behind_by_the_shell(self, tmp_path):
        # The shell has ended, but the program it started in the background still holds the output streams
        batch, pidfile = self.program_batch(tmp_path, background=True)
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        while batch.process.returncode is None:
            await asyncio.sleep(0.02)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 30)
        assert await self.gone(pid)

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_programs_get_ctrl_break_first(self, tmp_path):
        batch, pidfile = self.program_batch(tmp_path, mode='note-break')
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert await self.gone(pid)
        assert (tmp_path / 'child.pid.sigbreak').read_text() == 'SIGBREAK'

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_programs_that_ignore_ctrl_break_are_ended(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LocalSite, '_TERMINATE_GRACE', 0.5)
        batch, pidfile = self.program_batch(tmp_path, mode='ignore-break')
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        start = time.monotonic()
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert await self.gone(pid)
        assert time.monotonic() - start < 5

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_without_a_console_the_tree_is_ended_at_once(self, tmp_path, monkeypatch):
        def no_console(pid, sig):
            raise OSError('no console')

        monkeypatch.setattr(os, 'kill', no_console)
        batch, pidfile = self.program_batch(tmp_path, mode='ignore-break')
        running = asyncio.create_task(LocalSite().execute_batch(batch))
        pid = await self.started(pidfile)
        start = time.monotonic()
        await batch.cancel()
        await asyncio.wait_for(running, 30)
        assert await self.gone(pid)
        assert time.monotonic() - start < LocalSite._TERMINATE_GRACE

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_kill_tree_falls_back_to_the_shell(self, tmp_path):
        class Shell:
            pid, returncode, killed = 4_000_000, None, False  # no such process: taskkill fails

            def kill(self):
                self.killed, self.returncode = True, 1

            async def wait(self):
                return self.returncode

        batch = BatchFile(tmp_path / 'job')
        batch._process = Shell()
        # NB: Not through `_terminate_batch()`, whose Ctrl+Break to a group that doesn't exist could reach every
        #  process on the console
        await LocalSite._kill_tree(batch)
        assert batch.process.killed


MODULE_COMMAND = shutil.which('module')
"""Environment Modules' `module.cmd` on Windows, where a test can run it; elsewhere `module` is a shell function"""


@functools.cache
def has_envmodule() -> bool:
    """Whether xtl's PowerShell has Environment Modules' `envmodule`, from its profile, as batch files run."""
    if not (OS_WINDOWS and MODULE_COMMAND):
        return False
    probe = 'if (Get-Command envmodule -ErrorAction SilentlyContinue) { 0 } else { 1 }'
    result = subprocess.run(
        [PowerShell().executable, '-NonInteractive', '-Command', probe], capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip().endswith('0')


def skip_without_module_command(shell) -> None:
    """Skip a test in PowerShell if its profile doesn't set up Environment Modules."""
    if shell == Shell.POWERSHELL and not has_envmodule():
        pytest.skip("PowerShell's profile doesn't set up Environment Modules")


@pytest.fixture
def module_tree(tmp_path, monkeypatch):
    """A module `xtltool/1.0` that puts a program `xtl-tool` on PATH, in a module path of its own."""
    if not (OS_WINDOWS and MODULE_COMMAND):
        pytest.skip('Environment Modules is not installed')
    folder = tmp_path / 'tool bin'
    folder.mkdir()
    (folder / 'xtl-tool.bat').write_text('@echo ran\r\n')
    modulefile = tmp_path / 'modulefiles' / 'xtltool' / '1.0'
    modulefile.parent.mkdir(parents=True)
    modulefile.write_text(f'#%Module\nprepend-path PATH {{{folder.as_posix()}}}\n')
    monkeypatch.setenv('MODULEPATH', str(tmp_path / 'modulefiles'))


class XTLTool(DependencySettings):
    _provides = frozenset({'xtl-tool'})


class TestModulesSite:
    def test_common_modules_load_first(self, modules):
        settings.dependencies.modules = ['gcc/12', 'openmpi/4']
        settings.dependencies.autoproc.modules = ['autoPROC/2024']
        shell = BashShell()
        lines = ModulesSite().prepare_preamble(dependencies=['autoproc'], shell=shell).splitlines()
        assert lines == [
            shell.shebang,
            shell.module_setup,
            'module purge',
            'module load gcc/12 openmpi/4',
            'module load autoPROC/2024',
            shell.get_guard_command('aP_wf_process'),
            shell.get_guard_command('process'),
        ]

    def test_no_common_modules(self, modules):
        lines = ModulesSite().prepare_preamble(dependencies=None, shell=CmdShell()).split('\r\n')
        assert lines == [CmdShell().shebang, 'call module purge', '']

    def test_dependency_without_modules(self, modules):
        lines = ModulesSite().prepare_preamble(dependencies=[Tool()], shell=CmdShell()).split('\r\n')
        assert lines[1:3] == ['call module purge', CmdShell().get_guard_command('helper')]

    def test_paths_after_the_modules(self, modules):
        settings.dependencies.path = Path('C:/common')
        settings.dependencies.modules = ['gcc/12']
        shell = CmdShell()
        lines = ModulesSite().prepare_setup([Tool(path='C:/tool')], shell=shell).split('\r\n')
        assert lines == [
            'call module purge',
            'call module load gcc/12',
            shell.get_path_command(Path('C:/common')),
            shell.get_path_command(Path('C:/tool')),
            '',
        ]

    def test_powershell_uses_envmodule(self, modules):
        # Set up by the profile, which batch files run with
        settings.dependencies.modules = ['gcc/12']
        lines = ModulesSite().prepare_setup(None, shell=PowerShell()).splitlines()
        assert lines == ['envmodule purge', 'envmodule load gcc/12']

    def test_unsafe_module_name(self, modules):
        settings.dependencies.modules = ['gcc; rm -rf x']
        with pytest.raises(ValueError, match='Invalid module name'):
            ModulesSite().prepare_preamble(dependencies=None, shell=BashShell())

    def test_preamble_with_a_policy(self, modules):
        class TaggingModulesSite(ModulesSite):
            _policy = TaggingPolicy()

        assert TaggingModulesSite().prepare_preamble(dependencies=None, shell=BashShell()).endswith('[preamble]')

    def test_check_loads_each_module_then_looks_up_the_executables(self, modules, monkeypatch):
        scripts = Scripts(report('gcc/12', 'autoPROC/2024', kind='module') + report('process', 'aP_wf_process'))
        monkeypatch.setattr(ModulesSite, '_run_script', scripts)
        settings.dependencies.modules = ['gcc/12']
        settings.dependencies.autoproc.modules = ['autoPROC/2024']
        shell = CmdShell()

        (check,) = ModulesSite().check_dependencies(['autoproc'], shell=shell)
        assert check == DependencyCheck(dependency='autoproc', found=True)
        assert scripts.scripts[0].split('\r\n') == [
            shell.shebang,
            shell.get_found_report_command('module', kind='command'),
            'call module purge',
            shell.get_load_report_command('gcc/12'),
            shell.get_load_report_command('autoPROC/2024'),
            shell.get_found_report_command('aP_wf_process'),
            shell.get_found_report_command('process'),
            '',
        ]

    @pytest.mark.parametrize(
        'output, problems',
        [
            # A module that doesn't load, and so the executables it provides
            (
                (
                    report('module', kind='command')
                    + report('gcc/12', kind='module')
                    + report(missing=('autoPROC/2024',), kind='module')
                    + report(missing=('process', 'aP_wf_process')),
                    "ERROR: Unable to locate a modulefile for 'autoPROC/2024'\n",
                ),
                [
                    'module autoPROC/2024 not found (required for autoproc)',
                    'executables aP_wf_process, process not found (required for autoproc)',
                ],
            ),
            # A module that every job loads
            (
                report('module', kind='command')
                + report(missing=('gcc/12',), kind='module')
                + report('autoPROC/2024', kind='module')
                + report('process', 'aP_wf_process'),
                ['module gcc/12 not found (loaded for every job)'],
            ),
            # The modules load, but don't provide an executable
            (
                report('module', kind='command')
                + report('gcc/12', 'autoPROC/2024', kind='module')
                + report('process', missing=('aP_wf_process',)),
                ['executable aP_wf_process not found (required for autoproc)'],
            ),
            # No module command: nothing loads
            (
                report(missing=('module',), kind='command')
                + report(missing=('gcc/12', 'autoPROC/2024'), kind='module')
                + report(missing=('process', 'aP_wf_process')),
                [
                    'module command `module` not found (required for autoproc, to load gcc/12, autoPROC/2024)',
                    'executables aP_wf_process, process not found (required for autoproc)',
                ],
            ),
        ],
        ids=['module missing', 'common module missing', 'executable missing', 'no module command'],
    )
    def test_check_reports_what_is_missing(self, modules, monkeypatch, output, problems):
        monkeypatch.setattr(ModulesSite, '_run_script', Scripts(output))
        settings.dependencies.modules = ['gcc/12']
        settings.dependencies.autoproc.modules = ['autoPROC/2024']
        (check,) = ModulesSite().check_dependencies(['autoproc'], shell=BashShell())
        assert not check.found
        assert check.problems() == problems

    def test_check_keeps_the_error_of_a_module(self, modules, monkeypatch):
        error = "ERROR: Unable to locate a modulefile for 'autoPROC/2024'"
        output = (
            report(missing=('autoPROC/2024',), kind='module') + report(missing=('process', 'aP_wf_process')),
            error,
        )
        monkeypatch.setattr(ModulesSite, '_run_script', Scripts(output))
        settings.dependencies.autoproc.modules = ['autoPROC/2024']
        (check,) = ModulesSite().check_dependencies(['autoproc'], shell=BashShell())
        assert check.details == error

    def test_check_with_an_unsafe_module_name(self, modules, monkeypatch):
        monkeypatch.setattr(ModulesSite, '_run_script', Scripts(lambda script: pytest.fail('ran')))
        settings.dependencies.autoproc.modules = ['autoPROC; rm -rf x']
        (check,) = ModulesSite().check_dependencies(['autoproc'], shell=BashShell())
        assert not check.found and check.error.startswith('Invalid module name')
        assert check.problems()[0].startswith("autoproc could not be checked: Invalid module name: 'autoPROC; rm")

    @pytest.mark.parametrize('shell', [CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_check_with_environment_modules(self, module_tree, modules, shell):
        skip_without_module_command(shell)
        (check,) = ModulesSite().check_dependencies(XTLTool(modules=['xtltool/1.0']), shell=shell)
        assert check.found, check.details
        # Not without the module
        (check,) = ModulesSite().check_dependencies(XTLTool(), shell=shell)
        assert (check.found, check.missing) == (False, ('xtl-tool',))
        assert check.problems() == ['executable xtl-tool not found (required for XTLTool)']

    @pytest.mark.parametrize('shell', [CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_check_with_a_missing_module(self, module_tree, modules, shell):
        skip_without_module_command(shell)
        (check,) = ModulesSite().check_dependencies(XTLTool(modules=['xtltool/9.9']), shell=shell)
        assert (check.found, check.missing) == (False, ('xtltool/9.9', 'xtl-tool'))
        assert check.problems() == [
            'module xtltool/9.9 not found (required for XTLTool)',
            'executable xtl-tool not found (required for XTLTool)',
        ]
        assert "Unable to locate a modulefile for 'xtltool/9.9'" in check.details

    @pytest.mark.parametrize('shell', [CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_batch_file_with_environment_modules(self, module_tree, modules, shell, tmp_path):
        # The preamble loads the module, and the guard lets the commands run
        skip_without_module_command(shell)
        dependency = XTLTool(modules=['xtltool/1.0'])
        text = ModulesSite().prepare_preamble([dependency], shell=shell) + 'xtl-tool' + shell.new_line_char
        batch = tmp_path / f'job{shell.batch_extension}'
        batch.write_text(text, encoding='utf-8', newline='')
        result = subprocess.run(shell.get_execute_batch_command(batch, as_list=True), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'ran' in result.stdout

    def test_powershell_without_envmodule(self, module_tree, modules):
        # A profile without Environment Modules: the modules don't load, and the check says why
        if has_envmodule():
            pytest.skip("PowerShell's profile sets up Environment Modules")
        (check,) = ModulesSite().check_dependencies(XTLTool(modules=['xtltool/1.0']), shell=PowerShell())
        assert (check.found, check.missing_command) == (False, 'envmodule')
        assert check.problems() == [
            'module command `envmodule` not found (required for XTLTool, to load xtltool/1.0)',
            'executable xtl-tool not found (required for XTLTool)',
        ]
        assert 'envmodule' in check.details


class TestSlurmSites:
    @pytest.fixture
    def directives(self, monkeypatch):
        monkeypatch.setattr(SlurmSite, '_prepare_slurm_preamble', staticmethod(lambda shell, config: '#SBATCH -n 1\n'))

    def test_directives_before_the_setup_and_guards(self, modules, directives):
        settings.dependencies.autoproc.modules = ['autoPROC/2024']
        shell = BashShell()
        lines = SlurmModulesSite().prepare_preamble(['autoproc'], shell=shell, config={'x': 1}).splitlines()
        assert lines == [
            shell.shebang,
            '#SBATCH -n 1',
            shell.module_setup,
            'module purge',
            'module load autoPROC/2024',
            shell.get_guard_command('aP_wf_process'),
            shell.get_guard_command('process'),
        ]

    def test_without_a_configuration(self, modules, directives):
        shell = BashShell()
        assert SlurmLocalSite().prepare_preamble(['autoproc'], shell=shell).splitlines() == [
            shell.shebang,
            shell.get_guard_command('aP_wf_process'),
            shell.get_guard_command('process'),
        ]

    def test_check_on_the_submit_host(self, modules, monkeypatch):
        scripts = Scripts(report('process', 'aP_wf_process'))
        monkeypatch.setattr(LocalSite, '_run_script', scripts)
        (check,) = SlurmLocalSite().check_dependencies(['autoproc'], shell=BashShell())
        assert check.found and len(scripts.scripts) == 1
