import importlib
import inspect
from pathlib import Path
import pickle
import pkgutil
import subprocess

import pytest
import typer
from typer.testing import CliRunner

from tests.jobs.helpers import POOLS
import xtl
from xtl import settings
from xtl.cli.utilities.common import (
    ConsoleOptions,
    JobOptions,
    PermissionsOptions,
    get_console_options,
    get_job_options,
)
from xtl.cli.utilities.decorators import CLI_DEPENDENCIES, attach_hook, requires, typer_async
from xtl.config.settings import AtsasSettings, XTLSettings
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.dependencies import DependencyCheck, dependency_name, site_key, unknown_dependencies
from xtl.jobs.exceptions import DependencyError
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.pools import AsyncPool
from xtl.jobs.results import JobStatus
from xtl.jobs.shells import BashShell, CmdShell, DefaultShell, Shell
from xtl.jobs.sites import ComputeSite, LocalSite, ModulesSite
from xtl.jobs.steps import StepSpec
from xtl.tui.console import ConsoleIO
from xtl_core.options import Option
from xtl_core.system.compatibility import OS_WINDOWS

SYSTEM_SHELL = Shell(DefaultShell().name)

MISSING = DependencyCheck(dependency='atsas', found=False, missing_executables=('datcmp',), details='datcmp: not found')
FOUND = DependencyCheck(dependency='atsas', found=True)


class Checks:
    """Stands in for the scripts that dependency checks run, reporting executables as found or not."""

    def __init__(self, found: bool):
        self.found = found
        self.scripts: list[str] = []

    # NB: Set on a site's class, an instance isn't bound to the site, so it gets only the script and the shell
    def __call__(self, script: str, shell) -> subprocess.CompletedProcess:
        self.scripts.append(script)
        code = 0 if self.found else 1
        stdout = ''.join(f'xtl-check executable {exe} {code}\n' for exe in sorted(settings.dependencies.atsas.provides))
        return subprocess.CompletedProcess(['script'], 0, stdout=stdout, stderr='' if self.found else 'not here')


@pytest.fixture(autouse=True)
def plain_settings(monkeypatch):
    """No remembered checks; ATSAS on the local site, without settings of its own; checks on."""
    settings.dependencies.clear_checks()
    monkeypatch.setattr(settings.jobs, 'compute_site', 'local')
    monkeypatch.setattr(settings.dependencies, 'check', True)
    for field in ('compute_site', 'shell', 'policy', 'path'):
        monkeypatch.setattr(settings.dependencies.atsas, field, None)
    monkeypatch.setattr(settings.dependencies.atsas, 'modules', [])
    yield
    settings.dependencies.clear_checks()


@pytest.fixture
def fake_atsas(tmp_path, monkeypatch):
    """ATSAS's executables, in a directory that `dependencies.atsas.path` points to, so that guards find them."""
    folder = tmp_path / 'atsas bin'
    folder.mkdir()
    for exe in settings.dependencies.atsas.provides:
        if OS_WINDOWS:
            (folder / f'{exe}.bat').write_text('@echo atsas\r\n')
        else:
            (folder / exe).write_text('#!/bin/sh\necho atsas\n')
            (folder / exe).chmod(0o755)
    monkeypatch.setattr(settings.dependencies.atsas, 'path', folder)


def checks(monkeypatch, found: bool) -> Checks:
    checks = Checks(found)
    monkeypatch.setattr(LocalSite, '_run_script', checks)
    return checks


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

    def test_stale_after_a_change_in_place(self):
        dep = settings.dependencies.atsas
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


# Jobs


class Echo(BatchJob[BatchJobConfig]):
    _dependencies = {'atsas'}


def echo_config(tmp_path) -> BatchJobConfig:
    return BatchJobConfig(job_directory=tmp_path / 'job', templates={SYSTEM_SHELL: 'echo ran'})


class ValueConfig(JobConfig):
    value: int = Option(default=0, desc='A value')


class Ran(Job[ValueConfig]):
    """Notes that it ran."""

    runs: list[int] = []

    async def _execute(self):
        Ran.runs.append(self.config.value)
        return self.config.value


class NeedsAtsas(Ran):
    _dependencies = 'atsas'


class Inner(SteppedJob[SteppedJobConfig]):
    _steps = (StepSpec(name='needs', job_cls=NeedsAtsas, config_cls=ValueConfig, defaults={'value': 2}),)


class Outer(SteppedJob[SteppedJobConfig]):
    _dependencies = {'autoproc'}
    _steps = (
        StepSpec(name='first', job_cls=Ran, config_cls=ValueConfig, defaults={'value': 1}),
        StepSpec(name='inner', job_cls=Inner, config_cls=SteppedJobConfig),
    )


class Both(BatchJob[BatchJobConfig]):
    # NB: At module level, where a pool's submission finds it
    _dependencies = {'atsas', 'easybragg'}


class TestDeclarations:
    def test_none(self):
        assert Job.dependencies() == frozenset()
        assert Ran.dependencies() == frozenset()

    def test_as_a_set_or_a_name(self):
        assert Echo.dependencies() == frozenset({'atsas'})
        assert NeedsAtsas.dependencies() == frozenset({'atsas'})
        assert isinstance(NeedsAtsas._dependencies, frozenset)

    def test_inherited(self):
        class Child(Echo):
            pass

        assert Child.dependencies() == frozenset({'atsas'})

    def test_stepped_jobs_collect_their_steps(self):
        # Its own and its steps', and their steps', without building a configuration
        assert Inner.dependencies() == frozenset({'atsas'})
        assert Outer.dependencies() == frozenset({'atsas', 'autoproc'})


def job_classes() -> list[type[Job]]:
    """Every job class that xtl defines, from the modules of its `jobs` packages."""
    for module in pkgutil.walk_packages(xtl.__path__, prefix='xtl.'):
        if '.jobs' in module.name:
            importlib.import_module(module.name)
    classes, todo = [], [Job]
    while todo:
        cls = todo.pop()
        todo.extend(cls.__subclasses__())
        if cls.__module__.startswith('xtl.'):
            classes.append(cls)
    return sorted(set(classes), key=lambda cls: f'{cls.__module__}.{cls.__qualname__}')


class TestJobsOfXtl:
    def test_found(self):
        names = {cls.__name__ for cls in job_classes()}
        assert {'DatcmpBatchJob', 'DatopBatchJob', 'NanoBraggBatchJob', 'SAXSCompareJob'} <= names

    @pytest.mark.parametrize('cls', job_classes(), ids=lambda cls: cls.__name__)
    def test_dependencies_are_known(self, cls):
        assert unknown_dependencies(cls.dependencies()) == []

    def test_atsas_jobs_run_executables_that_atsas_provides(self):
        from xtl.scattering.jobs.atsas_utils import ATSASOptions

        options = ATSASOptions.__subclasses__()
        assert options
        for cls in options:
            executable = cls.__private_attributes__['_executable'].get_default()
            assert executable in settings.dependencies.atsas.provides, cls.__name__

    def test_job_dependencies(self):
        from xtl.nanobragg.jobs.nanobragg import NanoBraggBatchJob, NanoBraggJob
        from xtl.scattering.jobs.atsas import DatcmpBatchJob, DatopBatchJob

        assert DatcmpBatchJob.dependencies() == DatopBatchJob.dependencies() == frozenset({'atsas'})
        assert NanoBraggBatchJob.dependencies() == NanoBraggJob.dependencies() == frozenset({'easybragg'})


class TestChecksBeforeRunning:
    @pytest.mark.asyncio
    async def test_found(self, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=True)
        result = await Echo(echo_config(tmp_path), job_id='e1').run()
        assert result.success, result.error
        assert result.data.stdout.strip() == 'ran'
        assert len(scripts.scripts) == 1

    @pytest.mark.asyncio
    async def test_missing(self, tmp_path, monkeypatch):
        checks(monkeypatch, found=False)
        result = await Echo(echo_config(tmp_path), job_id='e2').run()
        assert isinstance(result.error, DependencyError)
        assert result.error.dependencies == ('atsas',)
        assert 'executables datcmp, datop not found (required for atsas)' in str(result.error)
        assert 'not here' in str(result.error)
        # Before the batch file was written
        assert not (tmp_path / 'job').exists()

    @pytest.mark.asyncio
    async def test_checked_once(self, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=True)
        for i in range(3):
            assert (await Echo(echo_config(tmp_path / str(i)), job_id=f'e3{i}').run()).success
        assert len(scripts.scripts) == 1

    @pytest.mark.asyncio
    async def test_switched_off(self, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=False)
        monkeypatch.setattr(settings.dependencies, 'check', False)
        result = await Echo(echo_config(tmp_path), job_id='e4').run()
        assert result.success, result.error
        assert scripts.scripts == []

    @pytest.mark.asyncio
    async def test_switched_off_the_guards_still_stop_the_job(self, tmp_path, monkeypatch):
        # ATSAS isn't installed: no check, but the batch file stops before its first command
        monkeypatch.setattr(settings.dependencies, 'check', False)
        result = await Echo(echo_config(tmp_path), job_id='e7').run()
        assert 'failed with return code 127' in str(result.error)
        assert 'xtl: datcmp not found' in (tmp_path / 'job' / 'batch_job.stderr.log').read_text()
        assert 'ran' not in (tmp_path / 'job' / 'batch_job.stdout.log').read_text()

    @pytest.mark.asyncio
    async def test_batch_file_guards_the_executables(self, tmp_path, monkeypatch, fake_atsas):
        checks(monkeypatch, found=True)
        job = Echo(echo_config(tmp_path), job_id='e5')
        await job.run()
        # NB: The job tidied up its files, so the batch file's preamble as it was written
        batch = job._batch
        for executable in ('datcmp', 'datop'):
            assert batch.shell.get_guard_command(executable) in batch.get_preamble()

    @pytest.mark.asyncio
    async def test_without_dependencies(self, tmp_path, monkeypatch):
        scripts = checks(monkeypatch, found=False)
        result = await BatchJob(echo_config(tmp_path), job_id='e6').run()
        assert result.success, result.error
        assert scripts.scripts == []

    @pytest.mark.asyncio
    async def test_stepped_job_checks_before_its_first_step(self, monkeypatch):
        checks(monkeypatch, found=False)
        Ran.runs = []
        result = await Outer(SteppedJobConfig(), job_id='o1').run()
        assert isinstance(result.error, DependencyError)
        assert Ran.runs == []

    @pytest.mark.asyncio
    async def test_steps_use_the_outcome_of_the_check(self, monkeypatch):
        monkeypatch.setattr(settings.dependencies.autoproc, 'path', None)
        monkeypatch.setattr(settings.dependencies.autoproc, 'modules', [])
        scripts = checks(monkeypatch, found=True)
        Ran.runs = []
        result = await Inner(SteppedJobConfig(), job_id='i1').run()
        assert result.success, result.error
        assert Ran.runs == [2]
        # The step's own check found the outcome of the job's
        assert len(scripts.scripts) == 1


# Pools

pool_types = pytest.mark.parametrize('pool_cls', POOLS, ids=lambda cls: cls.__name__)


class TestPools:
    @pool_types
    @pytest.mark.asyncio
    async def test_checked_once_for_the_pool(self, pool_cls, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=True)
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Echo, [echo_config(tmp_path / str(i)) for i in range(3)], job_ids=['a', 'b', 'c'])
            results = await pool.launch()
        assert [result.success for result in results] == [True] * 3, [result.error for result in results]
        # Once, before the jobs started; they found the outcome (workers of a process pool, with the settings)
        assert len(scripts.scripts) == 1

    @pool_types
    @pytest.mark.asyncio
    async def test_missing_fails_the_jobs_without_running_them(self, pool_cls, tmp_path, monkeypatch):
        scripts = checks(monkeypatch, found=False)
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Echo, [echo_config(tmp_path / 'a'), echo_config(tmp_path / 'b')], job_ids=['a', 'b'])
            pool.submit(BatchJob, echo_config(tmp_path / 'c'), job_ids=['c'])
            results = await pool.launch()
        a, b, c = results
        for result in (a, b):
            assert isinstance(result.error, DependencyError)
            assert result.error.dependencies == ('atsas',)
        # A job without the dependency runs
        assert c.success, c.error
        assert pool.status == {'a': JobStatus.FAILED, 'b': JobStatus.FAILED, 'c': JobStatus.DONE}
        assert not (tmp_path / 'a').exists() and not (tmp_path / 'b').exists()
        assert len(scripts.scripts) == 1

    @pytest.mark.asyncio
    async def test_dependencies_that_disagree_fail_the_job(self, tmp_path, monkeypatch):
        scripts = checks(monkeypatch, found=True)
        monkeypatch.setattr(settings.dependencies.atsas, 'compute_site', 'local')
        monkeypatch.setattr(settings.dependencies.easybragg, 'compute_site', 'modules')
        async with AsyncPool() as pool:
            pool.submit(Both, echo_config(tmp_path), job_ids=['both'])
            (result,) = await pool.launch()
        assert isinstance(result.error, DependencyError)
        assert 'different values of `compute_site`' in str(result.error)
        assert scripts.scripts == []

    @pytest.mark.asyncio
    async def test_stream(self, tmp_path, monkeypatch):
        checks(monkeypatch, found=False)
        async with AsyncPool() as pool:
            pool.submit(Echo, echo_config(tmp_path), job_ids=['a'])
            results = [result async for result in await pool.launch(mode='stream')]
        assert isinstance(results[0].error, DependencyError)

    @pytest.mark.asyncio
    async def test_switched_off(self, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=False)
        monkeypatch.setattr(settings.dependencies, 'check', False)
        async with AsyncPool() as pool:
            pool.submit(Echo, echo_config(tmp_path), job_ids=['a'])
            (result,) = await pool.launch()
        assert result.success, result.error
        assert scripts.scripts == []

    @pytest.mark.asyncio
    async def test_each_launch_checks_its_jobs(self, tmp_path, monkeypatch, fake_atsas):
        scripts = checks(monkeypatch, found=True)
        async with AsyncPool() as pool:
            pool.submit(Echo, echo_config(tmp_path / 'a'), job_ids=['a'])
            await pool.launch()
            # Remembered: a later launch doesn't check again, unless the settings change
            pool.submit(Echo, echo_config(tmp_path / 'b'), job_ids=['b'])
            await pool.launch()
            assert len(scripts.scripts) == 1
            settings.dependencies.clear_checks()
            pool.submit(Echo, echo_config(tmp_path / 'c'), job_ids=['c'])
            await pool.launch()
        assert len(scripts.scripts) == 2


# Command line tools: a command written like xtl's, which declares its dependencies and checks them once

cli = typer.Typer()


@cli.command()
@requires('atsas')
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def echo(
    root: Path = typer.Option(..., '--root', help='Where the job works'),
    job_options: JobOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.apply_job_options(job_options)
    console.check_dependencies()
    async with AsyncPool() as pool:
        pool.submit(Echo, echo_config(root), job_ids=['a'])
        (result,) = await pool.launch()
    console.print(f'success: {result.success}')


@pytest.fixture
def restore_settings():
    """Put back the settings that a command changes."""
    before = settings.to_dict()
    yield
    settings.update_from(XTLSettings.from_dict(before))


def run(*args: str):
    # NB: The default of `--compute-site` is read when the command is defined, so it is always given
    return CliRunner().invoke(cli, ['--compute-site', 'local', *args], env={'COLUMNS': '200'})


class TestRequires:
    def test_names(self):
        assert echo.dependencies == ('atsas',)
        assert requires('atsas', 'easybragg', 'atsas')(lambda: None).dependencies == ('atsas', 'easybragg')

    @pytest.mark.parametrize('names, match', [((), 'at least one'), (('atsas', 'nope'), 'Unknown dependencies: nope')])
    def test_invalid(self, names, match):
        with pytest.raises(ValueError, match=match):
            requires(*names)

    def test_only_while_the_command_runs(self):
        @requires('easybragg')
        def command():
            return CLI_DEPENDENCIES.get()

        assert command() == ('easybragg',)
        assert CLI_DEPENDENCIES.get() == ()


class TestCommandLine:
    def test_checked_once_for_the_command(self, tmp_path, monkeypatch, fake_atsas, restore_settings):
        scripts = checks(monkeypatch, found=True)
        result = run('--root', str(tmp_path))
        assert result.exit_code == 0, result.output
        assert 'success: True' in result.output
        # The command's check; the pool and the job found its outcome
        assert len(scripts.scripts) == 1

    def test_missing_exits(self, tmp_path, monkeypatch, restore_settings):
        checks(monkeypatch, found=False)
        result = run('--root', str(tmp_path))
        assert result.exit_code == 1
        assert 'Missing dependencies' in result.output
        assert '- executables datcmp, datop not found (required for atsas)' in result.output
        assert 'not here' not in result.output
        assert 'Run with -v to see what the checks printed' in result.output
        # No job ran
        assert 'success' not in result.output and not (tmp_path / 'job').exists()

    def test_missing_with_details(self, tmp_path, monkeypatch, restore_settings):
        checks(monkeypatch, found=False)
        result = run('--root', str(tmp_path), '-v')
        assert result.exit_code == 1
        assert 'Checking dependency: atsas (LocalSite' in result.output
        assert 'What the check of atsas (LocalSite' in result.output
        assert 'not here' in result.output

    def test_switched_off(self, tmp_path, monkeypatch, fake_atsas, restore_settings):
        scripts = checks(monkeypatch, found=False)
        monkeypatch.setattr(settings.dependencies, 'check', False)
        result = run('--root', str(tmp_path))
        assert result.exit_code == 0, result.output
        assert scripts.scripts == []

    def test_dependencies_given(self, monkeypatch):
        checks(monkeypatch, found=False)
        with pytest.raises(typer.Exit):
            ConsoleIO().check_dependencies(['atsas'])
        # None declared, none given: nothing to check
        ConsoleIO().check_dependencies()

    def test_unknown_name(self, monkeypatch, capsys):
        scripts = checks(monkeypatch, found=True)
        with pytest.raises(typer.Exit):
            ConsoleIO(width=200).check_dependencies(['nope'])
        assert '- unknown dependency nope (not in the [dependencies] of xtl.toml)' in capsys.readouterr().out
        assert scripts.scripts == []

    def test_a_check_that_could_not_run(self, monkeypatch, capsys):
        monkeypatch.setattr(LocalSite, '_run_script', lambda site, script, shell: None)
        with pytest.raises(typer.Exit):
            ConsoleIO(width=200).check_dependencies(['atsas'])
        out = capsys.readouterr().out
        assert '- atsas could not be checked: the check in' in out
        assert 'not found' not in out


class TestModulesOption:
    def test_parse(self):
        assert JobOptions.parse_module_names(None) == {'extra': []}
        assert JobOptions.parse_module_names('gcc/12,atsas=atsas/4,atsas=x/1') == {
            'extra': ['gcc/12'],
            'atsas': ['atsas/4', 'x/1'],
        }
        with pytest.raises(typer.BadParameter):
            JobOptions.parse_module_names('a=b=c')

    def test_into_the_settings(self, monkeypatch, restore_settings):
        monkeypatch.setattr(settings.dependencies, 'modules', ['from/config'])
        monkeypatch.setattr(settings.dependencies.atsas, 'path', Path('/opt/atsas'))
        options = JobOptions(
            compute_site=ComputeSite.MODULES,
            permissions=PermissionsOptions(
                update=settings.jobs.permissions.update,
                files=settings.jobs.permissions.files,
                directories=settings.jobs.permissions.directories,
            ),
            modules=JobOptions.parse_module_names('gcc/12,atsas=atsas/4,unknown=other/1'),
            keep_temp=settings.jobs.keep_temp,
        )
        ConsoleIO().apply_job_options(options)
        # Modules not of a known dependency replace the common ones; a dependency's replace its own, and its path
        assert settings.dependencies.modules == ['gcc/12', 'other/1']
        assert (settings.dependencies.atsas.modules, settings.dependencies.atsas.path) == (['atsas/4'], None)
        assert settings.jobs.compute_site == ComputeSite.MODULES

    def test_no_default(self):
        # The default of `--modules` doesn't depend on the settings, or on which commands were imported
        option = inspect.signature(get_job_options).parameters['modules'].default
        assert option.default is None
