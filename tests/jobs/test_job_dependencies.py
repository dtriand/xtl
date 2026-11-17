import importlib
import pkgutil
import subprocess

import pytest

import xtl
from xtl import settings
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.dependencies import unknown_dependencies
from xtl.jobs.exceptions import DependencyError
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.shells import DefaultShell, Shell
from xtl.jobs.sites import LocalSite
from xtl.jobs.steps import StepSpec
from xtl_core.options import Option
from xtl_core.system.compatibility import OS_WINDOWS

SYSTEM_SHELL = Shell(DefaultShell().name)


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
        assert 'atsas: datcmp, datop' in str(result.error)
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
