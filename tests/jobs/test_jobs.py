"""
Tests for jobs: the job data that travels to workers, the base job, batch jobs and stepped jobs.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from tests.conftest import skipif_not_windows
from tests.jobs.helpers import Failing, Talker
from xtl import settings
from xtl.jobs import jobs as jobs_module
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.exceptions import SubprocessError
from xtl.jobs.jobs import BatchJob, Job, JobData, SteppedJob
from xtl.jobs.logging import JobFileHandlerConfig, LoggerConfig
from xtl.jobs.pools import SimplePool
from xtl.jobs.resources import Resources
from xtl.jobs.results import BatchResults, JobResults, SteppedJobResults
from xtl.jobs.shells import DefaultShell, Shell
from xtl.jobs.steps import StepSpec
from xtl_core.options import Option, OptionsValidationError
from xtl_core.system.compatibility import OS_WINDOWS

SYSTEM_SHELL = Shell(DefaultShell().name)
SLEEP_5 = 'ping -n 6 127.0.0.1 > nul' if OS_WINDOWS else 'sleep 5'
EXIT_3 = 'exit /b 3' if OS_WINDOWS else 'exit 3'


class Records(logging.Handler):
    """Collects the records of a job's logger."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int = logging.NOTSET) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


def recorded(job: Job) -> Records:
    """Replace the handlers of a job's logger with one that collects its records, down to DEBUG."""
    records = Records()
    job.logger.handlers[:] = [records]
    job.logger.setLevel(logging.DEBUG)
    return records


class ValueConfig(JobConfig):
    value: int = Option(default=0, desc='Returned by the job')
    should_fail: bool = Option(default=False, desc='Raise instead')
    tags: list[str] = Option(default_factory=list, desc='Returned by the job')


class ReturnsValue(Job[ValueConfig]):
    async def _execute(self):
        if self.config.should_fail:
            raise ValueError('Job execution failed')
        return self.config.value


class ReturnsTags(Job[ValueConfig]):
    async def _execute(self):
        self.config.tags.append('touched')  # must not reach the spec's defaults
        return self.config.tags


class Watches(Job[JobConfig]):
    """Returns whether it saw itself running."""

    async def _execute(self):
        return self.is_running, self.is_complete


class Waits(Job[JobConfig]):
    """Runs until it's cancelled."""

    async def _execute(self):
        await asyncio.sleep(30)


class Raises(Job[JobConfig]):
    error: BaseException = SubprocessError('the program failed', command=('prog', '--flag'))

    async def _execute(self):
        raise self.error


class KeptTalker(Talker):
    _keep_files = True


class Limited(Talker):
    _resources = Resources(jobs=2)


def batch_config(tmp_path: Path, *lines: str, **kwargs) -> BatchJobConfig:
    return BatchJobConfig(job_directory=tmp_path / 'job', templates={SYSTEM_SHELL: '\n'.join(lines)}, **kwargs)


class TwoSteps(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(name='one', job_cls=ReturnsValue, config_cls=ValueConfig),
        StepSpec(name='two', job_cls=ReturnsValue, config_cls=ValueConfig),
    )


class Configured(SteppedJob[SteppedJobConfig]):
    """Defaults, dynamic defaults from the first step's result, overrides, a condition and a post-processor."""

    _steps = (
        StepSpec(name='first', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 1}),
        StepSpec(
            name='second',
            job_cls=ReturnsValue,
            config_cls=ValueConfig,
            defaults={'value': 2},
            dynamic_defaults=lambda ctx: {'value': ctx.results['first'].data * 10},
            post_processor=lambda results, ctx: {'second_doubled': results.data * 2},
        ),
        StepSpec(
            name='skipped',
            job_cls=Failing,
            config_cls=JobConfig,
            condition=lambda ctx: ctx.data.get('second_doubled') == 0,
        ),
    )


def run_stepped(job_cls, steps=None, job_directory=None):
    job = job_cls(job_id=job_cls.__name__)
    job.configure(SteppedJobConfig(steps=steps or {}, job_directory=job_directory))
    return job.run()


def _raises(*args):
    raise RuntimeError('hook failed')


class TestJobData:
    def test_job_cls(self):
        data = JobData(job_cls='tests.jobs.helpers.Talker', job_id='a', config=None)
        assert data.get_job_cls() is Talker
        assert data.log_handlers == {}
        assert data.pool_label is None

    @pytest.mark.parametrize('path', ['tests.jobs.helpers.Nope', 'no.such.module.Job', 'Talker'])
    def test_job_cls_that_cannot_be_imported(self, path):
        with pytest.raises(OptionsValidationError, match='Invalid `job_cls`'):
            JobData(job_cls=path, job_id='a', config=None)

    @pytest.mark.parametrize('path', ['tests.jobs.helpers.POOLS', 'xtl.jobs.config.JobConfig'])
    def test_job_cls_that_is_not_a_job(self, path):
        with pytest.raises(TypeError, match='must be a subclass of Job'):
            JobData(job_cls=path, job_id='a', config=None)


class TestJob:
    def test_init(self):
        job = ReturnsValue(job_id='a1')
        assert job.job_id == 'a1'
        assert job.config is None
        assert job.pool is None
        assert (job.is_running, job.is_complete) == (False, False)
        assert job.logger.name == 'a1'
        assert job.logger_config.level == logging.INFO

    def test_random_job_id(self, monkeypatch):
        monkeypatch.setattr(settings.jobs, 'job_digits', 6)
        first, second = ReturnsValue(), ReturnsValue()
        assert len(first.job_id) == 6
        assert first.job_id != second.job_id

    def test_job_id_is_used_as_given(self):
        # Two jobs given the same id keep it, and share its logger
        first, second = ReturnsValue(job_id='a2'), ReturnsValue(job_id='a2')
        assert first.job_id == second.job_id == 'a2'
        assert first.logger is second.logger
        assert ReturnsValue(job_id=3).job_id == '3'

    def test_config_class(self):
        class Plain(Job):
            async def _execute(self): ...

        class Inherits(ReturnsValue):
            pass

        assert ReturnsValue._config_class is ValueConfig
        assert Plain._config_class is JobConfig
        assert Inherits._config_class is ValueConfig
        assert BatchJob._config_class is BatchJobConfig
        assert SteppedJob._config_class is SteppedJobConfig

    def test_config_class_must_be_a_job_config(self):
        with pytest.raises(TypeError, match='must be a subclass of JobConfig, got int'):

            class Wrong(Job[int]):
                async def _execute(self): ...

    def test_configure(self):
        job = ReturnsValue(job_id='a3')
        config = ValueConfig(value=1)
        job.configure(config)
        assert job.config is config
        with pytest.raises(TypeError, match='Expected config of type ValueConfig, got JobConfig'):
            job.configure(JobConfig())

    def test_with_config(self):
        config = ValueConfig()
        logger = logging.getLogger('a4_logger')
        job = ReturnsValue.with_config(config, job_id='a4', logger=logger)
        assert (job.job_id, job.config, job.logger) == ('a4', config, logger)
        assert ReturnsValue.with_config().config is None
        with pytest.raises(TypeError, match='Expected config of type ValueConfig'):
            ReturnsValue.with_config(JobConfig())

    def test_pool(self):
        job, pool = ReturnsValue(job_id='a5'), SimplePool()
        job.pool = pool
        assert job.pool is pool
        job.pool = None
        assert job.pool is None
        with pytest.raises(TypeError, match='`pool` must implement PoolProtocol, got str'):
            job.pool = 'pool'

    def test_get_resources(self):
        assert Talker(job_id='a6').get_resources() is None
        assert Limited(job_id='a6').get_resources() == Resources(jobs=2)
        job = Limited.with_config(JobConfig(resources=Resources(jobs=3)), job_id='a6')
        assert job.get_resources() == Resources(jobs=3)
        assert Limited.with_config(JobConfig(), job_id='a6').get_resources() == Resources(jobs=2)

    @pytest.mark.asyncio
    async def test_run(self):
        result = await ReturnsValue.with_config(ValueConfig(value=123), job_id='a7').run()
        assert isinstance(result, JobResults)
        assert (result.job_id, result.data, result.error, result.success) == ('a7', 123, None, True)

    @pytest.mark.asyncio
    async def test_run_failing(self):
        job = ReturnsValue.with_config(ValueConfig(should_fail=True), job_id='a8')
        records = recorded(job)
        result = await job.run()
        assert not result.success
        assert isinstance(result.error, ValueError)
        assert result.data is None
        assert 'Job failed due to an exception: Job execution failed' in records.messages(logging.ERROR)

    @pytest.mark.asyncio
    async def test_run_failing_program(self):
        job = Raises(job_id='a9')
        records = recorded(job)
        result = await job.run()
        assert result.error is Raises.error
        assert records.messages(logging.ERROR) == [
            "Job failed with subprocess error: the program failed, cmd_args: ('prog', '--flag')"
        ]

    @pytest.mark.parametrize('tracebacks', [True, False])
    @pytest.mark.asyncio
    async def test_run_failing_traceback(self, monkeypatch, tracebacks):
        monkeypatch.setattr(settings.jobs, 'tracebacks', tracebacks)
        job = Failing(job_id='a10')
        records = recorded(job)
        await job.run()
        [error] = [r for r in records.records if r.levelno == logging.ERROR]
        assert bool(error.exc_info) is tracebacks

    @pytest.mark.asyncio
    async def test_run_state(self):
        job = Watches(job_id='a11')
        result = await job.run()
        assert result.data == (True, False)
        assert (job.is_running, job.is_complete) == (False, True)

    @pytest.mark.asyncio
    async def test_run_only_once(self):
        job = Talker(job_id='a12')
        records = recorded(job)
        assert (await job.run()).success
        assert await job.run() is None
        assert 'Job has already completed' in records.messages(logging.WARNING)

    @pytest.mark.asyncio
    async def test_run_while_running(self):
        job = Waits(job_id='a13')
        records = recorded(job)
        task = asyncio.create_task(job.run())
        while not job.is_running:
            await asyncio.sleep(0.01)
        assert await job.run() is None
        assert 'Job is already running' in records.messages(logging.WARNING)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.parametrize('reason, message', [(None, 'Job was cancelled'), ('stop', 'with reason: stop')])
    @pytest.mark.asyncio
    async def test_run_cancelled(self, tmp_path, reason, message):
        job = Waits.with_config(JobConfig(job_directory=tmp_path), job_id='a14')
        records = recorded(job)
        task = asyncio.create_task(job.run())
        while not job.is_running:
            await asyncio.sleep(0.01)
        task.cancel(reason) if reason else task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert any(message in m for m in records.messages(logging.WARNING))
        assert (job.is_running, job.is_complete) == (False, False)
        # A cancelled job leaves its directory
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_tidy_up_removes_the_directory_on_success(self, tmp_path):
        directory = tmp_path / 'job'
        directory.mkdir()
        assert (await Talker.with_config(JobConfig(job_directory=directory), job_id='a15').run()).success
        assert not directory.exists()

    @pytest.mark.asyncio
    async def test_tidy_up_keeps_the_directory_on_failure(self, tmp_path):
        assert not (await Failing.with_config(JobConfig(job_directory=tmp_path), job_id='a16').run()).success
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_tidy_up_without_a_directory(self, tmp_path):
        assert (await Talker.with_config(JobConfig(job_directory=tmp_path / 'never'), job_id='a17').run()).success
        assert (await Talker(job_id='a17b').run()).success

    @pytest.mark.parametrize('kept_by', ['class', 'settings'])
    @pytest.mark.asyncio
    async def test_tidy_up_keeps_the_directory(self, tmp_path, monkeypatch, kept_by):
        monkeypatch.setattr(jobs_module, 'OS_POSIX', False)
        if kept_by == 'settings':
            monkeypatch.setattr(settings.jobs, 'keep_temp', True)
        job_cls = KeptTalker if kept_by == 'class' else Talker
        assert (await job_cls.with_config(JobConfig(job_directory=tmp_path), job_id='a18').run()).success
        assert tmp_path.exists()

    @pytest.mark.asyncio
    async def test_tidy_up_warns_about_what_it_cannot_remove(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jobs_module.shutil, 'rmtree', lambda path, ignore_errors=False: None)
        job = Talker.with_config(JobConfig(job_directory=tmp_path), job_id='a19')
        records = recorded(job)
        assert (await job.run()).success
        assert f'Job directory {tmp_path} could not be removed completely' in records.messages(logging.WARNING)
        assert 'Job directory cleaned up successfully' not in records.messages()

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_tidy_up_with_an_open_file(self, tmp_path):
        job = Talker.with_config(JobConfig(job_directory=tmp_path), job_id='a20')
        records = recorded(job)
        with open(tmp_path / 'held.txt', 'w'):
            assert (await job.run()).success
        assert any('could not be removed completely' in m for m in records.messages(logging.WARNING))

    @pytest.mark.asyncio
    async def test_tidy_up_updates_permissions(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(jobs_module, 'OS_POSIX', True)
        monkeypatch.setattr(jobs_module, 'chmod_recursively', lambda path, **kwargs: calls.append((path, kwargs)))
        monkeypatch.setattr(settings.jobs.permissions, 'update', True)
        assert (await KeptTalker.with_config(JobConfig(job_directory=tmp_path), job_id='a21').run()).success
        assert calls == [
            (
                tmp_path,
                {
                    'files_permissions': settings.jobs.permissions.files,
                    'directories_permissions': settings.jobs.permissions.directories,
                },
            )
        ]

        calls.clear()
        monkeypatch.setattr(settings.jobs.permissions, 'update', False)
        assert (await KeptTalker.with_config(JobConfig(job_directory=tmp_path), job_id='a22').run()).success
        assert calls == []

    @pytest.mark.asyncio
    async def test_tidy_up_permissions_that_cannot_be_updated(self, tmp_path, monkeypatch):
        def fails(path, **kwargs):
            raise PermissionError('not allowed')

        monkeypatch.setattr(jobs_module, 'OS_POSIX', True)
        monkeypatch.setattr(jobs_module, 'chmod_recursively', fails)
        monkeypatch.setattr(settings.jobs.permissions, 'update', True)
        job = KeptTalker.with_config(JobConfig(job_directory=tmp_path), job_id='a23')
        records = recorded(job)
        assert (await job.run()).success
        assert any('Failed to update permissions' in m for m in records.messages(logging.ERROR))

    @pytest.mark.asyncio
    async def test_lock(self):
        job = Talker(job_id='a24')
        async with job.lock():
            pass
        async with SimplePool() as pool:
            job.pool = pool
            lock = job.lock('named')
            assert lock is pool.get_lock('named')

    @pytest.mark.parametrize(
        'pool_type, dimension, max_jobs',
        [('simple', 'jobs', 1), ('async', 'jobs', 2), ('threads', 'threads', 2), ('processes', 'processes', 2)],
    )
    @pytest.mark.asyncio
    async def test_get_pool(self, pool_type, dimension, max_jobs):
        job = Talker(job_id='a25')
        records = recorded(job)
        async with job.get_pool(pool_type, max_jobs=1_000, resources=Resources(jobs=2, threads=2, processes=2)) as pool:
            # Capped to the resources (a simple pool runs one job at a time anyway)
            assert pool.max_jobs == max_jobs
            assert pool.logger.parent is job.logger
        [warning] = records.messages(logging.WARNING)
        assert f'exceeds granted {dimension}=2' in warning

    def test_get_logger(self):
        logger = ReturnsValue.get_logger(42)
        assert logger.name == '42'
        assert len(logger.handlers) == 1
        with pytest.raises(TypeError, match='Expected a LoggerConfig instance, got dict'):
            ReturnsValue.get_logger('a26', config={'level': 10})

    def test_serialize(self):
        job = ReturnsValue.with_config(ValueConfig(value=5), job_id='a27')
        data = job.serialize(as_dict=False)
        assert isinstance(data, JobData)
        assert (data.job_cls, data.job_id) == ('tests.jobs.test_jobs.ReturnsValue', 'a27')
        assert data.config['value'] == 5
        assert job.serialize() == data.to_dict()
        assert ReturnsValue(job_id='a27b').serialize()['config'] is None

    @pytest.mark.parametrize('as_dict', [True, False])
    def test_deserialize(self, as_dict):
        job = ReturnsValue.with_config(ValueConfig(value=5), job_id='a28')
        job._log_handlers = {'file': JobFileHandlerConfig()}
        job._pool_label = 'SimplePool'
        restored = Job.deserialize(job.serialize(as_dict=as_dict))
        assert type(restored) is ReturnsValue
        assert (restored.job_id, restored.config.value) == ('a28', 5)
        assert isinstance(restored._log_handlers['file'], JobFileHandlerConfig)
        assert restored._pool_label == 'SimplePool'
        assert restored.logger is job.logger

    def test_deserialize_logger_and_no_config(self):
        logger = logging.getLogger('a29_worker')
        restored = Job.deserialize(ReturnsValue(job_id='a29').serialize(), logger=logger)
        assert restored.logger is logger
        assert restored.config is None

    def test_deserialize_something_else(self):
        with pytest.raises(TypeError, match='Expected a JobData instance or dict, got str'):
            Job.deserialize('job')

    @pytest.mark.asyncio
    async def test_create_child(self):
        parent = ReturnsValue(job_id='a30')
        child = parent._create_child(ReturnsValue, ValueConfig(value=1), suffix='x')
        assert (child.job_id, child.logger.name, child.config.value) == ('a30.x', 'a30.x', 1)
        assert child.logger_config is parent.logger_config
        assert child.pool is None
        override = LoggerConfig(handlers=[])
        assert (
            parent._create_child(ReturnsValue, ValueConfig(), suffix=2, logger_config=override).logger_config
            is override
        )

        async with SimplePool() as pool:
            parent.pool = pool
            assert parent._create_child(ReturnsValue, ValueConfig(), suffix=3).pool is pool
            assert parent._create_child(ReturnsValue, ValueConfig(), suffix=4, inherit_pool=False).pool is None


class TestBatchJob:
    def test_with_config(self):
        # Only the arguments that aren't the constructor's are template context
        logger = logging.getLogger('b1')
        job = BatchJob.with_config(BatchJobConfig(), job_id='b1', logger=logger, batch_args=['a'], x=1)
        assert (job.job_id, job.logger) == ('b1', logger)
        assert job._batch_args == ['a']
        assert job._batch_context == {'x': 1}

    def test_context(self, tmp_path):
        config = batch_config(tmp_path, 'echo __WHO__')
        job = BatchJob.with_config(config, job_id='b2', who='me', nested={'a': 1})
        assert job.context == {'who': 'me', 'nested': {'a': 1}}
        # A copy: changing it leaves the job's alone
        job.context['nested']['a'] = 2
        assert job.context['nested'] == {'a': 1}
        with pytest.raises(ValueError, match='Job is not configured'):
            BatchJob(job_id='b2b').context

    @pytest.mark.asyncio
    async def test_run(self, tmp_path):
        job = BatchJob.with_config(batch_config(tmp_path, 'echo hello'), job_id='b3')
        result = await job.run()
        assert result.success, result.error
        assert isinstance(result.data, BatchResults)
        assert result.data.stdout.split() == ['hello']
        assert result.data.return_code == 0
        assert not (tmp_path / 'job').exists()

    @pytest.mark.asyncio
    async def test_run_with_arguments_and_context(self, tmp_path):
        line = 'echo [%~1] __WHO__' if OS_WINDOWS else 'echo "[$1]" __WHO__'
        job = BatchJob.with_config(batch_config(tmp_path, line), job_id='b4', batch_args=['a b'], WHO='me')
        result = await job.run()
        assert result.data.stdout.strip() == '[a b] me'

    @pytest.mark.asyncio
    async def test_run_creates_the_directory(self, tmp_path):
        config = BatchJobConfig(job_directory=tmp_path / 'a' / 'b', templates={SYSTEM_SHELL: EXIT_3})
        await BatchJob.with_config(config, job_id='b5').run()
        assert config.file.exists()
        assert config.stdout.exists() and config.stderr.exists()

    @pytest.mark.asyncio
    async def test_run_failing_program(self, tmp_path):
        job = BatchJob.with_config(batch_config(tmp_path, EXIT_3), job_id='b6')
        result = await job.run()
        assert isinstance(result.error, SubprocessError)
        assert 'failed with return code 3' in str(result.error)
        assert (tmp_path / 'job').exists()

    @pytest.mark.asyncio
    async def test_run_unconfigured(self):
        result = await BatchJob(job_id='b7').run()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == 'Job is not configured.'

    @pytest.mark.asyncio
    async def test_run_without_a_directory(self, tmp_path):
        (tmp_path / 'file').write_text('')
        job = BatchJob.with_config(BatchJobConfig(job_directory=tmp_path / 'file' / 'job'), job_id='b8')
        records = recorded(job)
        result = await job.run()
        assert isinstance(result.error, OSError)
        assert any('Failed to create batch directory' in m for m in records.messages(logging.ERROR))

    @pytest.mark.asyncio
    async def test_run_without_a_batch_file(self, tmp_path, monkeypatch):
        def fails(self, **kwargs):
            raise PermissionError('read-only')

        monkeypatch.setattr(BatchFile, 'save', fails)
        job = BatchJob.with_config(batch_config(tmp_path, 'echo'), job_id='b9')
        records = recorded(job)
        result = await job.run()
        assert isinstance(result.error, PermissionError)
        assert any('Failed to save batch file' in m for m in records.messages(logging.ERROR))

    @pytest.mark.asyncio
    async def test_run_without_log_files(self, tmp_path, monkeypatch):
        def fails(self, exist_ok=True):
            raise PermissionError('read-only')

        job = BatchJob.with_config(batch_config(tmp_path, 'echo'), job_id='b10')
        records = recorded(job)
        monkeypatch.setattr(Path, 'touch', fails)
        result = await job.run()
        assert isinstance(result.error, PermissionError)
        assert any('Failed to create log files' in m for m in records.messages(logging.ERROR))

    @pytest.mark.asyncio
    async def test_run_cancelled(self, tmp_path):
        job = BatchJob.with_config(batch_config(tmp_path, SLEEP_5), job_id='b11')
        records = recorded(job)
        task = asyncio.create_task(job.run())
        while job._batch is None or job._batch.status != BatchFileStatus.RUNNING:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        assert job._batch.status == BatchFileStatus.CANCELLED
        assert job._batch.process.returncode is not None
        # The site stops the batch; nothing tries to cancel it again
        assert not any(m.startswith('Cannot cancel batch file') for m in records.messages())


class TestSteppedJob:
    @pytest.mark.parametrize(
        'steps, error, match',
        [
            ({'three': {}}, ValueError, "Unknown step 'three' in `steps`; the steps of TwoSteps are: one, two"),
            ({'one': {'shuold_fail': True}}, ValueError, "Unknown options for step 'one'.*shuold_fail"),
            ({'one': [True]}, TypeError, "override of step 'one' must be a dict"),
        ],
    )
    def test_configure_checks_step_overrides(self, steps, error, match):
        with pytest.raises(error, match=match):
            TwoSteps.with_config(SteppedJobConfig(steps=steps))

    def test_configure_without_steps(self):
        class NoSteps(SteppedJob[SteppedJobConfig]):
            pass

        with pytest.raises(ValueError, match='the steps of NoSteps are: none'):
            NoSteps.with_config(SteppedJobConfig(steps={'one': {}}))

    def test_get_step_config(self):
        job = TwoSteps.with_config(SteppedJobConfig(steps={'two': {'should_fail': True}}))
        assert job._get_step_config('two', ctx=None).should_fail is True
        with pytest.raises(ValueError, match='Unknown job step: three'):
            job._get_step_config('three', ctx=None)
        with pytest.raises(ValueError, match='Job is not configured'):
            TwoSteps()._get_step_config('one', ctx=None)

    def test_duplicate_step_names(self):
        with pytest.raises(ValueError, match='Duplicate step names in Duplicate._steps: same'):

            class Duplicate(SteppedJob[SteppedJobConfig]):
                _steps = (
                    StepSpec(name='same', job_cls=ReturnsValue, config_cls=ValueConfig),
                    StepSpec(name='same', job_cls=ReturnsValue, config_cls=ValueConfig),
                )

    @pytest.mark.asyncio
    async def test_run(self):
        result = await run_stepped(Configured)
        assert isinstance(result, SteppedJobResults)
        assert result.success
        assert {name: r.data for name, r in result.steps.items()} == {'first': 1, 'second': 10}
        assert result.data == {'second_doubled': 20}

    @pytest.mark.asyncio
    async def test_run_with_overrides(self):
        # Overrides win over both kinds of defaults
        result = await run_stepped(Configured, steps={'first': {'value': 3}, 'second': {'value': 7}})
        assert {name: r.data for name, r in result.steps.items()} == {'first': 3, 'second': 7}

    @pytest.mark.asyncio
    async def test_run_skips_a_step(self):
        result = await run_stepped(Configured)
        assert 'skipped' not in result.steps

    @pytest.mark.asyncio
    async def test_run_copies_the_defaults(self):
        class Twice(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='a', job_cls=ReturnsTags, config_cls=ValueConfig, defaults={'tags': ['x']}),
                StepSpec(name='b', job_cls=ReturnsTags, config_cls=ValueConfig, defaults={'tags': ['x']}),
            )

        result = await run_stepped(Twice)
        assert [r.data for r in result.steps.values()] == [['x', 'touched'], ['x', 'touched']]
        assert Twice._steps[0].defaults == {'tags': ['x']}

    @pytest.mark.asyncio
    async def test_run_stops_at_a_failing_step(self):
        class Stops(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='ok', job_cls=ReturnsValue, config_cls=ValueConfig),
                StepSpec(name='fails', job_cls=Failing, config_cls=JobConfig),
                StepSpec(name='never', job_cls=ReturnsValue, config_cls=ValueConfig),
            )

        result = await run_stepped(Stops)
        assert not result.success
        assert list(result.steps) == ['ok', 'fails']
        assert isinstance(result.error, ValueError) and result.error is result.steps['fails'].error

    @pytest.mark.asyncio
    async def test_run_step_directories(self, tmp_path):
        class Keeps(SteppedJob[SteppedJobConfig]):
            _steps = (StepSpec(name='only', job_cls=ReturnsValue, config_cls=ValueConfig),)

        job = Keeps(job_id='k')
        job.configure(SteppedJobConfig(job_directory=tmp_path))
        directories = []
        original = job._create_child

        def spy(job_cls, config, suffix, **kwargs):
            directories.append(config.job_directory)
            return original(job_cls=job_cls, config=config, suffix=suffix, **kwargs)

        job._create_child = spy
        await job.run()
        assert directories == [tmp_path / '1_only']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('hook', ['condition', 'dynamic_defaults', 'post_processor'])
    async def test_run_keeps_earlier_steps_when_a_hook_raises(self, hook):
        class Hooked(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='done', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 5}),
                StepSpec(name='hooked', job_cls=ReturnsValue, config_cls=ValueConfig, **{hook: _raises}),
            )

        result = await run_stepped(Hooked)
        assert not result.success
        assert isinstance(result.error, RuntimeError)
        assert result.steps['done'].data == 5

    @pytest.mark.asyncio
    async def test_run_keeps_earlier_steps_with_a_bad_option_value(self):
        class Later(SteppedJob[SteppedJobConfig]):
            _steps = (
                StepSpec(name='done', job_cls=ReturnsValue, config_cls=ValueConfig, defaults={'value': 5}),
                StepSpec(name='bad', job_cls=ReturnsValue, config_cls=ValueConfig),
            )

        # Option names are checked when the job is configured, values when the step's configuration is built
        result = await run_stepped(Later, steps={'bad': {'value': 'not a number'}})
        assert not result.success
        assert result.steps['done'].data == 5

    @pytest.mark.asyncio
    async def test_run_only_once(self):
        job = TwoSteps.with_config(SteppedJobConfig(), job_id='s1')
        assert (await job.run()).success
        assert await job.run() is None

    @pytest.mark.asyncio
    async def test_run_unconfigured(self):
        result = await TwoSteps(job_id='s2').run()
        assert isinstance(result.error, ValueError)
        assert result.steps == {}

    @pytest.mark.asyncio
    async def test_run_checks_what_execute_returns(self):
        class Custom(SteppedJob[SteppedJobConfig]):
            async def _execute(self):
                return {'not': 'a tuple'}

        with pytest.raises(TypeError, match=r'Custom._execute\(\) must return a 2-tuple'):
            await Custom.with_config(SteppedJobConfig(), job_id='s3').run()

    @pytest.mark.asyncio
    async def test_run_without_a_pool_logs_no_warnings(self):
        job = TwoSteps.with_config(SteppedJobConfig(), job_id='s4')
        records = recorded(job)
        assert (await job.run()).success
        assert records.messages(logging.WARNING) == []
