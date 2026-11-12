"""
Tests for the logging of jobs and pools.

The tests assert the *intended* behavior. The ones that fail today are marked
``xfail(strict=True)`` with the bug as the reason; the fix removes the marker.

Records are captured with :class:`Capture`, an in-process handler, instead of caplog,
because job and pool loggers don't propagate to the root logger. Jobs are defined at
module level, so that worker processes can import them.
"""

import ast
import asyncio
from collections import defaultdict
import io
import logging
from pathlib import Path
import subprocess
import sys

import pytest

import xtl
import xtl_core
from xtl.jobs.batchfiles import BatchFile
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.pools import AsyncPool, BasePool, MultiprocessPool, SimplePool, ThreadedPool
from xtl.jobs.sites import LocalSite
from xtl.jobs.steps import StepSpec
from xtl.jobs.submissions import JobSubmission
from xtl.logging.config import FileHandlerConfig, HandlerConfig, LoggerConfig, StreamHandlerConfig
from xtl.tui.console import ConsoleIO
from xtl.tui.live_pool import LivePool
from xtl.tui.logging.handlers import BufferingHandler, LoggerHandlerPatcher

RECORDS: dict[str, list[logging.LogRecord]] = defaultdict(list)
"""Records captured by :class:`Capture`, per key. Only filled in the process that emits them."""

POOLS = [SimplePool, AsyncPool, ThreadedPool, MultiprocessPool]
WORKER_POOLS = [ThreadedPool, MultiprocessPool]


class Capture(logging.Handler):
    """Collects records in :data:`RECORDS`; a message that can't be formatted is stored as an error."""

    def __init__(self, key: str = 'default', level: int = logging.NOTSET):
        super().__init__(level)
        self.key = key

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record.captured = record.getMessage()
        except Exception as e:
            record.captured = f'<format error: {type(e).__name__}: {e}>'
        RECORDS[self.key].append(record)


def capture_config(key: str, level: int = logging.DEBUG) -> LoggerConfig:
    # `options` is the one part of a handler config that survives a dict round trip today
    return LoggerConfig(level=level, propagate=False, handlers=[HandlerConfig(handler=Capture, options={'key': key})])


def messages(key: str) -> list[str]:
    return [r.captured for r in RECORDS[key]]


def names(key: str) -> set[str]:
    return {r.name for r in RECORDS[key]}


class Talker(Job[JobConfig]):
    async def _execute(self):
        self.logger.debug('debug from %s', self.job_id)
        self.logger.info('info from %s', self.job_id)
        self.logger.warning('warning from %s', self.job_id)
        return self.job_id


class Failing(Job[JobConfig]):
    async def _execute(self):
        raise ValueError('boom')


class Stepped(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(name='one', job_cls=Talker, config_cls=JobConfig),
        StepSpec(name='two', job_cls=Talker, config_cls=JobConfig),
    )


class SteppedRaising(SteppedJob[SteppedJobConfig]):
    async def _execute(self):
        raise ValueError('stepped boom')


def _subclasses(cls: type) -> list[type]:
    found = [cls]
    for sub in cls.__subclasses__():
        found.extend(_subclasses(sub))
    return found


@pytest.fixture(autouse=True)
def fresh_logging():
    """
    Undo what jobs, pools and the TUI leak into `logging` and into class attributes: the
    class-level logger configs, `logging.getLogger`, the root logger, and every logger
    created during the test.
    """
    missing = object()
    classes = _subclasses(Job) + _subclasses(BasePool)
    configs = {cls: cls.__dict__.get('_logger_config', missing) for cls in classes}
    get_logger = logging.getLogger
    manager = logging.root.manager
    existing = set(manager.loggerDict)
    # The TUI swaps the handlers of every logger and doesn't put back `propagate`
    loggers = [logging.root] + [lg for lg in manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    states = {lg: (list(lg.handlers), lg.propagate, lg.level) for lg in loggers}
    RECORDS.clear()

    yield

    logging.getLogger = get_logger
    for cls, config in configs.items():
        if config is missing:
            if '_logger_config' in cls.__dict__:
                delattr(cls, '_logger_config')
        else:
            cls._logger_config = config
    for lg, (handlers, propagate, level) in states.items():
        lg.handlers[:] = handlers
        lg.propagate = propagate
        lg.setLevel(level)
    for name in set(manager.loggerDict) - existing:
        logger = manager.loggerDict.pop(name)
        for handler in getattr(logger, 'handlers', []):
            handler.close()
    manager._clear_cache()
    RECORDS.clear()


async def run_pool(pool: BasePool, job_cls: type[Job], job_ids: list[str], config: JobConfig | None = None):
    async with pool:
        jobs = pool.submit(job_cls, configs=[config for _ in job_ids], job_ids=job_ids)
        results = await pool.launch()
    return jobs, results


def default_config(job_cls: type[Job]) -> JobConfig:
    return SteppedJobConfig() if issubclass(job_cls, SteppedJob) else JobConfig()


class TestStandaloneJob:
    def test_default_logger(self):
        logger = Talker(job_id='a1').logger
        assert logger.name == 'a1'
        assert logger.propagate is False
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert getattr(logger.handlers[0], '_xtl_owned', False)

    @pytest.mark.asyncio
    async def test_levels(self):
        job = Talker(job_id='a2', logger=Talker.get_logger('a2', capture_config('a2', level=logging.INFO)))
        job.configure(JobConfig())
        await job.run()
        assert messages('a2') == ['info from a2', 'warning from a2']  # no debug, no lifecycle messages

    @pytest.mark.asyncio
    async def test_failing_job_logs_the_error(self):
        job = Failing(job_id='a3', logger=Failing.get_logger('a3', capture_config('a3')))
        job.configure(JobConfig())
        result = await job.run()
        assert not result.success
        assert any(m.startswith('Job failed due to an exception: boom') for m in messages('a3'))

    def test_reconfigure_keeps_foreign_handlers(self):
        logger = logging.getLogger('a4')
        foreign = Capture('a4_foreign')
        logger.addHandler(foreign)
        Talker.get_logger('a4', capture_config('a4'))
        Talker.get_logger('a4', capture_config('a4'))
        assert foreign in logger.handlers
        assert len([h for h in logger.handlers if getattr(h, '_xtl_owned', False)]) == 1

    def test_reconfigure_twice_with_two_handlers(self):
        two = LoggerConfig(handlers=[HandlerConfig(handler=Capture), HandlerConfig(handler=Capture)])
        logger = Talker.get_logger('a4b', two)
        Talker.get_logger('a4b', two)
        assert len(logger.handlers) == 2

    def test_configured_logger_is_left_alone(self):
        logger = Talker.get_logger('a5', capture_config('a5'))
        handlers = list(logger.handlers)
        assert Talker.get_logger('a5').handlers == handlers

    def test_logger_passed_in_is_used(self):
        custom = logging.getLogger('a5_custom')
        assert Talker(job_id='a5b', logger=custom).logger is custom


class TestChildJobs:
    @pytest.mark.asyncio
    async def test_steps_reach_the_parent_handlers(self):
        job = Stepped(job_id='b6', logger=Stepped.get_logger('b6', capture_config('b6')))
        job.configure(SteppedJobConfig())
        result = await job.run()
        assert result.success
        assert 'info from b6.1' in messages('b6')
        assert 'info from b6.2' in messages('b6')
        assert {'b6', 'b6.1', 'b6.2'} <= names('b6')

    @pytest.mark.asyncio
    async def test_stepped_job_error_is_returned(self):
        job = SteppedRaising(job_id='b7', logger=SteppedRaising.get_logger('b7', capture_config('b7')))
        job.configure(SteppedJobConfig())
        result = await job.run()
        assert isinstance(result.error, ValueError)

    @pytest.mark.xfail(strict=True, reason="sites log to their module logger, not the job's")
    @pytest.mark.asyncio
    async def test_site_messages_reach_the_job(self, tmp_path, monkeypatch):
        async def cannot_start(*args, **kwargs):
            raise OSError('cannot start')

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', cannot_start)
        job = BatchJob.with_config(
            BatchJobConfig(job_directory=tmp_path / 'b8'),
            job_id='b8',
            logger=BatchJob.get_logger('b8', capture_config('b8')),
        )
        result = await job.run()
        assert not result.success
        assert any(m.startswith('Error executing batch file') for m in messages('b8'))


class TestPools:
    @pytest.mark.asyncio
    async def test_pool_logger(self):
        pool = SimplePool(name='c9', logger_config=capture_config('c9'))
        await run_pool(pool, Talker, ['c9_job'], JobConfig())
        assert pool.logger.name == 'SimplePool|c9'
        assert pool.logger.propagate is False
        assert 'SimplePool|c9' in names('c9')

    @pytest.mark.asyncio
    async def test_job_config_defaults_to_the_pool_config(self):
        pool = SimplePool(name='c10', logger_config=capture_config('c10'))
        await run_pool(pool, Talker, ['c10_job'], JobConfig())
        assert 'info from c10_job' in messages('c10')

    @pytest.mark.parametrize(
        'pool_cls',
        [
            SimplePool,
            AsyncPool,
            ThreadedPool,
            pytest.param(
                MultiprocessPool,
                marks=pytest.mark.xfail(strict=True, reason='records stay in the worker process'),
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_job_records_reach_the_pool(self, pool_cls):
        key = f'c11_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, job_logger_config=capture_config(key))
        ids = [f'{key}_1', f'{key}_2']
        _, results = await run_pool(pool, Talker, ids, JobConfig())
        assert all(r.success for r in results)
        for job_id in ids:
            assert f'info from {job_id}' in messages(key)

    @pytest.mark.parametrize(
        'pool_cls',
        [
            SimplePool,
            AsyncPool,
            pytest.param(
                ThreadedPool,
                marks=pytest.mark.xfail(strict=True, reason='workers lose the stream, log to stderr'),
            ),
            pytest.param(
                MultiprocessPool,
                marks=pytest.mark.xfail(strict=True, reason='records stay in the worker process'),
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_stream_handler_in_pool(self, pool_cls):
        stream = io.StringIO()
        handler = StreamHandlerConfig(stream=stream)
        config = LoggerConfig(level=logging.INFO, handlers=[handler])
        key = f'c11b_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, job_logger_config=config)
        _, results = await run_pool(pool, Talker, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results)
        assert f'info from {key}_1' in stream.getvalue()

    @pytest.mark.asyncio
    async def test_tui_config_in_process_pool(self):
        # The TUI's lifeline: with no handlers, records still come back to the parent
        config = LoggerConfig(level=logging.INFO, handlers=[])
        pool = MultiprocessPool(name='c12', max_jobs=2, logger_config=config, job_logger_config=config)
        async with pool:
            jobs = pool.submit(Stepped, configs=[SteppedJobConfig()], job_ids=['c12_job'])
            for job in jobs:
                job.logger.addHandler(Capture('c12'))
            results = await pool.launch()
        assert all(r.success for r in results)
        # Under their own names (step 7 puts them under the pool's logger, hence endswith)
        for suffix in ('c12_job', 'c12_job.1', 'c12_job.2'):
            assert any(name.endswith(suffix) for name in names('c12')), names('c12')
        assert 'info from c12_job.1' in messages('c12')
        assert all(r.processName.startswith('Process') for r in RECORDS['c12'])

    @pytest.mark.xfail(strict=True, reason="a pool's config becomes the class default")
    @pytest.mark.asyncio
    async def test_no_config_leaks(self):
        pool = SimplePool(name='c13', logger_config=capture_config('c13'))
        await run_pool(pool, Talker, ['c13_job'], JobConfig())
        later_job = Talker(job_id='c13_later')
        later_pool = SimplePool(name='c13_later')
        for logger in (later_job.logger, later_pool.logger):
            assert not any(isinstance(h, Capture) for h in logger.handlers)

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS)
    @pytest.mark.xfail(strict=True, reason='file handler configs lose their filename in workers')
    @pytest.mark.asyncio
    async def test_file_handler_in_worker_pool(self, pool_cls, tmp_path):
        log = tmp_path / 'jobs.log'
        config = LoggerConfig(level=logging.INFO, handlers=[FileHandlerConfig(filename=str(log))])
        key = f'c14_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, job_logger_config=config)
        _, results = await run_pool(pool, Talker, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results), [r.error for r in results]
        assert f'info from {key}_1' in log.read_text(encoding='utf-8')

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS)
    @pytest.mark.asyncio
    async def test_job_without_config_in_worker_pool(self, pool_cls):
        key = f'c14b_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, job_logger_config=capture_config(key))
        _, results = await run_pool(pool, Talker, [f'{key}_1'], None)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_get_pool_warning_is_formatted(self):
        job = Talker(job_id='c14c', logger=Talker.get_logger('c14c', capture_config('c14c')))
        async with job.get_pool('simple', max_jobs=1_000_000):
            pass
        warnings = [r.captured for r in RECORDS['c14c'] if r.levelno == logging.WARNING]
        assert any('capping to' in m for m in warnings), warnings


class TestSerialization:
    def test_submission_carries_the_level(self):
        # Step 5 moves the level from `JobData.logging_config` to `JobSubmission.log_level`
        job = Talker(job_id='d16', logger=Talker.get_logger('d16', capture_config('d16', level=logging.DEBUG)))
        submission = JobSubmission.from_job(job)
        assert submission.data.logging_config['level'] == logging.DEBUG


class TestTUI:
    def test_patcher_captures_existing_and_new_loggers(self):
        buffer = BufferingHandler()
        patcher = LoggerHandlerPatcher(handler=buffer)
        existing = logging.getLogger('e17')
        patcher.patch(existing)
        logging.getLogger = patcher.patched_getLogger
        try:
            existing.warning('from existing')
            logging.getLogger('e17_new').warning('from new')
        finally:
            patcher.restore()
        assert [r.getMessage() for r in buffer._records] == ['from existing', 'from new']

    def test_restore_puts_back_handlers_and_get_logger(self):
        original_get_logger = logging.getLogger
        logger = logging.getLogger('e18')
        handler = Capture('e18')
        logger.addHandler(handler)
        patcher = LoggerHandlerPatcher(handler=BufferingHandler())
        patcher.patch(logger)
        logging.getLogger = patcher.patched_getLogger
        patcher.restore()
        assert logger.handlers == [handler]
        assert logging.getLogger is original_get_logger

    def test_restore_puts_back_propagate(self):
        logger = logging.getLogger('e18b')
        assert logger.propagate is True
        patcher = LoggerHandlerPatcher(handler=BufferingHandler())
        patcher.patch(logger)
        patcher.restore()
        assert logger.propagate is True

    @pytest.mark.parametrize('pool_type', ['simple', 'processes'])
    @pytest.mark.asyncio
    async def test_live_pool(self, pool_type):
        original_get_logger = logging.getLogger
        bystander = logging.getLogger('e19_bystander')
        handler = Capture('e19_bystander')
        bystander.addHandler(handler)

        console = ConsoleIO(file=io.StringIO(), width=100)
        live = LivePool(console=console, pool_type=pool_type, max_jobs=2)
        key = f'e19_{pool_type}'
        async with live:
            live.submit(Stepped, configs=[SteppedJobConfig()], job_ids=[key])
            results = await live.launch()
            records = list(live._buffer._records)

        assert all(r.success for r in results)
        assert f'info from {key}.1' in [r.getMessage() for r in records]
        assert logging.getLogger is original_get_logger
        assert bystander.handlers == [handler]


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)


class TestModuleLoggers:
    @pytest.mark.xfail(strict=True, reason='xtl calls logging.setLoggerClass()')
    def test_import_keeps_the_logger_class(self):
        result = _run_python('import logging, xtl; assert logging.getLoggerClass() is logging.Logger')
        assert result.returncode == 0, result.stderr

    @pytest.mark.xfail(strict=True, reason='Logger() rejects a plain logger that exists already')
    def test_plain_logger_created_first(self):
        result = _run_python("import logging; logging.getLogger('xtl.jobs.jobs'); import xtl.jobs.jobs")
        assert result.returncode == 0, result.stderr

    def test_import_is_quiet(self):
        result = _run_python('import logging, xtl; assert not logging.root.handlers')
        assert result.returncode == 0, result.stderr
        assert result.stdout == ''
        assert result.stderr == ''


class _Dummy:
    """Formats as anything: %s, %d, %f, %r."""

    def __str__(self):
        return 'x'

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __index__(self):
        return 0


LOGGING_METHODS = {'debug', 'info', 'warning', 'error', 'exception', 'critical'}


def format_problems(path: Path) -> list[str]:
    """Logging calls with a literal message that `LogRecord.getMessage()` couldn't format."""
    problems = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'warnings':
            continue
        if node.func.attr in LOGGING_METHODS:
            args = node.args
        elif node.func.attr == 'log':
            args = node.args[1:]
        else:
            continue
        if len(args) < 2 or not (isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)):
            continue
        msg, rest = args[0].value, args[1:]
        if len(rest) == 1 and isinstance(rest[0], ast.Dict):
            values = {k.value: _Dummy() for k in rest[0].keys if isinstance(k, ast.Constant)}
        elif len(rest) == 1 and isinstance(rest[0], ast.Name):
            continue  # might be a dict or a single value
        else:
            values = tuple(_Dummy() for _ in rest)
        try:
            msg % values
        except Exception as e:
            problems.append(f'{path}:{node.lineno}: {type(e).__name__}: {e}: {msg!r}')
    return problems


class TestFormatsAndTimeouts:
    def test_logging_format_strings(self):
        problems = []
        for package in (xtl, xtl_core):
            for path in sorted(Path(package.__file__).parent.rglob('*.py')):
                problems.extend(format_problems(path))
        assert not problems, '\n'.join(problems)

    @pytest.mark.asyncio
    async def test_cancel_message_is_formatted(self, tmp_path):
        logger = logging.getLogger('xtl.jobs.sites')
        logger.addHandler(Capture('g_cancel'))
        try:
            await LocalSite().cancel_batch(BatchFile(tmp_path / 'idle'))
        finally:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, Capture)]
        assert any(m.startswith('Cannot cancel batch file') for m in messages('g_cancel')), messages('g_cancel')

    def test_execute_command_timeout(self):
        logger = logging.getLogger('xtl.jobs.sites')
        logger.addHandler(Capture('g_timeout'))
        try:
            result = LocalSite.execute_command([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=1)
        finally:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, Capture)]
        assert result is None
        assert any('timed out' in m for m in messages('g_timeout'))
