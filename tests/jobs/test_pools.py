"""
Tests for pools: the pool types, what every pool does (one suite over the four types), the proxy that jobs in workers
get, and what each type adds.
"""

import asyncio
import logging
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time
import typing
from pathlib import Path

import pytest

from xtl import settings
from tests.jobs.helpers import POOLS, WORKER_POOLS, Failing, Naps, Sleeper, Talker, until
from xtl.jobs.config import BatchJobConfig, JobConfig, SiteResources, SteppedJobConfig
from xtl.jobs.exceptions import WorkerError
from xtl.jobs.ipc import AsyncIPCBackend, ThreadedIPCBackend
from xtl.jobs.ipc.base import IPCHandle, IPCHandleNames
from xtl.jobs.ipc.processes import _ShieldedStart
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.logging import HandlerConfig, JobFileHandlerConfig, LoggerConfig
from xtl.jobs.pools import (
    AsyncPool,
    BasePool,
    JobPool,
    MultiprocessPool,
    PoolType,
    ProxyPool,
    SimplePool,
    ThreadedPool,
    _picklable_results,
    _run_watched,
)
from xtl.jobs.resources import Resources, get_rc_manager
from xtl.jobs.results import JobResults, JobStatus, SteppedJobResults
from xtl.jobs.steps import StepSpec
from xtl.jobs.submissions import JobSubmission
from xtl.jobs.shells import Shell
from xtl_core.options import Option


class Records(logging.Handler):
    """Collects the records of a pool's logger."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int = logging.NOTSET) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


def recorded(pool_cls: type[BasePool], **kwargs) -> tuple[BasePool, Records]:
    """A pool whose records, down to DEBUG, are collected."""
    records = Records()
    pool = pool_cls(logger_config=LoggerConfig(level=logging.DEBUG, handlers=[records]), **kwargs)
    return pool, records


class Counter(Job[JobConfig]):
    """Increments a shared counter with a read-modify-write that only a lock makes safe."""

    async def _execute(self):
        state = self.pool.get_state('counter')
        async with self.lock():
            value = state['n']
            await asyncio.sleep(0.001)
            state['n'] = value + 1


class UsesLateLock(Job[JobConfig]):
    async def _execute(self):
        async with self.lock('late'):
            return True


class UsesUnknownLock(Job[JobConfig]):
    async def _execute(self):
        async with self.lock('unknown'):
            return True


class Producer(Job[JobConfig]):
    """Puts its id in the pool's queue `items`, a little after it starts."""

    async def _execute(self):
        await asyncio.sleep(0.05)
        await self.pool.get_queue('items').put(self.job_id)


class ReadsSettings(Job[JobConfig]):
    async def _execute(self):
        return settings.jobs.keep_temp, settings.jobs.permissions.update, settings.dependencies.modules


class WhereAmI(Job[JobConfig]):
    """Returns the process id and the name of the thread it runs in."""

    async def _execute(self):
        return os.getpid(), threading.current_thread().name


class Stubborn(Job[JobConfig]):
    """Ignores the first cancellation."""

    async def _execute(self):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)


class MarksConfig(JobConfig):
    marker: typing.Optional[Path] = Option(default=None, desc='Where the job notes what happened to it')
    seconds: float = Option(default=10, desc='How long the job runs')


class Marks(Job[MarksConfig]):
    """Notes in its marker file that it started, then, if it is cancelled, that it was."""

    async def _execute(self):
        self.config.marker.write_text('started')
        try:
            await asyncio.sleep(self.config.seconds)
        except asyncio.CancelledError:
            self.config.marker.write_text('cancelled')
            raise


class Blocks(Job[MarksConfig]):
    """Notes that it started, then blocks its event loop, so that no cancellation reaches it."""

    async def _execute(self):
        self.config.marker.write_text('started')
        time.sleep(self.config.seconds)


class Dies(Job[JobConfig]):
    """Ends its worker process."""

    async def _execute(self):
        os._exit(3)


class Picky(Exception):
    """Pickles, but can't be unpickled: pickling calls the class again with `args` only."""

    def __init__(self, text: str, code: int):
        self.code = code
        super().__init__(text)


class RaisesPicky(Job[JobConfig]):
    async def _execute(self):
        raise Picky('picky failure', code=7)


class ReturnsPicky(Job[JobConfig]):
    async def _execute(self):
        return Picky('picky data', code=7)


class SteppedPicky(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(name='fine', job_cls=Naps, config_cls=JobConfig),
        StepSpec(name='picky', job_cls=RaisesPicky, config_cls=JobConfig),
    )


ECHO = {Shell.BASH: 'echo "[$1] [$2]"', Shell.CMD: 'echo [%~1] [%~2]'}

pool_types = pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)


def run_ctrl_c_script(tmp_path: Path, script: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """
    Run a script that sends itself Ctrl+C, in a console or a session of its own, so that Ctrl+C reaches only the
    script and its processes; `ctrl_c()` is defined for it. A script that hangs prints the stacks of its threads, and
    is ended with its processes after `timeout` seconds, which fails the test with its output.
    """
    ctrl_c = textwrap.dedent(
        f"""
        import ctypes, faulthandler, os, signal


        def ctrl_c():
            if os.name == 'nt':
                os.kill(0, signal.CTRL_C_EVENT)
            else:
                os.killpg(0, signal.SIGINT)


        if __name__ == '__main__':
            # Ctrl+C may be off for children; not here in the pool's processes, which import this script again, and
            #  ignore Ctrl+C from their start
            if os.name == 'nt':
                ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
            faulthandler.dump_traceback_later({max(timeout - 20, timeout / 2)})
        """
    )
    path = tmp_path / 'script.py'
    path.write_text(ctrl_c + textwrap.dedent(script))
    if os.name == 'nt':
        # A console of its own (Ctrl+C reaches every process of a console), with its window hidden
        startupinfo = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=subprocess.SW_HIDE)
        options = {'creationflags': subprocess.CREATE_NEW_CONSOLE, 'startupinfo': startupinfo}
    else:
        options = {'start_new_session': True}
    root = str(Path(__file__).parents[2])
    cmd = [sys.executable, str(path), *args]
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [root, os.environ.get('PYTHONPATH')]))},
        **options,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Ending only the script leaves its pipes open in the pool's processes, and reading them hangs too
            if os.name == 'nt':
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(process.pid)], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                stdout, stderr = '', '(no output: processes of the script still run)'
            pytest.fail(f'The script hung for {timeout} s\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}')
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


class TestJobPool:
    @pytest.mark.parametrize(
        'member, cls',
        [
            (JobPool.SIMPLE, SimplePool),
            (JobPool.ASYNC, AsyncPool),
            (JobPool.THREADS, ThreadedPool),
            (JobPool.PROCESSES, MultiprocessPool),
        ],
    )
    def test_get(self, member, cls):
        assert member.get() is cls
        assert JobPool(member.value).get() is cls

    def test_unknown_pool_type(self):
        with pytest.raises(ValueError):
            JobPool('cluster')

    def test_pool_type_names_match_the_enum(self):
        # A type checker can't derive the strings from the enum, so they're written twice
        enum_cls, names = typing.get_args(PoolType)
        assert enum_cls is JobPool
        assert set(typing.get_args(names)) == {member.value for member in JobPool}


class TestBasePool:
    # Construction

    def test_init(self, monkeypatch):
        monkeypatch.setattr(settings.jobs, 'job_digits', 6)
        pool = AsyncPool(max_jobs=3)
        assert len(pool.pool_id) == 6
        assert (pool.name, pool.label) == (None, 'AsyncPool')
        assert pool.resources is None
        assert pool.max_jobs == 3
        assert pool.status == {}
        assert AsyncPool(name='fit').name == 'fit'

    def test_max_jobs_must_be_positive(self):
        with pytest.raises(ValueError, match='`max_jobs` must be at least 1, got 0'):
            AsyncPool(max_jobs=0)

    def test_logger_config_type(self):
        with pytest.raises(TypeError, match='Expected a LoggerConfig instance for `logger_config`, got dict'):
            AsyncPool(logger_config={'level': 10})

    def test_parent_logger(self):
        parent = logging.getLogger('parent_job')
        pool = AsyncPool(parent_logger=parent)
        assert pool.logger.parent is parent
        assert (pool.logger.handlers, pool.logger.propagate) == ([], True)

        records = Records()
        configured = AsyncPool(name='own', parent_logger=parent, logger_config=LoggerConfig(handlers=[records]))
        assert configured.logger.parent is parent
        assert configured.logger.handlers == [records]
        configured.logger.info('mine')
        assert records.records[0].job == 'AsyncPool|own'

    def test_subclass_needs_an_ipc_backend(self):
        with pytest.raises(TypeError, match='must define an `_ipc_cls`'):

            class NoBackend(BasePool):
                pass

        with pytest.raises(TypeError, match='must define an `_ipc_cls`'):

            class WrongBackend(BasePool):
                _ipc_cls = dict

    def test_get_logger(self):
        logger = AsyncPool.get_logger('pool_logger')
        assert len(logger.handlers) == 1
        assert logger.level == logging.INFO
        with pytest.raises(TypeError, match='Expected a LoggerConfig instance, got dict'):
            AsyncPool.get_logger('pool_logger', config={})

    # Context

    @pool_types
    @pytest.mark.asyncio
    async def test_resources_are_granted_and_released(self, pool_cls):
        available = await get_rc_manager().available()
        async with pool_cls(max_jobs=2) as pool:
            assert pool.resources is not None
            assert pool.max_jobs == pool.resources.jobs
            assert await get_rc_manager().available() != available
        assert pool.resources is None
        assert await get_rc_manager().available() == available

    @pytest.mark.asyncio
    async def test_cannot_be_entered_twice(self):
        pool = AsyncPool()
        async with pool:
            with pytest.raises(RuntimeError, match='already in a context manager'):
                await pool.__aenter__()
        with pytest.raises(RuntimeError, match='single-use'):
            await pool.__aenter__()

    @pytest.mark.asyncio
    async def test_primitives_only_in_the_context(self):
        pool = AsyncPool()
        for get in (pool.get_lock, pool.get_queue, pool.get_state):
            with pytest.raises(RuntimeError, match='only available from within the pool context manager'):
                get('x')
        async with pool:
            pool.get_queue('q')
            pool.get_state('s')
            assert pool._build_ipc_handle() is not None
        assert pool._build_ipc_handle() is None
        with pytest.raises(RuntimeError):
            pool.get_lock()

    # Submitting

    @pytest.mark.asyncio
    async def test_submit_only_in_the_context(self):
        with pytest.raises(RuntimeError, match='Jobs submitted outside the pool context'):
            AsyncPool().submit(Talker)

    @pytest.mark.parametrize(
        'job_cls, configs, match',
        [
            (JobConfig, None, 'Expected a subclass of Job'),
            (Talker, 'config', 'Invalid `configs` type: str'),
            (Talker, 3, 'Invalid `configs` type: int'),
        ],
    )
    @pytest.mark.asyncio
    async def test_submit_checks_its_arguments(self, job_cls, configs, match):
        async with AsyncPool() as pool:
            with pytest.raises(TypeError, match=match):
                pool.submit(job_cls, configs=configs)
            with pytest.raises(TypeError):
                pool.submit(Talker, job_id=['a'])  # misspelled

    @pytest.mark.asyncio
    async def test_submit_configs(self):
        async with AsyncPool() as pool:
            config = JobConfig()
            assert [j.config for j in pool.submit(Talker)] == [None]
            assert [j.config for j in pool.submit(Talker, configs=config)] == [config]
            assert len(pool.submit(Talker, configs=(JobConfig() for _ in range(3)))) == 3

    @pytest.mark.asyncio
    async def test_submit_wires_the_job(self):
        async with AsyncPool(name='wired') as pool:
            pool.add_job_handler(JobFileHandlerConfig(), name='file')
            (job,) = pool.submit(Talker, job_ids=['a'])
            assert job.pool is pool
            assert job.logger.parent is pool.logger
            assert list(job._log_handlers) == ['file']
            assert job._log_handlers is not pool._job_handlers
            assert job._pool_label == 'AsyncPool|wired'
            assert pool.status == {'a': JobStatus.SUBMITTED}

    @pytest.mark.asyncio
    async def test_job_ids(self, monkeypatch):
        monkeypatch.setattr(settings.jobs, 'job_digits', 6)
        async with AsyncPool() as pool:
            jobs = pool.submit(Talker, configs=[None] * 10)
            assert [j.job_id for j in jobs][::9] == ['Talker-01', 'Talker-10']
            # Shown as before, also in the names of their loggers
            assert [j.label for j in jobs][::9] == ['Talker|01', 'Talker|10']
            assert jobs[0].logger.name == f'{pool.logger.name}.Talker|01'
            (given, random) = pool.submit(Talker, configs=[None, None], job_ids=['given', None])
            assert given.job_id == 'given'
            assert len(random.job_id) == 6
            with pytest.raises(ValueError, match=r'Length of `job_ids` \(1\) must match length of `configs` \(2\)'):
                pool.submit(Talker, configs=[None, None], job_ids=['one'])

    @pytest.mark.asyncio
    async def test_duplicate_job_ids_raise(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(Talker, job_ids=['a'])
            with pytest.raises(ValueError, match='unique within the pool, got duplicates: a'):
                pool.submit(Talker, job_ids=['a'])
            with pytest.raises(ValueError, match='got duplicates: b'):
                pool.submit(Talker, configs=[None, None], job_ids=['b', 'b'])
            assert list(pool._jobs) == ['a']

    @pytest.mark.asyncio
    async def test_job_ids_that_cannot_be_file_names_raise(self):
        async with AsyncPool(max_jobs=2) as pool:
            with pytest.raises(ValueError, match='usable in file names and commands'):
                pool.submit(Talker, configs=[None, None], job_ids=['fine', 'Talker|1'])
            # None of them is submitted
            assert pool.status == {}

    @pytest.mark.asyncio
    async def test_generated_job_ids_skip_ids_given_to_other_jobs(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(Counter, job_ids=['Talker-1'])
            (job,) = pool.submit(Talker)
            assert job.job_id == 'Talker-2'

    @pytest.mark.asyncio
    async def test_pools_at_once_generate_the_same_job_ids(self):
        async with AsyncPool(max_jobs=2) as first, AsyncPool(max_jobs=2) as second:
            first_jobs = first.submit(Talker, configs=[JobConfig(), JobConfig()])
            second_jobs = second.submit(Talker, configs=[JobConfig(), JobConfig()])
            assert [j.job_id for j in first_jobs] == [j.job_id for j in second_jobs] == ['Talker-1', 'Talker-2']
            assert first_jobs[0].logger is not second_jobs[0].logger

    @pool_types
    @pytest.mark.asyncio
    async def test_submit_resources(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Talker, configs=[JobConfig(resources=Resources(jobs=4, threads=4, processes=1)), None])
            asked, not_asked = pool._submissions.values()
            # A job's request is capped by the pool; a job that asks for nothing gets the pool's
            assert asked.resources == Resources(jobs=4, threads=4, processes=1).cap_to(pool.resources)
            assert not_asked.resources == pool.resources
            results = await pool.launch()
        assert [r.success for r in results] == [True, True]

    @pytest.mark.asyncio
    async def test_batch_job_asks_pool_and_site_apart(self, tmp_path):
        config = BatchJobConfig(
            job_directory=tmp_path,
            resources=Resources(jobs=1, threads=2, processes=1),
            site_resources=SiteResources(cpus=8, memory=16),
        )
        assert '--cpus=8' in config.to_slurm()
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(BatchJob, configs=[config])
            submission = next(iter(pool._submissions.values()))
            assert submission.resources == Resources(jobs=1, threads=2, processes=1).cap_to(pool.resources)

    @pool_types
    @pytest.mark.asyncio
    async def test_batch_job_in_a_path_with_spaces(self, pool_cls, tmp_path):
        # Its arguments are in its configuration, so they reach worker threads and processes too
        config = BatchJobConfig(job_directory=tmp_path / 'my dir', templates=ECHO, arguments=['x y', "it's"])
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(BatchJob, configs=[config])
            (result,) = await pool.launch()
        assert result.error is None
        assert result.data.stdout.strip() == "[x y] [it's]"

    # Job handlers

    @pytest.mark.asyncio
    async def test_job_handlers(self):
        pool = AsyncPool()
        with pytest.raises(TypeError, match='Expected a HandlerConfig instance, got dict'):
            pool.add_job_handler({}, name='file')
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        pool.add_job_handler(HandlerConfig(), name='other')
        async with pool:
            (first,) = pool.submit(Talker)
            pool.remove_job_handler('file')
            pool.remove_job_handler('never added')
            (second,) = pool.submit(Talker)
        # Jobs submitted before keep what they got
        assert list(first._log_handlers) == ['file', 'other']
        assert list(second._log_handlers) == ['other']

    # Launching

    @pool_types
    @pytest.mark.asyncio
    async def test_launch(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, configs=[None, None], job_ids=['a', 'b'])
            results = await pool.launch()
        assert all(isinstance(r, JobResults) for r in results)
        assert [r.data for r in results] == ['a', 'b']

    @pool_types
    @pytest.mark.asyncio
    async def test_launch_stream(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, configs=[None, None], job_ids=['a', 'b'])
            results = [r async for r in await pool.launch('stream')]
        assert sorted(r.data for r in results) == ['a', 'b']

    @pool_types
    @pytest.mark.asyncio
    async def test_each_launch_runs_new_jobs_only(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, job_ids=['a'])
            first = await pool.launch()
            pool.submit(Naps, configs=[None, None], job_ids=['b', 'c'])
            second = await pool.launch()
            third = await pool.launch()
            fourth = [r async for r in await pool.launch('stream')]
        assert [r.data for r in first] == ['a']
        assert [r.data for r in second] == ['b', 'c']
        assert third == fourth == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'pool_cls, mode, expected',
        [
            (AsyncPool, 'all', ['slow', 'fast']),  # submission order
            (AsyncPool, 'stream', ['fast', 'slow']),  # completion order
            (SimplePool, 'all', ['slow', 'fast']),
            (SimplePool, 'stream', ['slow', 'fast']),  # one at a time, in submission order
        ],
    )
    async def test_result_order(self, pool_cls, mode, expected):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, configs=[None, None], job_ids=['slow', 'fast'])
            if mode == 'all':
                results = await pool.launch()
            else:
                results = [r async for r in await pool.launch('stream')]
        assert [r.data for r in results] == expected

    @pytest.mark.asyncio
    async def test_launch_checks_its_arguments(self):
        with pytest.raises(RuntimeError, match=r'AsyncPool.launch\(\) must be used within the context manager'):
            await AsyncPool().launch()
        async with AsyncPool() as pool:
            with pytest.raises(ValueError, match="Invalid mode: 'some'"):
                await pool.launch('some')

    @pytest.mark.asyncio
    async def test_launch_without_jobs(self):
        pool, records = recorded(AsyncPool)
        async with pool:
            assert await pool.launch() == []
            pool.submit(Naps)
            await pool.launch()
            assert await pool.launch() == []
        assert records.messages(logging.WARNING) == [
            'No jobs submitted to pool',
            'No jobs submitted to pool since the last launch',
        ]

    @pytest.mark.asyncio
    async def test_launch_turns_errors_of_the_pool_into_results(self, monkeypatch):
        async def breaks(self, submission):
            raise OSError('cannot reach the worker')

        monkeypatch.setattr(AsyncPool, '_execute_submission', breaks)
        pool, records = recorded(AsyncPool)
        async with pool:
            pool.submit(Talker, job_ids=['a'])
            (result,) = await pool.launch()
        assert (result.job_id, type(result.error)) == ('a', OSError)
        assert pool.status == {'a': JobStatus.FAILED}
        assert 'Job a failed with exception: cannot reach the worker' in records.messages(logging.ERROR)

    # Status

    @pool_types
    @pytest.mark.asyncio
    async def test_status_of_finished_jobs(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, job_ids=['a'])
            pool.submit(Failing, job_ids=['b'])
            assert pool.status == {'a': JobStatus.SUBMITTED, 'b': JobStatus.SUBMITTED}
            await pool.launch()
        # Still readable after the pool exits
        assert pool.status == {'a': JobStatus.DONE, 'b': JobStatus.FAILED}
        counts = pool.counts()
        assert set(counts) == set(JobStatus)
        assert (counts[JobStatus.DONE], counts[JobStatus.FAILED], sum(counts.values())) == (1, 1, 2)

    @pool_types
    @pytest.mark.asyncio
    async def test_status_queued_running_and_cancelled(self, pool_cls):
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(Sleeper, configs=[None, None], job_ids=['first', 'second'])
            launch = asyncio.create_task(pool.launch())
            for _ in range(100):  # until the first job has its slot (a process pool starts its workers first)
                await asyncio.sleep(0.02)
                if pool.status['first'] == JobStatus.RUNNING:
                    break
            assert pool.status == {'first': JobStatus.RUNNING, 'second': JobStatus.QUEUED}
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await launch
            # Running or waiting for a slot, both are dropped
            assert pool.status == {'first': JobStatus.CANCELLED, 'second': JobStatus.CANCELLED}

    def test_status_is_a_copy(self):
        pool = AsyncPool()
        pool.status['x'] = JobStatus.DONE
        pool.counts()[JobStatus.DONE] = 5
        assert pool.status == {} and pool.counts()[JobStatus.DONE] == 0

    # Exit

    @pool_types
    @pytest.mark.asyncio
    async def test_cancellation_propagates(self, pool_cls):
        after_block = False

        async def run():
            nonlocal after_block
            async with pool_cls(max_jobs=1) as pool:
                pool.submit(Sleeper)
                await pool.launch()
            after_block = True

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(run(), timeout=0.5)
        assert not after_block

    @pytest.mark.asyncio
    async def test_exceptions_propagate_and_cancel_the_jobs(self):
        pool, records = recorded(AsyncPool, max_jobs=1)
        with pytest.raises(ValueError, match='from the block'):
            async with pool:
                pool.submit(Sleeper, job_ids=['a'])
                launch = asyncio.create_task(pool.launch())
                await asyncio.sleep(0.05)
                raise ValueError('from the block')
        assert launch.cancelled() or launch.done()
        assert pool.status == {'a': JobStatus.CANCELLED}
        assert 'An exception occurred within the pool context: ValueError' in records.messages(logging.ERROR)
        assert 'Cancelling 1 running jobs' in records.messages(logging.WARNING)

    @pytest.mark.asyncio
    async def test_interrupt(self):
        pool, records = recorded(AsyncPool)
        with pytest.raises(KeyboardInterrupt):
            async with pool:
                raise KeyboardInterrupt
        assert 'Pool execution was interrupted by user' in records.messages(logging.ERROR)

    @pytest.mark.asyncio
    async def test_exit_waits_a_while_for_jobs_to_stop(self, monkeypatch):
        wait = asyncio.wait
        monkeypatch.setattr(asyncio, 'wait', lambda tasks, timeout=None: wait(tasks, timeout=0.05))
        pool, records = recorded(AsyncPool, max_jobs=1)
        with pytest.raises(ValueError):
            async with pool:
                pool.submit(Stubborn, job_ids=['a'])
                asyncio.create_task(pool.launch())
                await asyncio.sleep(0.05)
                raise ValueError
        # Cancelled again after the wait
        assert pool.status == {'a': JobStatus.CANCELLED}
        assert records.messages(logging.WARNING).count('Job was cancelled') == 1

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    @pytest.mark.asyncio
    async def test_exit_keeps_the_event_loop_going(self, pool_cls):
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        ticking = asyncio.create_task(ticker())
        with pytest.raises(ValueError):
            async with pool_cls(max_jobs=1) as pool:
                pool.submit(Sleeper, job_ids=['a'])
                asyncio.create_task(pool.launch())
                while pool.status['a'] != JobStatus.RUNNING:
                    await asyncio.sleep(0.02)
                raise ValueError
        ticking.cancel()
        # The loop goes on while the worker stops the job
        assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.5
        assert pool._executor is None
        assert pool.status == {'a': JobStatus.CANCELLED}

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    @pytest.mark.asyncio
    async def test_cancellation_reaches_jobs_in_workers(self, pool_cls, tmp_path):
        marker = tmp_path / 'marker'
        pool, records = recorded(pool_cls, max_jobs=1)

        async def run():
            async with pool:
                pool.submit(Marks, MarksConfig(marker=marker), job_ids=['a'])
                await pool.launch()

        task = asyncio.create_task(run())
        await until(marker.exists)
        start = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The job is stopped, not left to run its 10 s, and the pool says so before the job does
        assert time.monotonic() - start < 3
        assert marker.read_text() == 'cancelled'
        assert pool.status == {'a': JobStatus.CANCELLED}
        warnings = records.messages(logging.WARNING)
        said = warnings.index('Cancelling 1 running jobs; waiting up to 15 s for them to stop')
        assert said < warnings.index('Cancellation received, stopping') < warnings.index('Job was cancelled')

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    @pytest.mark.asyncio
    async def test_jobs_in_workers_get_the_reason(self, pool_cls, tmp_path):
        marker = tmp_path / 'marker'
        pool, records = recorded(pool_cls, max_jobs=1)
        with pytest.raises(ValueError):
            async with pool:
                pool.submit(Marks, MarksConfig(marker=marker), job_ids=['a'])
                asyncio.create_task(pool.launch())
                await until(marker.exists)
                raise ValueError
        assert marker.read_text() == 'cancelled'
        reason = 'Job cancellation requested with reason: An exception occurred in the pool context'
        assert reason in records.messages(logging.WARNING)

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    @pytest.mark.asyncio
    async def test_a_cancelled_launch_stops_jobs_in_workers(self, pool_cls, tmp_path):
        marker = tmp_path / 'marker'
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(Marks, MarksConfig(marker=marker), job_ids=['a'])
            launch = asyncio.create_task(pool.launch())
            await until(marker.exists)
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await launch
        # The pool goes on, and its exit waits for the worker to stop the job
        assert marker.read_text() == 'cancelled'

    @pytest.mark.asyncio
    async def test_threads_that_dont_stop_are_waited_for(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ThreadedPool, '_CANCEL_GRACE', 0.2)
        marker = tmp_path / 'marker'
        pool, records = recorded(ThreadedPool, max_jobs=1)
        start = time.monotonic()
        with pytest.raises(ValueError):
            async with pool:
                pool.submit(Blocks, MarksConfig(marker=marker, seconds=1.5), job_ids=['a'])
                asyncio.create_task(pool.launch())
                await until(marker.exists)
                raise ValueError
        assert time.monotonic() - start >= 1.5
        assert "1 jobs didn't stop within 0.2 s; waiting for their workers to finish" in records.messages(
            logging.WARNING
        )

    @pytest.mark.parametrize('pool', ['ThreadedPool', 'MultiprocessPool'])
    def test_ctrl_c_cancels_batch_jobs(self, tmp_path, pool):
        # The console's or terminal's Ctrl+C doesn't reach a batch file's programs, which would die of it before the
        #  worker cancels the job (failing it): the job is cancelled, and stops them
        done = run_ctrl_c_script(
            tmp_path,
            f"""
            import asyncio, sys

            from tests.jobs.helpers import until
            from xtl.jobs.config import BatchJobConfig
            from xtl.jobs.jobs import BatchJob
            from xtl.jobs.pools import {pool}
            from xtl.jobs.results import JobStatus
            from xtl.jobs.shells import Shell

            python = '"' + sys.executable + '"'
            sleep = python + ' -c "import time; time.sleep(30)"'


            async def main():
                pool = {pool}(max_jobs=2)
                try:
                    async with pool:
                        configs = [BatchJobConfig(templates={{Shell.BASH: sleep, Shell.CMD: sleep}}) for _ in range(2)]
                        pool.submit(BatchJob, configs, job_ids=['a', 'b'])
                        launch = asyncio.ensure_future(pool.launch())
                        await until(lambda: set(pool.status.values()) == {{JobStatus.RUNNING}})
                        await asyncio.sleep(2)  # the programs have started
                        ctrl_c()
                        await launch
                finally:
                    print('status:', sorted(s.value for s in pool.status.values()))


            if __name__ == '__main__':
                try:
                    asyncio.run(main())
                except KeyboardInterrupt:
                    print('interrupted')
            """,
        )
        assert done.returncode == 0, done.stderr
        assert "status: ['cancelled', 'cancelled']" in done.stdout, done.stdout + done.stderr
        assert 'subprocess error' not in done.stdout + done.stderr

    # Settings and shared primitives

    @pool_types
    @pytest.mark.asyncio
    async def test_jobs_see_settings_changed_in_code(self, pool_cls, monkeypatch):
        # A process worker started with spawn reads `xtl.toml` afresh; the job still sees the parent's settings
        monkeypatch.setattr(settings.jobs, 'keep_temp', not settings.jobs.keep_temp)
        monkeypatch.setattr(settings.jobs.permissions, 'update', not settings.jobs.permissions.update)
        monkeypatch.setattr(settings.dependencies, 'modules', ['gcc/12', 'openmpi/4'])
        expected = (settings.jobs.keep_temp, settings.jobs.permissions.update, ['gcc/12', 'openmpi/4'])
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(ReadsSettings)
            (result,) = await pool.launch()
        assert result.error is None
        assert tuple(result.data) == expected

    @pool_types
    @pytest.mark.asyncio
    async def test_default_lock_is_shared(self, pool_cls):
        async with pool_cls(max_jobs=4) as pool:
            state = pool.get_state('counter')
            state['n'] = 0
            pool.submit(Counter, configs=[JobConfig() for _ in range(16)])
            results = await pool.launch()
            assert all(r.success for r in results)
            assert state['n'] == 16

    @pool_types
    @pytest.mark.asyncio
    async def test_caller_reads_the_queue_while_jobs_run(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            q = pool.get_queue('items')
            pool.submit(Producer, configs=[None, None], job_ids=['a', 'b'])
            launch = asyncio.create_task(pool.launch())
            items = [await asyncio.wait_for(q.get(), timeout=30) for _ in range(2)]
            await launch
        assert sorted(items) == ['a', 'b']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    async def test_lock_registered_after_submit_is_shared(self, pool_cls):
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(UsesLateLock)
            pool.get_lock('late')
            (result,) = await pool.launch()
        assert result.error is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', WORKER_POOLS, ids=lambda c: c.__name__)
    async def test_unregistered_lock_raises_in_workers(self, pool_cls):
        async with pool_cls(max_jobs=1) as pool:
            pool.submit(UsesUnknownLock)
            (result,) = await pool.launch()
        assert isinstance(result.error, KeyError)
        assert "pool.get_lock('unknown')" in str(result.error)


class TestProxyPool:
    def test_pool_id(self):
        assert ProxyPool(AsyncIPCBackend()).pool_id == '<proxy>'

    def test_primitives_come_from_the_backend(self):
        backend = ThreadedIPCBackend()
        backend.start()
        handle = backend.to_handle(IPCHandleNames(locks={'l'}, queues={'q'}, states={'s'}))
        backend.get_lock('l'), backend.get_queue('q'), backend.get_state('s')
        proxy = ProxyPool(ThreadedIPCBackend.from_handle(handle))
        assert proxy.get_lock('l').name == 'l'
        proxy.get_state('s')['x'] = 1
        assert backend.get_state('s')['x'] == 1
        assert proxy.get_queue('q') is not None
        # A worker's backend only gives out what the pool registered
        for get in (proxy.get_lock, proxy.get_queue, proxy.get_state):
            with pytest.raises(KeyError, match='was not registered with the pool'):
                get('x')
        backend.stop()


class TestRunWatched:
    class Gone:
        """A shared state whose manager has gone away."""

        def __contains__(self, key):
            raise ConnectionError('gone')

    @pytest.mark.parametrize(
        'ipc', [None, ThreadedIPCBackend.from_handle(IPCHandle({}, {}, {}))], ids=['none', 'empty']
    )
    @pytest.mark.asyncio
    async def test_without_flags_the_job_just_runs(self, ipc):
        result = await _run_watched(Naps(job_id='a'), ipc)
        assert result.data == 'a'

    @pytest.mark.asyncio
    async def test_flags_that_go_away_leave_the_job_running(self):
        ipc = ThreadedIPCBackend.from_handle(IPCHandle({}, {}, {'_xtl_cancel': self.Gone()}))
        result = await _run_watched(Naps(job_id='a'), ipc)
        assert result.data == 'a'


class TestSimplePool:
    def test_resources(self):
        assert SimplePool(max_jobs=4)._rc_requested == Resources(jobs=1, threads=1, processes=1)
        requested = SimplePool(resources=Resources(jobs=3, threads=2, processes=2))._rc_requested
        assert requested == Resources(jobs=1, threads=2, processes=2)

    @pytest.mark.asyncio
    async def test_one_job_at_a_time(self):
        async with SimplePool(max_jobs=4) as pool:
            assert pool.max_jobs == 1
            pool.submit(Naps, configs=[None] * 3, job_ids=['slow1', 'slow2', 'slow3'])
            launch = asyncio.create_task(pool.launch())
            await asyncio.sleep(0.05)
            assert list(pool.status.values()).count(JobStatus.RUNNING) == 1
            await launch


class TestAsyncPool:
    def test_resources(self):
        assert AsyncPool(max_jobs=4)._rc_requested == Resources(jobs=4, threads=1, processes=1)
        requested = Resources(jobs=3, threads=2, processes=2)
        assert AsyncPool(max_jobs=4, resources=requested)._rc_requested == requested

    @pytest.mark.asyncio
    async def test_jobs_run_in_this_thread(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(WhereAmI)
            (result,) = await pool.launch()
        assert result.data == (os.getpid(), threading.current_thread().name)

    @pytest.mark.asyncio
    async def test_submission_without_a_job(self):
        async with AsyncPool() as pool:
            pool.submit(Talker, job_ids=['a'])
            submission = next(iter(pool._submissions.values()))
            pool._jobs.clear()
            with pytest.raises(KeyError, match="No job found for submission with job_id='a'"):
                await pool._execute_submission(submission)


class TestThreadedPool:
    def test_resources(self):
        assert ThreadedPool(max_jobs=4)._rc_requested == Resources(jobs=4, threads=4, processes=1)

    @pytest.mark.asyncio
    async def test_jobs_run_in_worker_threads(self):
        async with ThreadedPool(max_jobs=2) as pool:
            assert pool._executor is not None
            pool.submit(WhereAmI, configs=[None, None])
            results = await pool.launch()
        assert pool._executor is None
        assert {r.data[0] for r in results} == {os.getpid()}
        assert all(r.data[1].startswith('Thread') for r in results)
        assert threading.current_thread().name not in {r.data[1] for r in results}

    @pytest.mark.parametrize('name, renamed', [('ThreadPoolExecutor-0_3', 'Thread3'), ('worker', 'Thread0')])
    def test_rename_current_thread(self, name, renamed):
        names = []

        def run():
            ThreadedPool._rename_current_thread()
            names.append(threading.current_thread().name)

        thread = threading.Thread(target=run, name=name)
        thread.start()
        thread.join()
        assert names == [renamed]


class TestMultiprocessPool:
    def test_resources(self):
        assert MultiprocessPool(max_jobs=4)._rc_requested == Resources(jobs=4, threads=1, processes=4)

    @pytest.mark.asyncio
    async def test_jobs_run_in_worker_processes(self):
        async with MultiprocessPool(max_jobs=2) as pool:
            assert pool._executor is not None
            assert pool._listener is not None
            pool.submit(WhereAmI, configs=[None, None])
            results = await pool.launch()
        assert (pool._executor, pool._listener) == (None, None)
        assert all(r.error is None for r in results), [r.error for r in results]
        assert os.getpid() not in {r.data[0] for r in results}

    @pytest.mark.asyncio
    async def test_a_dying_worker_fails_its_job(self):
        async with MultiprocessPool(max_jobs=1) as pool:
            pool.submit(Dies, job_ids=['dies'])
            (result,) = await pool.launch()
        assert result.error is not None
        assert pool.status == {'dies': JobStatus.FAILED}

    @pytest.mark.asyncio
    async def test_an_error_that_cannot_be_unpickled(self):
        pool, records = recorded(MultiprocessPool, max_jobs=1)
        async with pool:
            pool.submit(RaisesPicky, job_ids=['picky'])
            pool.submit(Naps, job_ids=['next'])
            picky, following = await pool.launch()
        # The job fails with a stand-in, and the pool goes on
        assert isinstance(picky.error, WorkerError)
        assert str(picky.error) == 'tests.jobs.test_pools.Picky: picky failure'
        assert picky.error.error_type == 'tests.jobs.test_pools.Picky'
        assert 'Picky: picky failure' in picky.error.traceback
        assert following.success and following.data == 'next'
        assert pool.status == {'picky': JobStatus.FAILED, 'next': JobStatus.DONE}
        assert any("error (Picky) can't be sent back" in m for m in records.messages(logging.WARNING))

    @pytest.mark.asyncio
    async def test_data_that_cannot_be_unpickled(self):
        pool, records = recorded(MultiprocessPool, max_jobs=1)
        async with pool:
            pool.submit(ReturnsPicky, job_ids=['picky'])
            (result,) = await pool.launch()
        assert result.data is None
        assert isinstance(result.error, WorkerError)
        assert "data (Picky) can't be sent back" in str(result.error)
        assert any("data (Picky) can't be sent back" in m for m in records.messages(logging.ERROR))

    @pytest.mark.asyncio
    async def test_a_step_error_that_cannot_be_unpickled(self):
        async with MultiprocessPool(max_jobs=1) as pool:
            pool.submit(SteppedPicky, configs=[SteppedJobConfig()], job_ids=['s'])
            (result,) = await pool.launch()
        assert isinstance(result, SteppedJobResults)
        assert result.steps['fine'].success
        assert isinstance(result.steps['picky'].error, WorkerError)
        assert isinstance(result.error, WorkerError)

    def test_an_escaping_error_that_cannot_be_unpickled(self, monkeypatch):
        async def escapes(submission):
            raise Picky('escaped', code=7)

        monkeypatch.setattr(MultiprocessPool, '_bootstrap_process', staticmethod(escapes))
        payload = JobSubmission.from_job(Naps(job_id='e')).to_dict()
        with pytest.raises(WorkerError, match='tests.jobs.test_pools.Picky: escaped'):
            MultiprocessPool._run_in_process(payload)

        async def escapes_plainly(submission):
            raise ValueError('plain')

        monkeypatch.setattr(MultiprocessPool, '_bootstrap_process', staticmethod(escapes_plainly))
        with pytest.raises(ValueError, match='plain'):
            MultiprocessPool._run_in_process(payload)

    @pytest.mark.asyncio
    async def test_workers_that_dont_stop_are_ended(self, monkeypatch, tmp_path):
        monkeypatch.setattr(MultiprocessPool, '_CANCEL_GRACE', 0.5)
        marker = tmp_path / 'marker'
        pool, records = recorded(MultiprocessPool, max_jobs=1)
        with pytest.raises(ValueError):
            async with pool:
                pool.submit(Blocks, MarksConfig(marker=marker, seconds=30), job_ids=['a'])
                asyncio.create_task(pool.launch())
                await until(marker.exists)
                start = time.monotonic()
                raise ValueError
        assert time.monotonic() - start < 10
        assert pool.status == {'a': JobStatus.CANCELLED}
        warnings = records.messages(logging.WARNING)
        assert 'Cancelling 1 running jobs; waiting up to 0.5 s for them to stop' in warnings
        assert "Ending 1 workers that didn't stop" in warnings

    @pytest.mark.asyncio
    async def test_ending_workers_without_their_processes(self):
        # `ProcessPoolExecutor._processes` is private: without it, the pool can only wait
        pool, records = recorded(MultiprocessPool)
        await pool._end_workers(object(), busy=2)
        assert "2 jobs didn't stop within 15 s; waiting for their workers to finish" in records.messages(
            logging.WARNING
        )

    @pytest.mark.asyncio
    async def test_a_failed_listener_is_logged_on_exit(self):
        # Its error isn't raised again on exit: with Python 3.10, that was the Ctrl+C that had already reached
        #  `asyncio.run()`, and it broke the pool's exit
        pool, records = recorded(MultiprocessPool)

        async def fails():
            raise InterruptedError('lost')

        pool._listener = asyncio.ensure_future(fails())
        await asyncio.sleep(0)
        await pool._stop_listener()
        assert pool._listener is None
        assert "Log listener stopped early: InterruptedError('lost')" in records.messages(logging.WARNING)

    def test_ctrl_c_is_left_to_the_parent(self, tmp_path):
        # Ctrl+C reaches every process on the console (or in the terminal's process group); a worker waiting for work
        #  and the manager used to die of it, which broke the pool. Here the parent takes it and goes on.
        done = run_ctrl_c_script(
            tmp_path,
            """
            import asyncio

            from tests.jobs.helpers import Naps, until
            from xtl.jobs.pools import MultiprocessPool

            received = []


            async def main():
                async with MultiprocessPool(max_jobs=2) as pool:
                    pool.submit(Naps, job_ids=['before'])
                    await pool.launch()  # its worker now waits for work
                    state = pool.get_state('shared')
                    signal.signal(signal.SIGINT, lambda *args: received.append(True))
                    ctrl_c()
                    await until(lambda: received)
                    await asyncio.sleep(0.5)
                    state['x'] = 1  # the manager is still there
                    pool.submit(Naps, job_ids=['after'])
                    (result,) = await pool.launch()
                    print('after:', result.success, repr(result.error))


            if __name__ == '__main__':
                asyncio.run(main())
            """,
        )
        assert done.returncode == 0, done.stderr
        assert 'after: True None' in done.stdout, done.stdout + done.stderr

    @pytest.mark.parametrize('delay', [0.1, 0.4])
    def test_ctrl_c_while_the_workers_start(self, tmp_path, delay):
        # A worker started with spawn imports xtl for a second or so before its initializer runs; Ctrl+C then used to
        #  kill it, which broke the pool
        done = run_ctrl_c_script(
            tmp_path,
            """
            import asyncio, sys

            from tests.jobs.helpers import Naps
            from xtl.jobs.pools import MultiprocessPool


            async def main(delay):
                async with MultiprocessPool(max_jobs=4) as pool:
                    pool.submit(Naps, [None] * 8, job_ids=[f'slow{i}' for i in range(8)])
                    launch = asyncio.ensure_future(pool.launch())
                    await asyncio.sleep(delay)
                    ctrl_c()
                    await launch


            if __name__ == '__main__':
                try:
                    asyncio.run(main(float(sys.argv[1])))
                except KeyboardInterrupt:
                    print('interrupted')
            """,
            str(delay),
        )
        assert done.returncode == 0, done.stderr
        assert 'interrupted' in done.stdout
        assert 'Traceback' not in done.stderr, done.stderr

    @pytest.mark.asyncio
    async def test_workers_and_manager_ignore_ctrl_c_from_their_start(self):
        async with MultiprocessPool() as pool:
            assert issubclass(pool._executor._mp_context.Process, _ShieldedStart)
            assert issubclass(pool._ipc._manager._ctx.Process, _ShieldedStart)

    def test_ctrl_c_stops_the_jobs(self, tmp_path):
        # Only the parent acts on Ctrl+C, and the workers stop their jobs
        marker = tmp_path / 'marker'
        done = run_ctrl_c_script(
            tmp_path,
            """
                import asyncio, sys
                from pathlib import Path

                from tests.jobs.helpers import Naps, until
                from tests.jobs.test_pools import Marks, MarksConfig
                from xtl.jobs.pools import MultiprocessPool
                from xtl.jobs.results import JobStatus


                async def main(marker):
                    async with MultiprocessPool(max_jobs=2) as pool:
                        # A job that ends at once leaves its worker waiting for work, the other one runs
                        pool.submit(Naps, job_ids=['quick'])
                        pool.submit(Marks, MarksConfig(marker=marker, seconds=30), job_ids=['slow'])
                        launch = asyncio.ensure_future(pool.launch())
                        await until(lambda: marker.exists() and pool.status['quick'] == JobStatus.DONE)
                        ctrl_c()
                        await launch


                if __name__ == '__main__':
                    try:
                        asyncio.run(main(Path(sys.argv[1])))
                    except KeyboardInterrupt:
                        print('interrupted')
            """,
            str(marker),
        )
        assert done.returncode == 0, done.stderr
        assert 'interrupted' in done.stdout
        assert 'BrokenProcessPool' not in done.stderr
        assert 'handle is closed' not in done.stderr
        assert marker.read_text() == 'cancelled'

    def test_results_that_survive_pickling_are_left_alone(self):
        results = JobResults(job_id='a', data={'x': 1}, error=ValueError('fine'))
        assert _picklable_results(results, logging.getLogger('guard')) is results
        assert _picklable_results(None, logging.getLogger('guard')) is None

    def test_worker_error_survives_pickling(self):
        error = WorkerError('text', error_type='pkg.Cls', traceback='Traceback ...')
        restored = pickle.loads(pickle.dumps(error))
        assert (str(restored), restored.error_type, restored.traceback) == ('pkg.Cls: text', 'pkg.Cls', 'Traceback ...')

    @pytest.mark.asyncio
    async def test_flush_logs_gives_up_after_a_while(self):
        pool, records = recorded(MultiprocessPool)
        async with pool:
            await pool._stop_listener()
            await pool._flush_logs()  # no listener: nothing to wait for
            pool._listener = asyncio.create_task(asyncio.sleep(30))  # a listener that doesn't read the queue
            await pool._flush_logs(timeout=0.05)
            pool._listener.cancel()
            pool._listener = None
        assert pool._flushes == {}
        assert 'Log listener did not catch up within 0 s' in records.messages(logging.WARNING)

    @pytest.mark.asyncio
    async def test_stop_listener_gives_up_after_a_while(self, monkeypatch):
        wait_for = asyncio.wait_for
        monkeypatch.setattr(asyncio, 'wait_for', lambda awaitable, timeout: wait_for(awaitable, timeout=0.05))
        pool, records = recorded(MultiprocessPool)
        await pool._stop_listener()  # not started: nothing to stop
        pool._listener = listener = asyncio.create_task(asyncio.sleep(30))
        await pool._stop_listener()
        assert listener.cancelled()
        assert pool._listener is None
        assert 'Log listener did not shut down within timeout' in records.messages(logging.WARNING)

    @pytest.mark.parametrize('identity, renamed', [((3,), 'Process2'), ((), 'Process0')])
    def test_rename_current_process(self, monkeypatch, identity, renamed):
        import multiprocessing

        process = multiprocessing.current_process()
        monkeypatch.setattr(process, '_identity', identity)
        monkeypatch.setattr(process, 'name', process.name)
        MultiprocessPool._rename_current_process()
        assert process.name == renamed
