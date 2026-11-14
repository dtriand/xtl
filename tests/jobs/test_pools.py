"""
Tests for pools: the pool types, what every pool does (one suite over the four types), the proxy that jobs in workers
get, and what each type adds.
"""

import asyncio
import logging
import os
import threading
import time
import typing

import pytest

from xtl import settings
from tests.jobs.helpers import POOLS, WORKER_POOLS, Failing, Naps, Sleeper, Talker
from xtl.jobs.config import BatchJobConfig, JobConfig, SiteResources
from xtl.jobs.ipc import AsyncIPCBackend, ThreadedIPCBackend
from xtl.jobs.ipc.base import IPCHandleNames
from xtl.jobs.jobs import BatchJob, Job
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
)
from xtl.jobs.resources import Resources, get_rc_manager
from xtl.jobs.results import JobResults, JobStatus
from xtl.jobs.shells import Shell


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


class Dies(Job[JobConfig]):
    """Ends its worker process."""

    async def _execute(self):
        os._exit(3)


ECHO = {Shell.BASH: 'echo "[$1] [$2]"', Shell.CMD: 'echo [%~1] [%~2]'}

pool_types = pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)


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
            assert [j.job_id for j in pool.submit(Talker, configs=[None] * 10)][::9] == ['Talker|01', 'Talker|10']
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
    async def test_generated_job_ids_skip_ids_given_to_other_jobs(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(Counter, job_ids=['Talker|1'])
            (job,) = pool.submit(Talker)
            assert job.job_id == 'Talker|2'

    @pytest.mark.asyncio
    async def test_pools_at_once_generate_the_same_job_ids(self):
        async with AsyncPool(max_jobs=2) as first, AsyncPool(max_jobs=2) as second:
            first_jobs = first.submit(Talker, configs=[JobConfig(), JobConfig()])
            second_jobs = second.submit(Talker, configs=[JobConfig(), JobConfig()])
            assert [j.job_id for j in first_jobs] == [j.job_id for j in second_jobs] == ['Talker|1', 'Talker|2']
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

    @pytest.mark.asyncio
    async def test_batch_job_in_a_path_with_spaces(self, tmp_path):
        config = BatchJobConfig(job_directory=tmp_path / 'my dir', templates=ECHO)
        async with AsyncPool(max_jobs=1) as pool:
            (job,) = pool.submit(BatchJob, configs=[config])
            job._batch_args = ['x y', "it's"]
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
        assert 'Cancelling all running tasks' in records.messages(logging.WARNING)

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
        assert 'Waiting for 1 tasks to cancel' in records.messages(logging.WARNING)

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
        # The running job still finishes in its worker, but the loop goes on meanwhile
        assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.5
        assert pool._executor is None
        assert pool.status == {'a': JobStatus.CANCELLED}

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
