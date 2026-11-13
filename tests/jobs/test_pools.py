"""
Tests for submitting jobs to pools and sharing locks, queues and states with them.
"""

import asyncio

import pytest

from xtl import settings
from tests.jobs.test_logging import POOLS, WORKER_POOLS, Talker, fresh_logging  # noqa: F401 (autouse fixture)
from xtl.jobs.config import BatchJobConfig, JobConfig, SiteResources
from xtl.jobs.ipc.base import IPCHandleNames
from xtl.jobs.jobs import BatchJob, Job
from xtl.jobs.pools import AsyncPool, SimplePool, ThreadedPool
from xtl.jobs.resources import Resources
from xtl.jobs.shells import Shell


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


class Sleeper(Job[JobConfig]):
    async def _execute(self):
        await asyncio.sleep(2)


class Naps(Job[JobConfig]):
    """Returns its id; a job whose id starts with `slow` takes longer."""

    async def _execute(self):
        await asyncio.sleep(0.2 if self.job_id.startswith('slow') else 0.01)
        return self.job_id


class ReadsSettings(Job[JobConfig]):
    async def _execute(self):
        return settings.jobs.keep_temp, settings.jobs.permissions.update, settings.dependencies.modules


ECHO = {Shell.BASH: 'echo "[$1] [$2]"', Shell.CMD: 'echo [%~1] [%~2]'}


class TestSubmit:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)
    async def test_job_with_a_resource_request(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Talker, configs=[JobConfig(resources=Resources(jobs=4, threads=4, processes=1))])
            submission = next(iter(pool._submissions.values()))
            assert submission.resources == Resources(jobs=4, threads=4, processes=1).cap_to(pool.resources)
            results = await pool.launch()
        assert [r.success for r in results] == [True]

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


class TestLaunch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)
    async def test_each_launch_runs_new_jobs_only(self, pool_cls):
        async with pool_cls(max_jobs=2) as pool:
            pool.submit(Naps, job_ids=['a'])
            first = await pool.launch()
            pool.submit(Naps, configs=[None, None], job_ids=['b', 'c'])
            second = await pool.launch()
            third = await pool.launch()
        assert [r.data for r in first] == ['a']
        assert [r.data for r in second] == ['b', 'c']
        assert third == []

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


class TestExit:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)
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
    async def test_exceptions_propagate(self):
        with pytest.raises(ValueError, match='from the block'):
            async with AsyncPool(max_jobs=1):
                raise ValueError('from the block')


class TestSettings:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)
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


class TestJobIds:
    @pytest.mark.asyncio
    async def test_pools_at_once_generate_the_same_ids(self):
        async with AsyncPool(max_jobs=2) as first, AsyncPool(max_jobs=2) as second:
            first_jobs = first.submit(Talker, configs=[JobConfig(), JobConfig()])
            second_jobs = second.submit(Talker, configs=[JobConfig(), JobConfig()])
            assert [j.job_id for j in first_jobs] == [j.job_id for j in second_jobs] == ['Talker|1', 'Talker|2']
            assert first_jobs[0].logger is not second_jobs[0].logger

    @pytest.mark.asyncio
    async def test_duplicate_ids_raise(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(Talker, job_ids=['a'])
            with pytest.raises(ValueError, match='unique within the pool, got duplicates: a'):
                pool.submit(Talker, job_ids=['a'])
            with pytest.raises(ValueError, match='got duplicates: b'):
                pool.submit(Talker, configs=[None, None], job_ids=['b', 'b'])
            assert list(pool._jobs) == ['a']

    @pytest.mark.asyncio
    async def test_generated_ids_skip_ids_given_to_other_jobs(self):
        async with AsyncPool(max_jobs=2) as pool:
            pool.submit(Counter, job_ids=['Talker|1'])
            (job,) = pool.submit(Talker)
            assert job.job_id == 'Talker|2'


class TestSharedPrimitives:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)
    async def test_default_lock_is_shared(self, pool_cls):
        async with pool_cls(max_jobs=4) as pool:
            state = pool.get_state('counter')
            state['n'] = 0
            pool.submit(Counter, configs=[JobConfig() for _ in range(16)])
            results = await pool.launch()
            assert all(r.success for r in results)
            assert state['n'] == 16

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

    def test_worker_backend_cannot_create(self):
        backend = ThreadedPool._ipc_cls.from_handle(ThreadedPool._ipc_cls().to_handle(IPCHandleNames()))
        for get in (backend.get_lock, backend.get_queue, backend.get_state):
            with pytest.raises(KeyError, match='was not registered with the pool'):
                get('x')
