"""
Pools and jobs shared by the tests of :mod:`xtl.jobs`.

Jobs are defined at module level, so that worker processes can import them.
"""

import asyncio

from xtl.jobs.config import JobConfig
from xtl.jobs.jobs import Job
from xtl.jobs.pools import AsyncPool, MultiprocessPool, SimplePool, ThreadedPool

POOLS = [SimplePool, AsyncPool, ThreadedPool, MultiprocessPool]
WORKER_POOLS = [ThreadedPool, MultiprocessPool]


class Talker(Job[JobConfig]):
    """Logs one record per level and returns its id."""

    async def _execute(self):
        self.logger.debug('debug from %s', self.job_id)
        self.logger.info('info from %s', self.job_id)
        self.logger.warning('warning from %s', self.job_id)
        return self.job_id


class Failing(Job[JobConfig]):
    async def _execute(self):
        raise ValueError('boom')


class Sleeper(Job[JobConfig]):
    """Runs long enough to be cancelled."""

    async def _execute(self):
        await asyncio.sleep(2)


class Naps(Job[JobConfig]):
    """Returns its id; a job whose id starts with `slow` takes longer."""

    async def _execute(self):
        await asyncio.sleep(0.2 if self.job_id.startswith('slow') else 0.01)
        return self.job_id
