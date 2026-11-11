"""
Tests for the exceptions of :mod:`xtl.jobs`.
"""

import pytest

from xtl.exceptions import XTLError
from xtl.jobs.config import JobConfig
from xtl.jobs.exceptions import SubprocessError
from xtl.jobs.jobs import Job
from xtl_core.errors import XTLCoreError


class FailingJob(Job[JobConfig]):
    async def _execute(self):
        raise SubprocessError('Batch execution of run.sh failed with return code 2', command=('run.sh', '--fast'))


def test_subprocess_error():
    error = SubprocessError('Batch execution of run.sh failed with return code 2', command=('run.sh', '--fast'))
    assert isinstance(error, (XTLError, XTLCoreError, RuntimeError))
    assert str(error) == error.message == 'Batch execution of run.sh failed with return code 2'
    assert error.command == ('run.sh', '--fast')


def test_command_optional():
    assert SubprocessError('failed').command is None


def test_lazy_export():
    import xtl.jobs

    assert xtl.jobs.SubprocessError is SubprocessError


@pytest.mark.asyncio
async def test_job_captures_subprocess_error():
    job = FailingJob(job_id='test_subprocess_error')
    job.configure(JobConfig())
    result = await job.run()
    assert not result.success
    assert isinstance(result.error, SubprocessError)
    assert result.error.command == ('run.sh', '--fast')
