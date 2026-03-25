"""
Tests for the job execution functionality.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

from tests.conftest import skipif_not_windows, skipif_not_linux
from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig, BatchConfig
from xtl.automate.shells import BashShell, CmdShell, PowerShell
from xtl.common.options import Option


class SimpleJobConfig(JobConfig):
    should_fail: bool = Option(default=False)
    commands: list[str] = Option(default_factory=list)

class SimpleJob(Job[SimpleJobConfig]):

    async def _execute(self):
        if self.config.should_fail:
            raise ValueError('Job execution failed')
        await asyncio.sleep(0.2)  # Simulate some work
        return {'result': 123}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestJob:
    # Test basic job functionality
    class TestJobBasics:
        """Test basic functionality of the Job class."""

        def test_job_initialization(self):
            """Test that a job can be initialized correctly."""
            job = SimpleJob(job_id="test_001")
            assert job.job_id == "test_001"
            assert job.config is None

        def test_job_configuration(self):
            """Test that a job can be configured correctly."""
            job = SimpleJob(job_id="test_001")
            job.configure(SimpleJobConfig(should_fail=True))
            assert job.config is not None
            assert job.config.should_fail == True

        def test_job_with_config(self):
            """Test job creation with configuration."""
            config = SimpleJobConfig()
            job = SimpleJob.with_config(config, job_id="test_002")
            assert job.job_id == "test_002"
            assert job.config is config

        def test_job_map(self):
            """Test mapping multiple configurations to job instances."""
            configs = [SimpleJobConfig(should_fail=True) for _ in range(3)]
            jobs = SimpleJob.map(configs)
            assert len(jobs) == 3
            for job, config in zip(jobs, configs):
                assert job.config is config

        @pytest.mark.parametrize(
            'should_fail', [True, False],
        )
        @pytest.mark.asyncio
        async def test_job_run(self, should_fail):
            """Test successful job execution."""
            job = SimpleJob(job_id='test_success')
            job.configure(SimpleJobConfig(should_fail=should_fail))

            result = await job.run()  # captures the exception
            if should_fail:
                assert not result.success
                assert isinstance(result.error, ValueError)
                assert result.data is None
            else:
                assert result.success
                assert result.data == {'result': 123}
                assert result.error is None

        @pytest.mark.asyncio
        async def test_job_run_cancellation(self):
            """Test job cancellation during execution."""
            job = SimpleJob(job_id='test_cancel')
            job.configure(SimpleJobConfig())

            async def cancel_job():
                # Give the job a chance to start running
                await asyncio.sleep(0.1)
                # Raise cancellation
                raise asyncio.CancelledError('Test cancellation')

            with pytest.raises(asyncio.CancelledError):
                # Run both tasks concurrently
                await asyncio.gather(
                    job.run(),
                    cancel_job()
                )

        def test_job_logger(self):
            """Test job logger creation and access."""
            job = SimpleJob(job_id='test_logger')
            logger = job.logger
            assert logger is not None
            assert logger.name == job.job_id

        def test_class_logger(self):
            """Test static logger creation."""
            logger = SimpleJob.get_logger('static_logger')
            assert logger is not None
            assert logger.name == 'static_logger'

        def test_job_registry(self):
            """Test job registry functionality."""
            job_id = 'registry_test'
            job = SimpleJob(job_id=job_id)
            assert job_id in SimpleJob._registry
            assert SimpleJob._registry[job_id] is job

            # Test cleanup on deletion
            job.clear()
            assert job_id not in SimpleJob._registry
