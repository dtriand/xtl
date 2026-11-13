"""
Tests for the job execution functionality.
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path

from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.steps import StepSpec
from xtl_core.options import Option


class SimpleJobConfig(JobConfig):
    should_fail: bool = Option(default=False)
    commands: list[str] = Option(default_factory=list)


class SimpleJob(Job[SimpleJobConfig]):
    async def _execute(self):
        if self.config.should_fail:
            raise ValueError('Job execution failed')
        await asyncio.sleep(0.2)  # Simulate some work
        return {'result': 123}


class TwoSteps(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(name='one', job_cls=SimpleJob, config_cls=SimpleJobConfig),
        StepSpec(name='two', job_cls=SimpleJob, config_cls=SimpleJobConfig),
    )


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
            job = SimpleJob(job_id='test_001')
            assert job.job_id == 'test_001'
            assert job.config is None

        def test_job_configuration(self):
            """Test that a job can be configured correctly."""
            job = SimpleJob(job_id='test_001')
            job.configure(SimpleJobConfig(should_fail=True))
            assert job.config is not None
            assert job.config.should_fail is True

        def test_job_with_config(self):
            """Test job creation with configuration."""
            config = SimpleJobConfig()
            job = SimpleJob.with_config(config, job_id='test_002')
            assert job.job_id == 'test_002'
            assert job.config is config

        @pytest.mark.parametrize(
            'should_fail',
            [True, False],
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
                await asyncio.gather(job.run(), cancel_job())

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

        def test_batch_job_with_config_context(self):
            """Only the arguments that aren't the constructor's are template context."""
            logger = logging.getLogger('test_batch_context')
            job = BatchJob.with_config(
                BatchJobConfig(), job_id='test_batch_context', logger=logger, batch_args=['a'], x=1
            )
            assert job.job_id == 'test_batch_context'
            assert job.logger is logger
            assert job._batch_args == ['a']
            assert job._batch_context == {'x': 1}

        @pytest.mark.asyncio
        async def test_stepped_job_without_pool_logs_no_warnings(self, caplog):
            """A stepped job run on its own has nothing to warn about."""
            job = TwoSteps(job_id='test_stepped_no_pool', logger=logging.getLogger('test_stepped_no_pool'))
            job.configure(SteppedJobConfig())
            with caplog.at_level(logging.DEBUG, logger='test_stepped_no_pool'):
                result = await job.run()
            assert result.success
            assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

        def test_job_id_is_used_as_given(self):
            """Two jobs given the same id keep it, and share its logger."""
            first = SimpleJob(job_id='same_id')
            second = SimpleJob(job_id='same_id')
            assert first.job_id == second.job_id == 'same_id'
            assert first.logger is second.logger
