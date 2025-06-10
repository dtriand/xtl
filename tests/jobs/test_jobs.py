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

class EchoJob(Job[SimpleJobConfig]):

    async def _execute(self):
        results = await self._execute_batch(self.config.commands,
                                            args=['Hello', 'World!'])
        return results.data


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
            assert job.config.batch is None
            assert job.config.should_fail == True

            # Update configs
            job.configure(SimpleJobConfig(batch=BatchConfig(permissions='755')))
            assert job.config.batch is not None
            assert job.config.batch.permissions.octal == '0o755'

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


    # Test batch execution functionality
    class TestBatchExecution:
        """Test batch execution functionality of the Job class."""

        @skipif_not_windows
        @pytest.mark.parametrize(
            'shell,      cmd,             expected', [
            (CmdShell, 'echo "%2"', '"World!"\n'),
            (PowerShell, 'echo $args[1]', 'World!\n')
            ], ids=['cmd', 'powershell'])
        @pytest.mark.asyncio
        async def test_execute_batch_win(self, temp_dir, shell, cmd, expected):
            job = EchoJob(job_id='test_win')
            job.configure(
                SimpleJobConfig(
                    job_directory=temp_dir,
                    commands=[cmd],
                    batch=BatchConfig(),
                )
            )

            job.config.batch.shell = shell

            result = await job.run()

            assert result.data['stdout'] == expected
            assert result.data['stderr'] == ''

        @skipif_not_linux
        @pytest.mark.parametrize(
            'shell,     cmd,         expected', [
            (BashShell, 'echo "$2"', 'World!\n')
            ], ids=['bash'])
        @pytest.mark.asyncio
        async def test_execute_batch_posix(self, temp_dir, shell, cmd, expected):
            job = EchoJob(job_id='test_posix')
            job.configure(
                SimpleJobConfig(
                    job_directory=temp_dir,
                    commands=[cmd],
                    batch=BatchConfig(),
                )
            )

            job.config.batch.shell = shell

            result = await job.run()

            assert result.data['stdout'] == expected
            assert result.data['stderr'] == ''

        @pytest.mark.asyncio
        async def test_log_stream_to_file(self, temp_dir):
            """Test _log_stream_to_file functionality."""

            job = SimpleJob(job_id='test_log_stream_to_file')
            # Create a test file
            test_file = temp_dir / "test_log.txt"

            # Create a mock stream
            mock_stream = AsyncMock()
            mock_stream.read.side_effect = [b"line1\n", b"line2\n", b""]

            # Call _log_stream_to_file
            await job._log_stream_to_file(mock_stream, test_file)

            # Verify file contents
            assert test_file.exists()
            content = test_file.read_bytes()
            assert content == b"line1\nline2\n"

            # Verify stream read calls
            assert mock_stream.read.call_count == 3
