"""
Tests for the job execution functionality.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig, BatchConfig
from xtl.automate.shells import BashShell, DefaultShell
from xtl.automate.sites import LocalSite
from xtl.automate.batchfile import BatchFile

# Create a concrete implementation of the abstract Job class for testing
class TestJob(Job):
    """Test implementation of Job for unit testing."""

    def __init__(self, job_id=None, logger=None, should_fail=False):
        super().__init__(job_id=job_id, logger=logger)
        self.should_fail = should_fail
        self.execute_called = False
        self.execute_result = "test_result"

    async def _execute(self):
        """Implementation of the abstract _execute method for testing."""
        self.execute_called = True
        if self.should_fail:
            raise ValueError("Test job failure")
        return self.execute_result

# Create a job config with batch settings for testing batch execution
class TestBatchJobConfig(JobConfig): ...


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def batch_config(temp_dir):
    """Create a batch config for testing."""
    return BatchConfig(
        directory=temp_dir,
        filename='test_batch',
        compute_site=LocalSite(),
        default_shell=DefaultShell
    )


@pytest.fixture
def job_config(batch_config):
    """Create a job config for testing."""
    config = TestBatchJobConfig()
    config.batch = batch_config
    config.job_directory = batch_config.directory
    return config


@pytest.fixture
def test_job():
    """Create a test job instance."""
    return TestJob(job_id="test_job_001")


@pytest.fixture
def test_job_with_config(test_job, job_config):
    """Create a test job instance with configuration."""
    test_job.configure(job_config)
    return test_job


# Test basic job functionality
class TestJobBasics:
    """Test basic functionality of the Job class."""

    def test_job_initialization(self):
        """Test that a job can be initialized correctly."""
        job = TestJob(job_id="test_001")
        assert job.job_id == "test_001"
        assert not job.is_running
        assert not job.is_complete

    def test_job_configuration(self, test_job, job_config):
        """Test that a job can be configured correctly."""
        test_job.configure(job_config)
        assert test_job.config is not None
        assert test_job.config.batch is not None
        assert test_job.config.batch.directory == job_config.batch.directory

    def test_job_with_config(self):
        """Test job creation with configuration."""
        config = TestBatchJobConfig()
        job = TestJob.with_config(config, job_id="test_002")
        assert job.job_id == "test_002"
        assert job.config is config

    def test_job_map(self):
        """Test mapping multiple configurations to job instances."""
        configs = [TestBatchJobConfig() for _ in range(3)]
        jobs = TestJob.map(configs)
        assert len(jobs) == 3
        for job, config in zip(jobs, configs):
            assert job.config is config

    @pytest.mark.asyncio
    async def test_job_run_success(self, test_job):
        """Test successful job execution."""
        result = await test_job.run()
        assert test_job.execute_called
        assert result.job_id == test_job.job_id
        assert result.data == "test_result"
        assert result.error is None
        assert result.success
        assert test_job.is_complete
        assert not test_job.is_running

    @pytest.mark.asyncio
    async def test_job_run_failure(self):
        """Test job execution with failure."""
        job = TestJob(job_id="test_fail", should_fail=True)
        result = await job.run()
        assert job.execute_called
        assert result.job_id == job.job_id
        assert result.data is None
        assert isinstance(result.error, ValueError)
        assert not result.success
        assert not job.is_complete
        assert not job.is_running

    @pytest.mark.asyncio
    async def test_job_run_cancellation(self, test_job):
        """Test job cancellation during execution."""
        async def cancel_job():
            # Give the job a chance to start running
            await asyncio.sleep(0.1)
            # Raise cancellation
            raise asyncio.CancelledError("Test cancellation")

        with pytest.raises(asyncio.CancelledError):
            # Run both tasks concurrently
            await asyncio.gather(
                test_job.run(),
                cancel_job()
            )

    def test_job_logger(self, test_job):
        """Test job logger creation and access."""
        logger = test_job.logger
        assert logger is not None
        assert logger.name == test_job.job_id

    def test_class_logger(self):
        """Test static logger creation."""
        logger = TestJob.get_logger("static_logger")
        assert logger is not None
        assert logger.name == "static_logger"

    def test_job_registry(self):
        """Test job registry functionality."""
        job_id = "registry_test"
        job = TestJob(job_id=job_id)
        assert job_id in TestJob._registry
        assert TestJob._registry[job_id] is job

        # Test cleanup on deletion
        job.clear()
        assert job_id not in TestJob._registry


# Test batch execution functionality
class TestBatchExecution:
    """Test batch execution functionality of the Job class."""

    @pytest.mark.asyncio
    async def test_execute_batch_setup(self, test_job_with_config, temp_dir):
        """Test that _execute_batch sets up the batch execution environment correctly."""
        with patch.object(test_job_with_config, '_get_batch_execution_command') as mock_get_cmd, \
             patch('asyncio.create_subprocess_exec', new=AsyncMock()) as mock_exec, \
             patch.object(test_job_with_config, '_log_stream_to_file', new=AsyncMock()) as mock_log:

            # Setup mocks
            mock_get_cmd.return_value = ('executable', ['arg1', 'arg2'])
            process_mock = AsyncMock()
            process_mock.wait.return_value = 0
            mock_exec.return_value = process_mock

            # Call _execute_batch
            commands = ["echo 'test'", "ls -la"]
            result = await test_job_with_config._execute_batch(commands)

            # Verify batch file was created
            assert test_job_with_config._batch.file.exists()

            # Verify command was executed
            mock_exec.assert_called_once()
            mock_log.assert_called()
            assert result.success

            # Clean up
            test_job_with_config._batch.file.unlink()

    @pytest.mark.asyncio
    async def test_execute_batch_no_config(self, test_job):
        """Test _execute_batch with missing batch configuration."""
        with pytest.raises(ValueError) as excinfo:
            await test_job._execute_batch(["echo 'test'"])
        assert "batch settings" in str(excinfo.value)

    def test_get_batch_execution_command(self, test_job_with_config):
        """Test _get_batch_execution_command functionality."""
        mock_batch_file = MagicMock(spec=BatchFile)
        mock_batch_file.get_execute_command.return_value = ["executable", "arg1", "arg2"]

        executable, args = test_job_with_config._get_batch_execution_command(mock_batch_file)

        assert executable == "executable"
        assert args == ["arg1", "arg2"]
        mock_batch_file.get_execute_command.assert_called_once_with(arguments=None, as_list=True)

    @pytest.mark.asyncio
    async def test_log_stream_to_file(self, test_job, temp_dir):
        """Test _log_stream_to_file functionality."""
        # Create a test file
        test_file = temp_dir / "test_log.txt"

        # Create a mock stream
        mock_stream = AsyncMock()
        mock_stream.read.side_effect = [b"line1\n", b"line2\n", b""]

        # Call _log_stream_to_file
        await test_job._log_stream_to_file(mock_stream, test_file)

        # Verify file contents
        assert test_file.exists()
        content = test_file.read_bytes()
        assert content == b"line1\nline2\n"

        # Verify stream read calls
        assert mock_stream.read.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_batch_custom_logs(self, test_job_with_config, temp_dir):
        """Test _execute_batch with custom log paths."""
        with patch.object(test_job_with_config, '_get_batch_execution_command') as mock_get_cmd, \
             patch('asyncio.create_subprocess_exec', new=AsyncMock()) as mock_exec, \
             patch.object(test_job_with_config, '_log_stream_to_file', new=AsyncMock()) as mock_log:

            # Setup mocks
            mock_get_cmd.return_value = ('executable', ['arg1', 'arg2'])
            process_mock = AsyncMock()
            process_mock.wait.return_value = 0
            mock_exec.return_value = process_mock

            # Custom log paths
            custom_stdout = temp_dir / "custom_stdout.log"
            custom_stderr = temp_dir / "custom_stderr.log"

            # Call _execute_batch with custom log paths
            commands = ["echo 'test'"]
            await test_job_with_config._execute_batch(commands, stdout_log=custom_stdout, stderr_log=custom_stderr)

            # Verify log files were used
            mock_log.assert_any_call(process_mock.stdout, custom_stdout)
            mock_log.assert_any_call(process_mock.stderr, custom_stderr)
