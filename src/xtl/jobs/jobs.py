import abc
import asyncio
from dataclasses import dataclass
import logging
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar, TYPE_CHECKING, \
    get_args, Iterable

if TYPE_CHECKING:
    from xtl.jobs.pools import JobPool
    from xtl.jobs.batchfiles import BatchFile as BatchFile2

from xtl import settings, Logger
from xtl.jobs.config import _BaseJobConfig, JobConfig
from xtl.jobs.config2 import BatchJobConfig
from xtl.math.uuid import UUIDFactory
from xtl.logging.config import LoggerConfig, StreamHandlerConfig, LoggingFormat
from xtl.exceptions.base import SubprocessError, StderrError

uuid = UUIDFactory()
logger_ = Logger(__name__)


class _DummyPool:
    """
    A dummy context manager for job execution outside of a pool.
    """
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@dataclass
class JobResults:
    """
    Dataclass to hold the results of a single job.
    """
    job_id: str
    """The unique identifier of the job."""

    data: Any | None = None
    """Optional data returned by the job."""

    error: Any | None = None
    """Error that occurred during job execution, if any."""

    @property
    def success(self) -> bool:
        """
        Whether the job completed successfully without errors.
        """
        return self.error is None


BaseJobConfigType = TypeVar('BaseJobConfigType', bound=_BaseJobConfig)
JobConfigType = TypeVar('JobConfigType', bound=JobConfig)
BatchJobConfigType = TypeVar('BatchJobConfigType', bound=BatchJobConfig)


class Job(abc.ABC, Generic[JobConfigType]):
    _registry: ClassVar[dict[str, 'Job']] = {}
    """Registry of all alive jobs of this class."""

    # Note that this class variable is updated in __init_subclass__ when the subclass
    #  is defined with a generic parameter, e.g., Job[Config].
    _config_class: ClassVar[Type[JobConfig]] = JobConfig
    """The configuration class for this job type."""

    _logging_level: ClassVar[int] = logging.INFO
    """Logging level for jobs."""

    _logging_config: ClassVar[LoggerConfig] = LoggerConfig(
        level=_logging_level,
        propagate=False,
        handlers=[
            StreamHandlerConfig(
                format=LoggingFormat(
                    format='[%(asctime)s.%(msecs)03d:%(name)s] %(message)s',
                    datefmt='%H:%M:%S'
                )
            )
        ]
    )
    """Logging configuration for jobs."""

    _job_prefix: ClassVar[str] = 'xtl_job'
    """Job prefix"""

    _dependencies: ClassVar[set[str]] = set()
    """"External dependencies required for batch execution."""

    def __init__(self, job_id: str | None = None, logger: 'logging.Logger' = None):
        """
        Abstract base class for asynchronous jobs execution.

        :param job_id: Optional, a unique identifier for the job. If not provided,
            a unique ID will be generated.
        :param logger: Optional, a custom logger for the job. If not provided,
            a logger will be created using `job_id`.
        """
        # Create a unique job ID
        self._job_id = str(job_id) if job_id else \
            uuid.random(length=settings.automate.job_digits)
        while self._job_id in self._registry:
            # Regenerate if necessary
            logger_.debug('Regenerating job_id: %s', self._job_id)
            self._job_id = uuid.random(length=settings.automate.job_digits)

        # Attach a logger
        self._logger = logger or self.get_logger(self.job_id)

        # Register the job in the class registry
        self.__class__._registry[self._job_id] = self

        # Job state
        self._is_running = False
        self._is_complete = False
        self._error: Exception | None = None

        # Pool integration
        self._pool: Optional['JobPool'] = None

        # Initialize config
        self._config: JobConfigType | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check if the class was defined with a generic parameter
        config_class = None
        if hasattr(cls, '__orig_bases__'):
            # Iterate over the base classes
            for base in cls.__orig_bases__:
                # Check if the base class is a generic type, since Job is generic
                if hasattr(base, '__origin__'):
                    # Check if the base class is Job or a subclass of Job (e.g. BatchJob)
                    if issubclass(base.__origin__, Job):
                        # Get the type arguments of the Job generic
                        args = get_args(base)
                        if args and len(args) > 0:
                            # The first argument should be the JobConfig type
                            config_class = args[0]
                            break

        if config_class is not None:
            # Check if the config_class inherits from _BaseJobConfig
            if not issubclass(config_class, _BaseJobConfig):
                raise TypeError(f'{JobConfigType.__name__} must be a subclass of '
                                f'{JobConfig.__name__}, '
                                f'got {config_class.__name__}')
            # Update the _config_class of the subclass with the new config type
            cls._config_class = config_class
        else:
            # Default to JobConfig if no specific type is provided
            cls._config_class = JobConfig

    def configure(self, config: JobConfigType) -> None:
        """
        Pass configuration to the job.
        """
        config_class = self.__class__._config_class
        if not isinstance(config, config_class):
            raise TypeError(f'Expected config of type {config_class.__name__}, '
                            f'got {type(config).__name__}')
        self._config = config

    @property
    def config(self) -> JobConfigType | None:
        """
        Get the configuration of the job.
        """
        return self._config

    @classmethod
    def with_config(cls, config: JobConfigType = None, **kwargs) -> \
            'Job[JobConfigType]':
        """
        Create a preconfigured job instance.
        """
        if not isinstance(config, cls._config_class):
            raise TypeError(f'Expected config of type {cls._config_class.__name__}, '
                            f'got {type(config).__name__}')

        # Extract init parameters
        job_id = kwargs.pop('job_id', None)
        logger = kwargs.pop('logger', None)

        # Create job instance
        job = cls(job_id=job_id, logger=logger)

        # Set configuration if provided
        if config is not None:
            job.configure(config)

        return job

    @classmethod
    def map(cls, configs: Iterable[JobConfigType] | JobConfigType = None) -> \
            tuple['Job[JobConfigType]', ...]:
        """
        Map a list of configurations to job instances.
        """
        if not isinstance(configs, Iterable):
            if not isinstance(configs, cls._config_class):
                raise TypeError(f'Expected a list of {cls._config_class.__name__}, '
                                f'got {type(configs).__name__}')
            configs = (configs,)
        return tuple(cls.with_config(config) for config in configs)

    @property
    def job_id(self) -> str:
        """
        Get the unique identifier of the job.
        """
        return self._job_id

    @property
    def is_running(self) -> bool:
        """
        Check if the job is currently running.
        """
        return self._is_running

    @property
    def is_complete(self) -> bool:
        """
        Check if the job execution has completed.
        """
        return self._is_complete

    @property
    def pool(self) -> Optional['JobPool']:
        """
        Get the job pool associated with this job, if any.
        """
        return self._pool

    @pool.setter
    def pool(self, pool: Optional['JobPool']) -> None:
        # Avoid circular import by checking class name instead of importing JobPool
        if pool is not None and pool.__class__.__name__ != 'JobPool':
            raise TypeError(f'Expected a JobPool instance, '
                            f'got {type(pool).__name__}')
        self._pool = pool

    def clear(self) -> None:
        """
        Explicitly remove this job from the registry.
        """
        if self._job_id in self.__class__._registry:
            del self.__class__._registry[self._job_id]

    def __del__(self) -> None:
        self.clear()

    @abc.abstractmethod
    async def _execute(self) -> Any | None:
        """
        The actual task of the job. Needs to be implemented by subclasses.

        :returns: Optional, any data to be included in the JobResults.
        """
        ...

    async def run(self) -> JobResults | None:
        """
        Run the job asynchronously.
        """
        if self._is_running:
            self._logger.warning("Job is already running")
            return None

        if self._is_complete:
            self._logger.warning("Job has already completed")
            return None

        result = None
        try:
            self._is_running = True
            self._logger.debug('Launching job')
            result = await self._execute()
            self._is_complete = True
            self._logger.debug('Job completed successfully')
        except asyncio.CancelledError as e:
            self._error = e
            if e.args:
                self._logger.warning('Job cancellation requested with reason: %s',
                                     e.args[0])
            else:
                self._logger.warning('Job was cancelled')
            raise
        except Exception as e:
            self._error = e
            if isinstance(e, SubprocessError):
                self._logger.error(
                    'Job failed with subprocess error: %s, cmd_args: %s', e.message, repr(e.command),
                    exc_info=settings.jobs.tracebacks
                )
            elif isinstance(e, StderrError):
                self._logger.error(
                    'Job failed with stderr error: %s', repr(e.stderr),
                    exc_info=settings.jobs.tracebacks
                )
            else:
                self._logger.error(
                    'Job failed due to an exception: %s', str(e),
                    exc_info=settings.jobs.tracebacks
                )
        finally:
            self._is_running = False
            if self._error:
                self._logger.debug('Job aborted successfully')

        return JobResults(job_id=self._job_id, data=result, error=self._error)

    async def lock(self):
        """
        Context manager to acquire a lock during job execution.
        """
        if self._pool is None:
            # If no pool is set, return a dummy pool that does nothing
            return _DummyPool()
        # Use the pool's lock
        return self._pool.request_lock()

    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger associated with this job.
        """
        return self._logger

    @classmethod
    def get_logger(cls, job_id: str, config: LoggerConfig = None,
                   update_config: bool = True) -> logging.Logger:
        """
        Get or create a logger for the job with the given ID. If `config` is not
        specified, the default configuration is chosen.

        :param job_id: Unique identifier of the job.
        :param config: Optional logging configuration.
        :param update_config: If True, update the class-level logging configuration
            with the provided config.
        """
        # Recover existing loggers
        if job_id in cls._registry:
            return cls._registry[job_id].logger

        # Cast job_id to string
        if not isinstance(job_id, str):
            job_id = str(job_id)

        # Create and configure new logger
        logger = logging.getLogger(job_id)
        if config is not None:
            if not isinstance(config, LoggerConfig):
                raise TypeError(f'Expected a {LoggerConfig.__name__} instance, '
                                f'got {type(config).__name__}')
            config.configure(logger)
            if update_config:
                # This is required for propagating log configs to subjobs
                cls._logging_config = config
        else:
            cls._logging_config.configure(logger)

        return logger


class BatchJob(Job[BatchJobConfig], Generic[BatchJobConfigType]):

    def __init__(self, job_id: str | None = None, logger: 'logging.Logger' = None):
        super().__init__(job_id=job_id, logger=logger)

        self._config: BatchJobConfig | None
        self._batch: Optional[BatchFile2] = None
        self._batch_args: list[str] = []
        self._batch_context: dict = {}

    @property
    def config(self) -> BatchJobConfig | None:
        """
        Get the configuration of the batch job.
        """
        return self._config

    async def _execute(self) -> Any | None:
        # Check if batch configuration is available
        if not self.config:
            raise ValueError('Job configuration does not include batch settings')

        # Create the batch directory if it doesn't exist
        if not self.config.job_directory.exists():
            try:
                self._logger.debug('Creating directory for batch file: %s',
                                   self.config.job_directory)
                self.config.job_directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self._logger.error('Failed to create batch directory: %s',
                                   self.config.job_directory)
                raise e

        # Create batch file
        self._logger.debug('Creating batch file in %s',
                           self.config.job_directory)
        self._batch = self.config.get_batch(context=self.context)

        # Save batch file
        try:
            self._batch.save(update_permissions=True)
            self._logger.debug('Batch file created: %s', self._batch.file)
        except OSError as e:
            self._logger.error('Failed to save batch file: %s', self._batch.file)
            raise e

        # Make sure log files exist
        try:
            self.config.stdout.touch(exist_ok=True)
            self.config.stderr.touch(exist_ok=True)
            self._logger.debug('Log files initialized: stdout=%s, stderr=%s',
                               self.config.stdout, self.config.stderr)
        except OSError as e:
            self._logger.error('Failed to create log files: stdout=%s, stderr=%s',
                               self.config.stdout, self.config.stderr)
            raise e

        # Execute the batch file
        try:
            await self._batch.execute(
                stdout=self.config.stdout,
                stderr=self.config.stderr,
                batch_args=self._batch_args,
            )
        except asyncio.CancelledError as e:
            await self._batch.cancel()
            self.logger.error('Batch execution was cancelled by the user')
            raise e

        # Handle result
        result = {'stdout': self.config.stdout.read_text(encoding='utf-8'),
                  'stderr': self.config.stderr.read_text(encoding='utf-8')}

        self.logger.debug('Batch execution completed successfully')

        return result

    @classmethod
    def with_config(cls, config: BatchJobConfigType = None, **kwargs) -> \
            'BatchJob[BatchJobConfigType]':
        job = super().with_config(config=config, **kwargs)
        # Assign any additional kwargs that were not popped in the parent method
        #  as context for rendering batch file templates.
        job._batch_args = kwargs.pop('batch_args', [])
        job._batch_context |= kwargs
        return job

    @property
    def context(self) -> dict:
        """
        Get the context for rendering batch file templates.
        """
        return self._batch_context
