import abc
import asyncio
from dataclasses import dataclass
import logging
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar, TYPE_CHECKING, \
    get_args, Iterable

if TYPE_CHECKING:
    from xtl.jobs.pools import JobPool

from xtl import settings, Logger
from xtl.math.uuid import UUIDFactory
from xtl.logging.config import LoggerConfig, StreamHandlerConfig, LoggingFormat


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
    """Error that occured during job execution, if any."""

    @property
    def success(self) -> bool:
        """
        Whether the job completed successfully without errors.
        """
        return self.error is None


@dataclass
class JobConfig:
    """
    Base class for passing configuration to a job.
    """
    ...


JobConfigType = TypeVar('JobConfigType', bound=JobConfig)


class Job(abc.ABC, Generic[JobConfigType]):
    _registry: ClassVar[dict[str, 'Job']] = {}
    """Registry of all alive jobs of this class."""

    # Note that this class variable is updated in __init_subclass__ when the subclass
    #  is defined with a generic parameter, e.g., Job[Config].
    _config_class: ClassVar[Type[JobConfig]] = JobConfig
    """The configuration class for this job type."""

    _logging_level: ClassVar[int] = logging.INFO
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

    _job_prefix: ClassVar[str] = 'xtl_job'

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
                    # Check if the base class is Job
                    if base.__origin__ is Job:
                        # Get the type arguments of the Job generic
                        args = get_args(base)
                        if args and len(args) > 0:
                            # The first argument should be the JobConfig type
                            config_class = args[0]
                            break

        if config_class is not None:
            # Check if the config_class inherits from JobConfig
            if not issubclass(config_class, JobConfig):
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

    def __del__(self) -> None:
        del self.__class__._registry[self._job_id]

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
            self._logger.error('Job failed due to an exception %s', e.args)
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
    def get_logger(cls, job_id: str, config: LoggerConfig = None) -> logging.Logger:
        """
        Get or create a logger for the job with the given ID. If `config` is not
        specified, the default configuration is chosen.

        :param job_id: Unique identifier of the job.
        :param config: Optional logging configuration.
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
        else:
            cls._logging_config.configure(logger)

        return logger
