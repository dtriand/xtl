import abc
import asyncio
import contextlib
import contextvars
import logging
from typing import AsyncIterator, Literal, overload

from xtl import settings
from xtl.jobs.jobs import Job, JobResults
from xtl.jobs.logging import get_logger_config
from xtl.jobs.resources import Resources, ResourcesLease, ResourceManager, get_rc_manager
from xtl.jobs.submissions2 import JobSubmission
from xtl.logging.config import LoggerConfig
from xtl.math.uuid import UUIDFactory


uuid = UUIDFactory()


class BasePool(abc.ABC):

    _logging_level = logging.INFO
    _logging_config: LoggerConfig = get_logger_config(level=_logging_level)

    def __init__(self, name: str | None = None, max_jobs: int = 1, logger_config: LoggerConfig = None,
                 resources: Resources | None = None, resources_manager: ResourceManager | None = None, **kwargs):
        if max_jobs < 1:
            raise ValueError(f'`max_jobs` must be at least 1, got {max_jobs}')
        if logger_config and not isinstance(logger_config, LoggerConfig):
            raise TypeError(f'Expected a {LoggerConfig.__name__} instance for `logger_config`, '
                            f'got {type(logger_config).__name__}')

        # Resources allocation
        self._rc_requested: Resources = resources or Resources(jobs=max_jobs, threads=1, processes=1, cores=1)
        self._rc_manager: ResourceManager | None = resources_manager
        self._rc_lease: ResourcesLease | None = None
        self._resources: Resources | None = None

        # Set up pool_id and name
        self._pool_id = uuid.random(settings.jobs.job_digits)
        self._name = name

        # Configure logger for this pool
        self._logger = self.get_logger(
            logger_id=f'{self.__class__.__name__}|{self._name or self._pool_id}',
            config=logger_config or self._logging_config
        )

        # Initialize inter-process communication
        self._ipc = None

        # Context manager
        self._in_ctx: bool = False  # Managed by __enter__ and __exit__

        # Job tracking
        self._jobs: dict[str, Job] = {}
        self._running_jobs: set[str] = set()
        self._completed_jobs: set[str] = set()

        # Task tracking
        self._tasks: dict[str, asyncio.Task] = {}  # Managed by __exit__ and launchers

        # Submission tracking
        self._submissions: dict[str, JobSubmission] = {}  # Managed by submit() and launchers

        # Concurrency control
        self._semaphore: asyncio.Semaphore | None = None

        # Pool state
        self._is_running = False  # Managed by launcher methods
        self._error: Exception | None = None

    @property
    def pool_id(self) -> str:
        """
        Unique identifier for this pool.
        """
        return self._pool_id

    @property
    def name(self) -> str | None:
        """
        Name of this pool.
        """
        return self._name

    @property
    def resources(self) -> Resources | None:
        """
        The resources granted to this pool. Allocation happens within the context manager.
        """
        return self._resources

    @property
    def max_jobs(self) -> int:
        """
        Get the maximum number of concurrent jobs allowed in this pool.
        """
        if self._resources is None:
            return self._rc_requested.jobs
        return self._resources.jobs

    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger associated with this pool.
        """
        return self._logger

    @classmethod
    def get_logger(cls, logger_id: str, config: LoggerConfig = None, update_config: bool = True) -> logging.Logger:
        """
        Get or create a logger for the pool with the specified ID. If a `config` is not specified, the default
        configuration is chosen.

        :param logger_id: The ID for the logger (e.g., pool name)
        :param config: Optional logging configuration.
        :param update_config: If True, update the class-level logging configuration with the provided config.
        """
        logger = logging.getLogger(str(logger_id))

        if config is not None:
            if not isinstance(config, LoggerConfig):
                raise TypeError(f'Expected a {LoggerConfig.__name__} instance, got {type(config).__name__}')
            config.configure(logger)
            if update_config:
                cls._logging_config = config
        else:
            cls._logging_config.configure(logger)
        return logger

    async def __aenter__(self):
        """
        Enter the context manager.
        """
        if self._in_ctx:
            raise RuntimeError(f'{self.__class__.__name__} is already in a context manager and cannot be entered '
                               f'multiple times')

        # Acquire resources for this pool
        self.logger.debug('Requesting resources for the pool: %s', self._rc_requested)
        if self._rc_manager is None:
            self._rc_manager = get_rc_manager()
        self._rc_lease = await self._rc_manager.acquire(self._rc_requested)
        self._resources = self._rc_lease.granted
        self._semaphore = asyncio.Semaphore(self._resources.jobs)

        self._in_ctx = True
        self.logger.debug('Entering the pool context manager')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and handle any exceptions.
        """
        keyboard_interrupt = exc_type is KeyboardInterrupt or \
                             exc_type is asyncio.CancelledError
        # Exception handling
        if exc_val:
            # We use logger.error here instead of logger.exception to avoid
            #   printing the traceback
            if keyboard_interrupt:
                self.logger.error('Pool execution was interrupted by user')
            else:
                self.logger.error('An exception occurred within the pool context: %s',
                              exc_type.__name__, exc_info=settings.jobs.tracebacks)
            self._error = exc_val

            # Request cancellation of all running tasks
            self.logger.warning('Cancelling all running tasks')
            for job_id, task in list(self._tasks.items()):
                if keyboard_interrupt:
                    task.cancel('Execution interrupted by user')
                else:
                    task.cancel('An exception occurred in the pool context')

            self._in_ctx = False
            await self._drain_pool()
            if keyboard_interrupt:
                return True  # Suppress the exception
            return False  # Propagate other exceptions

        # Normal exit
        self._in_ctx = False
        await self._drain_pool()
        self.logger.debug('Exiting the pool context')
        return True

    async def _drain_pool(self):
        self._tasks.clear()

        # Release job ids from Job._registry, to allow reusing predictable job ids, e.g. Job1, Job2
        for job in self._jobs.values():
            with contextlib.suppress(Exception):
                job.clear()
        self._jobs.clear()
        self._submissions.clear()

        # Release resources
        if self._rc_lease is not None:
            await self._rc_lease.release()
            self._rc_lease = None
        self._resources = None
        self._semaphore = None

    def submit(self, job_cls, configs):
        ...

    @abc.abstractmethod
    def _execute_submission(self, submission) -> JobResults: ...

    async def _run_job(self, job) -> JobResults: ...

    @overload
    async def launch(self, results: Literal['stream']) -> AsyncIterator[JobResults]: ...

    @overload
    async def launch(self, results: Literal['all']) -> list[JobResults]: ...

    async def launch(self, results: Literal['stream', 'all'] = 'all') -> AsyncIterator[JobResults] | list[JobResults]:
        ...

    async def _launch_all(self) -> list[JobResults]:
        ...

    async def _launch_stream(self) -> AsyncIterator[JobResults]:
        ...