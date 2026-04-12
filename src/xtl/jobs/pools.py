from __future__ import annotations

import abc
import asyncio
import contextlib
from enum import Enum
import logging
from logging.handlers import QueueHandler
import multiprocessing
import re
import threading
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor
from typing import AsyncIterator, Literal, overload, Type, Iterable, Protocol, runtime_checkable

from xtl import settings
from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.misc import is_picklable
from xtl.jobs.config import JobConfig
from xtl.jobs.jobs import Job
from xtl.jobs.results import JobResults
from xtl.jobs.logging import get_logger_config
from xtl.jobs.ipc import (IPCBackend, IPCHandle, IPCHandleNames, IPCLock, IPCQueue, IPCState, AsyncIPCBackend,
                          ThreadedIPCBackend, ProcessIPCBackend)
from xtl.jobs.resources import Resources, ResourcesLease, ResourceManager, get_rc_manager
from xtl.jobs.submissions import JobSubmission
from xtl.logging.config import LoggerConfig
from xtl.math.uuid import UUIDFactory

if PY310_OR_LESS:
    class StrEnum(str, Enum):
        pass

    from typing_extensions import Self
else:
    from enum import StrEnum
    from typing import Self


__all__ = ['PoolProtocol', 'JobPool', 'BasePool', 'SimplePool', 'AsyncPool', 'ThreadedPool', 'MultiprocessPool']


uuid = UUIDFactory()


class JobPool(StrEnum):
    """
    Enum for available job pools.
    """

    SIMPLE = 'simple'
    ASYNC = 'async'
    THREADS = 'threads'
    PROCESSES = 'processes'

    def get(self) -> Type[BasePool]:
        """
        Get the class of the requested job pool.
        """
        if self == JobPool.SIMPLE:
            return SimplePool
        if self == JobPool.ASYNC:
            return AsyncPool
        if self == JobPool.THREADS:
            return ThreadedPool
        if self == JobPool.PROCESSES:
            return MultiprocessPool
        raise ValueError(f'Unsupported {JobPool.__name__} type: {self!r}')


@runtime_checkable
class PoolProtocol(Protocol):
    """
    Protocol for pools.
    """

    def get_lock(self, name: str | None = None) -> IPCLock: ...
    def get_queue(self, name: str) -> IPCQueue: ...
    def get_state(self, name: str) -> IPCState: ...

    @property
    def pool_id(self) -> str: ...


class BasePool(PoolProtocol, abc.ABC):

    _logging_level = logging.INFO
    """Default logging level for the pool's logger configuration. Setting after initialization has no effect"""

    _logger_config: LoggerConfig = get_logger_config(level=_logging_level)
    """Default logging config for the pool's logger."""

    _executor_cls: Executor | None = None
    """The executor class to use for this pool. If None, the pool does not use an executor and runs jobs in the 
    main thread."""

    _ipc_cls: type[IPCBackend]
    """The IPCBackend class to use for this pool."""

    def __init__(self, name: str | None = None, max_jobs: int = 1, logger_config: LoggerConfig = None,
                 resources: Resources | None = None, resources_manager: ResourceManager | None = None, **kwargs):
        """
        Abstract base class for job pools.

        :param name: Optional name for pool
        :param max_jobs: The maximum number of concurrent jobs to run. This is capped by the available quota
            (default: 1).
        :param logger_config: Optional configuration for the pool's logger.
        :param resources: Optional resources to request for this pool. This is capped by the available quota
            (default: None, get all available)
        :param resources_manager: Optional ResourceManager to use for resource allocation. If not provided, the default
            global ResourceManager will be used.
        :param kwargs: Additional keyword arguments:
            - `job_logger_config`: Configuration for the Job loggers (default: same as pool)
        :raises ValueError: If an invalid `max_jobs` is provided
        :raises TypeError: If `logger_config` is not of the correct type
        """
        if max_jobs < 1:
            raise ValueError(f'`max_jobs` must be at least 1, got {max_jobs}')
        if logger_config and not isinstance(logger_config, LoggerConfig):
            raise TypeError(f'Expected a {LoggerConfig.__name__} instance for `logger_config`, '
                            f'got {type(logger_config).__name__}')

        # Set up pool_id and name
        self._pool_id = uuid.random(settings.jobs.job_digits)
        self._name = name

        # Configure logger for this pool
        self._logger = self.get_logger(
            logger_id=f'{self.__class__.__name__}|{self._name or self._pool_id}',
            config=logger_config or self._logger_config
        )
        self._job_logger_config = kwargs.get('job_logger_config', self._logger_config)

        # Resources allocation
        self._rc_requested: Resources = self._resolve_resources(max_jobs=max_jobs, requested=resources)
        self._rc_manager: ResourceManager | None = resources_manager
        self._rc_lease: ResourcesLease | None = None
        self._resources: Resources | None = None

        # Pool executor backend
        self._executor = None  # Managed by __aenter__ and __aexit__

        # Initialize inter-process communication
        self._ipc: IPCBackend | None = None  # Managed by __aenter__ and __aexit__
        self._ipc_handle_names: IPCHandleNames = IPCHandleNames()

        # Concurrency control
        self._semaphore: asyncio.Semaphore | None = None

        # Context manager
        self._in_ctx: bool = False  # Managed by __aenter__ and __aexit__

        # Job tracking
        self._jobs: dict[str, Job] = {}
        self._running_jobs: set[str] = set()
        self._completed_jobs: set[str] = set()

        # Task tracking
        self._tasks: dict[str, asyncio.Task] = {}  # Managed by __aexit__ and launchers

        # Submission tracking
        self._submissions: dict[str, JobSubmission] = {}  # Managed by submit() and launchers

        # Pool state
        self._is_running = False  # Managed by launcher methods
        self._error: Exception | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_ipc_cls') or not issubclass(cls._ipc_cls, IPCBackend):
            raise TypeError(f'Subclasses of {BasePool.__name__} must define an `_ipc_cls` attribute that is a '
                            f'subclass of {IPCBackend.__name__}')

    @abc.abstractmethod
    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        """
        Translate a ``max_jobs`` count and an optional explicit ``Resources`` request in the
        resources to request from the ``ResourceManager``. Subclasses define their own
        behaviour depending on their intent.
        """
        ...

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
    def get_logger(cls, logger_id: str, config: LoggerConfig | None = None, update_config: bool = True) -> logging.Logger:
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
                cls._logger_config = config
        else:
            cls._logger_config.configure(logger)
        return logger

    async def __aenter__(self) -> Self:
        """
        Enter the context manager.
        """
        if self._in_ctx:
            raise RuntimeError(f'{self.__class__.__name__} is already in a context manager and cannot be entered '
                               f'multiple times')

        # Acquire resources for this pool
        self.logger.debug('Requesting resources for the pool: %s', self._rc_requested.__dict__)
        if (manager := self._rc_manager) is None:
            manager = get_rc_manager()
            self._rc_manager = manager

        lease = await manager.acquire(self._rc_requested)
        self._rc_lease = lease
        self._resources = lease.granted
        self.logger.debug('Resources granted for the pool: %s', lease.granted.__dict__)
        self._semaphore = asyncio.Semaphore(lease.granted.jobs)

        # Start the IPC backend
        ipc = self._ipc_cls()
        ipc.start()
        self._ipc = ipc

        self._in_ctx = True
        self.logger.debug('Entering the pool context manager')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and handle any exceptions.
        """
        interrupted = bool(exc_type and issubclass(exc_type, (KeyboardInterrupt, asyncio.CancelledError)))
        # Exception handling
        if exc_val:
            # We use logger.error here instead of logger.exception to avoid
            #   printing the traceback
            if interrupted:
                self.logger.error('Pool execution was interrupted by user')
            else:
                self.logger.error('An exception occurred within the pool context: %s',
                              exc_type.__name__, exc_info=settings.jobs.tracebacks)
            self._error = exc_val

            # Request cancellation of all running tasks
            self.logger.warning('Cancelling all running tasks')
            for job_id, task in list(self._tasks.items()):
                if interrupted:
                    task.cancel('Execution interrupted by user')
                else:
                    task.cancel('An exception occurred in the pool context')

            # Await cancellation
            pending = [t for t in self._tasks.values() if not t.done()]
            if pending:
                self.logger.warning('Waiting for %d tasks to cancel', len(pending))
                done, pending = await asyncio.wait(pending, timeout=5)
                if pending:
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

            self._in_ctx = False
            await self._drain_pool()
            if interrupted:
                return True  # Suppress the exception
            return False  # Propagate other exceptions

        # Normal exit
        self._in_ctx = False
        await self._drain_pool()
        self.logger.debug('Exiting the pool context')
        return True

    async def _drain_pool(self):
        """
        Pool tear down procedure.
        """
        self._tasks.clear()

        # Release job ids from Job._registry, to allow reusing job ids
        #  NB: This is especially needed when using predictable ids, such as Job1, Job2, etc.
        for job in self._jobs.values():
            with contextlib.suppress(Exception):
                job.clear()
        self._jobs.clear()
        self._submissions.clear()

        # Tear down IPC backend
        if self._ipc is not None:
            self._ipc.stop()
            self._ipc = None
        self._ipc_handle_names = IPCHandleNames()

        # Release resources
        if self._rc_lease is not None:
            self.logger.debug('Releasing resources from pool: %s', self._rc_lease.granted.__dict__)
            await self._rc_lease.release()
            self._rc_lease = None
        self._resources = None
        self._semaphore = None

    def _generate_job_ids(self, job_cls: Type[Job], n: int) -> list[str]:
        """
        Generate a list of `n` unique sequential job ids while preventing semantic collisions upon multiple calls for
        the same `job_cls`.

        :param job_cls: the job class to generate job ids for
        :param n: the number of job ids to generate
        """
        prefix = f'{job_cls.__name__}|'
        pattern = re.compile(rf'^{re.escape(prefix)}(\d+)$')

        # Find the indices of all existing job ids
        taken: set[int] = set()
        for job in self._jobs.values():
            if not isinstance(job, job_cls):
                # Skip for different Job types
                continue
            m = pattern.match(job.job_id)
            if not m:
                # Skip for randomized or user-provided job_ids
                continue
            taken.add(int(m.group(1)))

        digits = len(str(max(1, n)))

        def make_job_id(x: int) -> str:
            return f'{prefix}{str(x).zfill(digits)}'

        # Generate new job_ids
        job_ids: list[str] = []
        i = 1
        for _ in range(n):
            job_id = make_job_id(i)
            while i in taken:
                i += 1
                job_id = make_job_id(i)
            job_ids.append(job_id)
            taken.add(i)
            i += 1
        return job_ids


    def submit(self, job_cls: Type[Job], configs: JobConfig | None | Iterable[JobConfig | None] = None, **kwargs) -> \
            list[Job]:
        """
        Submit a job to the pool for execution. This method must be called from within the pool's context manager.

        Example:

            .. code-block:: python

                async with pool:
                    pool.submit(MyJob, configs=[config1, config2])
                    results = pool.launch()

        :param job_cls: The job type to submit. Must be a subclass of `Job`.
        :param configs: The configs to submit to the pool.
        :param kwargs: Additional keyword arguments to pass to `job_cls`.
            - `job_ids`: Optional list of job IDs to assign to the submitted jobs. If not provided, job IDs will be
                generated in the format `<job_cls>|<i>`, where i is an incrementing index
        :raises RuntimeError: If the method is called outside the pool's context manager
        :raises TypeError: If `job_cls` is not a subclass of `Job`.
        :raises TypeError: If one of the `configs` is not a `JobConfig` instance or None.
        :raises ValueError: If `job_ids` is provided but does not have the same length as `configs`
        :return: A list of the configured submitted jobs.
        """
        if not self._in_ctx:
            raise RuntimeError(f'Jobs submitted outside the pool context')

        if not issubclass(job_cls, Job):
            raise TypeError(f'Expected a subclass of {Job.__name__} for `job_cls`, got {job_cls}')

        # Cast configs to a list
        if configs is None:
            configs_list: list[JobConfig | None] = [None]
        elif isinstance(configs, JobConfig):
            configs_list = [configs]
        elif isinstance(configs, Iterable) and not isinstance(configs, (str, bytes, bytearray)):
            configs_list = list(configs)
        else:
            raise TypeError(f'Invalid `configs` type: {type(configs).__name__}')

        # Generate job IDs if not provided
        if (job_ids := kwargs.get('job_ids', None)) is None:
            job_ids = self._generate_job_ids(job_cls, len(configs_list))
        job_ids = list(job_ids)

        # Check for length consistency
        if len(job_ids) != len(configs_list):
            raise ValueError(f'Length of `job_ids` ({len(job_ids)}) must match length of `configs` ({len(configs_list)})')

        # Create and register jobs
        created_jobs: list[Job] = []
        for job_id, config in zip(job_ids, configs_list):
            if job_id is None:
                # In case None was passed explicitly in job_ids
                job_id = uuid.random(settings.jobs.job_digits)

            # Create logger
            logger = job_cls.get_logger(job_id, self._job_logger_config)

            # Create job instance
            job = job_cls(job_id=job_id, logger=logger)
            if config is not None:
                job.configure(config)
            job.pool = self

            # Create submission
            submission = JobSubmission.from_job(job)
            submission.ipc = self._build_ipc_handle()

            # Allocate resources
            job_request = job.get_resources()
            if job_request and self._resources:
                # Ensure the granted resources for the submission are capped by the pool's resources
                submission.resources = job_request.cap(self._resources)
            elif job_request:
                # If the pool has no limit, then grant whatever has been requested
                submission.resources = job_request
            else:
                # If the job has no requirement, then give everything the pool has
                submission.resources = self._resources

            # Register submission and job
            self._submissions[submission.submission_id] = submission
            self._jobs[job.job_id] = job
            created_jobs.append(job)
            self.logger.debug('Submitted job %s to %s', job.job_id, self.__class__.__name__)

        return created_jobs

    @abc.abstractmethod
    async def _execute_submission(self, submission: JobSubmission) -> JobResults | None:
        """
        Execute a job submission.

        :param submission: The submission to execute.
        """
        ...

    async def _process_submission(self, submission: JobSubmission) -> JobResults | None:
        """
        Process a job submission and ensure that the concurrency limit is respected.

        :param submission: The submission to execute.
        :raises RuntimeError: If the method is called outside the pool's context manager.
        :return: The result of the job execution.
        """
        if self._semaphore is None:
            raise RuntimeError(f'Pool semaphore is not initialized')

        async with self._semaphore:
            job_id = submission.data.job_id
            self._running_jobs.add(job_id)
            try:
                result = await self._execute_submission(submission)
                self._completed_jobs.add(job_id)
                return result
            except Exception as exc:
                self.logger.error('Job %s failed with exception: %s', job_id, exc,
                                  exc_info=settings.jobs.tracebacks)
                return JobResults(job_id=job_id, data=None, error=exc)
            finally:
                self._running_jobs.discard(job_id)

    @overload
    async def launch(self, mode: Literal['stream']) -> AsyncIterator[JobResults]: ...

    @overload
    async def launch(self, mode: Literal['all']) -> list[JobResults]: ...

    @overload
    async def launch(self) -> list[JobResults]: ...

    async def launch(self, mode: Literal['stream', 'all'] = 'all') -> AsyncIterator[JobResults] | list[JobResults]:
        """
        Launch all submitted jobs in the pool. By default, the results are returned only when all jobs have completed.
        If `mode` is set to `'stream'`, then the results are yielded as soon as each job completes.

        :param mode: How to return the results from the running jobs:
            - `'all'`: Return all results after all jobs have completed (default)
            - `'stream'`: Yield results as soon as they become available
        :raises RuntimeError: If called outside the pool's context manager.
        :raises ValueError: If an invalid `mode` is passed.
        :return: The results of the job execution.
        """
        if not self._in_ctx:
            raise RuntimeError(f'{self.__class__.__name__}.launch() must be used within the context manager.')
        if mode == 'all':
            return await self._launch_all()
        elif mode == 'stream':
            return self._launch_stream()
        raise ValueError(f'Invalid mode: {mode!r}. Expected \'all\' or \'stream\'.')

    async def _launch_all(self) -> list[JobResults]:
        """
        Launch all submitted jobs and return their aggregated results.
        """
        if not self._submissions:
            self.logger.warning('No jobs submitted to pool')
            return []

        # Activate the pool
        self._is_running = True
        self.logger.debug('Activating pool with %d submissions', len(self._submissions))
        try:
            # Create tasks
            for submission in self._submissions.values():
                task = asyncio.create_task(self._process_submission(submission))
                self._tasks[submission.submission_id] = task

            # Wait for all tasks to complete and gather results
            self.logger.debug('Launching jobs...')
            results = list(await asyncio.gather(*self._tasks.values()))
            self.logger.debug('All jobs completed')
            return results
        finally:
            # No exception handling at this stage, this is managed by __aexit__
            self.logger.debug('Deactivating pool')
            self._is_running = False

    async def _launch_stream(self) -> AsyncIterator[JobResults]:
        """
        Launch all submitted jobs and yield the results as they become available.
        """
        if not self._submissions:
            self.logger.warning('No jobs submitted to pool')
            return

        # Activate the pool
        self._is_running = True
        self.logger.debug('Activating pool with %d submissions', len(self._submissions))
        try:
            # Create tasks
            for submission in self._submissions.values():
                task = asyncio.create_task(self._process_submission(submission))
                self._tasks[submission.submission_id] = task

            self.logger.debug('Launching jobs...')
            for task in asyncio.as_completed(self._tasks.values()):
                self.logger.debug('A job has completed, yielding result...')
                yield await task
            self.logger.debug('All jobs completed')
        finally:
            # No exception handling at this stage, this is managed by __aexit__
            self.logger.debug('Deactivating pool')
            self._is_running = False

    def get_lock(self, name: str | None = None) -> IPCLock:
        """
        Get a lock from the pool. If the requested lock does not exist, it will be created.
        If a name is not specified, the default lock will be returned.

        :param name: The name of the lock to retrieve, or None for the default lock.
        :raises RuntimeError: If the pool is not currently active (i.e., not within the context manager).
        :return: An IPCLock instance corresponding to the requested lock.
        """
        if self._ipc is None:
            raise RuntimeError('Locks are only available from within the pool context manager')
        lock = self._ipc.get_lock(name)
        self._ipc_handle_names.locks.add(lock.name)
        return lock

    def get_queue(self, name: str, maxsize: int = 0) -> IPCQueue:
        """
        Get a queue from the pool. If the requested queue does not exist, it will be created.

        :param name: The name of the queue to retrieve.
        :param maxsize: The maximum size of the queue (default: 0 for unlimited).
        :raises RuntimeError: If the pool is not currently active (i.e., not within the context manager).
        :return: An IPCQueue instance corresponding to the requested queue.
        """
        if self._ipc is None:
            raise RuntimeError('Queues are only available from within the pool context manager')
        queue = self._ipc.get_queue(name, maxsize=maxsize)
        self._ipc_handle_names.queues.add(name)
        return queue

    def get_state(self, name: str) -> IPCState:
        """
        Get a shared state dict from the pool. If the requested state does not exist, it will be created.
        Per-operation atomicity is guaranteed. Any compound read-modify-write (RMW) operations (e.g. in-place
        modification) are NOT atomic and must be protected with an explicit lock.

        Example usage:

        .. code-block:: python

            state = pool.get_state('my_state')
            # Atomic operation
            state['counter'] = 0
            # Compound RMW - unsafe
            state['counter'] += 1  # 3 separate operations: get, modify, set - not atomic!
            # Compound RMW - safe with lock
            async with pool.get_lock():
                state['counter'] += 1  # Protected by lock, safe to perform compound RMW

        :param name: The name of the state to retrieve.
        :raises RuntimeError: If the pool is not currently active (i.e., not within the context manager).
        :return: An IPCState instance corresponding to the requested state.
        """
        if self._ipc is None:
            raise RuntimeError('Shared states are only available from within the pool context manager')
        state = self._ipc.get_state(name)
        self._ipc_handle_names.states.add(name)
        return state

    def _build_ipc_handle(self) -> IPCHandle | None:
        """
        Build a picklable IPCHandle from all registered IPC primitives.
        """
        if self._ipc is None:
            return None
        return self._ipc.to_handle(self._ipc_handle_names)


class ProxyPool(PoolProtocol):

    _PROXY_SENTINEL = '<proxy>'

    def __init__(self, ipc: IPCBackend):
        """
        A minimal pool shim injected as Job.pool inside a worker process or thread.
        It exposes only the methods needed for a Job to run, backed by the ProxyIPCBackend.

        :param ipc: An IPCBackend instance.
        """
        self._ipc = ipc

    @property
    def pool_id(self) -> str:
        """
        Sentinel pool_id for logging. Returns \'<proxy>\' to indicate this is a
        worker-side shim rather than a live BasePool.
        """
        return self._PROXY_SENTINEL

    def get_lock(self, name: str | None = None) -> IPCLock:
        return self._ipc.get_lock(name)

    def get_queue(self, name: str, maxsize: int = 0) -> IPCQueue:
        return self._ipc.get_queue(name, maxsize=maxsize)

    def get_state(self, name: str) -> IPCState:
        return self._ipc.get_state(name)


class AsyncPool(BasePool):
    """
    A pool for running jobs asynchronously. The jobs are executed within the same process and same thread (and event
    loop). This is well suited for running I/O-bound task, e.g. fetching data over network, executing batch jobs. It is
    the most responsive pool type, since it does not add any IPC overhead.
    """

    _ipc_cls = AsyncIPCBackend

    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        return requested or Resources(jobs=max_jobs, threads=1, processes=1)

    async def _execute_submission(self, submission: JobSubmission) -> JobResults | None:
        job = self._jobs.get(submission.data.job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={submission.data.job_id!r}')
        return await job.run()


class SimplePool(AsyncPool):
    """
    A pool for running jobs synchronously. This is a thin wrapper around AsyncPool, that enforces a sequential job
    execution (`max_jobs=1`). Suitable for running a single job, or when concurrency is risky. It can also be used for
    debugging, since it enforces a deterministic operations order.
    """

    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        # Hard cap max_jobs to 1, since this pool is meant for sequential execution
        r = requested or Resources(jobs=1, threads=1, processes=1)
        return Resources(jobs=1, threads=r.threads, processes=r.processes)

    async def _launch_all(self) -> list[JobResults]:
        """
        Launch all submitted jobs and return their aggregated results.
        """
        if not self._submissions:
            self.logger.warning('No jobs submitted to pool')
            return []

        # Activate the pool
        self._is_running = True
        self.logger.debug('Activating pool with %d submissions', len(self._submissions))
        try:
            # Create tasks
            for submission in self._submissions.values():
                task = asyncio.create_task(self._process_submission(submission))
                self._tasks[submission.submission_id] = task

            # Wait for all tasks to complete and gather results
            self.logger.debug('Launching jobs...')
            results = []
            for task in self._tasks.values():
                results.append(await task)
            self.logger.debug('All jobs completed')
            return results
        finally:
            # No exception handling at this stage, this is managed by __aexit__
            self.logger.debug('Deactivating pool')
            self._is_running = False

    async def _launch_stream(self) -> AsyncIterator[JobResults]:
        if not self._submissions:
            self.logger.warning('No jobs submitted to pool')
            return

        # Activate the pool
        self._is_running = True
        self.logger.debug('Activating pool with %d submissions', len(self._submissions))
        try:
            # Create tasks
            for submission in self._submissions.values():
                task = asyncio.create_task(self._process_submission(submission))
                self._tasks[submission.submission_id] = task

            self.logger.debug('Launching jobs...')
            for task in self._tasks.values():
                yield await task
            self.logger.debug('All jobs completed')
        finally:
            # No exception handling at this stage, this is managed by __aexit__
            self.logger.debug('Deactivating pool')
            self._is_running = False


class ThreadedPool(BasePool):
    """
    A pool for running jobs in separate threads. The jobs are executed within the same process but in separate threads.
    This pool is well suited for running I/O-blocking tasks, e.g. non-async APIs, filesystem I/O. Not ideal for
    CPU-bound tasks, since Python (at least < 3.14) is not truly threaded. Internally, the spawned threads are managed
    by a ThreadPoolExecutor.
    """

    _executor_cls = ThreadPoolExecutor
    _executor: ThreadPoolExecutor | None
    _ipc_cls = ThreadedIPCBackend

    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        return requested or Resources(jobs=max_jobs, threads=max_jobs, processes=1)

    async def __aenter__(self):
        await super().__aenter__()
        self._executor = self._executor_cls(max_workers=self._resources.jobs)  # type: ignore
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        await super().__aexit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _rename_current_thread() -> None:
        """
        Rename the current thread to Thread<N>, where N is a unique index extracted from the original thread name.
        """
        thread = threading.current_thread()

        # Grab the trailing digits
        match = re.compile(r'.*?(\d+)$').match(thread.name)
        if match:
            idx = int(match.group(1))
        else:
            idx = 0

        # Rename the thread
        new_name = f'Thread{idx}'
        if thread.name != new_name:
            thread.name = new_name

    @staticmethod
    async def _bootstrap_thread(submission: JobSubmission) -> JobResults | None:
        # Initialize the worker thread's ResourceManager with the granted budget.
        get_rc_manager(total=submission.resources)

        # Deserialize the job
        job = submission.to_job()

        # Reconstruct the IPC backend
        if submission.ipc is not None:
            ipc = ThreadedPool._ipc_cls.from_handle(submission.ipc)
            job._pool = ProxyPool(ipc)

        return await job.run()

    @staticmethod
    def _run_in_thread(submission: JobSubmission) -> JobResults | None:
        ThreadedPool._rename_current_thread()
        return asyncio.run(ThreadedPool._bootstrap_thread(submission))

    async def _execute_submission(self, submission: JobSubmission) -> JobResults | None:
        job_id = submission.data.job_id
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={job_id!r}')

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_in_thread, submission)


class MultiprocessPool(BasePool):
    """
    A pool for running jobs in separate processes. This pool is well suited for running CPU-bound tasks, e.g. heavy
    numerical calculations. It can also be used to impose stronger isolation between worker processes. It has the
    highest overhead among the pool types upon initialization. Internally, the spawned subprocesses are managed by
    a ProcessPoolExecutor.
    """

    _executor_cls = ProcessPoolExecutor
    _executor: ProcessPoolExecutor | None
    _ipc_cls = ProcessIPCBackend


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Eavesdropping task for capturing logs from worker processes
        self._eavesdropping: asyncio.Task | None = None  # Managed by __aexit__ and _ensure_picklable_log_handlers
        self._eavesdropping_qname = f'{self.pool_id}_log_queue'

    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        return requested or Resources(jobs=max_jobs, threads=1, processes=max_jobs)

    def _eavesdrop(self) -> None:
        """
        Start an asynchronous task that listens to the log queue for log records emitted by worker processes and handles
        them with the pool's logger. This allows logs from worker processes to be captured and displayed in the main
        process.
        """
        if self._eavesdropping is not None and not self._eavesdropping.done():
            # Eavesdropping is already active, no need to start another listener
            return

        # Get the log queue for this pool
        #  NB: The queue is first created in __aenter__, therefore is guaranteed to exist
        #      on every submit() call
        log_queue = self.get_queue(self._eavesdropping_qname)

        async def _listen() -> None:
            """
            Continuously listen for log records on the log queue and handle them with the pool's logger.
            """
            while True:
                record = await log_queue.get()
                if record is None:
                    # Sentinel value indicating shutdown
                    break
                if isinstance(record, dict):
                    record = logging.makeLogRecord(record)
                if isinstance(record, logging.LogRecord):
                    logging.getLogger(record.name).handle(record)

        # Start the eavesdropping task
        self.logger.debug('Starting eavesdropper...')
        self._eavesdropping = asyncio.create_task(
            _listen(),
            name=f'{self.__class__.__name__}:{self.pool_id}-Eavesdropper'
        )

    @contextlib.asynccontextmanager
    async def _ensure_picklable_log_handlers(self, submission: JobSubmission) -> AsyncIterator[None]:
        """
        Ensure job logging handlers are picklable for process execution. If no usable
        handlers remain (including the case handlers=[]), inject a dummy marker so
        the worker installs a QueueHandler and forwards logs to this pool.
        """
        popped_handlers: list[dict] = []
        dummy: dict | None = None

        # Ensure config structure exists
        if submission.data.logging_config is None:
            submission.data.logging_config = {}
        log_cfg = submission.data.logging_config
        handlers_configs = list(log_cfg.get('handlers', []))

        # Keep only picklable handlers; stash removed ones for restoration
        picklable_handlers: list[dict] = []
        for handler_config in handlers_configs:
            safe, _ = is_picklable(handler_config)
            if safe:
                picklable_handlers.append(handler_config)
                continue

            popped_handlers.append(handler_config)
            handler = handler_config.get('handler', '<unknown>')
            handler_name = getattr(handler, '__name__', str(handler))
            self.logger.debug(
                'Removed unpicklable log handler <%s> from submission of job: %s',
                handler_name,
                submission.data.job_id,
            )

        # Update submission to only contain picklable handlers for process handoff
        log_cfg['handlers'] = picklable_handlers

        # Critical: if no handlers remain, still inject queue relay
        need_relay = bool(popped_handlers) or not picklable_handlers
        try:
            if need_relay:
                dummy = {
                    'handler': 'XTL_DUMMY_QUEUE_HANDLER',
                    'config': {'eavesdrop_queue': self._eavesdropping_qname},
                }
                log_cfg['handlers'].append(dummy)
                self._eavesdrop()

            yield

        finally:
            # Remove dummy relay marker if we added one
            if dummy is not None and dummy in log_cfg.get('handlers', []):
                log_cfg['handlers'].remove(dummy)

            # Restore any removed unpicklable handlers
            if popped_handlers:
                self.logger.debug(
                    'Reattaching popped handlers to submission for job: %s',
                    submission.data.job_id,
                )
                log_cfg['handlers'].extend(popped_handlers)
                popped_handlers.clear()

    async def __aenter__(self):
        await super().__aenter__()

        # Initialize the process pool executor
        self._executor = self._executor_cls(
            max_workers=self._resources.jobs,  # type: ignore
            initializer=self._rename_current_process
        )

        # Ensure that the log queue and default lock are created before any jobs are submitted
        self.get_queue(self._eavesdropping_qname)
        self.get_lock(None)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Check if eavesdropper is running and signal it to stop
        if self._eavesdropping is not None:
            self.logger.debug('Stopping eavesdropper')
            with contextlib.suppress(Exception):
                # Send sentinel value to unblock the eavesdropper if it's waiting on the queue
                log_queue = self.get_queue(self._eavesdropping_qname)
                await log_queue.put(None)

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._eavesdropping, timeout=2.0)

            if not self._eavesdropping.done():
                self.logger.warning('Eavesdropper did not shut down within timeout')
                self._eavesdropping.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._eavesdropping

            self._eavesdropping = None

        # Shutdown the process pool executor
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        await super().__aexit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _rename_current_process() -> None:
        """
        Rename the current process to Process<N>, where N is a unique index extracted from the original process name.
        """
        process = multiprocessing.current_process()
        identity = process._identity[0] if process._identity else 1
        idx = max(0, identity - 1)

        # Rename the process
        new_name = f'Process{idx}'
        if process.name != new_name:
            process.name = new_name

    @staticmethod
    async def _bootstrap_process(submission: JobSubmission) -> JobResults | None:
        # Initialize the worker process's ResourceManager with the granted budget.
        get_rc_manager(total=submission.resources)

        # Filter out dummy queue handlers from the logging config
        valid_handlers = []
        eavesdropper: dict | None = None
        for h in submission.data.logging_config.get('handlers', []):
            if h['handler'] == 'XTL_DUMMY_QUEUE_HANDLER':
                eavesdropper = h
                continue
            valid_handlers.append(h)
        # Update the logging config to only include valid handlers
        submission.data.logging_config['handlers'] = valid_handlers

        # Deserialize the job
        job = submission.to_job()

        # Reconstruct the IPC backend
        if submission.ipc is not None:
            ipc = MultiprocessPool._ipc_cls.from_handle(submission.ipc)
            job._pool = ProxyPool(ipc)

            # Install relay queue handler if eavesdropper config was found
            if eavesdropper is not None:
                config: dict = eavesdropper.get('config', {})
                qname: str | None = config.get('eavesdrop_queue', None)
                if qname is None:
                    raise ValueError('Eavesdropper config is missing the required "eavesdrop_queue" field')

                # Get the actual queue object from the IPC backend
                try:
                    queue = submission.ipc.queues[qname]
                except KeyError as e:
                    raise KeyError(f'Eavesdropper queue {qname!r} not found in IPC') from e

                # Add a QueueHandler to the job's logger to forward logs
                handler = QueueHandler(queue)
                job.logger.addHandler(handler)

        return await job.run()

    @staticmethod
    def _run_in_process(payload: dict) -> JobResults | None:
        submission = JobSubmission.from_dict(payload)

        # NB: No need to clear CURRENT_LEASE here, because we are not
        #  using copy_context to propagate the lease

        return asyncio.run(MultiprocessPool._bootstrap_process(submission))

    async def _execute_submission(self, submission: JobSubmission) -> JobResults | None:
        job_id = submission.data.job_id
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={job_id!r}')

        async with self._ensure_picklable_log_handlers(submission):
            payload = submission.to_dict()
            pickle_safe, reason = is_picklable(payload, path=JobSubmission.__name__)
            if not pickle_safe:
                path, obj_type, obj_repr = reason
                raise TypeError(
                    f'Job submission with job_id={job_id!r} contains an unpicklable object and cannot be executed in a '
                    f'process pool\n'
                    f'Offending object found at {path}: type={obj_type}, repr={obj_repr}'
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._run_in_process, payload)
