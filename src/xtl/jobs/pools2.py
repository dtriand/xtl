from __future__ import annotations

import abc
import asyncio
import contextlib
import contextvars
from enum import Enum
import logging
import multiprocessing
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import AsyncIterator, Literal, overload, Type, Iterable, TYPE_CHECKING

from xtl import settings
from xtl.common.compatibility import PY310_OR_LESS
from xtl.jobs.config import JobConfig
from xtl.jobs.jobs import Job, JobResults
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
else:
    from enum import StrEnum


uuid = UUIDFactory()


class JobPool(StrEnum):
    SIMPLE = 'simple'
    ASYNC = 'async'
    THREADS = 'threads'
    PROCESSES = 'processes'

    def get(self) -> Type[BasePool]:
        if self == JobPool.SIMPLE:
            return SimplePool
        if self == JobPool.ASYNC:
            return AsyncPool
        if self == JobPool.THREADS:
            return ThreadedPool
        if self == JobPool.PROCESSES:
            return MultiprocessPool
        raise ValueError(f'Unsupported {JobPool.__name__} type: {self!r}')


class BasePool(abc.ABC):

    _logging_level = logging.INFO
    _logging_config: LoggerConfig = get_logger_config(level=_logging_level)
    _executor_cls = None
    _ipc_cls: type[IPCBackend]

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

        # Pool executor backend
        self._executor: BasePool._executor_cls | None = None  # Managed by __aenter__ and __aexit__

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

        # Logger configuration
        if logger_config and not isinstance(logger_config, LoggerConfig):
            raise TypeError(f'Expected a {LoggerConfig.__name__} instance for `logger_config`, '
                            f'got {type(logger_config).__name__}')
        self._logger_config = logger_config or self._logging_config

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
        self.logger.debug('Requesting resources for the pool: %s', self._rc_requested.__dict__)
        if self._rc_manager is None:
            self._rc_manager = get_rc_manager()
        self._rc_lease = await self._rc_manager.acquire(self._rc_requested)
        self.logger.debug('Resources granted for the pool: %s', self._rc_lease.granted.__dict__)
        self._resources = self._rc_lease.granted
        self._semaphore = asyncio.Semaphore(self._resources.jobs)

        # Start the IPC backend
        self._ipc = self._ipc_cls()
        self._ipc.start()

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

    def submit(self, job_cls: Type[Job], configs: Iterable[JobConfig | None] = None, **kwargs):
        if not issubclass(job_cls, Job):
            raise TypeError(f'Expected a subclass of {Job.__name__} for `job_cls`, got {job_cls}')

        # Cast configs to a list
        if not isinstance(configs, Iterable):
            configs = [configs]
        configs = list(configs)

        # Generate job IDs if not provided
        if (job_ids := kwargs.get('job_ids', None)) is None:
            digits = len(str(max(1, len(configs))))
            job_ids = [f'{job_cls.__name__}|{str(i + 1).zfill(digits)}' for i in range(len(configs))]
        job_ids = list(job_ids)

        # Check for length consistency
        if len(job_ids) != len(configs):
            raise ValueError(f'Length of `job_ids` ({len(job_ids)}) must match length of `configs` ({len(configs)})')

        # Create and register jobs
        created_jobs: list[Job] = []
        for job_id, config in zip(job_ids, configs):
            if job_id is None:
                # In case None was passed explicitly in job_ids
                job_id = uuid.random(settings.jobs.job_digits)

            # Create logger
            logger = job_cls.get_logger(job_id, self._logger_config)


            # Create job instance
            job = job_cls(job_id=job_id, logger=logger)
            if config is not None:
                job.configure(config)
            job.pool = self

            # Create submission
            submission = JobSubmission.from_job(job)
            submission.ipc = self._build_ipc_handle()

            # Register submission and job
            self._submissions[submission.submission_id] = submission
            self._jobs[job.job_id] = job
            created_jobs.append(job)
            self.logger.debug('Submitted job %s to %s', job.job_id, self.__class__.__name__)

        return created_jobs

    @abc.abstractmethod
    async def _execute_submission(self, submission: JobSubmission) -> JobResults: ...

    @staticmethod
    async def _run_with_context(executor, fn, *args):
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()

        def _runner(_unused=None):
            return ctx.run(fn, *args)

        return await loop.run_in_executor(executor, _runner, None)

    async def _process_submission(self, submission: JobSubmission) -> JobResults:
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

    async def launch(self, mode: Literal['stream', 'all'] = 'all') -> AsyncIterator[JobResults] | list[JobResults]:
        if not self._in_ctx:
            raise RuntimeError(f'{self.__class__.__name__}.launch() must be used within the context manager.')
        if mode == 'all':
            return await self._launch_all()
        if mode == 'stream':
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
        self._ipc_handle_names.locks.add(name)
        return lock

    def get_queue(self, name: str | None, maxsize: int = 0) -> IPCQueue:
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

    def get_state(self, name: str | None) -> IPCState:
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
        Build a pickleable IPCHandle from all registered IPC primitives.
        """
        if self._ipc is None:
            return None
        return self._ipc.to_handle(self._ipc_handle_names)


class ProxyPool:
    """
    A minimal pool shim injected as Job.pool inside a worker process or thread.
    It exposes only the methods needed for a Job to run, backed by the ProxyIPCBackend.
    """

    def __init__(self, ipc: IPCBackend):
        self._ipc = ipc

    def get_lock(self, name: str | None = None) -> IPCLock:
        return self._ipc.get_lock(name)

    def get_queue(self, name: str | None, maxsize: int = 0) -> IPCQueue:
        return self._ipc.get_queue(name, maxsize=maxsize)

    def get_state(self, name: str) -> IPCState:
        return self._ipc.get_state(name)


class AsyncPool(BasePool):

    _ipc_cls = AsyncIPCBackend

    async def _execute_submission(self, submission: JobSubmission) -> JobResults:
        job = self._jobs.get(submission.data.job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={submission.job_id!r}')
        return await job.run()


class SimplePool(AsyncPool):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rc_requested = Resources(jobs=1, threads=self._rc_requested.threads,
                                       processes=self._rc_requested.processes, cores=self._rc_requested.cores)

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

    _executor_cls = ThreadPoolExecutor
    _executor: ThreadPoolExecutor | None
    _ipc_cls = ThreadedIPCBackend

    async def __aenter__(self):
        await super().__aenter__()
        self._executor = self._executor_cls(max_workers=self._resources.jobs)
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
    def _run_in_thread(job: Job, handle: IPCHandle | None):
        ThreadedPool._rename_current_thread()
        if handle is not None:
            ipc = ThreadedPool._ipc_cls.from_handle(handle)
            job._pool = ProxyPool(ipc)
        return asyncio.run(job.run())

    async def _execute_submission(self, submission: JobSubmission) -> JobResults:
        job_id = submission.data.job_id
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={job_id!r}')

        return await self._run_with_context(self._executor, self._run_in_thread, job, submission.ipc)


class MultiprocessPool(BasePool):

    _executor_cls = ProcessPoolExecutor
    _executor: ProcessPoolExecutor | None
    _ipc_cls = ProcessIPCBackend

    async def __aenter__(self):
        await super().__aenter__()
        self._executor = self._executor_cls(
            max_workers=self._resources.jobs,
            initializer=self._rename_current_process
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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
    def _run_in_process(payload: dict) -> JobResults:
        submission = JobSubmission.from_dict(payload)
        job = submission.to_job()

        if submission.ipc is not None:
            ipc = MultiprocessPool._ipc_cls.from_handle(submission.ipc)
            job._pool = ProxyPool(ipc)

        return asyncio.run(job.run())

    async def _execute_submission(self, submission: JobSubmission) -> JobResults:
        job_id = submission.data.job_id
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f'No job found for submission with job_id={job_id!r}')

        payload = submission.to_dict()
        try:
            pickle.dumps(payload)
        except Exception as exc:
            raise ValueError(f'Job submission with job_id={job_id!r} is not pickleable and cannot be executed '
                             f'in a process pool') from exc

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_in_process, payload)
