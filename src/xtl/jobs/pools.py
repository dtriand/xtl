import asyncio
import contextlib
from time import sleep
from typing import List, Type, Iterable, Literal, AsyncGenerator, overload

from xtl import Logger, settings
from xtl.jobs.jobs import Job, JobResults, JobConfig
from xtl.logging.config import LoggerConfig
from xtl.math.uuid import UUIDFactory


uuid = UUIDFactory()
logger_ = Logger(__name__)


class JobPool:

    _cancellation_timeout: int = 5
    """Timeout in seconds for cancelling tasks and cleaning up"""

    def __init__(self, max_jobs: int = 1, logger_config: LoggerConfig = None):
        """
        A pool for executing jobs with concurrency limits and control.

        :param max_jobs: Maximum number of jobs to execute concurrently (default: 1)
        :param logger_config: Optional logging configuration.
        """
        # Concurrency control
        self._max_jobs = max_jobs
        self._semaphore = asyncio.Semaphore(max_jobs)
        self._global_lock = asyncio.Lock()

        # Context manager
        self._in_ctx: bool = False  # Managed by __enter__ and __exit__

        # Job tracking
        self._jobs: dict[str, Job] = {}
        self._running_jobs: set[str] = set()
        self._completed_jobs: set[str] = set()

        # Task tracking
        self._tasks: dict[str, asyncio.Task] = {}  # Managed by __exit__ and launchers

        # Logger configuration
        if logger_config is not None and not isinstance(logger_config, LoggerConfig):
            raise TypeError(f'\'logger_config\' must be an {LoggerConfig.__name__} '
                            f'instance')
        self._logger_config: LoggerConfig | None = logger_config

        # Pool state
        self._is_running = False  # Managed by launcher methods
        self._error: Exception | None = None

    def __enter__(self):
        """
        Enter the context manager.
        """
        if self._in_ctx:
            raise RuntimeError(f'{JobPool.__name__} is already in a context manager')
        self._in_ctx = True
        logger_.debug('Entering the pool context manager')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
                logger_.error('Pool execution was interrupted by user')
            else:
                logger_.error('An exception occurred within the pool context: %s',
                              exc_type.__name__)
            self._error = exc_val

            # Request cancellation of all running tasks
            logger_.warning('Cancelling all running tasks')
            for job_id, task in list(self._tasks.items()):
                if keyboard_interrupt:
                    task.cancel('Execution interrupted by user')
                else:
                    task.cancel('An exception occurred in the pool context')

            if keyboard_interrupt:
                return True  # Suppress the exception

            # Propagate other exceptions
            return False

        # Normal exit
        logger_.debug('Exiting the JobPool context')
        self._in_ctx = False
        self._tasks.clear()
        return True

    def submit(self, job_class: Type[Job],
               configs: Iterable[JobConfig | None] = None,
               job_ids: Iterable[str] = None) -> list[Job]:
        """
        Submit jobs to the pool for execution.

        :param job_class: The Job class to instantiate
        :param configs: Optional list of configurations to pass to `Job.configure()`
        :param job_ids: Optional list of job IDs to assign to the jobs.
        """
        if not issubclass(job_class, Job):
            raise TypeError(f'Expected {job_class.__name__} to be a '
                            f'subclass of {Job.__name__}, '
                            f'got {type(job_class).__name__}')

        # Create a list to track created jobs
        created_jobs = []

        # Cast config to a list
        if not isinstance(configs, Iterable):
            configs = [configs]

        if job_ids is None:
            # Generate job ids
            digits = len(str(len(configs)))
            job_ids = [f'{job_class.__name__}|{str(i + 1).zfill(digits)}'
                       for i in range(len(configs))]
        if len(job_ids) != len(configs):
            raise ValueError(f'Length of \'job_ids\' must match length of \'configs\', '
                             f'{len(job_ids)} != {len(configs)}')

        for config, job_id in zip(configs, job_ids):
            if job_id is None:
                # In case None was passed within job_ids
                job_id = uuid.random(length=settings.automate.job_digits)

            logger = job_class.get_logger(job_id, self._logger_config)

            # Create the job instance and configure it
            job = job_class(job_id=job_id, logger=logger)

            if config is not None:
                job.configure(config)

            # Link pool to the job
            job.pool = self

            # Add job to the pool
            self._jobs[job.job_id] = job
            created_jobs.append(job)
            logger_.debug('Submitted job \'%s\' to pool', job.job_id)

        return created_jobs

    @contextlib.asynccontextmanager
    async def request_lock(self):
        """
        Request a concurrency lock for jobs that need exclusive access to resources.

        This is an async context manager that ensures only one job can run
        at a time within the context of the lock.
        """
        try:
            await self._global_lock.acquire()
            yield
        finally:
            self._global_lock.release()

    async def _run_job(self, job: Job) -> JobResults:
        """
        Run a single jobs asynchronously with concurrency control and logging.
        """
        async with self._semaphore:
            job_id = job.job_id
            self._running_jobs.add(job_id)

            try:
                # Execute the job
                logger_.debug('Starting job \'%(job_id)s\'', {'job_id': job_id})
                results = await job.run()
                self._completed_jobs.add(job_id)
                logger_.debug('Job \'%(job_id)s\' completed', {'job_id': job_id})
                return results
            except Exception as e:
                # Exceptions are logged but not raised, so that the pool continues running
                logger_.error('Job \'%(job_id)s\' raised an exception: %(e)s',
                              {'job_id': job_id, 'e': e})
                return JobResults(job_id=job_id, data=None, error=e)
            finally:
                # Update the job state before actually returning
                if job_id in self._running_jobs:
                    self._running_jobs.remove(job_id)
                    logger_.debug('Job \'%(job_id)s\' removed from running jobs',
                                  {'job_id': job_id})

    @overload
    async def launch(self, results: Literal['all'] = 'all') -> \
            List[JobResults | None]: ...

    @overload
    async def launch(self, results: Literal['stream']) -> \
            AsyncGenerator[JobResults, None]: ...

    async def launch(self, results: Literal['all', 'stream'] = 'all') -> \
            List[JobResults] | AsyncGenerator[JobResults, None]:
        """
        Launch all submitted jobs in the pool. By default the results are returned only
        when all jobs have completed. If `results` is set to 'stream', then the results
        are yielded as each of the jobs completes.

        :param results: How to return results from the running jobs:
            - 'all': Return all results when all jobs have completed (default)
            - 'stream': Yield results as they become available
        :raises RuntimeError: If called outside the context manager
        """
        # Check if the context manager is active
        if not self._in_ctx:
            raise RuntimeError(f'{JobPool.__name__}.launch() must be used '
                               f'within the context manager.')

        if results == 'all':
            return await self._launch_all()
        elif results == 'stream':
            return self._launch_stream()
        else:
            raise ValueError(f'Invalid {results=}. Expected \'all\' or '
                             f'\'stream\'.')

    async def _launch_all(self) -> List[JobResults]:
        """
        Run all jobs in the pool and return the aggregated results.
        """
        if not self._jobs:
            logger_.warning('No jobs submitted to pool')
            return []

        # Activate pool
        self._is_running = True
        logger_.debug('Activating pool with %d jobs', len(self._jobs))
        try:
            # Create tasks
            for job_id, job in self._jobs.items():
                task = asyncio.create_task(self._run_job(job))
                self._tasks[job_id] = task
            pending = set(self._tasks.values())

            # Wait for all tasks to complete
            logger_.debug('Launching jobs...')
            done, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
            logger_.debug('All jobs completed')

            # Collect results
            results = []
            logger_.debug('Collecting results from completed jobs')
            for task in done:
                # Task errors have already been casted to JobResults in _run_job
                #  so we can safely call result() here
                result = task.result()
                results.append(result)

                # Append job to completed jobs
                self._completed_jobs.add(result.job_id)
            logger_.debug('Collected %d results', len(results))
            return results
        finally:
            # No exception handling at this stage, this is managed by __exit__
            logger_.debug('Deactivating pool')
            self._is_running = False

    async def _launch_stream(self) -> AsyncGenerator[JobResults, None]:
        """
        Run all jobs in the pool but yield the results as the jobs complete.
        """
        if not self._jobs:
            logger_.warning('No jobs submitted to pool')
            return

        # Activate pool
        self._is_running = True
        logger_.debug('Activating pool with %d jobs', len(self._jobs))
        try:
            # Create tasks
            for job_id, job in self._jobs.items():
                task = asyncio.create_task(self._run_job(job))
                self._tasks[job_id] = task
            pending = set(self._tasks.values())

            # Launch jobs and yield results as they complete
            logger_.debug('Launching jobs...')
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Yield results from completed jobs
                logger_.debug('Some jobs completed, yielding results')
                for task in done:
                    # Task errors have already been casted to JobResults in _run_job
                    #  so we can safely call result() here
                    result = task.result()

                    # Append job to completed jobs
                    self._completed_jobs.add(result.job_id)

                    yield result
            logger_.debug('All jobs completed')
        finally:
            # No exception handling at this stage, this is managed by __exit__
            logger_.debug('Deactivating pool')
            self._is_running = False
