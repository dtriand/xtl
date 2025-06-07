import asyncio
from typing import List, Optional, Set, Dict, Any, Type, Sequence, TYPE_CHECKING, \
    Literal, AsyncGenerator, overload
import contextlib

from xtl import Logger, settings
from xtl.jobs.jobs import Job, JobResults, JobConfig
from xtl.math.uuid import UUIDFactory

if TYPE_CHECKING:
    import logging


uuid = UUIDFactory()
logger_ = Logger(__name__)


class JobPool:
    """
    A pool for executing jobs with concurrency limits.

    The JobPool manages job execution, logger configuration, and provides
    locking mechanisms for jobs that need exclusive access to resources.
    """

    def __init__(self, max_jobs: int = 5, logger_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a JobPool with concurrency limits and logger configuration.

        Args:
            max_jobs: Maximum number of jobs to run concurrently
            logger_config: Configuration for job loggers (optional)
        """
        # Concurrency control
        self._max_jobs = max_jobs
        self._semaphore = asyncio.Semaphore(max_jobs)
        self._global_lock = asyncio.Lock()

        # Context manager
        self._in_ctx: bool = False

        # Job tracking
        self._jobs: List[Job] = []
        self._running_jobs: Set[str] = set()
        self._completed_jobs: Set[str] = set()

        # Task tracking
        self._tasks: Dict[str, asyncio.Task] = {}

        # Logger configuration
        self._logger_config = logger_config or {}

        # Pool state
        self._is_running = False
        self._error: Optional[Exception] = None

    def __enter__(self):
        """Enter the context manager."""
        if self._in_ctx:
            raise RuntimeError(f'{JobPool.__name__} is already in a context manager')
        self._in_ctx = True
        logger_.debug('Entering the pool context manager')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, handling any exceptions."""
        self._in_ctx = False
        logger_.debug('Exiting the pool context manager')
        if exc_val:
            logger_.exception('Exception in JobPool context: %s', exc_val)
            self._error = exc_val
        if exc_type is KeyboardInterrupt:
            logger_.warning('KeyboardInterrupt received, cancelling all running tasks')
            self._cancel_tasks()
        return True  # Suppress exceptions

    def submit(self, job_class: Type[Job],
               configs: Optional[Sequence[JobConfig]] = None, **kwargs) -> List[Job]:
        """
        Submit jobs to the pool for execution.

        :param job_class: The Job class to instantiate
        :param configs: Optional list of configurations to pass to `Job.configure()`
        """
        if not issubclass(job_class, Job):
            raise TypeError(f'Expected {job_class.__name__} to be a '
                            f'subclass of {Job.__name__}, '
                            f'got {type(job_class).__name__}')

        # Create a list to track created jobs
        created_jobs = []

        if not isinstance(configs, Sequence):
            configs = [configs]

        job_ids = kwargs.pop('job_ids', [None] * len(configs))

        for config, job_id in zip(configs, job_ids):
            if not job_id:
                job_id = uuid.random(length=settings.automate.job_digits)

            logger = job_class.get_logger(job_id, **self._logger_config)

            # Create the job instance and configure it
            job = job_class(job_id=job_id, logger=logger, **kwargs)

            if config is not None:
                job.configure(config)

            # Set pool reference in the job
            job.pool = self

            # Add job to the pool
            self._jobs.append(job)
            created_jobs.append(job)

            logger_.debug('Job %(job_id)s created with config and submitted to pool',
                          {'job_id': job.job_id})

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
                logger_.debug('Starting job %(job_id)s', {'job_id': job_id})
                results = await job.run()
                logger_.debug('Job %(job_id)s completed', {'job_id': job_id})
                return results
            except Exception as e:
                # Log exception but do not raise so that the pool continues running
                logger_.error('Job %(job_id)s raised an exception: %(e)s',
                              {'job_id': job_id, 'e': e})
                # Return JobResults with the error
                return JobResults(job_id=job_id, data=None, error=e)
            finally:
                # Update the job state before actually returning
                self._running_jobs.remove(job_id)
                logger_.debug('Job %(job_id)s removed from running jobs',
                              {'job_id': job_id})

                self._completed_jobs.add(job_id)
                logger_.debug('Job %(job_id)s added to completed jobs',
                              {'job_id': job_id})

    def _cancel_tasks(self):
        """
        Cancel all running tasks in the pool.
        """
        for job_id, task in self._tasks.items():
            # Cancel tasks that are still running
            if not task.done() and not task.cancelled():
                logger_.warning('Cancelling task for job %s', job_id)
                task.cancel()
                logger_.debug('Task %s cancelled', job_id)

        # Clear the task dictionary
        self._tasks.clear()

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
            for job in self._jobs:
                task = asyncio.create_task(self._run_job(job))
                self._tasks[job.job_id] = task
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
            logger_.debug('Collected %d results', len(results))
            return results
        finally:
            # Clean up task references
            self._tasks.clear()
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
            for job in self._jobs:
                task = asyncio.create_task(self._run_job(job))
                self._tasks[job.job_id] = task
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
                    yield result
            logger_.debug('All jobs completed')
        finally:
            # Clean up task references
            self._tasks.clear()
            logger_.debug('Deactivating pool')
            self._is_running = False




if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')

    class TestJob(Job):
        async def _execute(self):
            self.logger.info('Going to sleep...')
            await asyncio.sleep(5)
            self.logger.info('Woke up!')

    async def main():
        with JobPool(max_jobs=5) as pool:
            jobs = pool.submit(TestJob, configs=[None for _ in range(10)])
            results = await pool.launch()

    asyncio.run(main())
