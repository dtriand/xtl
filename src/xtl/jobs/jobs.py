import abc
import asyncio
import contextlib
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar, TYPE_CHECKING, \
    get_args, Iterable, overload, Literal

from pydantic import field_validator

if TYPE_CHECKING:
    from xtl.jobs.pools import PoolProtocol
    from xtl.jobs.batchfiles import BatchFile
    from xtl.jobs.steps import StepSpec, JobContext

from xtl import settings, Logger
from xtl.math.uuid import UUIDFactory
from xtl.common.options import Option, Options
from xtl.common.misc import deepmerge
from xtl.logging.config import LoggerConfig, StreamHandlerConfig, LoggingFormat
from xtl.exceptions.base import SubprocessError, StderrError
from xtl.jobs.config import JobConfig, BatchJobConfig
from xtl.jobs.logging import get_logger_config
from xtl.jobs.results import JobResults, BatchResults, SteppedJobResults


uuid = UUIDFactory()
logger_ = Logger(__name__)


class _DummyLock:
    """
    A dummy lock context manager for job execution outside of a pool.
    """
    async def __aenter__(self): return None

    async def __aexit__(self, exc_type, exc_val, exc_tb): return None


class JobData(Options):
    """
    Serializable data class for transporting job information and configuration.
    """

    job_cls: str = \
        Option(
            ...,
            desc='Import path of the Job class, e.g. "xtl.foo.bar.MyJob"'
        )

    job_id: str = \
        Option(
            ...,
            desc='Unique identifier for this job instance'
        )

    config: dict[str, Any] | None = \
        Option(
            ...,
            desc='Job configuration'
        )

    logging_config: dict[str, Any] | None = \
        Option(
            ...,
            desc='Logging configuration for this job'
        )

    @staticmethod
    def _import_symbol(path: str):
        """
        Dynamically import a symbol from a module given its import path.
        """
        module_path, _, attr = path.rpartition('.')
        if not module_path or not attr:
            raise ValueError(f'Invalid import path: {path!r}')
        module = __import__(module_path, fromlist=[attr])
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise ImportError(f'Could not resolve symbol {path!r}') from exc

    def get_job_cls(self) -> Type['Job']:
        """
        Dynamically import and return the Job class specified by `job_cls`.
        """
        return self._import_symbol(self.job_cls)

    @field_validator('job_cls', mode='before')
    @classmethod
    def validate_job_cls(cls, value: Any) -> str:
        try:
            job_cls = cls._import_symbol(value)
        except (ImportError, ValueError) as exc:
            raise ValueError(f'Invalid `job_cls`: {value!r}') from exc
        if not isinstance(job_cls, type) or not issubclass(job_cls, Job):
            raise TypeError(f'`job_cls` must be a subclass of {Job.__name__}, got {value!r}')
        return value


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

    _logger_config: ClassVar[LoggerConfig] = get_logger_config(level=_logging_level)
    """Logging configuration for jobs."""

    _keep_files: ClassVar[bool] = False
    """Whether to keep any files created by this job"""

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
            uuid.random(length=settings.jobs.job_digits)
        while self._job_id in self._registry:
            # Regenerate if necessary
            logger_.debug('Regenerating job_id: %s', self._job_id)
            self._job_id = uuid.random(length=settings.jobs.job_digits)

        # Attach a logger
        self._logger = logger or self.get_logger(self.job_id)

        # Register the job in the class registry
        self.__class__._registry[self._job_id] = self

        # Job state
        self._is_running = False
        self._is_complete = False
        self._error: Exception | None = None

        # Pool integration
        self._pool: Optional['PoolProtocol'] = None

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
    def pool(self) -> Optional['PoolProtocol']:
        """
        Get the job pool associated with this job, if any.
        """
        return self._pool

    @pool.setter
    def pool(self, pool: Optional['PoolProtocol']) -> None:
        from xtl.jobs.pools import PoolProtocol

        if pool is not None and not isinstance(pool, PoolProtocol):
            raise TypeError(f'`pool` must implement {PoolProtocol.__name__}, got {type(pool).__name__}')
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
            else:
                # Clean files when no errors occurred
                await self._tidy_up()

        return JobResults(job_id=self._job_id, data=result, error=self._error)

    async def _tidy_up(self) -> None:
        if not self.config.job_directory or not self.config.job_directory.exists():
            # Skip when no job_directory was created
            return

        if not self._keep_files and not settings.jobs.keep_temp:
            try:
                self.logger.debug('Cleaning up job directory: %s', self.config.job_directory)
                async with await self.lock():
                    shutil.rmtree(self.config.job_directory, ignore_errors=True)
                self.logger.debug('Job directory cleaned up successfully')
            except OSError as e:
                self.logger.error('Failed to clean up job directory: %s', self.config.job_directory)
                self.logger.error('Error details: %s', str(e))
        else:
            from xtl.common.compatibility import OS_POSIX

            if not OS_POSIX:
                # Skip permission update on non-POSIX systems
                return

            from xtl.common.os import chmod_recursively
            try:
                self.logger.debug('Updating permissions in job directory: %s', self.config.job_directory)
                async with await self.lock():
                    chmod_recursively(
                        self.config.job_directory,
                        files_permissions=settings.jobs.permissions.files,
                        directories_permissions=settings.jobs.permissions.directories
                    )
                self.logger.debug('Permissions updated successfully')
            except OSError as e:
                self.logger.error('Failed to update permissions in job directory: %s', self.config.job_directory)
                self.logger.error('Error details: %s', str(e))

    async def lock(self, name: str | None = None) -> Any:
        """
        Context manager to acquire a lock during job execution.

        :param name: Optional name of the lock.
        """
        if self._pool is None:
            # If no pool is set, return a dummy lock that does nothing
            self.logger.warning('Requested lock for job outside a pool context manager')
            return _DummyLock()
        # Use the pool's lock
        return self._pool.get_lock(name)

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
        # Avoid duplicate emission when the same logger id is configured repeatedly.
        if logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.close()
        if config is not None:
            if not isinstance(config, LoggerConfig):
                raise TypeError(f'Expected a {LoggerConfig.__name__} instance, '
                                f'got {type(config).__name__}')
            config.configure(logger)
            if update_config:
                # This is required for propagating log configs to subjobs
                cls._logger_config = config
        else:
            cls._logger_config.configure(logger)

        return logger

    @overload
    def serialize(self, as_dict: Literal[True]) -> dict[str, Any]: ...

    @overload
    def serialize(self, as_dict: Literal[False]) -> JobData: ...

    def serialize(self, as_dict: bool = True) -> dict[str, Any] | JobData:
        """
        Serialize the job to a dictionary for storage or transmission.
        """
        job_data = JobData(
            job_cls=f'{self.__class__.__module__}.{self.__class__.__name__}',
            job_id=self.job_id,
            config=self.config.to_dict() if self.config else None,
            logging_config=self._logger_config.to_dict() if self._logger_config else None
        )
        if as_dict:
            return job_data.to_dict()
        return job_data

    @classmethod
    def deserialize(cls, data: JobData | dict[str, Any]) -> 'Job[JobConfigType]':
        """
        Deserialize a job from a dictionary.
        """
        if isinstance(data, dict):
            data = JobData.from_dict(data)
        elif not isinstance(data, JobData):
            raise TypeError(f'Expected a {JobData.__name__} instance or dict, got {type(data).__name__}')
        job_cls = data.get_job_cls()
        config_cls = job_cls._config_class
        config = config_cls.from_dict(data.config) if data.config else None

        job = job_cls.with_config(
            config=config,
            job_id=data.job_id,
            logger=job_cls.get_logger(
                data.job_id,
                config=LoggerConfig.from_dict(data.logging_config) if data.logging_config else None
            )
        )
        return job


class BatchJob(Job[BatchJobConfig], Generic[BatchJobConfigType]):

    def __init__(self, job_id: str | None = None, logger: 'logging.Logger' = None):
        super().__init__(job_id=job_id, logger=logger)

        self._config: BatchJobConfig | None
        self._batch: Optional[BatchFile] = None
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
        results = BatchResults(
            stdout=self.config.stdout.read_text(encoding='utf-8'),
            stderr=self.config.stderr.read_text(encoding='utf-8'),
            return_code=self._batch.process.returncode
        )

        # Check for errors
        if results.return_code != 0:
            raise SubprocessError(
                message=f'Batch execution failed with return code {results.return_code}',
                command=(self._batch.file, ) + tuple(self._batch_args),
                raiser=self._batch.file
            )

        self.logger.debug('Batch execution completed successfully')

        return results

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
        batch_context = deepcopy(self._batch_context)
        config_context = self.config.get_context()
        return deepmerge(batch_context, config_context)


class SteppedJob(Job[JobConfig], Generic[JobConfigType]):

    _steps: ClassVar[tuple['StepSpec', ...]] = ()

    def _get_step_config(self, step_name: str, ctx: 'JobContext') -> JobConfig | BatchJobConfig:
        spec = next((s for s in self._steps if s.name == step_name), None)
        if spec is None:
            raise ValueError(f'Unknown job step: {step_name}')
        defaults = deepcopy(spec.defaults)
        self.logger.debug('Determining dynamic defaults')
        dynamic_defaults = spec.dynamic_defaults(ctx)
        overrides = deepcopy(self.config.steps.get(spec.name, {}))
        payload = deepmerge(defaults | dynamic_defaults, overrides)
        return spec.config_cls(**payload)

    async def _execute(self) -> Any | None:
        from xtl.jobs.steps import JobContext

        # Initialize a JobContext to pass to steps for sharing state and results
        ctx = JobContext(parent=self)

        # Execute steps sequentially
        for i, spec in enumerate(self._steps, start=1):
            self.logger.info('Executing step %(i)d/%(n)d: %(step)s', {'i': i, 'n': len(self._steps), 'step': spec.name})
            self.logger.debug('Preparing %(config_cls)s', {'config_cls': spec.config_cls.__name__})

            # Get the config for the current step, merging defaults and dynamic defaults from the StepSpec with any
            #  overrides from the JobConfig
            config = self._get_step_config(spec.name, ctx=ctx)

            # Propagate job directory, if specified for the main job
            if self.config.job_directory is not None:
                config.job_directory = self.config.job_directory / f'{i}_{spec.name}'

            self.logger.debug('Preparing %(job_cls)s', {'job_cls': spec.job_cls.__name__})

            # Create preconfigured job
            step_id = f'{self.job_id}.{i}'
            logger = spec.job_cls.get_logger(job_id=step_id, config=self._logger_config)  # propagate logging config
            job = spec.job_cls.with_config(config, job_id=step_id, logger=logger)

            # Run the job step
            self.logger.debug('Running %(job_cls)s', {'job_cls': spec.job_cls.__name__})
            ctx.step = job  # set the current step in the context
            results = await job.run()

            # Check for errors and abort if the step failed
            if results and not results.success:
                self.logger.error('Aborting job due to an error in step: %(step)s', {'step': spec.name})
                ctx.results[spec.name] = results
                return ctx.results, ctx.data

            # Apply post-processing if specified in the StepSpec
            if spec.post_processor is not None:
                self.logger.debug('Post-processing results for step: %(step)s', {'step': spec.name})
                ctx.data |= spec.post_processor(results, ctx=ctx)

            # Store results in the context for access by subsequent steps
            ctx.results[spec.name] = results
            ctx.step = None  # clear the current step from the context
            self.logger.debug('Step %(i)d/%(n)d completed: %(step)s', {'i': i, 'n': len(self._steps), 'step': spec.name})

        self.logger.info('All steps completed')
        return ctx.results, ctx.data

    async def run(self) -> SteppedJobResults | None:
        results = await super().run()
        if results is None:
            return None

        # Cast the results to SteppedJobResults
        stepped_results = SteppedJobResults(
            job_id=results.job_id,
            steps=results.data[0],
            data=results.data[1],
            error=results.error
        )

        # Check for errors in any of the steps and set the overall error if any step failed
        for step in stepped_results.steps.values():
            if step.error:
                stepped_results.error = step.error
                break

        return stepped_results
