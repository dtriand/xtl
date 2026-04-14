import asyncio
import logging
from typing import Any, Literal, Type, Union, TYPE_CHECKING, Iterable

import rich.console
import rich.live
import rich.logging
import rich.panel
import rich.progress
import rich.text
import rich.theme
from typer import BadParameter

from xtl.logging.config import LoggerConfig
from xtl.jobs.pools import PoolProtocol, JobPool, BasePool, JOB_POOL_TYPES
from xtl.common.compatibility import PY310_OR_LESS
from xtl.tui.highlighters import JobHighlighter, PoolHighlighter
from xtl.tui.logging.handlers import BufferingHandler, LoggerHandlerPatcher
from xtl.tui.renderers import LogRenderer
from xtl.tui.progress import ProgressBar
from xtl.tui.styles import JOB_STYLES

if PY310_OR_LESS:
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:
    from xtl.jobs import Job, JobConfig
    from xtl.jobs.results import JobResults
    from xtl.tui.console import ConsoleIO


class JobOverviewColumn(rich.progress.ProgressColumn):

    def __init__(self, sep: str = '/'):
        super().__init__()
        self._sep = sep

    def render(self, task: rich.progress.Task) -> rich.console.RenderableType:
        fmt = ','
        completed = int(task.completed)
        total = f'{int(task.total):{fmt}}' if task.total is not None else '?'

        success = int(task.fields.get('success', 0))
        failed = int(task.fields.get('failed', 0))
        if (success + failed != completed) or (completed == 0):
            # Assume the fields are not getting updated properly
            text = f'[dim]{completed}{self._sep}{total}[/dim]'
        else:
            text = ''
            if success:
                text += f'[green]{success:{fmt}}[/green][dim]{self._sep}[/dim]'
            if failed:
                text += f'[red]{failed:{fmt}}[/red][dim]{self._sep}[/dim]'
            text += f'[dim]{total}[/dim]'

        return rich.text.Text.from_markup(text)


class PoolProgress(ProgressBar):

    def __init__(self, *, console: 'ConsoleIO') -> None:
        # Setting transient=False, since it's managed by the Live instance
        super().__init__(console=console, transient=False)

        # Expand the BarColumn to the console width
        col: rich.progress.BarColumn = self.columns[2]  # type: ignore
        if isinstance(col, rich.progress.BarColumn):
            col.bar_width = self._console.width

        # Create single task
        self._job_task = self.add_task(
            name='jobs',
            description='Waiting for jobs...',
            success=0,
            failed=0,
        )

    @classmethod
    def get_default_columns(cls) -> tuple[rich.progress.ProgressColumn, ...]:
        return (
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn('[progress.description]{task.description}'),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            JobOverviewColumn(),
            rich.progress.TimeElapsedColumn(),
        )

    def start(self):
        self._job_task.description = 'Running jobs'
        self._job_task.start()

    def complete_job(self, result: 'JobResults'):
        self._job_task.advance(1)
        if result.success:
            self._job_task.advance(1, field='success')
        else:
            self._job_task.advance(1, field='failed')

    @property
    def total(self) -> float | None:
        return self._job_task.total

    @total.setter
    def total(self, total: float) -> None:
        self._job_task.total = total

    @property
    def description(self) -> str:
        return self._job_task.description

    @description.setter
    def description(self, description: str) -> None:
        self._job_task.description = description



class LogPanel:

    def __init__(
            self,
            handler: BufferingHandler,
            console: 'ConsoleIO',
            *,
            title: str = None,
            subtitle: str = None,
            **kwargs
    ) -> None:
        self._handler = handler
        self._console = console
        self._title = title
        self._subtitle = subtitle
        self._renderer = LogRenderer(
            console=console,
            log_fmt=kwargs.pop('log_fmt', None),
            highlighters={
                'job': JobHighlighter(),
                'pool': PoolHighlighter(),
             },
            **kwargs
        )

    def __rich__(self):
        logs = [self._renderer.render(record) for record in self._handler.tail_records()]

        if not logs:
            logs = [
                rich.text.Text('Waiting for logs...', style='dim italic', justify='center')
            ]

        return rich.panel.Panel(
            rich.console.Group(*logs),
            title=self._title,
            border_style='dim',
            subtitle=self._subtitle,  # Updated from LivePool.__aenter__
            subtitle_align='right',
        )

    def replay(self) -> None:
        records = self._handler._records
        limit = self._handler._records.maxlen

        if len(records) >= limit:
            self._console.print(f'[dim italic]Displaying the latest {limit:,} records[/]', justify='center', highlight=False)

        for record in records:
            log = self._renderer.render(record, time_format='%Y-%m-%d %H:%M:%S,%f')
            self._console.print(log)


class LivePool(PoolProtocol):

    def __init__(
            self,
            console: 'ConsoleIO',
            pool_type: JobPool | JOB_POOL_TYPES | None = None,
            max_jobs: int = 1,
            progress: bool = True,
    ) -> None:
        """
        Α job pool with interactive rich elements on the console.

        :param console: The console to render the pool in.
        :param pool_type: The type of the pool to use. If None, a simple pool is created.
        :param max_jobs: The maximum number of jobs to run concurrently in the pool.
        :param progress: Whether or not to display a progress bar.
        """
        from xtl import settings

        self._console: 'ConsoleIO' = console
        self._theme = rich.theme.Theme(JOB_STYLES)
        # Buffer size between 10-20, depending on the console height
        self._buffer_size = min(20, max(10, self._console.height - 8))
        self._buffer = BufferingHandler(tail_size=self._buffer_size)

        # Log formatting
        if settings.jobs.threading_debug:
            log_fmt = '[%(name)s<%(processName)s:%(threadName)s>] %(message)s'
        else:
            log_fmt = '[%(name)s] %(message)s'

        # Rich components
        self._log_panel = LogPanel(
            console=self._console,
            handler=self._buffer,
            title='Job logs',
            log_fmt=log_fmt
        )
        self._progress = PoolProgress(
            console=self._console,
        ) if progress else None
        self._live: rich.live.Live | None = None

        # Pool and loggers
        self._pool = self._create_pool(pool_type, max_jobs)
        self._patcher = LoggerHandlerPatcher(
            handler=self._buffer
        )

        # Job submissions
        self._tasks: dict[str, str] = {}
        self._results = []

    def _create_pool(
            self,
            pool_type: JobPool | JOB_POOL_TYPES | None = None,
            max_jobs: int = 1
    ) -> BasePool:
        """
        Create a preconfigured job pool.

        :param pool_type: The type of the pool to use. If None, a simple pool is created.
        :param max_jobs: The maximum number of jobs to run concurrently in the pool.
        """
        logger_config = LoggerConfig(
            level=logging.DEBUG if self._console.debug else logging.INFO,
            propagate=False,
            handlers=[]
        )
        if pool_type is None:
            pool_type = JobPool.SIMPLE
        pool_cls = JobPool(pool_type).get()
        return pool_cls(
            name='root',
            max_jobs=max_jobs,
            logger_config=logger_config,
            job_logger_config=logger_config
        )

    async def __aenter__(self) -> Self:
        from xtl import settings

        # Update the console theme
        self._console.push_theme(self._theme)

        # Create renderables
        components: list[Any] = [self._log_panel]
        if self._progress:
            components.append(self._progress)

        renderable = rich.console.Group(*components)

        # Create Live instance and start
        live = rich.live.Live(
            renderable=renderable,
            console=self._console,
            refresh_per_second=settings.cli.max_fps,
            transient=True
        )
        self._live = live
        live.start()

        # Install special buffer handler for redirecting logs
        self._install_buffer()

        # Enter the actual pool
        await self._pool.__aenter__()

        # Resources allocation
        #  NB: This is only available from within the pool context
        rc = self._pool.resources
        if rc is not None and self._console.verbose >= 1:
            subtitle = f'J:{rc.jobs}|T:{rc.threads}|P:{rc.processes}'
            self._log_panel._subtitle = subtitle

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Check if the exit was caused by Ctrl+C
        interrupted = bool(exc_type and issubclass(exc_type, (KeyboardInterrupt, asyncio.CancelledError)))

        # Check for job errors
        job_errors = self._results and any(r.error for r in self._results)

        suppressed = False
        try:
            # Stop rich.Live and clear output
            if live := self._live:
                live.stop()

            # Replay logs if there were any errors on the jobs
            if exc_val or job_errors:
                self._console.print('An error occurred while executing jobs', style='red')
                if (not self._console.is_terminal or
                        self._console.confirm('Would you like to print the job logs?', default=True)):
                    self._replay_logs()

            # Let the pool exit normally
            suppressed = await self._pool.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            # Empty the handler's buffer
            self._buffer.clear()

            # Restore patched loggers
            self._patcher.restore()
            self._patcher.remove_handler()

            # Remove console theme
            self._console.pop_theme()

        # Inform about manual interruption
        if interrupted:
            self._console.print('User cancelled the job execution.', style='yellow')
            return False  # never suppress Ctrl+C/cancel

        return suppressed

    def _install_buffer(self) -> None:
        """
        Install a buffering logging handler and patch all available loggers
        """
        manager = logging.root.manager

        # Grab the default logger
        all_loggers = [logging.getLogger()]

        # Also grap every other registered logger
        for logger in manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                all_loggers.append(logger)

        # Patch all loggers
        self._patcher.patch(*all_loggers)

        # Patch the default logger getter to ensure consistency with any loggers
        #  that will be spawned in the future
        logging.getLogger = self._patcher.patched_getLogger

    def _replay_logs(self) -> None:
        """
        Replay all logs to the console.
        """
        if not self._buffer.has_records:
            return

        self._console.print('\n[dim]--- Job logs ---[/]', justify='center')
        self._log_panel.replay()
        self._console.print(f'[dim]--- End of job logs ---[/]', justify='center')

    def submit(
            self,
            job_cls: Type['Job'],
            configs: Union['JobConfig', None, Iterable[Union['JobConfig', None]]] = None,
            **kwargs
    ) -> list['Job']:
        """
        Submit one or more jobs to the pool.

        :param job_cls: Job class to submit to.
        :param configs: Job configs to iterate over.
        :param kwargs: Extra keyword arguments to pass to the underlying pool's `.submit` method.
        """
        jobs = self._pool.submit(job_cls, configs=configs, **kwargs)

        for job in jobs:
            self._tasks[job.job_id] = job.__class__.__name__

        # Update progress bar
        if self._progress:
            # Count total tasks, not just the jobs currently submitted
            self._progress.total = len(self._tasks)

        return jobs

    async def launch(self, mode: Literal['all', 'stream'] = 'all'):
        """
        Launch all submitted jobs and track their progress.

        :param mode: Whether to return all results together (`'all'`) or yield them as each job completes (`'stream'`).
        """
        if mode not in ['all', 'stream']:
            raise BadParameter(f'`mode` must be one of: \'all\', \'stream\'')

        # Start progress bar
        if self._progress:
            self._progress.total = len(self._tasks)
            self._progress.start()

        if mode == 'stream':
            return self._launch_stream()
        return await self._launch_all()

    async def _launch_all(self):
        # NB: Internally we stream the results, so that we can update the
        #  progress bar accordingly
        async for _ in self._launch_stream():
            pass
        return self._results

    async def _launch_stream(self):
        async for result in await self._pool.launch(mode='stream'):
            if self._progress:
                self._progress.complete_job(result)
            self._results.append(result)
            yield result

    def get_lock(self, name: str | None = None):
        return self._pool.get_lock(name)

    def get_queue(self, name: str, maxsize: int = 0):
        return self._pool.get_queue(name, maxsize=maxsize)

    def get_state(self, name: str):
        return self._pool.get_state(name)
