import logging
from typing import Any, Literal, Type, TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from typer import BadParameter

from xtl.logging.config import LoggerConfig
from xtl.jobs.pools import PoolProtocol, JobPool, BasePool
from xtl.common.compatibility import PY310_OR_LESS
from xtl.tui.highlighters import JobHighlighter, PoolHighlighter
from xtl.tui.logging.handlers import BufferingHandler, LoggerHandlerPatcher
from xtl.tui.renderers import LogRenderer
from xtl.tui.progress import ProgressBar
from xtl.tui.styles import JOB_STYLES

if PY310_OR_LESS:
    from typing_extensions import Self
else:
    from typing import Self, Any, Type, Literal

if TYPE_CHECKING:
    from xtl.jobs import Job, JobConfig
    from xtl.tui.console import ConsoleIO


import rich.progress
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
        )

    @classmethod
    def get_default_columns(cls) -> tuple[rich.progress.ProgressColumn, ...]:
        return (
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn('[progress.description]{task.description}'),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeElapsedColumn(),
        )

    def start(self):
        self._job_task.description = 'Running jobs'
        self._job_task.start()

    def complete_job(self, result: Any):
        # TODO: Do additional checking for errors in results here
        self._job_task.advance(1)

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

    def __init__(self, handler: BufferingHandler, console: 'ConsoleIO', **kwargs) -> None:
        self._handler = handler
        self._console = console
        self._title = kwargs.pop('title', None)
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
                Text('Waiting for logs...', style='dim italic', justify='center')
            ]

        return Panel(
            Group(*logs),
            title=self._title,
            border_style='dim'
        )


class LivePool(PoolProtocol):

    def __init__(
            self,
            console: 'ConsoleIO',
            pool_type: JobPool | str | None = None,
            max_jobs: int = 1,
            progress: bool = True,
    ) -> None:
        from xtl import settings

        self._console: 'ConsoleIO' = console
        self._theme = Theme(JOB_STYLES)
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
        self._live: Live | None = None

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
            pool_type: JobPool | str | None = None,
            max_jobs: int = 1
    ) -> BasePool:
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
        self._console.push_theme(self._theme)

        components: list[Any] = [self._log_panel]
        if self._progress:
            components.append(self._progress)

        renderable = Group(*components)

        live = Live(
            renderable=renderable,
            console=self._console,
            refresh_per_second=20,
            transient=True
        )
        self._live = live
        live.start()

        self._install_buffer()

        await self._pool.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        suppressed = await self._pool.__aexit__(exc_type, exc_val, exc_tb)

        # Check for errors
        # ...

        try:
            # Do something depending on error
            # if exc_val:
            #     # Stop rich.Live but persist on screen
            #     if self._live:
            #         self._live.transient = False
            #         self._live.stop()
            #     raise typer.Abort()

            self._buffer.clear()

            # Stop rich.Live and clear output
            if self._live:
                self._live.transient = True
                self._live.stop()

            return suppressed
        finally:
            self._patcher.restore()
            self._patcher.remove_handler()
            self._console.pop_theme()

    def _install_buffer(self) -> None:
        manager = logging.root.manager

        all_loggers = [logging.getLogger()]
        for logger in manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                all_loggers.append(logger)

        self._patcher.patch(*all_loggers)
        logging.getLogger = self._patcher.patched_getLogger

    def submit(
            self,
            job_cls: Type['Job'],
            configs: 'JobConfig | None | Iterable[JobConfig | None]' = None,
            **kwargs
    ) -> list['Job']:

        jobs = self._pool.submit(job_cls, configs=configs, **kwargs)

        for job in jobs:
            self._tasks[job.job_id] = job.__class__.__name__

        # Update progress bar
        if self._progress:
            # Count total tasks, not just the jobs currently submitted
            self._progress.total = len(self._tasks)

        return jobs

    async def launch(self, mode: Literal['all', 'stream'] = 'stream'):
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
        results = await self._pool.launch(mode='all')
        for result in results:
            if self._progress:
                self._progress.complete_job(result)
        self._results = results
        return results

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
