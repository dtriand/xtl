from __future__ import annotations

import contextlib
from collections import deque
import logging
from datetime import datetime

import typer
from rich._log_render import LogRender
from rich.console import Console, ConsoleRenderable, Group
from rich.highlighter import Highlighter
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, \
    TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback

from xtl import settings
from xtl.common.compatibility import PY310_OR_LESS
from xtl.jobs import Job
from xtl.jobs.ipc import IPCLock
from xtl.jobs.pools import PoolProtocol, JobPool, BasePool
from xtl.cli.utilities.logging import JobHighlighter, JobPoolHighlighter
from xtl.logging.config import LoggerConfig

if PY310_OR_LESS:
    from typing_extensions import Self
else:
    from typing import Self, Any, Type, Literal


class BufferingHandler(logging.Handler):

    def __init__(self, tail_size: int = 20):
        super().__init__(level=logging.NOTSET)
        self._records: list[logging.LogRecord] = []
        self._tail: deque[logging.LogRecord] = deque(maxlen=tail_size)

    def emit(self, record: logging.LogRecord) -> None:
        record.getMessage()
        self._records.append(record)
        self._tail.append(record)

    def clear(self) -> None:
        self._records.clear()
        self._tail.clear()

    def flush_to(self, handler: logging.Handler) -> None:
        for record in self._records:
            with contextlib.suppress(Exception):
                handler.handle(record)
        self._records.clear()

    def tail_records(self) -> list[logging.LogRecord]:
        return list(self._tail)


class LogRenderer:

    def __init__(self, console: Console, **kwargs) -> None:
        self._console = console

        # TODO: Centralize these fmt strings in jobs.logging
        if settings.jobs.threading_debug:
            fmt = '[%(name)s<%(processName)s:%(threadName)s>] %(message)s'
        else:
            fmt = '[%(name)s] %(message)s'
        self._formatter = logging.Formatter(fmt=fmt)

        # TODO: Reorganize highlighters too
        self._highlighters: dict[str, Highlighter] = {
            'Job': JobHighlighter(),
            'Pool': JobPoolHighlighter(),
        }

        self._render = LogRender(
            show_time=kwargs.get('show_time', True),
            show_level=kwargs.get('show_level', True),
            show_path=kwargs.get('show_path', False),
            time_format=kwargs.get('time_format', '%X'),
            omit_repeated_times=kwargs.get('omit_repeated_times', True),
            level_width=kwargs.get('level_width', 8),
        )

    def _create_renderables(self, record: logging.LogRecord):
        formatted = self._formatter.format(record)
        message = Text(formatted)

        for name, highlighter in self._highlighters.items():
            if name in record.name:
                message = highlighter(message)

        renderables: list[ConsoleRenderable] = [message]
        if record.exc_info:
            renderables.append(
                Traceback.from_exception(
                    *record.exc_info,
                    show_locals=False
                )
            )

        return renderables

    def _create_level(self, record: logging.LogRecord) -> Text:
        return Text(
            f'{record.levelname:<{self._render.level_width}}',
            style=f'logging.level.{record.levelname.lower()}',
        )

    def render(self, record: logging.LogRecord, **kwargs) -> Table:
        level = self._create_level(record)
        renderables = self._create_renderables(record)
        return self._render(
            console=self._console,
            renderables=renderables,
            log_time=datetime.fromtimestamp(record.created),
            time_format=self._render.time_format,
            level=level,
            path=record.pathname if self._render.show_path else None,
            line_no=record.lineno if self._render.show_path else None,
            link_path=record.pathname if self._render.show_path else None,
        )


class LoggerHandlerPatcher:

    def __init__(self, handler: logging.Handler) -> None:
        self._logging_getLogger = logging.getLogger
        self._handler = handler

        self._original_handlers: dict[logging.Logger, list[logging.Handler]] = {}
        self._attached_loggers: list[logging.Logger] = []

    def patched_getLogger(self, name: str = None) -> logging.Logger:
        logger = self._logging_getLogger(name)
        self.patch(logger)
        return logger

    def patch(self, *loggers: logging.Logger) -> None:
        for logger in loggers:
            # Avoid double patching
            if logger in self._original_handlers:
                continue

            # Save original handlers
            self._original_handlers[logger] = list(logger.handlers)

            # Remove all handlers
            logger.handlers = []

            # Attach our own handler
            logger.addHandler(self._handler)
            self._attached_loggers.append(logger)

            # Ensure propagation
            logger.propagate = True

    def restore(self):
        # Restore the original handlers to the loggers
        for logger, handlers in self._original_handlers.items():
            with contextlib.suppress(Exception):
                logger.handlers = handlers
        self._original_handlers.clear()

        # Restore the original logging.getLogger signature
        if self._logging_getLogger:
            logging.getLogger = self._logging_getLogger
            self._logging_getLogger = None

    def remove_handler(self):
        for logger in self._attached_loggers:
            with contextlib.suppress(Exception):
                logger.handlers.remove(self._handler)
        self._attached_loggers.clear()


class LogPanel:

    def __init__(self, handler: BufferingHandler, console: Console, **kwargs) -> None:
        self._handler = handler
        self._console = console
        self._title = kwargs.pop('title', None)
        self._renderer = LogRenderer(
            console=console,
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


class PoolProgress:

    def __init__(self, console: Console, **kwargs) -> None:
        self._console = console

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(bar_width=self._console.width),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )

        self._task_id = self._progress.add_task(
            description='Waiting for jobs...',
            total=None
        )
        self._task = self._progress.tasks[self._task_id]

    def start(self):
        self._progress.start_task(self._task_id)

    def complete_job(self, result: Any):
        # TODO: Do additional checking for errors in results here
        self._progress.update(
            task_id=self._task_id,
            advance=1
        )

    @property
    def total(self) -> float | None:
        return self._progress.tasks[self._task_id].total

    @total.setter
    def total(self, total: float) -> None:
        self._progress.update(
            task_id=self._task_id,
            total=total
        )

    @property
    def description(self) -> str:
        return self._progress.tasks[self._task_id].description

    @description.setter
    def description(self, description: str) -> None:
        self._progress.update(
            task_id=self._task_id,
            description=description
        )

    def __rich__(self):
        return self._progress


class LivePool(PoolProtocol):

    def __init__(
            self,
            console: Console,
            pool_type: JobPool | str | None = None,
            max_jobs: int = 1,
            progress: bool = True,
    ) -> None:
        self._console: Console = console
        # TODO: determine tail size based on console height
        self._buffer = BufferingHandler(tail_size=20)

        # Rich components
        self._log_panel = LogPanel(
            console=self._console,
            handler=self._buffer,
            title='Job logs'
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
            job_cls: Type[Job],
            configs: 'JobConfig | None | Iterable[JobConfig | None]' = None,
            **kwargs
    ) -> list[Job]:

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
            raise typer.BadParameter(f'`mode` must be one of: \'all\', \'stream\'')

        # Start progress bar
        if self._progress:
            self._progress.total = len(self._tasks)
            self._progress.description = 'Running jobs'
            self._progress.start()

        if mode == 'stream':
            return self._launch_stream()
        return await self._launch_all()

    async def _launch_all(self):
        results = await self._pool.launch(mode='all')
        for result in results:
            self._progress.complete_job(result)
        self._results = results
        return results

    async def _launch_stream(self):
        async for result in await self._pool.launch(mode='stream'):
            self._progress.complete_job(result)
            self._results.append(result)
            yield result

    def get_lock(self, name: str | None = None):
        return self._pool.get_lock(name)

    def get_queue(self, name: str, maxsize: int = 0):
        return self._pool.get_queue(name, maxsize=maxsize)

    def get_state(self, name: str):
        return self._pool.get_state(name)
