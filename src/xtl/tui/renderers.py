from datetime import datetime
import logging
from typing import TYPE_CHECKING

import rich.console
import rich.highlighter
import rich.table
import rich.text
import rich.traceback
from rich._log_render import LogRender

if TYPE_CHECKING:
    from xtl.tui.console import ConsoleIO


class LogRenderer:

    def __init__(
            self,
            console: 'ConsoleIO',
            *,
            log_fmt: str = '[%(name)s] %(message)s',
            highlighters: dict[str, rich.highlighter.Highlighter] | None = None,
        **kwargs
    ) -> None:
        self._console = console
        self._formatter = logging.Formatter(fmt=log_fmt)

        self._highlighters: dict[str, rich.highlighter.Highlighter] = {}
        if highlighters is not None:
            self._highlighters.update(highlighters)

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
        message = rich.text.Text(formatted)

        for highlighter in self._highlighters.values():
            message = highlighter(message)

        renderables: list[rich.console.ConsoleRenderable] = [message]
        if record.exc_info:
            renderables.append(
                rich.traceback.Traceback.from_exception(
                    *record.exc_info,
                    show_locals=False
                )
            )

        return renderables

    def _create_level(self, record: logging.LogRecord) -> rich.text.Text:
        return rich.text.Text(
            f'{record.levelname:<{self._render.level_width}}',
            style=f'logging.level.{record.levelname.lower()}',
        )

    def render(self, record: logging.LogRecord, **kwargs) -> rich.table.Table:
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