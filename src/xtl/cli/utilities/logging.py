from functools import partial

from rich.console import Console
from rich.logging import RichHandler
from rich.highlighter import RegexHighlighter, Highlighter

from xtl.logging.config import LoggerConfig, LoggingFormat, HandlerConfig


class JobHighlighter(RegexHighlighter):
    """
    Highlighter for xtl.jobs.job.Job loggers.
    """
    base_style = 'xtl.jobs.job.'
    highlights = [
        r'(?P<job_id>\[([^\]]+)\])',
    ]


class JobPoolHighlighter(RegexHighlighter):
    """
    Highlighter for xtl.jobs.pools.JobPool loggers.
    """
    base_style = 'xtl.jobs.pool.'
    highlights = [
        r'(?P<name>\[([^\]]+)\])',
    ]


def get_rich_logger_config(console: Console = None,
                           highlighter: Highlighter = None) -> LoggerConfig:
    """
    Returns a logger config with rich.logging.RichHandler
    """
    config = LoggerConfig(
        handlers=[
            HandlerConfig(
                handler=RichHandler,
                format=LoggingFormat(
                    format='[%(name)s] %(message)s',
                ),
                options={
                    'show_path': False,
                    'console': console,
                    'highlighter': highlighter,
                    'log_time_format': '%X'
                }
            )
        ]
    )
    return config


JobLoggerConfig = partial(get_rich_logger_config, highlighter=JobHighlighter())
JobPoolLoggerConfig = partial(get_rich_logger_config, highlighter=JobPoolHighlighter())