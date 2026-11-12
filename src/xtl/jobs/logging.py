"""
Logging of jobs and pools: the handler and logger config models, the defaults, the filter that names jobs on screen,
and the handler that sends the records of worker processes back to the parent.
"""

import contextlib
import logging
from logging.handlers import QueueHandler
import sys
from typing import Any

from xtl_core.options import Option, Options


class LoggingFormat(Options):
    """
    Configuration for the logging format.
    """

    format: str = Option(
        default='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', desc='Format string for log messages'
    )
    datefmt: str = Option(default='%H:%M:%S', desc='Date format string for log messages')

    def get_formatter(self) -> logging.Formatter:
        """
        Create a logging formatter based on the current configuration.

        :return: A logging.Formatter instance configured with the specified format
        """
        return logging.Formatter(fmt=self.format, datefmt=self.datefmt)


class HandlerConfig(Options):
    """
    Base configuration for a logging handler.
    """

    handler: type(logging.Handler) = Option(default=logging.Handler, desc='Handler class for logging')
    format: LoggingFormat = Option(default=LoggingFormat(), desc='Format configuration for the handler')
    options: dict = Option(default=dict(), desc='Additional keyword arguments to pass to the handler constructor')

    def get_handler(self) -> logging.Handler:
        """
        Create a logging handler based on the current configuration.

        :return: A logging.Handler instance configured with the specified class,
                 format, and options
        """
        handler = getattr(self, 'handler', logging.Handler)(**self.options)
        handler.setFormatter(self.format.get_formatter())
        return handler


class StreamHandlerConfig(HandlerConfig):
    """
    Configuration for a stream handler.
    """

    handler: type(logging.Handler) = Option(default=logging.StreamHandler, desc='Handler class for the stream')
    stream: Any = Option(default=sys.stdout, desc='Stream to which log messages are sent')

    def get_handler(self) -> logging.Handler:
        """
        Create a stream handler based on the current configuration.

        :return: A logging.StreamHandler instance configured with the specified stream,
                 format, and options
        """
        handler = getattr(self, 'handler', logging.StreamHandler)(stream=self.stream)
        handler.setFormatter(self.format.get_formatter())
        return handler


class FileHandlerConfig(HandlerConfig):
    """
    Configuration for a file handler.
    """

    handler: type(logging.Handler) = Option(default=logging.FileHandler, desc='Handler class for the file')
    filename: str = Option(default='job.log', desc='File to which log messages are written')
    mode: str = Option(default='a', desc='File mode for the log file (e.g., "a" for append)')
    encoding: str = Option(default='utf-8', desc='Encoding for the log file')

    def get_handler(self) -> logging.Handler:
        """
        Create a file handler based on the current configuration.

        :return: A logging.FileHandler instance configured with the specified filename,
                 mode, encoding, format, and options
        """
        handler = getattr(self, 'handler', logging.FileHandler)(
            filename=self.filename, mode=self.mode, encoding=self.encoding
        )
        handler.setFormatter(self.format.get_formatter())
        return handler


class LoggerConfig(Options):
    """
    Configuration for a Job logger.
    """

    level: int = Option(
        default=logging.INFO, desc='Logging level for the job logger', ge=0, le=50
    )  # logging.UNSET to logging.CRITICAL
    propagate: bool = Option(default=False, desc='Whether to propagate log messages to parent loggers')
    handlers: list[HandlerConfig] = Option(
        default_factory=lambda: [StreamHandlerConfig()], desc='List of handlers for the job logger'
    )

    def configure(self, logger: logging.Logger, label: str | None = None) -> None:
        """
        Configure the given logger based on this configuration. Handlers installed by this
        method are tagged with a ``_xtl_owned`` attribute and are replaced on consecutive
        calls of this method. Other handlers are not modified. Each installed handler gets a
        :class:`JobNameFilter`, so that formats can use ``%(job)s``.

        :param logger: The logger to configure
        :param label: Optional, how the logger's own records are named on screen, e.g. a pool's label. If not
            provided, records are named relative to the logger's parent, which suits jobs.
        """
        # Remove previously installed handlers by this method (iterate over a copy, since
        #  removing from the list being iterated skips every other handler)
        for handler in list(logger.handlers):
            if getattr(handler, '_xtl_owned', False):
                logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.close()

        logger.setLevel(self.level)
        logger.propagate = self.propagate

        name_filter = JobNameFilter.for_logger(logger, label)
        for handler_config in self.handlers:
            handler = handler_config.get_handler()
            handler.addFilter(name_filter)
            # Mark handlers created by this method
            handler._xtl_owned = True
            logger.addHandler(handler)

        # Ensure at least a NullHandler is present
        if not logger.hasHandlers():
            null = logging.NullHandler()
            null._xtl_owned = True
            logger.addHandler(null)


def get_logger_config(level: int = logging.INFO) -> LoggerConfig:
    """
    Get a LoggerConfig instance for configuring job and pool loggers.
    """
    from xtl import settings

    if settings.jobs.threading_debug:
        fmt = '[%(asctime)s.%(msecs)03d:%(processName)s:%(threadName)s:%(name)s] %(message)s'
    else:
        fmt = '[%(asctime)s.%(msecs)03d:%(job)s] %(message)s'

    config = LoggerConfig(
        level=level,
        propagate=False,
        handlers=[StreamHandlerConfig(format=LoggingFormat(format=fmt, datefmt='%H:%M:%S'))],
    )
    return config


def pass_through(logger: logging.Logger) -> None:
    """
    Make a logger pass its records up the tree: remove the handlers xtl installed on it (others are kept), reset its
    level, so that it inherits its parent's, and let it propagate.

    :param logger: The logger to reset.
    """
    for handler in list(logger.handlers):
        if getattr(handler, '_xtl_owned', False):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


_POOL_CLASSES: set[str] = set()
"""Names of the pool classes, filled by ``BasePool.__init_subclass__``, so that a pool's segment of a logger name can
be told apart from a job's."""


def register_pool_class(name: str) -> None:
    """
    Register the name of a pool class, so that :class:`JobNameFilter` recognizes the loggers of its pools.

    :param name: The class name.
    """
    _POOL_CLASSES.add(name)


def pool_label(segment: str) -> str | None:
    """
    How a pool's logger segment is shown on screen: ``<Cls>|<name>`` for a named pool, ``<Cls>`` for one named by its
    id (``<Cls>#<pool_id>``).

    :param segment: One dot-separated part of a logger name.
    :return: The label, or None if the segment isn't a pool's.
    """
    cls_name, sep, _ = segment.partition('#')
    if sep and cls_name in _POOL_CLASSES:
        return cls_name
    cls_name, sep, _ = segment.partition('|')
    if sep and cls_name in _POOL_CLASSES:
        return segment
    return None


class JobNameFilter(logging.Filter):
    """
    Sets ``record.job``, the name of the record's logger as shown on screen, relative to a base logger:

    - the base's own records get its label, e.g. ``MultiprocessPool``;
    - records from below it get their name relative to it, with each nested pool hidden and shown as a ``>`` instead,
      e.g. ``Talker|01.2`` or ``Outer|01>Talker|03``; a nested pool's own records end in its label;
    - records from loggers outside it keep their full name.

    The filter lets every record through.
    """

    def __init__(self, base: str, label: str | None = None):
        """
        :param base: Name of the base logger; an empty string names records relative to the root logger.
        :param label: How the base's own records are named. If not provided, they get the base's name.
        """
        super().__init__()
        self.base = base
        self.label = label if label is not None else base

    @classmethod
    def for_logger(cls, logger: logging.Logger, label: str | None = None) -> 'JobNameFilter':
        """
        The filter for the handlers of a logger: with a label, the logger is the base (a pool); without one, its
        parent is, so that a job is named by its id and its steps by ``<job_id>.<i>``.

        :param logger: The logger whose handlers get the filter.
        :param label: Optional, how the logger's own records are named.
        """
        if label is not None:
            return cls(base=logger.name, label=label)
        parent = logger.parent
        return cls(base='' if parent is None or parent is logging.root else parent.name)

    def display_name(self, name: str) -> str:
        """
        The name of a logger as shown on screen.

        :param name: The logger's name.
        """
        if name == self.base:
            return self.label
        if not self.base:
            parts = name.split('.')
        elif name.startswith(self.base + '.'):
            parts = name[len(self.base) + 1 :].split('.')
        else:
            return name

        levels: list[str] = []
        current: list[str] = []
        for i, part in enumerate(parts):
            label = pool_label(part)
            if label is None:
                current.append(part)
                continue
            # A nested pool closes the current level; it's only shown for its own records
            if current:
                levels.append('.'.join(current))
                current = []
            if i == len(parts) - 1:
                levels.append(label)
        if current:
            levels.append('.'.join(current))
        return '>'.join(levels)

    def filter(self, record: logging.LogRecord) -> bool:
        record.job = self.display_name(record.name)
        return True


_JSON_TYPES = (str, int, float, bool, type(None))


class RelayHandler(QueueHandler):
    """
    The sink of a worker process: puts every record on a queue as a JSON-safe dict (:meth:`record_to_dict`), for the
    parent to re-emit.
    """

    def prepare(self, record: logging.LogRecord) -> dict[str, Any]:
        return self.record_to_dict(record)

    @staticmethod
    def record_to_dict(record: logging.LogRecord) -> dict[str, Any]:
        """
        Convert a log record to a JSON-safe dict, for the parent process to re-emit
        (:meth:`xtl.jobs.pools.MultiprocessPool._relay_log`). The message is formatted with its arguments, the
        traceback is kept as text, and any other value that JSON can't hold is turned into a string.

        :param record: The record to convert.
        :return: A dict that :func:`logging.makeLogRecord` turns back into a record.
        """
        data = dict(record.__dict__)
        data['msg'] = record.getMessage()
        data['args'] = None
        if record.exc_info and not record.exc_text:
            data['exc_text'] = logging.Formatter().formatException(record.exc_info)
        data['exc_info'] = None
        data.pop('message', None)
        return {key: value if isinstance(value, _JSON_TYPES) else str(value) for key, value in data.items()}
