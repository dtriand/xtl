"""
Logging of jobs and pools: the handler and logger config models, the defaults, and the handler that sends the records
of worker processes back to the parent.
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

    def configure(self, logger: logging.Logger) -> None:
        """
        Configure the given logger based on this configuration. Handlers installed by this
        method are tagged with a ``_xtl_owned`` attribute and are replaced on consecutive
        calls of this method. Other handlers are not modified.

        :param logger: The logger to configure
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

        for handler_config in self.handlers:
            handler = handler_config.get_handler()
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
        fmt = '[%(asctime)s.%(msecs)03d:%(name)s] %(message)s'

    config = LoggerConfig(
        level=level,
        propagate=False,
        handlers=[StreamHandlerConfig(format=LoggingFormat(format=fmt, datefmt='%H:%M:%S'))],
    )
    return config


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
