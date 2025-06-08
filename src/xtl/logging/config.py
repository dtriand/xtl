import logging
import sys
from typing import Any

from xtl.common.options import Option, Options


class LoggingFormat(Options):
    """
    Configuration for the logging format.
    """
    format: str = Option(default='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
                         desc='Format string for log messages')
    datefmt: str = Option(default='%H:%M:%S',
                          desc='Date format string for log messages')

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
    handler: type(logging.Handler) = Option(default=logging.Handler,
                                            desc='Handler class for logging')
    format: LoggingFormat = Option(default=LoggingFormat(),
                                   desc='Format configuration for the handler')
    options: dict = Option(default=dict(),
                           desc='Additional keyword arguments to pass to the '
                                'handler constructor')

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
    handler: type(logging.Handler) = Option(default=logging.StreamHandler,
                                            desc='Handler class for the stream')
    stream: Any = Option(default=sys.stdout,
                         desc='Stream to which log messages are sent')

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
    handler: type(logging.Handler) = Option(default=logging.FileHandler,
                                      desc='Handler class for the file')
    filename: str = Option(default='job.log', desc='File to which log messages are '
                                                   'written')
    mode: str = Option(default='a', desc='File mode for the log file (e.g., "a" for '
                                         'append)')
    encoding: str = Option(default='utf-8', desc='Encoding for the log file')

    def get_handler(self) -> logging.Handler:
        """
        Create a file handler based on the current configuration.

        :return: A logging.FileHandler instance configured with the specified filename,
                 mode, encoding, format, and options
        """
        handler = getattr(self, 'handler', logging.FileHandler)(filename=self.filename,
                                                                mode=self.mode,
                                                                encoding=self.encoding)
        handler.setFormatter(self.format.get_formatter())
        return handler


class LoggerConfig(Options):
    """
    Configuration for a Job logger.
    """
    level: int = Option(default=logging.INFO, desc='Logging level for the job logger',
                        ge=0, le=50)  # logging.UNSET to logging.CRITICAL
    propagate: bool = Option(default=False, desc='Whether to propagate log messages to '
                                                 'parent loggers')
    handlers: list[HandlerConfig] = Option(default_factory=\
                                               lambda: [StreamHandlerConfig()],
                                           desc='List of handlers for the job logger')

    def configure(self, logger: logging.Logger) -> None:
        """
        Configure the given logger based on this configuration.

        :param logger: The logger to configure
        """
        logger.setLevel(self.level)
        logger.propagate = self.propagate

        for handler_config in self.handlers:
            handler = handler_config.get_handler()
            logger.addHandler(handler)

        # Ensure at least a NullHandler is present
        if not logger.hasHandlers():
            logger.addHandler(logging.NullHandler())
