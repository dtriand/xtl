"""
Logging of jobs and pools: the handler and logger config models, the defaults, the filter that names jobs on screen,
and the handler that sends the records of worker processes back to the parent.
"""

import contextlib
import importlib
import logging
from logging.handlers import QueueHandler
from pathlib import Path
import sys
from typing import Any, ClassVar

from pydantic import SerializationInfo, field_serializer, field_validator

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


class MissingPlaceholderError(ValueError):
    """
    A handler spec uses a placeholder that has no value for this job, e.g. ``{job_directory}`` for a job without a
    directory.
    """


_STREAMS = {'stdout': lambda: sys.stdout, 'stderr': lambda: sys.stderr}
"""Streams that a config can name, resolved when the handler is made."""


class HandlerConfig(Options):
    """
    Base configuration for a logging handler.

    A config is data: dumped, it names the handler class by its import path and its own config class under
    ``config_cls``, so that :class:`LoggerConfig` restores the right subclass from a dict or JSON.
    """

    handler: type(logging.Handler) = Option(default=logging.Handler, desc='Handler class for logging')
    format: LoggingFormat = Option(default=LoggingFormat(), desc='Format configuration for the handler')
    options: dict = Option(default=dict(), desc='Additional keyword arguments to pass to the handler constructor')
    level: int = Option(
        default=logging.NOTSET, desc="Level of the handler (default: NOTSET, the logger config's level)", ge=0, le=50
    )

    @staticmethod
    def _import_path(obj: type) -> str:
        """
        The import path of a class, e.g. ``'logging.FileHandler'``.
        """
        return f'{obj.__module__}.{obj.__qualname__}'

    @staticmethod
    def _import_symbol(path: str) -> Any:
        """
        Import a symbol from its import path, e.g. ``'logging.FileHandler'``.

        :raises ImportError: If the path can't be resolved.
        """
        module_path, _, attr = path.rpartition('.')
        if not module_path or not attr:
            raise ImportError(f'Invalid import path: {path!r}')
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attr)
        except AttributeError as e:
            raise ImportError(f'Could not resolve {path!r}') from e

    @field_validator('handler', mode='before')
    @classmethod
    def _import_handler(cls, value: Any) -> Any:
        return cls._import_symbol(value) if isinstance(value, str) else value

    @field_serializer('handler', when_used='json')
    def _handler_path(self, value: type[logging.Handler]) -> str:
        return self._import_path(value)

    def to_spec(self, mode: str = 'python') -> dict[str, Any]:
        """
        Dump this config with its config class under ``config_cls``, for :meth:`from_spec` to restore.

        :param mode: ``'python'`` keeps classes and objects, ``'json'`` turns them into JSON values.
        """
        data = self.model_dump(mode=mode)
        data['config_cls'] = self._import_path(type(self))
        return data

    @staticmethod
    def from_spec(data: dict[str, Any]) -> 'HandlerConfig':
        """
        Restore a config dumped by :meth:`to_spec`, as the class named by ``config_cls`` (default:
        :class:`HandlerConfig`).

        :raises TypeError: If ``config_cls`` doesn't name a :class:`HandlerConfig` subclass.
        """
        data = dict(data)
        config_cls = data.pop('config_cls', None)
        cls = HandlerConfig._import_symbol(config_cls) if config_cls else HandlerConfig
        if not (isinstance(cls, type) and issubclass(cls, HandlerConfig)):
            raise TypeError(f'{config_cls!r} is not a {HandlerConfig.__name__} subclass')
        return cls.model_validate(data)

    def get_handler(self) -> logging.Handler:
        """
        Create a logging handler based on the current configuration.

        :return: A logging.Handler instance configured with the specified class,
                 format, and options
        """
        handler = getattr(self, 'handler', logging.Handler)(**self.options)
        handler.setFormatter(self.format.get_formatter())
        if self.level:
            handler.setLevel(self.level)
        return handler

    PLACEHOLDERS: ClassVar[tuple[str, ...]] = ('job_directory', 'job_id', 'job_class', 'pool')
    """The placeholders that :meth:`fill` replaces, e.g. ``{job_directory}``."""

    def fill(self, values: dict[str, str | None]) -> 'HandlerConfig':
        """
        A copy of this config with the placeholders filled in: in every string field, except ``format``, and in the
        string values of ``options``. Only :attr:`PLACEHOLDERS` are replaced; other braces are left alone.

        :param values: The value of each placeholder; None if it has none (e.g. a job without a directory).
        :raises MissingPlaceholderError: If a placeholder that is used has no value.
        """

        def fill_text(text: str) -> str:
            for key in self.PLACEHOLDERS:
                token = '{' + key + '}'
                if token not in text:
                    continue
                value = values.get(key)
                if value is None:
                    raise MissingPlaceholderError(f'No value for the placeholder {token} in {text!r}')
                text = text.replace(token, str(value))
            return text

        update: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if name == 'format':
                continue
            if isinstance(value, str):
                update[name] = fill_text(value)
            elif name == 'options':
                update[name] = {k: fill_text(v) if isinstance(v, str) else v for k, v in value.items()}
        return self.model_copy(update=update)


class StreamHandlerConfig(HandlerConfig):
    """
    Configuration for a stream handler.
    """

    handler: type(logging.Handler) = Option(default=logging.StreamHandler, desc='Handler class for the stream')
    stream: Any = Option(
        default='stdout',
        desc="Stream to which log messages are sent: 'stdout' or 'stderr', resolved when the handler is made, or a "
        "stream object (which JSON can't hold)",
    )

    @field_serializer('stream', when_used='json')
    def _stream_name(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        for name, get_stream in _STREAMS.items():
            if value is get_stream():
                return name
        raise ValueError(f"A stream object can't be serialized, use 'stdout' or 'stderr' instead: {value!r}")

    def get_handler(self) -> logging.Handler:
        """
        Create a stream handler based on the current configuration.

        :return: A logging.StreamHandler instance configured with the specified stream,
                 format, and options
        """
        stream = self.stream
        if isinstance(stream, str):
            if stream not in _STREAMS:
                raise ValueError(f"Unknown stream {stream!r}, expected 'stdout' or 'stderr'")
            stream = _STREAMS[stream]()
        handler = getattr(self, 'handler', logging.StreamHandler)(stream=stream)
        handler.setFormatter(self.format.get_formatter())
        if self.level:
            handler.setLevel(self.level)
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
        if self.level:
            handler.setLevel(self.level)
        return handler


class JobFileHandlerConfig(FileHandlerConfig):
    """
    A log file in each job's directory, with the job's and its steps' records down to DEBUG. Add it to a pool with
    ``pool.add_job_handler(JobFileHandlerConfig(), name='file')``. The file goes with the directory when the job
    succeeds (unless files are kept) and stays when it fails.
    """

    filename: str = Option(default='{job_directory}/xtl.log', desc='File to which log messages are written')
    mode: str = Option(default='w', desc='File mode for the log file (e.g., "a" for append)')
    level: int = Option(default=logging.DEBUG, desc='Level of the handler', ge=0, le=50)
    format: LoggingFormat = Option(
        default=LoggingFormat(format='[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'),
        desc='Format configuration for the handler',
    )

    def get_handler(self) -> logging.Handler:
        """
        Create the file handler, and the file's directory if it doesn't exist yet.
        """
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        return super().get_handler()


def _remove_xtl_handlers(logger: logging.Logger) -> None:
    """
    Remove the handlers that xtl added to a logger: those built from configs are closed, live ones (passed in by the
    user) are only detached. Other handlers are kept.
    """
    # Iterate over a copy, since removing from the list being iterated skips every other handler
    for handler in list(logger.handlers):
        if getattr(handler, '_xtl_owned', False):
            logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
        elif getattr(handler, '_xtl_live', False):
            logger.removeHandler(handler)


class LoggerConfig(Options):
    """
    Configuration for a Job logger.
    """

    level: int = Option(
        default=logging.INFO, desc='Logging level for the job logger', ge=0, le=50
    )  # logging.UNSET to logging.CRITICAL
    propagate: bool = Option(default=False, desc='Whether to propagate log messages to parent loggers')
    handlers: list[HandlerConfig | logging.Handler] = Option(
        default_factory=lambda: [StreamHandlerConfig()],
        desc='Handlers for the job logger: configs, or live handlers (which stay in this process and are not '
        'serialized)',
    )

    @field_validator('handlers', mode='before')
    @classmethod
    def _restore_handler_configs(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [HandlerConfig.from_spec(item) if isinstance(item, dict) else item for item in value]
        return value

    @field_serializer('handlers')
    def _dump_handlers(self, value: list[HandlerConfig | logging.Handler], info: SerializationInfo) -> list[Any]:
        dumped = []
        for item in value:
            if isinstance(item, HandlerConfig):
                dumped.append(item.to_spec(mode='json' if info.mode_is_json() else 'python'))
            elif info.mode_is_json():
                raise ValueError(f"A live handler can't be serialized, use a {HandlerConfig.__name__}: {item!r}")
            else:
                dumped.append(item)
        return dumped

    def configure(self, logger: logging.Logger, label: str | None = None) -> None:
        """
        Configure the given logger based on this configuration. Handlers built from configs
        are tagged with a ``_xtl_owned`` attribute, and live handlers with ``_xtl_live``; both
        are replaced on consecutive calls of this method (only the former are closed). Other
        handlers are not modified. Each installed handler gets a :class:`JobNameFilter`, unless
        it has one already, so that formats can use ``%(job)s``.

        :param logger: The logger to configure
        :param label: Optional, how the logger's own records are named on screen, e.g. a pool's label. If not
            provided, records are named relative to the logger's parent, which suits jobs.
        """
        _remove_xtl_handlers(logger)

        logger.propagate = self.propagate

        name_filter = JobNameFilter.for_logger(logger, label)
        levels = [self.level]
        for item in self.handlers:
            if isinstance(item, logging.Handler):
                handler = item
                handler._xtl_live = True
            else:
                handler = item.get_handler()
                handler._xtl_owned = True
            # A handler without a level of its own gets the config's, so that a lower one elsewhere doesn't widen it
            if handler.level == logging.NOTSET:
                handler.setLevel(self.level)
            levels.append(handler.level)
            if not any(isinstance(f, JobNameFilter) for f in handler.filters):
                handler.addFilter(name_filter)
            logger.addHandler(handler)

        # The logger only pre-filters: it lets through what its most verbose handler wants
        logger.setLevel(min(levels) if self.handlers else self.level)

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
    _remove_xtl_handlers(logger)
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
    How a pool's logger segment is shown on screen, without the pool's id: ``<Cls>|<name>`` for a named pool
    (``<Cls>|<name>#<pool_id>``), ``<Cls>`` for an unnamed one (``<Cls>#<pool_id>``).

    :param segment: One dot-separated part of a logger name.
    :return: The label, or None if the segment isn't a pool's.
    """
    label, sep, _ = segment.partition('#')
    if not sep:
        return None
    if label in _POOL_CLASSES or label.partition('|')[0] in _POOL_CLASSES:
        return label
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
