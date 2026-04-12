from collections import deque
import contextlib
import logging


class BufferingHandler(logging.Handler):

    def __init__(self, tail_size: int = 20):
        super().__init__(level=logging.NOTSET)
        self._records: deque[logging.LogRecord] = deque(maxlen=10_000)  # NB: Limit the maximum number of stored records
        self._tail: deque[logging.LogRecord] = deque(maxlen=tail_size)

    @property
    def has_records(self) -> bool:
        return len(self._records) > 0

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
