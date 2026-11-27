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
