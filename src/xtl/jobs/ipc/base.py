from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

POLL_FIRST = 0.001
"""Seconds before the first retry of :func:`poll`."""

POLL_MAX = 0.05
"""Longest time between two retries of :func:`poll`."""

NOT_YET = object()
"""What an attempt passed to :func:`poll` returns when it should be retried."""


async def poll(attempt: Callable[[], Any]) -> Any:
    """
    Wait for a thread or process primitive without blocking the event loop: call ``attempt`` until it returns
    something other than :data:`NOT_YET`, sleeping between tries, twice as long each time up to :data:`POLL_MAX`.

    Unlike waiting in an executor thread, which can't be stopped, a cancelled or timed-out wait leaves nothing behind:
    no thread that later takes the lock or an item, and nothing for ``asyncio.run()`` to wait for on exit.

    :param attempt: A non-blocking try, e.g. ``lock.acquire(blocking=False)`` or ``queue.get_nowait()``.
    :return: What the successful attempt returned.
    """
    delay = POLL_FIRST
    while (result := attempt()) is NOT_YET:
        await asyncio.sleep(delay)
        delay = min(2 * delay, POLL_MAX)
    return result


class IPCLock(abc.ABC):
    """
    Unified async context manager for locks
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    async def acquire(self) -> None: ...

    @abc.abstractmethod
    async def release(self) -> None: ...

    async def __aenter__(self) -> 'IPCLock':
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release()


class IPCQueue(abc.ABC):
    """
    Unified async interface for queues
    """

    @abc.abstractmethod
    async def put(self, item: Any) -> None: ...

    @abc.abstractmethod
    async def get(self) -> Any: ...

    @abc.abstractmethod
    def empty(self) -> bool: ...

    @abc.abstractmethod
    def qsize(self) -> int: ...


class IPCState(abc.ABC):
    """
    Unified interface for dict-like shared state
    """

    @abc.abstractmethod
    def __getitem__(self, key: str) -> Any: ...

    @abc.abstractmethod
    def __setitem__(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    def __delitem__(self, key: str) -> None: ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def keys(self) -> list[str]: ...

    @abc.abstractmethod
    def values(self) -> list[Any]: ...

    @abc.abstractmethod
    def items(self) -> list[tuple[str, Any]]: ...

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, mapping: dict[str, Any]) -> None:
        # Each key is updated atomically, but the entire update is not atomic.
        for key, value in mapping.items():
            self[key] = value

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({dict(self.items())})'


class IPCBackend(abc.ABC):
    """
    Abstract IPC backend interface for creating locks, queues, and shared state.
    """

    _default_lock_name = '__xtl_default_lock__'

    @abc.abstractmethod
    def start(self) -> None:
        """
        Start any background infrastructure needed for this backend.
        """
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """
        Tear down any background infrastructure and clean up resources.
        """
        ...

    @abc.abstractmethod
    def get_lock(self, name: str = None) -> IPCLock:
        """
        Get a lock by name. If name is None, return the default lock.
        """
        ...

    @abc.abstractmethod
    def get_queue(self, name: str, maxsize: int = 0) -> IPCQueue:
        """
        Get a queue by name.
        """
        ...

    @abc.abstractmethod
    def get_state(self, name: str) -> IPCState:
        """
        Get a shared state dict by name.
        """
        ...

    @abc.abstractmethod
    def _raw_lock(self, name: str) -> Any:
        """
        Get the raw (pickleable) lock object by name, for use in IPCHandle.
        """
        ...

    @abc.abstractmethod
    def _raw_queue(self, name: str) -> Any:
        """
        Get the raw (pickleable) queue object by name, for use in IPCHandle.
        """
        ...

    @abc.abstractmethod
    def _raw_state(self, name: str) -> Any:
        """
        Get the raw (pickleable) state dict object by name, for use in IPCHandle.
        """
        ...

    def to_handle(self, names: IPCHandleNames) -> IPCHandle:
        """
        Create an IPCHandle containing the requested locks, queues and state dicts.
        """
        return IPCHandle(
            locks={name: self._raw_lock(name) for name in names.locks},
            queues={name: self._raw_queue(name) for name in names.queues},
            state={name: self._raw_state(name) for name in names.states},
        )

    @classmethod
    @abc.abstractmethod
    def from_handle(cls, handle: IPCHandle) -> IPCBackend:
        """
        Reconstruct a backend from an IPCHandle inside a worker. The returned backend
        is a lightweight shim: it does not own the underlying IPC primitives and cannot
        create new ones.
        """
        ...

    @staticmethod
    def _unregistered(kind: str, name: str) -> KeyError:
        """
        The error of a worker-side backend asked for a primitive that the pool didn't register.

        :param kind: ``'lock'``, ``'queue'`` or ``'state'``.
        :param name: The name of the primitive.
        """
        return KeyError(
            f'{kind.capitalize()} {name!r} was not registered with the pool. '
            f'Call pool.get_{kind}({name!r}) before launching jobs.'
        )


@dataclass
class IPCHandleNames:
    """
    Names of IPC primitives to include in an IPCHandle.
    """

    locks: set[str] = field(default_factory=set)
    queues: set[str] = field(default_factory=set)
    states: set[str] = field(default_factory=set)


class IPCHandle:
    """
    Pickleable envelope for transporting IPC primitives across process boundaries.
    """

    def __init__(self, locks: dict[str, Any], queues: dict[str, Any], state: Any):
        self._locks = locks
        self._queues = queues
        self._state = state

    @property
    def locks(self) -> dict[str, Any]:
        return self._locks

    @property
    def queues(self) -> dict[str, Any]:
        return self._queues

    @property
    def state(self) -> Any:
        return self._state
