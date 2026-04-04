import asyncio
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from typing import Any

from .base import IPCLock, IPCQueue, IPCState, IPCBackend, IPCHandle


class ProcessLock(IPCLock):
    """
    Process context manager for locks

    Wraps a multiprocessing.Manager().Lock() proxy with an async interface.
    """

    def __init__(self, lock):
        self._lock = lock

    async def acquire(self) -> None:
        # Manager lock acquisition is blocking; run it in an executor thread
        #  to avoid blocking the main event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._lock.acquire)

    async def release(self) -> None:
        self._lock.release()


class ProcessQueue(IPCQueue):
    """
    Process interface for queues

    Wraps a multiprocessing.Manager().Queue() proxy with an async interface.
    """

    def __init__(self, queue):
        self._q = queue

    async def put(self, item: Any) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._q.put, item)

    async def get(self) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._q.get)

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


class ProcessState(IPCState):
    """
    Process interface for dict-like shared state

    Wraps a multiprocessing.Manager().dict() proxy with a dict-like interface.
    """

    def __init__(self, data_):
        self._data = data_

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def values(self) -> list[Any]:
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self._data.items())


class ProcessIPCBackend(IPCBackend):
    """
    Process backend for IPC primitives

    Wraps a multiprocessing.Manager() to provide locks, queues,
    and dict-like state that can be shared across processes.
    """

    def __init__(self):
        self._manager: SyncManager | None = None
        self._locks: dict[str, ProcessLock] = {}
        self._queues: dict[str, ProcessQueue] = {}
        self._states: dict[str, ProcessState] = {}

        # Raw manager proxies for pickling into handles
        self._raw_locks: dict[str, Any] = {}
        self._raw_queues: dict[str, Any] = {}
        self._raw_states: dict[str, Any] = {}

    def start(self) -> None:
        self._manager = mp.Manager()

    def stop(self) -> None:
        self._locks.clear()
        self._queues.clear()
        self._states.clear()
        self._raw_locks.clear()
        self._raw_queues.clear()
        self._raw_states.clear()

        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def get_lock(self, name: str | None = None) -> ProcessLock:
        if name is None:
            name = self._default_lock_name
        if name not in self._locks:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting locks')
            raw_lock = self._manager.Lock()
            self._raw_locks[name] = raw_lock
            self._locks[name] = ProcessLock(raw_lock)
        return self._locks[name]

    def get_queue(self, name: str | None, maxsize: int = 0) -> ProcessQueue:
        if name not in self._queues:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting queues')
            raw_queue = self._manager.Queue(maxsize=maxsize)
            self._raw_queues[name] = raw_queue
            self._queues[name] = ProcessQueue(raw_queue)
        return self._queues[name]

    def get_state(self, name: str | None) -> ProcessState:
        if name not in self._states:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting state')
            raw_state = self._manager.dict()
            self._raw_states[name] = raw_state
            self._states[name] = ProcessState(raw_state)
        return self._states[name]

    def _raw_lock(self, name: str | None) -> Any:
        if name is None:
            name = self._default_lock_name
        return self._raw_locks.get(name)

    def _raw_queue(self, name: str | None) -> Any:
        return self._raw_queues.get(name)

    def _raw_state(self, name: str | None) -> Any:
        return self._raw_states.get(name)

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> 'ProxyProcessIPCBackend':
        return ProxyProcessIPCBackend(handle)


class ProxyProcessIPCBackend(IPCBackend):
    """
    Lightweight backend shim reconstructed inside a worker process from
    IPCHandles. It wraps the Manager proxies, but cannot create new
    primitives, only access pre-registered ones.
    """

    def __init__(self, handle: IPCHandle):
        self._locks: dict[str, ProcessLock] = {
            n: ProcessLock(h) for n, h in handle.locks.items()
        }
        self._queues: dict[str, ProcessQueue] = {
            n: ProcessQueue(h) for n, h in handle.queues.items()
        }
        self._states: dict[str, ProcessState] = {
            n: ProcessState(h) for n, h in handle.state.items()
        }

    # The manager is owned by the parent process, so we don't need
    #  to do anything to start/stop the backend in the worker process.
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_lock(self, name: str | None = None) -> ProcessLock:
        if name is None:
            name = self._default_lock_name
        try:
            return self._locks[name]
        except KeyError:
            raise KeyError(
                f'Lock {name!r} not pre-registered in the parent process. '
                f'Call pool.get_lock({name!r}) before launching jobs.'
            )

    def get_queue(self, name: str | None, maxsize: int = 0) -> ProcessQueue:
        try:
            return self._queues[name]
        except KeyError:
            raise KeyError(
                f'Queue {name!r} not pre-registered in the parent process. '
                f'Call pool.get_queue({name!r}) before launching jobs.'
            )

    def get_state(self, name: str | None) -> ProcessState:
        try:
            return self._states[name]
        except KeyError:
            raise KeyError(
                f'State {name!r} not pre-registered in the parent process. '
                f'Call pool.get_state({name!r}) before launching jobs.'
            )

    def _raw_lock(self, name: str | None) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    def _raw_queue(self, name: str | None) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    def _raw_state(self, name: str | None) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> 'ProxyProcessIPCBackend':
        return cls(handle)