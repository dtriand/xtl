import queue
import threading
from typing import Any

from .base import IPCLock, IPCQueue, IPCState, IPCBackend, IPCHandle


class ThreadedLock(IPCLock):
    """
    Threaded context manager for locks
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._lock = threading.RLock()

    async def acquire(self) -> None:
        self._lock.acquire()

    async def release(self) -> None:
        self._lock.release()


class ThreadedQueue(IPCQueue):
    """
    Threaded interface for queues
    """

    def __init__(self, maxsize: int = 0):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    async def put(self, item: Any) -> None:
        self._q.put(item)

    async def get(self) -> Any:
        return self._q.get()

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


class ThreadedState(IPCState):
    """
    Threaded interface for dict-like shared state
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> list[Any]:
        with self._lock:
            return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return list(self._data.items())


class ThreadedIPCBackend(IPCBackend):
    """
    Threaded backend for IPC primitives
    """

    def __init__(self):
        self._locks: dict[str, ThreadedLock] = {}
        self._queues: dict[str, ThreadedQueue] = {}
        self._states: dict[str, ThreadedState] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._locks.clear()
        self._queues.clear()
        self._states.clear()

    def get_lock(self, name: str = None) -> ThreadedLock:
        if name is None:
            name = self._default_lock_name
        if name not in self._locks:
            self._locks[name] = ThreadedLock(name=name)
        return self._locks[name]

    def get_queue(self, name: str, maxsize: int = 0) -> ThreadedQueue:
        if name not in self._queues:
            self._queues[name] = ThreadedQueue(maxsize=maxsize)
        return self._queues[name]

    def get_state(self, name: str) -> ThreadedState:
        if name not in self._states:
            self._states[name] = ThreadedState()
        return self._states[name]

    # Threading primitives are also not pickleable. ThreadedPool always runs in the
    #  same process, so we share the backend by reference.
    def _raw_lock(self, name: str) -> Any:
        return self.get_lock(name)

    def _raw_queue(self, name: str) -> Any:
        return self.get_queue(name)

    def _raw_state(self, name: str) -> Any:
        return self.get_state(name)

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> 'ThreadedIPCBackend':
        backend = cls()
        # Iterate over lists to ensure atomic operations
        for name, lock in handle.locks.items():
            backend._locks[name] = lock
        for name, queue in handle.queues.items():
            backend._queues[name] = queue
        for name, state in handle.state.items():
            backend._states[name] = state
        return backend
