import queue
import threading
from typing import Any

from .base import IPCLock, IPCQueue, IPCState, IPCBackend, IPCHandle, NOT_YET, poll


class ThreadedLock(IPCLock):
    """
    Threaded context manager for locks

    Not reentrant, like the asyncio and process locks: it excludes coroutines on the same thread too.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._lock = threading.Lock()

    async def acquire(self) -> None:
        await poll(lambda: True if self._lock.acquire(blocking=False) else NOT_YET)

    async def release(self) -> None:
        self._lock.release()


class ThreadedQueue(IPCQueue):
    """
    Threaded interface for queues
    """

    def __init__(self, maxsize: int = 0):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def _try_put(self, item: Any) -> Any:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            return NOT_YET

    def _try_get(self) -> Any:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return NOT_YET

    async def put(self, item: Any) -> None:
        await poll(lambda: self._try_put(item))

    async def get(self) -> Any:
        return await poll(self._try_get)

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
        # A worker's backend (see from_handle) only gives out what the pool registered
        self._shim: bool = False

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
            if self._shim:
                raise self._unregistered('lock', name)
            self._locks[name] = ThreadedLock(name=name)
        return self._locks[name]

    def get_queue(self, name: str, maxsize: int = 0) -> ThreadedQueue:
        if name not in self._queues:
            if self._shim:
                raise self._unregistered('queue', name)
            self._queues[name] = ThreadedQueue(maxsize=maxsize)
        return self._queues[name]

    def get_state(self, name: str) -> ThreadedState:
        if name not in self._states:
            if self._shim:
                raise self._unregistered('state', name)
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
        backend._shim = True
        # Iterate over lists to ensure atomic operations
        for name, lock in handle.locks.items():
            backend._locks[name] = lock
        for name, q in handle.queues.items():
            backend._queues[name] = q
        for name, state in handle.state.items():
            backend._states[name] = state
        return backend
