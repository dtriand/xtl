import asyncio
from typing import Any

from .base import IPCLock, IPCQueue, IPCBackend, IPCHandle
from .threads import ThreadedState


class AsyncLock(IPCLock):
    """
    Async context manager for locks
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._lock.acquire()

    async def release(self) -> None:
        self._lock.release()


class AsyncQueue(IPCQueue):
    """
    Async interface for queues
    """

    def __init__(self, maxsize: int = 0):
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: Any) -> None:
        await self._q.put(item)

    async def get(self) -> Any:
        return await self._q.get()

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


# NB: AsyncState is implemented as a ThreadedState, because using an asyncio.Lock to
#  protect access to the underlying dict would require all methods to be async and
#  would be incompatible with the synchronous interface of IPCState.
class AsyncState(ThreadedState):
    ...


class AsyncIPCBackend(IPCBackend):
    """
    Async backend for IPC primitives
    """

    def __init__(self):
        self._locks: dict[str, AsyncLock] = {}
        self._queues: dict[str, AsyncQueue] = {}
        self._states: dict[str, AsyncState] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._locks.clear()
        self._queues.clear()
        self._states.clear()

    def get_lock(self, name: str | None = None) -> AsyncLock:
        if name is None:
            name = self._default_lock_name
        if name not in self._locks:
            self._locks[name] = AsyncLock()
        return self._locks[name]

    def get_queue(self, name: str | None, maxsize: int = 0) -> AsyncQueue:
        if name not in self._queues:
            self._queues[name] = AsyncQueue(maxsize=maxsize)
        return self._queues[name]

    def get_state(self, name: str | None) -> AsyncState:
        if name not in self._states:
            self._states[name] = AsyncState()
        return self._states[name]

    # Asyncio primitives are not pickleable. Since SimplePool and AsyncPool always run on
    #  the same event loop, we share the backend by reference.
    def _raw_lock(self, name: str | None) -> Any:
        return self.get_lock(name)

    def _raw_queue(self, name: str | None) -> Any:
        return self.get_queue(name)

    def _raw_state(self, name: str | None) -> Any:
        return self.get_state(name)

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> 'AsyncIPCBackend':
        backend = cls()
        # Iterate over lists to ensure atomic operations
        for name, lock in handle.locks.items():
            backend._locks[name] = lock
        for name, queue in handle.queues.items():
            backend._queues[name] = queue
        for name, state in handle.state.items():
            backend._states[name] = state
        return backend
