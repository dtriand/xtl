import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import multiprocessing as mp
import multiprocessing.context
from multiprocessing.managers import SyncManager
import queue
import signal
import threading
from typing import Any, Callable

from xtl_core.system.compatibility import OS_WINDOWS

from .base import IPCLock, IPCQueue, IPCState, IPCBackend, IPCHandle, NOT_YET, poll


def _on_interrupt(signum, frame) -> None:
    pass


def ignore_interrupts() -> None:
    """
    Make the current process ignore Ctrl+C (``SIGINT``), for the processes of a pool: Ctrl+C reaches every process on
    the console, and is the parent's to handle. On POSIX with a handler that does nothing rather than ``SIG_IGN``,
    which programs started from the process would inherit (it isn't reset on ``exec``), so that a batch file's
    programs still stop on ``SIGINT``; on Windows with ``SIG_IGN``, since a handler would still interrupt the
    process's waits.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN if OS_WINDOWS else _on_interrupt)
    if not OS_WINDOWS:
        # Blocked since the process was created (see `_ShieldedStart`); a Ctrl+C meanwhile is pending,
        #  and now goes to the handler. Programs started from here get it unblocked.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})


class _ShieldedStart:
    """
    A process that ignores Ctrl+C from its start, until :func:`ignore_interrupts` takes over in its initializer: before
    that, a process started with spawn imports xtl for a second or so, and Ctrl+C would kill it (which breaks a process
    pool). On POSIX, ``SIGINT`` is blocked in the current thread while the process is created, which the child
    inherits; a Ctrl+C for this process stays pending until the block ends. On Windows, Ctrl+C is switched off for this
    process meanwhile (``SetConsoleCtrlHandler(NULL, TRUE)``), which the child inherits; a Ctrl+C in those few
    milliseconds is lost.
    """

    def start(self) -> None:
        if OS_WINDOWS:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCtrlHandler(None, True)
            try:
                super().start()  # type: ignore[misc]
            finally:
                kernel32.SetConsoleCtrlHandler(None, False)
        else:
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                super().start()  # type: ignore[misc]
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)


# NB: Processes started with spawn are pickled for the child, so their classes are defined at module level
class _ShieldedSpawnProcess(_ShieldedStart, multiprocessing.context.SpawnProcess):
    pass


class _ShieldedSpawnContext(multiprocessing.context.SpawnContext):
    Process = _ShieldedSpawnProcess


_SHIELDED_CONTEXTS: dict[str, type] = {'spawn': _ShieldedSpawnContext}

if not OS_WINDOWS:

    class _ShieldedForkProcess(_ShieldedStart, multiprocessing.context.ForkProcess):
        pass

    class _ShieldedForkContext(multiprocessing.context.ForkContext):
        Process = _ShieldedForkProcess

    _SHIELDED_CONTEXTS['fork'] = _ShieldedForkContext

    if hasattr(multiprocessing.context, 'ForkServerContext'):

        class _ShieldedForkServerProcess(_ShieldedStart, multiprocessing.context.ForkServerProcess):
            pass

        class _ShieldedForkServerContext(multiprocessing.context.ForkServerContext):
            Process = _ShieldedForkServerProcess

        _SHIELDED_CONTEXTS['forkserver'] = _ShieldedForkServerContext


def shielded_context() -> multiprocessing.context.BaseContext:
    """
    A ``multiprocessing`` context for the processes of a pool (its workers and its manager), with the default start
    method, whose processes ignore Ctrl+C from their start; their initializer must call :func:`ignore_interrupts`.
    """
    return _SHIELDED_CONTEXTS[mp.get_start_method()]()


class _DirectCalls:
    """
    Calls to the manager, made from the calling thread: for the workers, which ignore Ctrl+C.
    """

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """
        Call the manager through ``fn(*args)`` and return what it returns.
        """
        return fn(*args)

    def attempt(self, fn: Callable[[], Any], undo: Callable[[Any], Any] | None = None) -> Any:
        """
        A try of :func:`~xtl.jobs.ipc.base.poll`: what ``fn()`` returns, or an awaitable of it.

        :param fn: A non-blocking try, which returns :data:`~xtl.jobs.ipc.base.NOT_YET` when it got nothing.
        :param undo: Gives back what ``fn()`` got, for a try that ends after its wait was cancelled.
        """
        return fn()

    def stop(self) -> None:
        pass


class _ThreadCalls(_DirectCalls):
    """
    Calls to the manager, made one at a time from a thread of their own, while the caller waits: for the pool's
    process. Ctrl+C interrupts only the main thread's waits; a call interrupted there (``KeyboardInterrupt``, or
    ``InterruptedError`` on Windows with a handler that doesn't raise) left its reply unread on the thread's connection
    to the manager, for the next call to read. Here it only interrupts the wait, and the call ends in its thread.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='xtl-manager', initializer=self._started)
        self._thread: int | None = None
        # The thread starts now, with the pool, rather than on its first call: starting it waits as `Future.result()`
        #  does (see `call()`)
        self.call(lambda: None)

    def _started(self) -> None:
        self._thread = threading.get_ident()

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        # NB: Waits on a bare lock, not with `Future.result()`: Ctrl+C in a `Condition.wait()` can leave the
        #  condition's lock released (seen with Python 3.10 on Windows), and its `with` block then raises
        #  `RuntimeError: release unlocked lock` instead of `KeyboardInterrupt`
        done = threading.Lock()
        done.acquire()
        future = self._executor.submit(fn, *args)
        future.add_done_callback(lambda _: done.release())
        done.acquire()
        return future.result()

    async def attempt(self, fn: Callable[[], Any], undo: Callable[[Any], Any] | None = None) -> Any:
        future = self._executor.submit(fn)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # A try that hadn't started is dropped; one that had still ends, and gives back what it got
            if undo is not None:
                future.add_done_callback(lambda done: self._undo(done, undo))
            raise

    def _undo(self, done: Future, undo: Callable[[Any], Any]) -> None:
        """
        Give back what a try got, from the calls' thread (the callback of a try that is already done runs in the
        caller's thread).
        """
        if threading.get_ident() != self._thread:
            with contextlib.suppress(RuntimeError):  # stopped meanwhile
                self._executor.submit(self._undo, done, undo)
            return
        if done.cancelled() or done.exception() is not None:
            return
        if (result := done.result()) is not NOT_YET:
            with contextlib.suppress(Exception):
                undo(result)

    def stop(self) -> None:
        """
        Wait for the calls that were started, then end the thread.
        """
        self._executor.shutdown(wait=True)


class ProcessLock(IPCLock):
    """
    Process context manager for locks

    Wraps a multiprocessing.Manager().Lock() proxy with an async interface. Each try to acquire it is a call to the
    manager process.
    """

    def __init__(self, lock, name: str, calls: _DirectCalls | None = None):
        super().__init__(name)
        self._lock = lock
        self._calls = calls or _DirectCalls()

    def _try_acquire(self) -> Any:
        return True if self._lock.acquire(False) else NOT_YET

    async def acquire(self) -> None:
        await poll(lambda: self._calls.attempt(self._try_acquire, undo=lambda _: self._lock.release()))

    async def release(self) -> None:
        self._calls.call(self._lock.release)


class ProcessQueue(IPCQueue):
    """
    Process interface for queues

    Wraps a multiprocessing.Manager().Queue() proxy with an async interface. Each try to put or get an item is a call
    to the manager process.
    """

    def __init__(self, queue, calls: _DirectCalls | None = None):
        self._q = queue
        self._calls = calls or _DirectCalls()

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
        await poll(lambda: self._calls.attempt(lambda: self._try_put(item)))

    async def get(self) -> Any:
        # An item taken by a try whose wait was cancelled goes back at the end of the queue
        return await poll(lambda: self._calls.attempt(self._try_get, undo=self._q.put_nowait))

    def empty(self) -> bool:
        return self._calls.call(self._q.empty)

    def qsize(self) -> int:
        return self._calls.call(self._q.qsize)


class ProcessState(IPCState):
    """
    Process interface for dict-like shared state

    Wraps a multiprocessing.Manager().dict() proxy with a dict-like interface.
    """

    def __init__(self, data_, calls: _DirectCalls | None = None):
        self._data = data_
        self._calls = calls or _DirectCalls()

    def __getitem__(self, key: str) -> Any:
        return self._calls.call(self._data.__getitem__, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._calls.call(self._data.__setitem__, key, value)

    def __delitem__(self, key: str) -> None:
        self._calls.call(self._data.__delitem__, key)

    def __contains__(self, key: str) -> bool:
        return self._calls.call(self._data.__contains__, key)

    def __len__(self) -> int:
        return self._calls.call(self._data.__len__)

    def keys(self) -> list[str]:
        return self._calls.call(lambda: list(self._data.keys()))

    def values(self) -> list[Any]:
        return self._calls.call(lambda: list(self._data.values()))

    def items(self) -> list[tuple[str, Any]]:
        return self._calls.call(lambda: list(self._data.items()))


class ProcessIPCBackend(IPCBackend):
    """
    Process backend for IPC primitives

    Wraps a multiprocessing.Manager() to provide locks, queues,
    and dict-like state that can be shared across processes.
    """

    def __init__(self):
        self._manager: SyncManager | None = None
        # The pool's calls to the manager, which Ctrl+C mustn't interrupt: from a thread of their own once started
        self._calls: _DirectCalls = _DirectCalls()
        self._locks: dict[str, ProcessLock] = {}
        self._queues: dict[str, ProcessQueue] = {}
        self._states: dict[str, ProcessState] = {}

        # Raw manager proxies for pickling into handles
        self._raw_locks: dict[str, Any] = {}
        self._raw_queues: dict[str, Any] = {}
        self._raw_states: dict[str, Any] = {}

    def start(self) -> None:
        # As `mp.Manager()`, but the manager process leaves Ctrl+C to the pool
        self._manager = SyncManager(ctx=shielded_context())
        self._manager.start(initializer=ignore_interrupts)
        self._calls = _ThreadCalls()

    def stop(self) -> None:
        self._locks.clear()
        self._queues.clear()
        self._states.clear()
        self._raw_locks.clear()
        self._raw_queues.clear()
        self._raw_states.clear()

        self._calls.stop()
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

    def get_lock(self, name: str = None) -> ProcessLock:
        if name is None:
            name = self._default_lock_name
        if name not in self._locks:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting locks')
            raw_lock = self._manager.Lock()
            self._raw_locks[name] = raw_lock
            self._locks[name] = ProcessLock(raw_lock, name=name, calls=self._calls)
        return self._locks[name]

    def get_queue(self, name: str, maxsize: int = 0) -> ProcessQueue:
        if name not in self._queues:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting queues')
            raw_queue = self._manager.Queue(maxsize=maxsize)
            self._raw_queues[name] = raw_queue
            self._queues[name] = ProcessQueue(raw_queue, calls=self._calls)
        return self._queues[name]

    def get_state(self, name: str) -> ProcessState:
        if name not in self._states:
            if self._manager is None:
                raise RuntimeError(f'{ProcessIPCBackend.__name__} must be started before getting state')
            raw_state = self._manager.dict()
            self._raw_states[name] = raw_state
            self._states[name] = ProcessState(raw_state, calls=self._calls)
        return self._states[name]

    def _raw_lock(self, name: str) -> Any:
        if name is None:
            name = self._default_lock_name
        return self._raw_locks.get(name)

    def _raw_queue(self, name: str) -> Any:
        return self._raw_queues.get(name)

    def _raw_state(self, name: str) -> Any:
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
        self._locks: dict[str, ProcessLock] = {n: ProcessLock(h, name=n) for n, h in handle.locks.items()}
        self._queues: dict[str, ProcessQueue] = {n: ProcessQueue(h) for n, h in handle.queues.items()}
        self._states: dict[str, ProcessState] = {n: ProcessState(h) for n, h in handle.state.items()}

    # The manager is owned by the parent process, so we don't need
    #  to do anything to start/stop the backend in the worker process.
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_lock(self, name: str = None) -> ProcessLock:
        if name is None:
            name = self._default_lock_name
        try:
            return self._locks[name]
        except KeyError:
            raise self._unregistered('lock', name) from None

    def get_queue(self, name: str, maxsize: int = 0) -> ProcessQueue:
        try:
            return self._queues[name]
        except KeyError:
            raise self._unregistered('queue', name) from None

    def get_state(self, name: str) -> ProcessState:
        try:
            return self._states[name]
        except KeyError:
            raise self._unregistered('state', name) from None

    def _raw_lock(self, name: str | None) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    def _raw_queue(self, name: str) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    def _raw_state(self, name: str) -> Any:
        raise NotImplementedError(f'{ProxyProcessIPCBackend.__name__} cannot create new primitives')

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> 'ProxyProcessIPCBackend':
        return cls(handle)
