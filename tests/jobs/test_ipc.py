"""
Tests for the IPC backends: one suite over the backend of each pool type, and the worker-side backends rebuilt from a
handle. The process backend starts a `multiprocessing` manager per test (about 1 s on Windows).
"""

import asyncio
import signal
import threading
import time

import pytest

from tests.conftest import skipif_not_linux

from xtl.jobs.ipc import (
    AsyncIPCBackend,
    IPCBackend,
    IPCHandle,
    IPCHandleNames,
    ProcessIPCBackend,
    ThreadedIPCBackend,
)
from xtl.jobs.ipc.processes import (
    ProcessLock,
    ProcessQueue,
    ProcessState,
    ProxyProcessIPCBackend,
    _DirectCalls,
    _ThreadCalls,
    ignore_interrupts,
    shielded_context,
)
from xtl_core.system.compatibility import OS_WINDOWS

BACKENDS = [AsyncIPCBackend, ThreadedIPCBackend, ProcessIPCBackend]


@pytest.fixture(params=BACKENDS, ids=lambda c: c.__name__)
def backend(request):
    backend = request.param()
    backend.start()
    yield backend
    backend.stop()


@pytest.fixture
def worker(backend):
    """A worker-side backend, given a lock, a queue and a state registered by the pool."""
    backend.get_lock()
    backend.get_lock('l')
    backend.get_queue('q')
    backend.get_state('s')
    names = IPCHandleNames(locks={backend._default_lock_name, 'l'}, queues={'q'}, states={'s'})
    return backend.from_handle(backend.to_handle(names))


class TestLocks:
    def test_by_name(self, backend):
        lock = backend.get_lock('a')
        assert lock.name == 'a'
        assert backend.get_lock('a') is lock
        assert backend.get_lock('b') is not lock

    def test_default(self, backend):
        lock = backend.get_lock()
        assert lock.name == IPCBackend._default_lock_name
        assert backend.get_lock() is lock

    @pytest.mark.asyncio
    async def test_context_manager(self, backend):
        lock = backend.get_lock('a')
        async with lock as entered:
            assert entered is lock
        # Released: it can be taken again
        await asyncio.wait_for(lock.acquire(), timeout=5)
        await lock.release()

    @pytest.mark.asyncio
    async def test_excludes_concurrent_holders(self, backend):
        # Coroutines on the same thread too: a thread lock that belongs to the thread would let them all in
        lock = backend.get_lock('a')
        inside, peak = 0, 0

        async def critical():
            nonlocal inside, peak
            async with lock:
                inside += 1
                peak = max(peak, inside)
                await asyncio.sleep(0.02)
                inside -= 1

        await asyncio.wait_for(asyncio.gather(*(critical() for _ in range(3))), timeout=10)
        assert peak == 1


def run_in_new_loop(main, timeout: float = 10):
    """
    Run ``main()`` with ``asyncio.run()`` in a thread, as a job runs in a worker, and fail if it doesn't return in
    time (a thread left waiting in the loop's executor keeps ``asyncio.run()`` from returning).
    """
    outcome = {}

    def target():
        try:
            outcome['result'] = asyncio.run(main())
        except BaseException as e:
            outcome['error'] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), 'asyncio.run() did not return'
    if 'error' in outcome:
        raise outcome['error']
    return outcome['result']


class TestWaits:
    """Waiting doesn't block the event loop, and a wait that is given up leaves nothing behind."""

    def test_lock_waits_without_blocking_the_loop(self, backend):
        lock = backend.get_lock('a')

        async def main():
            await lock.acquire()
            waiting = asyncio.ensure_future(lock.acquire())
            await asyncio.sleep(0.1)  # the loop keeps running while the lock is taken
            assert not waiting.done()
            await lock.release()
            await asyncio.wait_for(waiting, timeout=5)
            await lock.release()

        run_in_new_loop(main)

    def test_timed_out_acquire_leaves_the_lock_free(self, backend):
        lock = backend.get_lock('a')

        async def main():
            await lock.acquire()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(lock.acquire(), timeout=0.1)
            await lock.release()
            await asyncio.wait_for(lock.acquire(), timeout=1)
            await lock.release()

        run_in_new_loop(main)

    def test_timed_out_get_takes_nothing(self, backend):
        q = backend.get_queue('q')

        async def main():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.1)
            await q.put('item')
            await asyncio.sleep(0.2)
            return await asyncio.wait_for(q.get(), timeout=1)

        assert run_in_new_loop(main) == 'item'

    def test_put_waits_while_full(self, backend):
        q = backend.get_queue('bounded', maxsize=1)

        async def main():
            await q.put('first')
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.put('dropped'), timeout=0.1)
            waiting = asyncio.ensure_future(q.put('second'))
            await asyncio.sleep(0.1)
            assert not waiting.done()
            assert await q.get() == 'first'
            await asyncio.wait_for(waiting, timeout=1)
            return await q.get(), q.empty()

        assert run_in_new_loop(main) == ('second', True)


class TestQueues:
    @pytest.mark.asyncio
    async def test_fifo(self, backend):
        q = backend.get_queue('q')
        assert q.empty()
        for item in ('a', 'b', {'c': 1}):
            await q.put(item)
        assert q.qsize() == 3
        assert not q.empty()
        assert [await q.get() for _ in range(3)] == ['a', 'b', {'c': 1}]
        assert q.empty()

    def test_by_name(self, backend):
        q = backend.get_queue('q')
        assert backend.get_queue('q') is q
        assert backend.get_queue('other') is not q


class TestStates:
    def test_dict_interface(self, backend):
        state = backend.get_state('s')
        state['a'] = 1
        state.update({'b': 2, 'c': 3})
        del state['c']

        assert state['a'] == 1
        assert 'b' in state and 'c' not in state
        assert len(state) == 2
        assert sorted(state.keys()) == ['a', 'b']
        assert sorted(state.values()) == [1, 2]
        assert sorted(state.items()) == [('a', 1), ('b', 2)]
        assert sorted(state) == ['a', 'b']
        assert state.get('a') == 1
        assert state.get('missing', 'default') == 'default'
        assert repr(state).endswith(f'({dict(state.items())})')

    def test_missing_key(self, backend):
        state = backend.get_state('s')
        with pytest.raises(KeyError):
            state['missing']
        with pytest.raises(KeyError):
            del state['missing']

    def test_by_name(self, backend):
        state = backend.get_state('s')
        state['a'] = 1
        assert backend.get_state('s')['a'] == 1
        assert 'a' not in backend.get_state('other')


class TestWorkerBackend:
    @pytest.mark.asyncio
    async def test_shares_the_pools_primitives(self, backend, worker):
        backend.get_state('s')['from'] = 'pool'
        await backend.get_queue('q').put('from pool')

        assert worker.get_state('s')['from'] == 'pool'
        assert await worker.get_queue('q').get() == 'from pool'

        worker.get_state('s')['from'] = 'worker'
        await worker.get_queue('q').put('from worker')
        assert backend.get_state('s')['from'] == 'worker'
        assert await backend.get_queue('q').get() == 'from worker'

    @pytest.mark.asyncio
    async def test_shares_the_pools_locks(self, backend, worker):
        async with backend.get_lock('l'):
            waiting = asyncio.ensure_future(worker.get_lock('l').acquire())
            await asyncio.sleep(0.1)
            assert not waiting.done()
        await asyncio.wait_for(waiting, timeout=5)
        await worker.get_lock('l').release()

    def test_default_lock(self, worker):
        assert worker.get_lock().name == IPCBackend._default_lock_name

    @pytest.mark.parametrize('kind', ['lock', 'queue', 'state'])
    def test_unregistered_names_raise(self, worker, kind):
        with pytest.raises(KeyError, match=f"{kind.capitalize()} 'unknown' was not registered") as e:
            getattr(worker, f'get_{kind}')('unknown')
        assert f"pool.get_{kind}('unknown')" in str(e.value)

    def test_start_and_stop_leave_the_pools_primitives(self, backend, worker):
        backend.get_state('s')['a'] = 1
        worker.start()
        worker.stop()
        assert backend.get_state('s')['a'] == 1

    def test_handle_holds_only_the_named_primitives(self, backend):
        backend.get_lock('l')
        backend.get_queue('q')
        backend.get_state('s')
        backend.get_state('not shared')
        handle = backend.to_handle(IPCHandleNames(locks={'l'}, queues={'q'}, states={'s'}))
        assert isinstance(handle, IPCHandle)
        assert set(handle.locks) == {'l'}
        assert set(handle.queues) == {'q'}
        assert set(handle.state) == {'s'}


def report_blocked(queue) -> None:
    """In a child process: whether SIGINT is blocked, as a shielded process starts."""
    queue.put(signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, set()))


class TestProcessIPCBackend:
    @pytest.mark.parametrize('kind', ['lock', 'queue', 'state'])
    def test_must_be_started(self, kind):
        with pytest.raises(RuntimeError, match='must be started'):
            getattr(ProcessIPCBackend(), f'get_{kind}')('a')

    def test_stop_shuts_the_manager_down(self):
        backend = ProcessIPCBackend()
        backend.start()
        backend.get_state('s')
        backend.stop()
        assert backend._manager is None
        with pytest.raises(RuntimeError, match='must be started'):
            backend.get_state('s')
        backend.stop()  # again: nothing to do

    def test_ignore_interrupts(self):
        previous = signal.getsignal(signal.SIGINT)
        try:
            ignore_interrupts()
            handler = signal.getsignal(signal.SIGINT)
        finally:
            signal.signal(signal.SIGINT, previous)
        if OS_WINDOWS:
            assert handler is signal.SIG_IGN
        else:
            # A handler, not SIG_IGN, which the programs a worker starts would inherit
            assert callable(handler)
            handler(signal.SIGINT, None)
            assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, set())

    @skipif_not_linux
    def test_shielded_processes_start_with_sigint_blocked(self):
        ctx = shielded_context()
        queue = ctx.SimpleQueue()
        process = ctx.Process(target=report_blocked, args=(queue,))
        process.start()
        process.join(30)
        assert queue.get() is True
        # The parent's mask is back as it was
        assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def test_worker_backend_cannot_create_primitives(self):
        worker = ProcessIPCBackend.from_handle(IPCHandle(locks={}, queues={}, state={}))
        assert isinstance(worker, ProxyProcessIPCBackend)
        assert isinstance(ProxyProcessIPCBackend.from_handle(IPCHandle({}, {}, {})), ProxyProcessIPCBackend)
        for method in (worker._raw_lock, worker._raw_queue, worker._raw_state):
            with pytest.raises(NotImplementedError):
                method('a')


class Blocks:
    """Stands in for a manager proxy whose calls take as long as the test wants, and records where they ran."""

    def __init__(self):
        self.started = threading.Event()
        self.go_on = threading.Event()
        self.threads: list[threading.Thread] = []
        self.calls: list[str] = []

    def _call(self, name: str, result=None):
        self.threads.append(threading.current_thread())
        self.started.set()
        assert self.go_on.wait(5)
        self.calls.append(name)
        return result

    # A dict
    def __setitem__(self, key, value):
        self._call(f'set {key}={value}')

    # A lock
    def acquire(self, blocking=True):
        return self._call('acquire', True)

    def release(self):
        self._call('release')

    # A queue
    def get_nowait(self):
        return self._call('get', 'item')

    def put_nowait(self, item):
        self._call(f'put {item}')


class TestParentCalls:
    """
    The pool's process calls the manager from a thread of its own: Ctrl+C interrupts only the main thread, and a call
    interrupted there left its reply unread on the connection, for the next call to read.
    """

    def test_backend_calls_from_a_thread(self):
        backend = ProcessIPCBackend()
        backend.start()
        try:
            primitives = [backend.get_lock('l'), backend.get_queue('q'), backend.get_state('s')]
            assert all(isinstance(p._calls, _ThreadCalls) for p in primitives)
            assert len({id(p._calls) for p in primitives}) == 1
        finally:
            backend.stop()

    def test_workers_call_directly(self, worker):
        # Workers ignore Ctrl+C
        if isinstance(worker, ProxyProcessIPCBackend):
            primitives = [worker.get_lock('l'), worker.get_queue('q'), worker.get_state('s')]
            assert all(isinstance(p._calls, _DirectCalls) for p in primitives)

    def test_calls_run_outside_the_main_thread(self):
        proxy, calls = Blocks(), _ThreadCalls()
        proxy.go_on.set()
        try:
            ProcessState(proxy, calls)['a'] = 1
        finally:
            calls.stop()
        assert proxy.calls == ['set a=1']
        assert proxy.threads[0] is not threading.current_thread()

    def test_ctrl_c_leaves_the_call_whole(self):
        proxy, calls = Blocks(), _ThreadCalls()
        state = ProcessState(proxy, calls)

        def ctrl_c():
            assert proxy.started.wait(5)
            signal.raise_signal(signal.SIGINT)
            time.sleep(0.2)
            proxy.go_on.set()

        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        thread = threading.Thread(target=ctrl_c)
        try:
            thread.start()
            with pytest.raises(KeyboardInterrupt):
                state['a'] = 1
                time.sleep(5)  # Windows: a lock's wait isn't interrupted, so Ctrl+C comes once the call returns
        finally:
            thread.join(5)
            calls.stop()
            signal.signal(signal.SIGINT, previous)
        assert proxy.calls == ['set a=1']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'make, expected',
        [
            (lambda proxy, calls: ProcessLock(proxy, 'l', calls).acquire(), ['acquire', 'release']),
            (lambda proxy, calls: ProcessQueue(proxy, calls).get(), ['get', 'put item']),
        ],
        ids=['acquire', 'get'],
    )
    async def test_cancelled_wait_gives_back_what_it_took(self, make, expected):
        # The try in flight when the wait is cancelled still ends in the thread; what it took is given back
        proxy, calls = Blocks(), _ThreadCalls()
        waiting = asyncio.ensure_future(make(proxy, calls))
        assert await asyncio.to_thread(proxy.started.wait, 5)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        proxy.go_on.set()
        await asyncio.to_thread(calls.stop)
        assert proxy.calls == expected
        assert all(t is proxy.threads[0] for t in proxy.threads)

    @pytest.mark.asyncio
    async def test_cancelled_wait_before_its_try_takes_nothing(self):
        proxy, calls = Blocks(), _ThreadCalls()
        state = ProcessState(proxy, calls)
        busy = asyncio.ensure_future(asyncio.to_thread(state.__setitem__, 'a', 1))
        assert await asyncio.to_thread(proxy.started.wait, 5)
        waiting = asyncio.ensure_future(ProcessLock(proxy, 'l', calls).acquire())
        await asyncio.sleep(0.1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        proxy.go_on.set()
        await busy
        await asyncio.to_thread(calls.stop)
        assert proxy.calls == ['set a=1']


@pytest.mark.parametrize('cls', [AsyncIPCBackend, ThreadedIPCBackend])
def test_in_process_backends_stop_forgets_the_primitives(cls):
    backend = cls()
    backend.start()
    backend.get_state('s')['a'] = 1
    backend.stop()
    assert 'a' not in backend.get_state('s')
