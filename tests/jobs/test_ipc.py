"""
Tests for the IPC backends: one suite over the backend of each pool type, and the worker-side backends rebuilt from a
handle. The process backend starts a `multiprocessing` manager per test (about 1 s on Windows).
"""

import asyncio
import threading

import pytest

from xtl.jobs.ipc import (
    AsyncIPCBackend,
    IPCBackend,
    IPCHandle,
    IPCHandleNames,
    ProcessIPCBackend,
    ThreadedIPCBackend,
)
from xtl.jobs.ipc.processes import ProxyProcessIPCBackend

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

    def test_worker_backend_cannot_create_primitives(self):
        worker = ProcessIPCBackend.from_handle(IPCHandle(locks={}, queues={}, state={}))
        assert isinstance(worker, ProxyProcessIPCBackend)
        assert isinstance(ProxyProcessIPCBackend.from_handle(IPCHandle({}, {}, {})), ProxyProcessIPCBackend)
        for method in (worker._raw_lock, worker._raw_queue, worker._raw_state):
            with pytest.raises(NotImplementedError):
                method('a')


@pytest.mark.parametrize('cls', [AsyncIPCBackend, ThreadedIPCBackend])
def test_in_process_backends_stop_forgets_the_primitives(cls):
    backend = cls()
    backend.start()
    backend.get_state('s')['a'] = 1
    backend.stop()
    assert 'a' not in backend.get_state('s')
