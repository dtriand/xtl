import asyncio
from typing import Any

import pytest

from xtl.jobs.pools2 import BasePool
from xtl.jobs.ipc import IPCBackend, IPCLock, IPCQueue, IPCState, IPCHandle
from xtl.jobs.resources import CURRENT_LEASE, ResourceManager, Resources


class _DummyIPCBackend(IPCBackend):
    def start(self): pass
    def stop(self): pass
    def get_lock(self, name: str = None) -> IPCLock: pass
    def get_queue(self, name: str, maxsize: int = 0) -> IPCQueue: pass
    def get_state(self, name: str) -> IPCState: pass
    def _raw_lock(self, name: str) -> Any: pass
    def _raw_queue(self, name: str) -> Any: pass
    def _raw_state(self, name: str) -> Any: pass

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> IPCBackend: return cls()


class _DummyPool(BasePool):

    _ipc_cls = _DummyIPCBackend

    async def _execute_submission(self, submission): return None


@pytest.mark.asyncio
async def test_root_acquire_release_restores_available():
    manager = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    lease = await manager.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))
    available_during = await manager.available()
    assert available_during == Resources(jobs=2, threads=2, processes=2, cores=2)

    await lease.release()
    available_after = await manager.available()
    assert available_after == Resources(jobs=4, threads=4, processes=4, cores=4)


@pytest.mark.asyncio
async def test_nested_acquire_from_parent_scope():
    manager = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    root = await manager.acquire(Resources(jobs=4, threads=4, processes=4, cores=4))
    # Global capacity is now exhausted, but nested acquire should still work via the parent lease.
    child = await manager.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))

    assert child.granted == Resources(jobs=2, threads=2, processes=2, cores=2)

    await child.release()
    await root.release()


@pytest.mark.asyncio
async def test_nested_timeout_when_parent_budget_exhausted():
    manager = ResourceManager(Resources(jobs=2, threads=2, processes=2, cores=2))

    root = await manager.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))
    child1 = await manager.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))

    # Force sibling acquisition attempt from root scope; root has no child budget left.
    token = CURRENT_LEASE.set(root)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await manager.acquire(Resources(jobs=1, threads=1, processes=1, cores=1), timeout=0.05)
    finally:
        CURRENT_LEASE.reset(token)

    await child1.release()
    await root.release()


@pytest.mark.asyncio
async def test_nested_manager_mismatch_raises():
    manager1 = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))
    manager2 = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    root = await manager1.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))
    with pytest.raises(RuntimeError, match='different ResourceManager'):
        await manager2.acquire(Resources(jobs=1, threads=1, processes=1, cores=1))

    await root.release()


@pytest.mark.asyncio
async def test_released_parent_cannot_acquire_child():
    manager = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    root = await manager.acquire(Resources(jobs=2, threads=2, processes=2, cores=2))
    await root.release()

    with pytest.raises(RuntimeError, match='released lease'):
        await root.acquire_child(Resources(jobs=1, threads=1, processes=1, cores=1))


@pytest.mark.asyncio
async def test_current_lease_context_restored_on_nested_release():
    manager = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    root = await manager.acquire(Resources(jobs=3, threads=3, processes=3, cores=3))
    assert CURRENT_LEASE.get() is root

    child = await manager.acquire(Resources(jobs=1, threads=1, processes=1, cores=1))
    assert CURRENT_LEASE.get() is child

    await child.release()
    assert CURRENT_LEASE.get() is root

    await root.release()
    assert CURRENT_LEASE.get() is None


@pytest.mark.asyncio
async def test_nested_pool_contexts_share_manager_budget():
    manager = ResourceManager(Resources(jobs=4, threads=4, processes=4, cores=4))

    async with _DummyPool(max_jobs=4, resources_manager=manager) as outer:
        assert outer.resources == Resources(jobs=4, threads=1, processes=1, cores=1)
        # Outer pool consumed global budget.
        assert await manager.available() == Resources(jobs=0, threads=3, processes=3, cores=3)

        async with _DummyPool(max_jobs=2, resources_manager=manager) as inner:
            assert inner.resources == Resources(jobs=2, threads=1, processes=1, cores=1)
            # Nested pool should draw from parent lease only, not global manager again.
            assert await manager.available() == Resources(jobs=0, threads=3, processes=3, cores=3)

        # Releasing inner pool should not change global availability while outer is active.
        assert await manager.available() == Resources(jobs=0, threads=3, processes=3, cores=3)

    # Releasing outer pool restores global availability.
    assert await manager.available() == Resources(jobs=4, threads=4, processes=4, cores=4)


@pytest.mark.asyncio
async def test_nested_pool_context_restores_current_lease_scope():
    manager = ResourceManager(Resources(jobs=3, threads=3, processes=3, cores=3))

    async with _DummyPool(max_jobs=3, resources_manager=manager):
        outer_lease = CURRENT_LEASE.get()
        assert outer_lease is not None

        async with _DummyPool(max_jobs=1, resources_manager=manager):
            inner_lease = CURRENT_LEASE.get()
            assert inner_lease is not None
            assert inner_lease is not outer_lease

        # Exiting inner scope restores the outer lease in context.
        assert CURRENT_LEASE.get() is outer_lease

    assert CURRENT_LEASE.get() is None

