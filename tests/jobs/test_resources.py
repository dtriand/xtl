import asyncio
from typing import Any

import pytest

from xtl import settings
from xtl.jobs.pools import BasePool
from xtl.jobs.ipc import IPCBackend, IPCLock, IPCQueue, IPCState, IPCHandle
from xtl.jobs.ipc.async_ import AsyncLock
from xtl.jobs.resources import CURRENT_LEASE, ResourceManager, Resources, get_rc_manager, get_total_resources


def each(n: int) -> Resources:
    return Resources(jobs=n, threads=n, processes=n)


class _DummyIPCBackend(IPCBackend):
    def start(self):
        pass

    def stop(self):
        pass

    def get_lock(self, name: str = None) -> IPCLock:
        return AsyncLock(name or self._default_lock_name)

    def get_queue(self, name: str, maxsize: int = 0) -> IPCQueue:
        pass

    def get_state(self, name: str) -> IPCState:
        pass

    def _raw_lock(self, name: str) -> Any:
        pass

    def _raw_queue(self, name: str) -> Any:
        pass

    def _raw_state(self, name: str) -> Any:
        pass

    @classmethod
    def from_handle(cls, handle: IPCHandle) -> IPCBackend:
        return cls()


class _DummyPool(BasePool):
    _ipc_cls = _DummyIPCBackend

    def _resolve_resources(self, max_jobs: int, requested: Resources | None) -> Resources:
        return requested or Resources(jobs=max_jobs, threads=1, processes=1)

    async def _execute_submission(self, submission):
        return None


@pytest.fixture(autouse=True)
def no_lease_in_scope():
    """Each test starts, and must end, with no lease in scope."""
    token = CURRENT_LEASE.set(None)
    yield
    CURRENT_LEASE.reset(token)


class TestResources:
    def test_cap_to(self):
        assert Resources(4, 1, 8).cap_to(Resources(2, 2, 2)) == Resources(2, 1, 2)

    @pytest.mark.parametrize(
        'other, fits',
        [
            (Resources(2, 2, 2), True),
            (Resources(3, 3, 3), True),
            (Resources(1, 2, 2), False),
            (Resources(2, 2, 1), False),
        ],
    )
    def test_fits_in(self, other, fits):
        assert Resources(2, 2, 2).fits_in(other) is fits

    def test_clamp_min(self):
        assert Resources(0, 1, 5).clamp_min(2) == Resources(2, 2, 5)
        assert Resources(-1, 0, 1).clamp_min(-3) == Resources(0, 0, 1)  # a negative minimum counts as 0

    def test_add_and_subtract(self):
        assert Resources(1, 2, 3) + Resources(1, 1, 1) == Resources(2, 3, 4)
        assert Resources(3, 2, 1) - Resources(1, 1, 2) == Resources(2, 1, 0)  # never below 0

    def test_total_from_settings(self, monkeypatch):
        monkeypatch.setattr(settings.jobs.resources, 'max_jobs', 3)
        monkeypatch.setattr(settings.jobs.resources, 'max_threads', 5)
        monkeypatch.setattr(settings.jobs.resources, 'max_processes', 7)
        assert get_total_resources() == Resources(3, 5, 7)


class TestResourceManager:
    @pytest.mark.asyncio
    async def test_acquire_release_restores_available(self):
        manager = ResourceManager(each(4))

        lease = await manager.acquire(each(2))
        assert lease.granted == each(2)
        assert await manager.available() == each(2)

        await lease.release()
        assert await manager.available() == each(4)

    @pytest.mark.asyncio
    async def test_total_at_least_one(self):
        manager = ResourceManager(Resources(0, 0, 0))
        assert await manager.available() == each(1)

    @pytest.mark.asyncio
    async def test_request_capped_to_total_and_at_least_one(self):
        manager = ResourceManager(Resources(2, 3, 4))
        lease = await manager.acquire(Resources(10, 0, -1))
        assert lease.granted == Resources(2, 1, 1)
        await lease.release()

    @pytest.mark.asyncio
    async def test_waits_until_released(self):
        manager = ResourceManager(each(2))
        first = await manager.acquire(each(2))
        CURRENT_LEASE.set(None)  # a request from outside the first lease

        waiting = asyncio.create_task(manager.acquire(each(1)))
        await asyncio.sleep(0.05)
        assert not waiting.done()

        await first.release()
        second = await asyncio.wait_for(waiting, timeout=1)
        assert second.granted == each(1)

    @pytest.mark.asyncio
    async def test_timeout_leaves_the_budget_untouched(self):
        manager = ResourceManager(each(1))
        first = await manager.acquire(each(1))
        CURRENT_LEASE.set(None)

        with pytest.raises(asyncio.TimeoutError):
            await manager.acquire(each(1), timeout=0.05)
        assert await manager.available() == each(0)

        CURRENT_LEASE.set(first)
        await first.release()
        assert await manager.available() == each(1)
        second = await manager.acquire(each(1))
        assert second.granted == each(1)
        await second.release()

    @pytest.mark.asyncio
    async def test_lease_of_another_manager_in_scope_raises(self):
        manager1 = ResourceManager(each(4))
        manager2 = ResourceManager(each(4))

        root = await manager1.acquire(each(2))
        with pytest.raises(RuntimeError, match='different ResourceManager'):
            await manager2.acquire(each(1))

        await root.release()


class TestResourcesLease:
    @pytest.mark.asyncio
    async def test_nested_acquire_from_parent_scope(self):
        manager = ResourceManager(each(4))

        root = await manager.acquire(each(4))
        # The manager has nothing left, but a nested request draws from the lease in scope
        child = await manager.acquire(each(2))
        assert child.granted == each(2)
        assert await manager.available() == each(0)

        await child.release()
        await root.release()
        assert await manager.available() == each(4)

    @pytest.mark.asyncio
    async def test_child_capped_to_parent(self):
        manager = ResourceManager(each(8))
        root = await manager.acquire(each(2))
        child = await root.acquire_child(Resources(4, 1, 0))
        assert child.granted == Resources(2, 1, 1)
        await child.release()
        await root.release()

    @pytest.mark.asyncio
    async def test_child_waits_for_sibling(self):
        manager = ResourceManager(each(2))
        root = await manager.acquire(each(2))
        first = await root.acquire_child(each(2))

        waiting = asyncio.create_task(root.acquire_child(each(1)))
        await asyncio.sleep(0.05)
        assert not waiting.done()

        CURRENT_LEASE.set(first)
        await first.release()
        second = await asyncio.wait_for(waiting, timeout=1)
        assert second.granted == each(1)

    @pytest.mark.asyncio
    async def test_nested_timeout_when_parent_budget_exhausted(self):
        manager = ResourceManager(each(2))

        root = await manager.acquire(each(2))
        child = await manager.acquire(each(2))

        # A sibling request from the root's scope: the root has no budget left for children
        CURRENT_LEASE.set(root)
        with pytest.raises(asyncio.TimeoutError):
            await manager.acquire(each(1), timeout=0.05)

        CURRENT_LEASE.set(child)
        await child.release()
        await root.release()

    @pytest.mark.asyncio
    async def test_released_parent_cannot_acquire_child(self):
        manager = ResourceManager(each(4))

        root = await manager.acquire(each(2))
        await root.release()

        with pytest.raises(RuntimeError, match='released lease'):
            await root.acquire_child(each(1))

    @pytest.mark.asyncio
    async def test_release_twice(self):
        manager = ResourceManager(each(4))
        lease = await manager.acquire(each(2))
        await lease.release()
        await lease.release()
        assert await manager.available() == each(4)

    @pytest.mark.asyncio
    async def test_context_manager_releases(self):
        manager = ResourceManager(each(4))
        async with await manager.acquire(each(3)) as lease:
            assert await manager.available() == each(1)
        assert lease._released
        assert await manager.available() == each(4)

    @pytest.mark.asyncio
    async def test_current_lease_restored_on_release(self):
        manager = ResourceManager(each(4))

        root = await manager.acquire(each(3))
        assert CURRENT_LEASE.get() is root

        child = await manager.acquire(each(1))
        assert CURRENT_LEASE.get() is child

        await child.release()
        assert CURRENT_LEASE.get() is root

        await root.release()
        assert CURRENT_LEASE.get() is None

    @pytest.mark.asyncio
    async def test_release_from_another_task_returns_the_resources_once(self):
        manager = ResourceManager(each(2))
        lease = await manager.acquire(each(1))

        # The lease's scope belongs to the task that acquired it, so it can't be reset from another
        with pytest.raises(ValueError, match='different Context'):
            await asyncio.create_task(lease.release())
        assert await manager.available() == each(2)

        await lease.release()
        assert await manager.available() == each(2)


class TestGetRcManager:
    def test_one_per_loop(self):
        async def managers():
            return get_rc_manager(), get_rc_manager()

        first, same = asyncio.run(managers())
        other, _ = asyncio.run(managers())
        assert first is same
        assert other is not first

    def test_total_applies_to_a_new_manager_only(self):
        async def managers():
            first = get_rc_manager(total=each(2))
            second = get_rc_manager(total=each(8))
            return first, second, await first.available()

        first, second, available = asyncio.run(managers())
        assert first is second
        assert available == each(2)

    def test_total_defaults_to_the_settings(self, monkeypatch):
        monkeypatch.setattr(settings.jobs.resources, 'max_jobs', 3)

        async def available():
            return await get_rc_manager().available()

        assert asyncio.run(available()).jobs == 3

    def test_needs_a_running_loop(self):
        with pytest.raises(RuntimeError):
            get_rc_manager()


class TestPoolLeases:
    @pytest.mark.asyncio
    async def test_nested_pools_share_the_budget(self):
        manager = ResourceManager(each(4))

        async with _DummyPool(max_jobs=4, resources_manager=manager) as outer:
            assert outer.resources == Resources(jobs=4, threads=1, processes=1)
            assert await manager.available() == Resources(jobs=0, threads=3, processes=3)

            # Opened in the same task, the inner pool draws from the outer pool's lease, not the manager
            async with _DummyPool(max_jobs=2, resources_manager=manager) as inner:
                assert inner.resources == Resources(jobs=2, threads=1, processes=1)
                assert await manager.available() == Resources(jobs=0, threads=3, processes=3)

            assert await manager.available() == Resources(jobs=0, threads=3, processes=3)

        assert await manager.available() == each(4)

    @pytest.mark.asyncio
    async def test_nested_pools_restore_the_lease_in_scope(self):
        manager = ResourceManager(each(3))

        async with _DummyPool(max_jobs=3, resources_manager=manager):
            outer_lease = CURRENT_LEASE.get()
            assert outer_lease is not None

            async with _DummyPool(max_jobs=1, resources_manager=manager):
                inner_lease = CURRENT_LEASE.get()
                assert inner_lease is not None
                assert inner_lease is not outer_lease

            assert CURRENT_LEASE.get() is outer_lease

        assert CURRENT_LEASE.get() is None

    @pytest.mark.asyncio
    async def test_pools_in_separate_tasks_wait_for_each_other(self):
        manager = ResourceManager(each(2))
        order = []

        async def use(name: str, hold: float):
            async with _DummyPool(max_jobs=2, resources_manager=manager):
                order.append(f'{name} in')
                await asyncio.sleep(hold)
                order.append(f'{name} out')

        first = asyncio.create_task(use('first', 0.1))
        await asyncio.sleep(0.02)
        await asyncio.gather(first, use('second', 0))
        assert order == ['first in', 'first out', 'second in', 'second out']
        assert await manager.available() == each(2)

    @pytest.mark.asyncio
    async def test_pool_capped_to_the_total(self):
        manager = ResourceManager(each(2))
        async with _DummyPool(max_jobs=8, resources_manager=manager) as pool:
            assert pool.resources.jobs == 2
