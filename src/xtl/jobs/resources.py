import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass
import threading
from typing import Optional
import weakref

from xtl import settings


__all__ = ['Resources', 'ResourceManager', 'ResourcesLease', 'get_total_resources', 'get_rc_manager']


@dataclass(frozen=True)
class Resources:
    """
    Represents a set of computational resources that can be requested and allocated for a job or a pool.
    """
    jobs: int = 1
    threads: int = 1
    processes: int = 1

    def cap_to(self, other: 'Resources') -> 'Resources':
        """
        Returns a new Resources instance where each field is capped to the corresponding field in `other`.

        :param other: The Resources instance to cap against.
        :return: A new Resources instance with capped values.
        """
        return Resources(
            jobs=min(self.jobs, other.jobs),
            threads=min(self.threads, other.threads),
            processes=min(self.processes, other.processes)
        )

    def fits_in(self, other: 'Resources') -> bool:
        """
        Checks if this Resources instance can fit within another Resources instance.

        :param other: The Resources instance to compare against.
        :return: True if this instance can fit within `other`, False otherwise.
        """
        return (
            self.jobs <= other.jobs and
            self.threads <= other.threads and
            self.processes <= other.processes
        )

    def clamp_min(self, minimum: int = 1) -> 'Resources':
        """
        Returns a new Resources instance where each field is clamped to be at least `minimum`.

        :param minimum: The minimum value for each resource field.
        :return: A new Resources instance with clamped values.
        """
        minimum = max(0, int(minimum))
        return Resources(
            jobs=max(minimum, self.jobs),
            threads=max(minimum, self.threads),
            processes=max(minimum, self.processes)
        )

    def __sub__(self, other: 'Resources') -> 'Resources':
        return Resources(
            jobs=max(0, self.jobs - other.jobs),
            threads=max(0, self.threads - other.threads),
            processes=max(0, self.processes - other.processes)
        )

    def __add__(self, other: 'Resources') -> 'Resources':
        return Resources(
            jobs=self.jobs + other.jobs,
            threads=self.threads + other.threads,
            processes=self.processes + other.processes
        )


CURRENT_LEASE: ContextVar['ResourcesLease | None'] = ContextVar('CURRENT_LEASE', default=None)
"""
Context variable to track the current ResourcesLease in scope.
"""


class ResourcesLease:
    """
    Represents a lease of resources granted by a ResourceManager. Supports nested leases for hierarchical resource
    management.

    :param manager: The ResourceManager that granted this lease.
    :param granted: The Resources that were granted for this lease.
    :param token: The ContextVar token for this lease, used to restore previous lease scope on release.
    :param parent: The parent ResourcesLease if this is a nested lease, or None if this is a root lease.
    """

    def __init__(self, manager: 'ResourceManager', granted: Resources, token: Optional[Token] = None,
                 parent: Optional['ResourcesLease'] = None):
        self._manager = manager
        self.granted = granted
        self._token = token
        self._parent = parent
        self._released = False

        # Resources available to nested pools created within this lease scope.
        self._child_available = granted
        self._child_condition = asyncio.Condition()

    async def acquire_child(self, requested: Resources, timeout: float | None = None) -> 'ResourcesLease':
        """
        Acquire a child lease from this current lease. The requested resources will be capped to the granted resources
        of this lease.

        :param requested: The resources being requested for the child lease.
        :param timeout: Optional timeout in seconds for acquiring the child lease. If None, wait indefinitely.
        :return: A ResourcesLease representing the granted child resources.
        """
        if self._released:
            raise RuntimeError('Cannot acquire child resources from a released lease')

        requested = requested.clamp_min(1)
        scoped = requested.cap_to(self.granted)

        async with self._child_condition:
            waiter = self._child_condition.wait_for(lambda: scoped.fits_in(self._child_available))
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=timeout)
            self._child_available = self._child_available - scoped

        child = ResourcesLease(manager=self._manager, granted=scoped, parent=self)
        child._token = CURRENT_LEASE.set(child)
        return child

    async def _release_child(self, granted: Resources) -> None:
        """
        Release child resources back to this lease. This is called by child leases when they are released.

        :param granted: The resources being released by the child lease.
        """
        async with self._child_condition:
            self._child_available = self._child_available + granted
            self._child_condition.notify_all()

    async def release(self) -> None:
        """
        Release this lease and return the granted resources back to the manager or parent lease.
        """
        if self._released:
            return

        if self._parent is None:
            await self._manager._release_global(self.granted)
        else:
            await self._parent._release_child(self.granted)

        if self._token is not None:
            CURRENT_LEASE.reset(self._token)
        self._released = True

    async def __aenter__(self) -> 'ResourcesLease':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class ResourceManager:
    """
    Manages a global pool of resources and grants leases to requesters. Supports hierarchical resource management
    through nested leases. Each ResourceManager is intended to be used within a single event loop to avoid sharing
    asyncio primitives across loops.

    :param total: The total resources available for allocation. Each field will be clamped to a minimum of 1.
    """
    def __init__(self, total: Resources):
        self._total = total.clamp_min(1)
        self._available = self._total
        self._condition = asyncio.Condition()

    async def acquire(self, requested: Resources, timeout: float | None = None) -> ResourcesLease:
        """
        Acquire a lease of resources from the manager. If there is a current lease in scope, this will attempt to
        acquire a child lease from it instead, ensuring that nested pools share the same manager budget.

        :param requested: The resources being requested.
        :param timeout: Optional timeout in seconds for acquiring the resources. If None, wait indefinitely.
        :return: A ResourcesLease representing the granted resources.
        """
        requested = requested.clamp_min(1)

        parent = CURRENT_LEASE.get()
        if parent is not None:
            if parent._manager is not self:
                raise RuntimeError('Nested pool uses a different ResourceManager than its parent lease')
            return await parent.acquire_child(requested=requested, timeout=timeout)

        scoped = requested.cap_to(self._total)

        async with self._condition:
            waiter = self._condition.wait_for(lambda: scoped.fits_in(self._available))
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=timeout)
            self._available = self._available - scoped

        lease = ResourcesLease(manager=self, granted=scoped)
        lease._token = CURRENT_LEASE.set(lease)
        return lease

    async def _release_global(self, granted: Resources) -> None:
        """
        Release resources back to the global pool. This is called by root leases when they are released.

        :param granted: The resources being released by the root lease.
        """
        async with self._condition:
            self._available = self._available + granted
            self._condition.notify_all()

    async def available(self) -> Resources:
        """
        Get the currently available resources in the manager.
        """
        async with self._condition:
            return self._available


def get_total_resources() -> Resources:
    """
    Get the total resources available for job execution, as defined in the settings.
    """
    return Resources(
        jobs=settings.jobs.resources.max_jobs,
        threads=settings.jobs.resources.max_threads,
        processes=settings.jobs.resources.max_processes,
    )


_MANAGERS_BY_LOOP: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ResourceManager]" = weakref.WeakKeyDictionary()
_MANAGERS_LOCK = threading.Lock()


def get_rc_manager(total: Resources | None = None) -> ResourceManager:
    """
    Returns one ResourceManager per running event loop. Avoids sharing asyncio primitives across loops.

    :param total: Optional total resources for the manager. If None, uses get_total_resources().
    :return: A ResourceManager instance for the current event loop.
    """
    loop = asyncio.get_running_loop()
    with _MANAGERS_LOCK:
        manager = _MANAGERS_BY_LOOP.get(loop)
        if manager is None:
            manager = ResourceManager(total=total or get_total_resources())
            _MANAGERS_BY_LOOP[loop] = manager
        return manager
