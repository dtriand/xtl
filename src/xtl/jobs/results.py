from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xtl_core.system.compatibility import PY310_OR_LESS

if PY310_OR_LESS:

    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


__all__ = ['JobStatus', 'JobResults', 'SteppedJobResults', 'BatchResults']


class JobStatus(StrEnum):
    """
    Where a job submitted to a pool stands. The pool sets it, in the process that opened the pool, the same way in every
    pool type.
    """

    SUBMITTED = 'submitted'
    """Submitted, not launched yet"""
    QUEUED = 'queued'
    """Launched, waiting for a slot"""
    RUNNING = 'running'
    """Has a slot: running in the pool's event loop, or handed to a worker thread or process"""
    DONE = 'done'
    """Finished without an error"""
    FAILED = 'failed'
    """Finished with an error, or the pool failed to run it"""
    CANCELLED = 'cancelled'
    """Cancelled; in a thread or process pool, the worker may still finish the job, but its result is dropped"""

    @property
    def is_finished(self) -> bool:
        """
        Whether the job has stopped for the pool: done, failed or cancelled.
        """
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass
class JobResults:
    """
    Dataclass to hold the results of a single job.
    """

    job_id: str
    """The unique identifier of the job."""

    data: Any | None = None
    """Optional data returned by the job."""

    error: Any | None = None
    """Error that occurred during job execution, if any."""

    @property
    def success(self) -> bool:
        """
        Whether the job completed successfully without errors.
        """
        return self.error is None


@dataclass(frozen=True)
class BatchResults:
    """
    Dataclass to hold the results of a batch file execution.
    """

    stdout: str
    """Standard output captured from the batch execution."""

    stderr: str
    """Standard error captured from the batch execution."""

    return_code: int
    """Return code from the batch execution."""


@dataclass
class SteppedJobResults(JobResults):
    """
    Dataclass to hold the results of a stepped job, which includes results and data from all steps.
    """

    steps: dict[str, JobResults] = field(default_factory=dict)
    """A dictionary mapping step names to their results."""
