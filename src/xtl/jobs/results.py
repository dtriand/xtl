from dataclasses import dataclass, field
from typing import Any


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