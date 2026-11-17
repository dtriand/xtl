import logging

from xtl import settings
from xtl_core.options import Option, Options
from xtl.math.uuid import UUIDFactory
from xtl.jobs.jobs import Job, JobData
from xtl.jobs.ipc import IPCHandle
from xtl.jobs.resources import Resources


__all__ = ['JobSubmission']


uuid = UUIDFactory()


class JobSubmission(Options):
    """
    Serializable envelope for transporting job execution requests.
    """

    submission_id: str = Option(
        default_factory=lambda: uuid.random(settings.jobs.job_digits),
        desc='Unique identifier for this submission payload',
    )

    data: JobData = Option(..., desc='Serialized job data, including job class, config, and metadata')

    ipc: IPCHandle | None = Option(
        default=None,
        desc='Optional `IPCHandle` containing interprocess communication primitives for worker communication',
    )
    resources: Resources | None = Option(default=None, desc='Optional resources granted to the current pool')

    log_level: int = Option(
        default=logging.NOTSET, desc='Level of the job logger in the worker; records below it are not sent back'
    )

    logger_name: str | None = Option(
        default=None, desc="Full name of the job's logger, also used in the worker (default: the job's id)"
    )

    xtl_settings: dict | None = Option(
        default=None,
        desc="Optional snapshot of the parent's `xtl.settings`, for a worker that reads them afresh (see "
        '`apply_settings()`)',
    )

    dependency_checks: dict | None = Option(
        default=None,
        desc="Optional dependency checks made by the parent, which the settings snapshot doesn't hold (see "
        '`apply_settings()`)',
    )

    @classmethod
    def from_job(cls, job: 'Job') -> 'JobSubmission':
        data = job.serialize(as_dict=False)
        return cls(data=data, log_level=job.logger.getEffectiveLevel(), logger_name=job.logger.name)

    @property
    def job_logger_name(self) -> str:
        """
        Name of the job's logger: `logger_name`, or the job's id if it isn't set.
        """
        return self.logger_name or self.data.job_id

    def apply_settings(self) -> None:
        """
        Apply the snapshot of the parent's settings, if any, to this process's ``xtl.settings``, in place, and add the
        parent's dependency checks to it.
        """
        if self.xtl_settings is not None:
            from xtl.config.settings import XTLSettings

            settings.update_from(XTLSettings.from_dict(self.xtl_settings))
        if self.dependency_checks:
            settings.dependencies.update_checks(self.dependency_checks)

    def to_job(self, logger: logging.Logger | None = None) -> 'Job':
        return Job.deserialize(self.data, logger=logger)
