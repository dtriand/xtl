__all__ = ['JobSubmission']

from xtl import settings
from xtl.common.options import Option, Options
from xtl.math.uuid import UUIDFactory
from xtl.jobs.jobs import Job, JobData
from xtl.jobs.ipc import IPCHandle


uuid = UUIDFactory()


class JobSubmission(Options):
    """
    Serializable envelope for transporting job execution requests.
    """

    submission_id: str = \
        Option(
            default_factory=lambda: uuid.random(settings.jobs.job_digits),
            desc='Unique identifier for this submission payload'
        )

    data: JobData = \
        Option(
            ...,
            desc='Serialized job data, including job class, config, and metadata'
        )

    ipc: IPCHandle | None = \
        Option(
            default=None,
            desc='Optional `IPCHandle` containing interprocess communication primitives '
                 'for worker communication'
        )


    @classmethod
    def from_job(cls, job: 'Job') -> 'JobSubmission':
        data = job.serialize(as_dict=False)
        return cls(data=data)

    def to_job(self) -> 'Job':
        return Job.deserialize(self.data)
