import abc
from pathlib import Path

from xtl.common.options import Option
from xtl.jobs.jobs import Job
from xtl.jobs.config import JobConfig
from xtl.saxs.jobs.atsas_utils import ATSASOptions, DatcmpOptions


class ATSASJobConfig(JobConfig, abc.ABC):
    """
    Base configuration for ATSAS jobs.
    """
    options: ATSASOptions

    @abc.abstractmethod
    def get_command(self, as_list: bool = False) -> list[str] | str:
        ...


class DatcmpJobConfig(ATSASJobConfig):
    files: list[Path] = \
        Option(
            ...,
            desc='List of data files to compare',
            min_length=2,
            path_exists=True)
    options: DatcmpOptions = \
        Option(
            default_factory=DatcmpOptions,
            desc='Options for `datcmp`'
        )

    def get_command(self, as_list: bool = False) -> list[str] | str:
        parts = list(map(str, [self.options.executable,
                               *self.files,
                               *self.options.get_args()]))
        if as_list:
            return parts
        return ' '.join(parts)


class DatcmpJob(Job[DatcmpJobConfig]):
    """
    Job to compare SAXS datasets using ATSAS datcmp.
    """

    _job_prefix = 'datcmp'
    _dependencies = {'atsas'}

    async def _execute(self):
        if self.config is None:
            self.logger.error('Job is not configured.')
            raise ValueError('Job is not configured.')

        # Execute the command
        results = await self._execute_batch(self.config.get_command())

        if not results.success or 'stdout' not in results.data:
            self.logger.error('An error occurred during execution of the batch file %s',
                              results.error)
            return results

        return results.data['stdout']
