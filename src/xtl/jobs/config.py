from pathlib import Path
from typing import Optional, Type
import tempfile

from pydantic import computed_field, PrivateAttr, model_validator, Field, \
    field_validator

from xtl import Logger
from xtl.automate.shells import Shell, DefaultShell
from xtl.automate.sites import ComputeSiteType, LocalSite, BiotixHPC
from xtl.common.options import Option, Options
from xtl.common.os import FilePermissions


logger = Logger(__name__)


class BatchConfig(Options):
    """
    Configuration for execution of batch files.
    """
    filename: str = Option(default='batch_job', desc='Batch file name (without '
                                                     'extension)')
    directory: Path = Option(default_factory=lambda: Path(tempfile.gettempdir()),
                             desc='Directory for dumping batch file and logs')
    permissions: FilePermissions = Option(default=FilePermissions(0o700),
                                          desc='Permissions for the batch file in '
                                               'octal format (e.g., 700)')
    compute_site: ComputeSiteType = Option(default=LocalSite(), desc='Compute site')
    default_shell: Optional[Shell] = Option(default=None,
                                            desc='Default shell to use for batch '
                                                 'execution.')
    compatible_shells: set[Shell] = Option(default_factory=set,
                                           desc='List of compatible shell types for '
                                                'this batch job')

    _shell: Shell | None = PrivateAttr(None)

    @model_validator(mode='after')
    def _determine_shell(self):
        # Set the default shell if not provided
        if self._shell is None:
            if not self.compute_site.supported_shells:
                # If the compute site doesn't have any requirements, use the default
                #  from the job
                self._shell = self.default_shell
            elif not self.compatible_shells:
                # If the job doesn't have any compatible shells, use the default one
                self._shell = self.compute_site.default_shell
            else:
                # Both job and compute site specify compatible shells
                common = self.compatible_shells & \
                         set(self.compute_site.supported_shells)
                if self.default_shell in common:
                    # Choose the default shell if it is supported by both
                    self._shell = self.default_shell
                elif common:
                    # Otherwise, choose one of the common shells
                    self._shell = next(iter(common))

            # If still no shell is found
            if self._shell is None:
                self._shell = DefaultShell
        elif not isinstance(self._shell, Shell):
            raise ValueError(f'\'_shell\' must be an instance of {Shell.__name__}')

        # Raise warnings for incompatible shells
        if self.compatible_shells and self._shell not in self.compatible_shells:
            logger.warning('Shell %s is not compatible with %s', self._shell,
                           self.compatible_shells)
        if not self.compute_site.is_valid_shell(self._shell):
            logger.warning('Shell %s is not supported by compute site %s', self._shell,
                           self.compute_site)

        # Model validators must return self
        return self

    @computed_field
    def stdout(self) -> Path:
        """
        Path to the standard output log file for the batch job.
        """
        f = 'stdout.log'
        if self.filename:
            f = f'{self.filename}.{f}'
        return self.directory / f

    @computed_field
    def stderr(self) -> Path:
        """
        Path to the standard error log file for the batch job.
        """
        f = 'stderr.log'
        if self.filename:
            f = f'{self.filename}.{f}'
        return self.directory / f

    @computed_field
    def shell(self) -> Shell:
        """
        Returns the shell that will be used to execute the batch file.
        """
        return self._shell


class JobConfig(Options):
    """
    Base class for passing configuration to a job.
    """
    job_directory: Optional[Path] = Option(default=None,
                                           desc='Directory for job execution and '
                                                'results')
    batch: Optional[BatchConfig] = Option(default=None,
                                          desc='Configuration for execution of batch'
                                               'files')

    @model_validator(mode='after')
    def _propagate_job_directory(self):
        """
        Propagate job_directory to batch.batch_directory if both are set
        """
        if self.job_directory is not None and self.batch is not None:
            self.batch.directory = self.job_directory
        return self


if __name__ == '__main__':
    # Example usage
    class TestConfig(JobConfig):
        batch: BatchConfig = Option(
            default_factory=BatchConfig,
            desc='Configuration for the batch file execution'
        )

    config = TestConfig(job_directory=Path('/path/to/job'),
                        batch=BatchConfig(filename='test_batch'))

    from pprint import pprint
    pprint(config)
