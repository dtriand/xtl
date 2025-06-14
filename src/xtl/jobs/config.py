from pathlib import Path
from typing import Optional
import tempfile
from typing import TYPE_CHECKING

from pydantic import computed_field, PrivateAttr, model_validator

from xtl import Logger
from xtl.automate.shells import Shell, DefaultShell, WslShell
from xtl.automate.sites import ComputeSiteType, LocalSite
from xtl.common.options import Option, Options
from xtl.common.os import FilePermissions
from xtl.common.serializers import PermissionOctal

if TYPE_CHECKING:
    from xtl.automate.batchfile import BatchFile


logger = Logger(__name__)


class BatchConfig(Options):
    """
    Configuration for execution of batch files.
    """
    filename: str = \
        Option(
            default='batch_job',
            desc='Batch file name (without extension)'
        )
    directory: Path = \
        Option(
            default_factory=lambda: Path(tempfile.mkdtemp()),
            desc='Directory for dumping batch file and logs'
        )
    permissions: FilePermissions | str | int = \
        Option(
            default=FilePermissions(0o700),
            desc='Permissions for the batch file in octal format (e.g., 700)',
            cast_as=FilePermissions,
            formatter=PermissionOctal
        )
    compute_site: ComputeSiteType = \
        Option(
            default=LocalSite(),
            desc='Compute site'
        )
    default_shell: Optional[Shell] = \
        Option(
            default=None,
            desc='Default shell to use for batch execution.'
        )
    compatible_shells: set[Shell] = \
        Option(
            default_factory=set,
            desc='List of compatible shell types for this batch job'
        )
    dependencies: set[str] = \
        Option(
            default_factory=set,
            desc='List of dependencies required for this batch job'
        )

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
    @property
    def shell(self) -> Shell:
        """
        Returns the shell that will be used to execute the batch file.
        """
        return self._shell

    @shell.setter
    def shell(self, value: Shell):
        """
        Sets the shell to be used for executing the batch file.
        """
        if not isinstance(value, (Shell, WslShell)):
            raise ValueError(f'shell must be an instance of {Shell.__name__}')
        self._shell = value

    def get_batch(self) -> 'BatchFile':
        """
        Returns a BatchFile instance configured with this BatchConfig.
        """
        from xtl.automate.batchfile import BatchFile
        batch = BatchFile(filename=self.directory/self.filename,
                          compute_site=self.compute_site,
                          shell=self.shell,
                          dependencies=self.dependencies)
        batch.permissions = self.permissions
        return batch


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
