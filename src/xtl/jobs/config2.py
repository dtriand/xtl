from pathlib import Path
from typing import Optional

from pydantic import PrivateAttr, model_validator, computed_field, field_serializer

from xtl import settings, Logger
from xtl.common.options import Option
from xtl.common.os import FilePermissions
from xtl.common.serializers import PermissionOctal
from xtl.common.validators import cast_as_temp_dir_if_none
from xtl.jobs import Shell
from xtl.jobs.config import JobConfig
from xtl.jobs.sites import ComputeSite


logger = Logger(__name__)


class BatchJobConfig(JobConfig):

    job_directory: Optional[Path] = Option(
        default_factory=lambda: cast_as_temp_dir_if_none(None, prefix='xtl_batch_'),
        desc='Directory for job execution and results',
        cast_as=lambda x: cast_as_temp_dir_if_none(x, prefix='xtl_batch_'),
    )
    filename: str = \
        Option(
            default='batch_job',
            desc='Batch file name (without extension)'
        )
    name: Optional[str] = \
        Option(
            default=None,
            desc='Name of the batch job'
        )  # for slurm --job-name
    description: Optional[str] = \
        Option(
            default=None,
            desc='Description of the batch job'
        )  # for slurm --comment
    permissions: FilePermissions | str | int = \
        Option(
            default=FilePermissions(0o700),
            desc='Permissions for the batch file in octal format (e.g., 700)',
            cast_as=FilePermissions,
            formatter=PermissionOctal
        )
    compute_site: ComputeSite = \
        Option(
            default_factory=lambda: ComputeSite(settings.jobs.compute_site),
            desc='Compute site',
            cast_as=ComputeSite
        )
    dependencies: set[str] = \
        Option(
            default_factory=set,
            desc='List of dependencies required for this batch job'
        )
    default_shell: Optional[Shell] = \
        Option(
            default=None,
            desc='Default shell to use for batch execution'
        )
    templates: dict[Shell, str] = \
        Option(
            default_factory=dict,
            desc='Templates for the content of the batch file for different shells'
        )

    _shell: Shell | None = PrivateAttr(None)
    """The shell that will be used to execute the batch file"""

    @model_validator(mode='after')
    def _determine_shell(self):
        compute_site = ComputeSite(self.compute_site).get()
        # Set the default shell if not provided
        if self._shell is None:
            if not compute_site.supported_shells:
                # If the compute site doesn't have any requirements, use the default
                #  from the job
                self._shell = self.default_shell
            elif not self.compatible_shells:
                # If the job doesn't have any compatible shells, use the default one
                self._shell = compute_site.default_shell
            else:
                # Both job and compute site specify compatible shells
                common = self.compatible_shells & \
                         set(compute_site.supported_shells)
                if self.default_shell in common:
                    # Choose the default shell if it is supported by both
                    self._shell = self.default_shell
                elif common:
                    # Otherwise, choose one of the common shells
                    self._shell = next(iter(common))

            # If still no shell is found
            if self._shell is None:
                # Workaround to ensure that we get the actual underlying shell saved
                self._shell = Shell(Shell.DEFAULT.get().name)
        elif not isinstance(self._shell, Shell):
            raise ValueError(f'`_shell` must be an instance of {Shell.__name__}')

        # Raise warnings for incompatible shells
        if self.compatible_shells and self._shell not in self.compatible_shells:
            raise RuntimeError(f'Incompatible shell selected: {self._shell}. '
                               f'Compatible shells are: {self.compatible_shells}')
        if not compute_site.is_valid_shell(self._shell):
            logger.warning('Shell %s is not supported by compute site %s', self._shell,
                           self.compute_site)

        # Ensure _shell is a Shell enum
        self._shell = Shell(self._shell)

        # Model validators must return self
        return self

    @property
    def compatible_shells(self) -> set[Shell]:
        """
        List of compatible shell types for this batch job. This is inferred from the
        `templates` field.
        """
        if not self.templates:
            return set()
        return set(self.templates.keys())

    @computed_field
    @property
    def shell(self) -> Shell:
        """
        Returns the underlying Shell that will be used to execute the batch file.
        """
        return self._shell

    @field_serializer('shell')
    def _serialize_shell(self, shell: Shell, _info) -> str:
        return shell.value

    @computed_field
    @property
    def file(self) -> Path:
        """
        Returns the Path object for the batch file.
        """
        return self.job_directory / f'{self.filename}{self.shell.get().batch_extension}'

    @computed_field
    @property
    def stdout(self) -> Path:
        """
        Path to the standard output log file for the batch job.
        """
        f = 'stdout.log'
        if self.filename:
            f = f'{self.filename}.{f}'
        return self.job_directory / f

    @computed_field
    @property
    def stderr(self) -> Path:
        """
        Path to the standard error log file for the batch job.
        """
        f = 'stderr.log'
        if self.filename:
            f = f'{self.filename}.{f}'
        return self.job_directory / f

    def get_template(self) -> Optional[str]:
        """
        Get the template for the content of the batch file for the selected shell.
        """
        return self.templates.get(self.shell, None)
