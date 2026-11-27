from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, TYPE_CHECKING

from pydantic import computed_field, field_serializer, field_validator

from xtl import settings
from xtl_core.options import Option, Options
from xtl_core.system.os.permissions import FilePermissions
from xtl_core.options.serializers import PermissionOctal
from xtl_core.options.validators import cast_as_temp_dir_if_none
from xtl.jobs.shells import Shell
from xtl.jobs.sites import ComputeSite
from xtl.jobs.resources import Resources
from xtl.jobs.logging import HandlerConfig

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings
    from xtl.jobs.batchfiles import BatchFile
    from xtl.jobs.shells import ShellType
    from xtl.jobs.sites import ComputeSiteType


__all__ = ['JobConfig', 'BatchJobConfig', 'SteppedJobConfig']


logger = logging.getLogger(__name__)


class JobConfig(Options):
    """
    Base class for job configuration, containing fields common to all jobs.
    """

    job_directory: Optional[Path] = Option(default=None, desc='Directory for job execution and results')

    resources: Optional[Resources] = Option(
        default=None, desc='Resources to allocate for this job. Overrides job-defined resources'
    )

    log_handlers: dict[str, Optional[HandlerConfig]] = Option(
        default_factory=dict,
        desc="Handlers of this job's own, by name: a spec adds a handler, or replaces the pool's of the same name; "
        "None switches the pool's off",
    )

    @field_validator('log_handlers', mode='before')
    @classmethod
    def _restore_log_handlers(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: HandlerConfig.from_spec(v) if isinstance(v, dict) else v for k, v in value.items()}
        return value

    @field_serializer('log_handlers')
    def _dump_log_handlers(self, value: dict[str, Optional[HandlerConfig]], info) -> dict[str, Any]:
        mode = 'json' if info.mode_is_json() else 'python'
        return {k: v.to_spec(mode=mode) if v is not None else None for k, v in value.items()}


class SiteResources(Options):
    """
    Resources that a job asks of its compute site (e.g. a Slurm allocation), as opposed to
    ``JobConfig.resources``, which it asks of its pool.
    """

    # TODO: Custom formatters & aliases for SLURM
    # TODO: Custom validators for SLURM-like input
    # Formatting should be handled within different methods, e.g. to_slurm()
    # Maybe we implement a _aliases and _formatters PrivateAttr in the SiteResources for this purpose?
    cpus: int = Option(
        default=1,
        ge=1,
        desc='Number of CPU cores required for the job',
        # alias='cpus-per-task'
    )

    memory: float = Option(
        default=1.0,
        ge=0.0,
        desc='Amount of memory (in GB) required for the job',
        # alias='mem',
        # formatter=lambda x: f'{x}G'
    )

    timeout: float | str | timedelta | None = Option(
        default=None,
        desc='Maximum runtime for the job (in minutes or D-HH:MM:SS format)',
        # alias='time',
        # cast_as=...,
        # formatter=...
    )

    gpus: int = Option(
        default=0,
        ge=0,
        desc='Number of GPUs required for the job',
    )

    no_tasks: int = Option(
        default=1,
        ge=1,
        desc='Number of tasks required for the job (only used in MPI jobs)',
        # alias='ntasks'
    )

    no_nodes: int = Option(
        default=1,
        ge=1,
        desc='Number of nodes required for the job (only used in MPI jobs)',
        # alias='nodes'
    )

    def to_slurm(self) -> list[str]:
        args = []
        for field, value in self.to_dict(by_alias=True).items():
            if value:
                args.append(f'--{field}={value}')
        return args


class BatchJobConfig(JobConfig):
    # Batch jobs always require a job directory
    job_directory: Optional[Path] = Option(
        default_factory=lambda: cast_as_temp_dir_if_none(None, prefix='xtl_batch_'),
        desc='Directory for job execution and results',
        cast_as=lambda x: cast_as_temp_dir_if_none(x, prefix='xtl_batch_'),
    )
    filename: str = Option(default='batch_job', desc='Batch file name (without extension)')
    name: Optional[str] = Option(default=None, desc='Name of the batch job')  # for slurm --job-name
    description: Optional[str] = Option(default=None, desc='Description of the batch job')  # for slurm --comment
    permissions: FilePermissions | str | int = Option(
        default_factory=lambda: settings.jobs.permissions.scripts,
        desc='Permissions for the batch file in octal format (e.g., 700)',
        cast_as=FilePermissions,
        formatter=PermissionOctal,
    )
    compute_site: Optional[ComputeSite] = Option(
        default=None,
        desc="Compute site (default: the one the job's dependencies set in the settings, else jobs.compute_site)",
    )
    default_shell: Optional[Shell] = Option(
        default=None,
        desc="Shell to use for batch execution (default: the one the job's dependencies set in the settings, else "
        "the compute site's, else the system's)",
    )
    templates: dict[Shell, str] = Option(
        default_factory=dict, desc='Templates for the content of the batch file for different shells'
    )
    arguments: list[str] = Option(
        default_factory=list, desc='Arguments of the batch file on its command line, each passed as one argument'
    )
    site_resources: SiteResources = Option(
        default_factory=SiteResources, desc='Resources that the batch job asks of its compute site'
    )

    @property
    def compatible_shells(self) -> set[Shell]:
        """
        List of compatible shell types for this batch job. This is inferred from the
        `templates` field.
        """
        if not self.templates:
            return set()
        return set(self.templates.keys())

    def select_site_and_shell(
        self, dependencies: Iterable[DependencySettings | str] | None = None
    ) -> tuple[ComputeSiteType, ShellType]:
        """
        The compute site and shell that run this batch job, from the configuration (``compute_site``,
        ``default_shell``, the shells of the ``templates``) and the job's dependencies (see
        :func:`~xtl.jobs.dependencies.select_site_and_shell`).

        :param dependencies: Names or settings of the job's dependencies.
        :raises DependencyError: If the dependencies ask for different compute sites, shells or policies.
        :raises ValueError: If the shell asked for isn't one that the site and the templates allow.
        """
        from xtl.jobs.dependencies import select_site_and_shell

        return select_site_and_shell(
            dependencies, compute_site=self.compute_site, default_shell=self.default_shell, shells=self.templates
        )

    def get_file(self, shell: ShellType) -> Path:
        """
        The path of the batch file in a shell: the job directory, the ``filename`` and the shell's extension.

        :param shell: The shell.
        """
        return self.job_directory / f'{self.filename}{shell.batch_extension}'

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

    def get_batch(
        self,
        context: dict = None,
        dependencies: Iterable[DependencySettings | str] | None = None,
        compute_site: Optional[ComputeSiteType] = None,
        shell: Optional[ShellType] = None,
    ) -> 'BatchFile':
        """
        A batch file for this configuration, with the template of its shell rendered.

        :param context: Optional dictionary of additional context variables to be used when rendering the batch file
            template.
        :param dependencies: Names or settings of the job's dependencies, set up and guarded in the preamble.
        :param compute_site: The compute site, if already chosen; otherwise chosen with the shell (see
            :meth:`select_site_and_shell`).
        :param shell: The shell, if already chosen (together with ``compute_site``).
        """
        from xtl.jobs.batchfiles import BatchFile

        if compute_site is None or shell is None:
            compute_site, shell = self.select_site_and_shell(dependencies)
        return BatchFile.from_config(
            self, context=context, compute_site=compute_site, shell=shell, dependencies=dependencies
        )

    def get_template(self, shell: Shell | ShellType) -> Optional[str]:
        """
        The template for the content of the batch file in a shell, if there is one.

        :param shell: The shell.
        """
        return self.templates.get(Shell(shell.name) if not isinstance(shell, str) else Shell(shell), None)

    def get_context(self) -> dict:
        """
        Get the context variables for rendering the batch file template.
        """
        return {}

    def to_slurm(self) -> list[str]:
        """
        Convert the batch job configuration to a list of SLURM command-line arguments.
        """
        args = []
        if self.name:
            args.append(f'--job-name={self.name}')
        if self.description:
            args.append(f'--comment={self.description}')
        args.extend(self.site_resources.to_slurm())
        args.append(f'--output={self.stdout}')
        args.append(f'--error={self.stderr}')
        return args


class SteppedJobConfig(JobConfig):
    """
    Class for passing configuration to a job.
    """

    steps: dict[str, Any] = Option(default_factory=dict, desc='Configuration for each step of the job.')
