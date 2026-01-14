from __future__ import annotations
import asyncio
from enum import Enum
from pathlib import Path
from string import Template
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings
    from xtl.jobs.sites import ComputeSiteType, LocalSite, SchedulerSite
    from xtl.jobs.config2 import BatchJobConfig
from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.os import FilePermissions
from xtl.jobs.shells import Shell, ShellType, DefaultShell


__all__ = ['BatchFile', 'BatchFileStatus']


if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class BatchFileStatus(StrEnum):
    """
    Enumeration of possible statuses for a BatchFile.
    """

    IDLE = 'idle'
    """The batch file has been instantiated. It may not be saved to disk yet."""

    SCHEDULED = 'scheduled'
    """The batch file has been scheduled for execution on a compute site."""

    RUNNING = 'running'
    """The batch file is currently being executed on a compute site."""

    COMPLETED = 'completed'
    """The batch file has completed execution successfully."""

    FAILED = 'failed'
    """The batch file execution has failed."""

    CANCELLED = 'cancelled'
    """The batch file execution has been cancelled."""


class BatchTemplate(Template):
    """
    A template subclass for batch files using __VAR__ style placeholders.
    """

    delimiter = '__'
    pattern = r'''
            __(?:
                (?P<escaped>__)             |  # Escape sequence of two delimiters
                (?P<named>[a-zA-Z_]\w*)__   |  # Delimiter and a Python identifier
                (?P<braced>[a-zA-Z_]\w*)__  |  # Delimiter and a braced identifier
                (?P<invalid>)                  # Other ill-formed delimiter expressions
            )
        '''


class BatchFile:

    def __init__(self,
                 filename: str | Path,
                 compute_site: ComputeSiteType = None,
                 shell: ShellType = None,
                 dependencies: str | DependencySettings |
                               Iterable[str | DependencySettings] = None,
                 permissions: str | FilePermissions = None):
        """
        A class for programmatically creating batch/script files. The behavior of the
        batch file can be customized by specifying the compute site and shell to use.

        :param filename: The name of the batch file. The file extension will be
                            automatically set based on the Shell.
        :param compute_site: The ComputeSite where the batch file will be executed.
        :param shell: The Shell that will be used to execute the batch file.
        :param dependencies: A list of dependencies that are required by this batch
                                file.
        :param permissions: The file permissions for the batch file in octal format
                            (e.g., '700').
        :raises TypeError: If `compute_site` or `shell` are of incorrect types.
        :raises ValueError: If `permissions` do not allow owner execute access.
        """
        # Imports to prevent circular dependencies
        from xtl import settings
        from xtl.jobs.sites import ComputeSiteType, LocalSite

        # Check compute_site
        if compute_site and not isinstance(compute_site, ComputeSiteType):
            raise TypeError(f'`compute_site` must be an instance of ComputeSite, '
                            f'not {type(compute_site)}')
        self._compute_site = compute_site or LocalSite()

        # Check shell
        if shell and not isinstance(shell, ShellType):
            raise TypeError(f'`shell` must be an instance of Shell, not {type(shell)}')
        self._shell = shell or DefaultShell()

        # Set permissions
        _permissions = FilePermissions(permissions or settings.jobs.batch.permissions)
        if not _permissions.owner.can_execute:
            raise ValueError(f'`permissions` must allow owner execute access, '
                             f'not {_permissions}')
        else:
            self._permissions = _permissions

        # Initialize remaining attributes
        self._dependencies = self.compute_site.resolve_dependencies(dependencies)
        self._filename = Path(filename).with_suffix(self.shell.batch_extension)
        self._lines = []
        self._status = BatchFileStatus.IDLE
        self._saved = False
        self._process: asyncio.subprocess.Process | None = None

    @property
    def file(self) -> Path:
        """
        Returns the batch file `Path` object.
        """
        return self._filename

    @property
    def shell(self) -> ShellType:
        """
        Returns the underlying `Shell` that will be used to execute the batch file.
        """
        return self._shell

    @property
    def compute_site(self) -> ComputeSiteType:
        """
        Returns the `ComputeSite` where the batch file will be executed
        """
        return self._compute_site

    @property
    def dependencies(self) -> set[DependencySettings]:
        """
        Returns a set of `DependencySettings` for the batch file.
        """
        return self._dependencies

    @property
    def permissions(self) -> FilePermissions:
        """
        Returns the file permissions for the batch file.
        """
        return self._permissions

    @property
    def status(self) -> BatchFileStatus:
        """
        Returns the current status of the batch file.
        """
        return self._status

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        """
        Returns the underlying `asyncio.subprocess.Process` if the batch file is being
        executed, otherwise None.
        """
        return self._process

    def _add_line(self, line: str) -> None:
        if line:
            self._lines.append(str(line))

    def add_line(self, line: str) -> None:
        """
        Add a single line to the batch file.

        :param line: The line to add to the batch file.
        """
        self._add_line(line)

    def add_lines(self, lines: list[str]) -> None:
        """
        Add one or more lines to the batch file.

        :param lines: A list of lines to add to the batch file.
        """
        if isinstance(lines, str):
            lines = [lines]
        for line in lines:
            if not isinstance(line, str):
                raise TypeError(f'`lines` must be a list of strings, not {type(line)}')
            self.add_line(line)

    def add_comment(self, comment: str) -> None:
        """
        Add a comment line to the batch file.

        :param comment: The comment text to add, without the comment character.
        """
        char = self.shell.comment_char
        if not comment.startswith(char):
            comment = f'{char} {comment}'
        self._add_line(comment)

    def get_preamble(self) -> str:
        """
        Returns the preamble for the batch file.
        """
        return self.compute_site.prepare_preamble(dependencies=self.dependencies,
                                                  shell=self.shell)

    def get_content(self) -> str:
        """
        Returns the main content of the batch file.
        """
        commands = [self.compute_site.prepare_command(line) for line in self._lines]
        content = self.shell.new_line_char.join(commands)
        return self.compute_site.prepare_content(content)

    def get_postamble(self) -> str:
        """
        Returns the postamble for the batch file.
        """
        return self.compute_site.prepare_postamble()

    def save(self, overwrite: bool = False, update_permissions: bool = True):
        """
        Save the batch file to disk.

        :param overwrite: Whether to overwrite the file if it already exists.
        :param update_permissions: Whether to update the file permissions after saving.
        :raises FileExistsError: If the file already exists and `overwrite` is False.
        """
        # Delete the file if it already exists
        if self.file.exists() and not overwrite:
            raise FileExistsError(f'Batch file {self.file} already exists. '
                                  f'Set overwrite=True to overwrite it.')
        self.file.unlink(missing_ok=True)

        # Prepare the contents
        text = ''
        if preamble := self.get_preamble():
            text += preamble + self.shell.new_line_char
        if content := self.get_content():
            text += content + self.shell.new_line_char
        if postamble := self.get_postamble():
            text += postamble + self.shell.new_line_char

        # Write contents to file
        self.file.write_text(data=text, encoding='utf-8',
                             newline=self.shell.new_line_char)

        # Update file permissions
        if update_permissions:
            self.file.chmod(self.permissions.decimal)

        self._saved = True

    # Aliases for execution and cancellation
    #  These methods provide an alternative API for interacting with batch files, rather
    #  than going through the compute site directly.
    async def execute(self, schedule: bool = False):
        """
        Execute or schedule the batch file on the compute site.

        :param schedule: Whether to schedule the batch file for later execution
                         (if supported by the compute site).
        :raises RuntimeError: If the batch file has not been saved yet.
        """
        if not self._saved:
            raise RuntimeError('Batch file has not been saved yet. '
                               'Call `save()` before `execute()`.')
        if schedule and isinstance(self.compute_site, SchedulerSite):
            await self.compute_site.schedule_batch(self)
        else:
            await self.compute_site.execute_batch(self)

    async def cancel(self):
        """
        Cancel the execution or scheduling of the batch file on the compute site.
        """
        await self.compute_site.cancel_batch(self)

    @staticmethod
    def _render_template(template: str, context: dict) -> str:
        """
        Render a batch file template with the given context.

        :param template: The template string to render.
        :param context: A dictionary of context variables for rendering.
        :return: The rendered template string.
        :raises KeyError: If a placeholder in the `template` is not found in the
            `context`.
        :raises ValueError: If the `template` contains invalid placeholders.
        """
        batch_template = BatchTemplate(template)
        return batch_template.substitute(context or {})

    @classmethod
    def from_config(cls, config: BatchJobConfig | dict, context: dict = None) \
            -> BatchFile:
        """
        Create a pre-configured BatchFile instance from a BatchJobConfig. If the config
        includes an appropriate template, the batch file contents will be populated and
        any placeholder variables will be replaced using the provided context.

        Placeholder variables in the template should be in the format `__VAR__`, where
        `VAR` corresponds to a key in the `context` dictionary.

        :param config: A BatchJobConfig instance or dictionary.
        :param context: A dictionary of context variables for rendering the template.
        :return: A BatchFile instance configured according to the provided config.
        ️ :raises KeyError: If a placeholder in the template is not found in the
            context.
        :raises ValueError: If the template contains invalid placeholders.
        """
        from xtl.jobs.config2 import BatchJobConfig
        from xtl.jobs.sites import ComputeSite

        if isinstance(config, dict):
            config = BatchJobConfig(**config)

        batch = cls(
            filename=config.file,
            compute_site=ComputeSite(config.compute_site).get(),
            shell=Shell(config.shell).get(),
            dependencies=config.dependencies,
            permissions=config.permissions
        )

        # Render template if provided
        if template := config.get_template():
            content = batch._render_template(template, context or {})
            batch.add_lines(content.splitlines())

        return batch
