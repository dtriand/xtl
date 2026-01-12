from __future__ import annotations
import asyncio
from enum import Enum
from pathlib import Path
from typing import Iterable

from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.os import FilePermissions
from xtl.config.settings import DependencySettings
from xtl.jobs.shells import ShellType, DefaultShell
from xtl.jobs.sites import ComputeSiteType, LocalSite, SchedulerSite

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class BatchFileStatus(StrEnum):
    IDLE = 'idle'
    SCHEDULED = 'scheduled'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class BatchFile:

    def __init__(self,
                 filename: str | Path,
                 compute_site: ComputeSiteType = None,
                 shell: ShellType = None,
                 dependencies: str | DependencySettings |
                               Iterable[str | DependencySettings] = None,
                 permissions: str | FilePermissions = '700'):
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
        _permissions = FilePermissions(permissions)
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
        Returns the batch file Path object.
        """
        return self._filename

    @property
    def shell(self) -> ShellType:
        """
        Returns the underlying Shell that will be used to execute the batch file.
        """
        return self._shell

    @property
    def compute_site(self) -> ComputeSiteType:
        return self._compute_site

    @property
    def dependencies(self) -> set[DependencySettings]:
        return self._dependencies

    @property
    def permissions(self) -> FilePermissions:
        return self._permissions

    @property
    def status(self) -> BatchFileStatus:
        return self._status

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    def _add_line(self, line: str) -> None:
        if line:
            self._lines.append(str(line))

    def add_line(self, line: str) -> None:
        self._add_line(line)

    def add_lines(self, lines: list[str]) -> None:
        if isinstance(lines, str):
            lines = [lines]
        for line in lines:
            if not isinstance(line, str):
                raise TypeError(f'lines must be a list of strings, not {type(line)}')
            self.add_line(line)

    def add_comment(self, comment: str) -> None:
        char = self.shell.comment_char
        if not comment.startswith(char):
            comment = f'{char} {comment}'
        self._add_line(comment)

    def get_preamble(self) -> str:
        return self.compute_site.prepare_preamble(dependencies=self.dependencies,
                                                  shell=self.shell)

    def get_content(self) -> str:
        commands = [self.compute_site.prepare_command(line) for line in self._lines]
        content = self.shell.new_line_char.join(commands)
        return self.compute_site.prepare_content(content)

    def get_postamble(self) -> str:
        return self.compute_site.prepare_postamble()

    def save(self, overwrite: bool = False, update_permissions: bool = True):
        # Delete the file if it already exists
        if self.file.exists() and not overwrite:
            raise FileExistsError(f'Batch file {self.file} already exists. '
                                  f'Set overwrite=True to overwrite it.')
        self.file.unlink(missing_ok=True)

        # Prepare the contents
        text = self.shell.new_line_char.join(
            [self.get_preamble(), self.get_content(), self.get_postamble()]
        ) + self.shell.new_line_char

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
        if not self._saved:
            raise RuntimeError('Batch file has not been saved yet. '
                               'Call `save()` before `execute()`.')
        if schedule and isinstance(self.compute_site, SchedulerSite):
            await self.compute_site.schedule_batch(self)
        else:
            await self.compute_site.execute_batch(self)

    async def cancel(self):
        await self.compute_site.cancel_batch(self)

    # @classmethod
    # def from_config(cls, config: 'BatchConfig' | dict) -> BatchFile: ...

