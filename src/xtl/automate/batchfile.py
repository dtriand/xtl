__all__ = ['BatchFile']

from pathlib import Path, PurePosixPath
from typing import Any, Iterable, TYPE_CHECKING

from xtl import Logger
from xtl.automate.sites import ComputeSite, LocalSite
from xtl.automate.shells import Shell, DefaultShell, WslShell
from xtl.common.os import FilePermissions
from xtl.common.compatibility import OS_POSIX

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings

logger = Logger(__name__)


class BatchFile:

    def __init__(self, filename: str | Path, compute_site: ComputeSite = None,
                 shell: Shell | WslShell = DefaultShell,
                 dependencies: str | Iterable[str] = None):
        """
        A class for programmatically creating batch files. Additional configuration can be done by passing a ComputeSite
        instance.

        :param filename: The name of the batch file. The file extension will be automatically set based on the Shell.
        :param compute_site: The ComputeSite where the batch file will be executed
        :param shell: The Shell that will be used to execute the batch file
        """
        if not isinstance(shell, Shell | WslShell):
            raise TypeError(f'\'shell\' must be an instance of Shell, not {type(shell)}')
        self._shell = shell

        self._filename = Path(filename).with_suffix(self.shell.file_ext)

        # Set the filename to be read by WSL
        if isinstance(shell, WslShell):
            wsl_filename = self._filename.as_posix().replace(f'//wsl.localhost/{shell.distro}', '')
            self._wsl_filename = PurePosixPath(wsl_filename)
        else:
            self._wsl_filename = None

        # Set compute_site
        if compute_site is None:
            compute_site = LocalSite()
        elif not isinstance(compute_site, ComputeSite):
            raise TypeError(f"compute_site must be an instance of ComputeSite, not {type(compute_site)}")
        self._compute_site = compute_site

        # Set dependencies
        if dependencies is None:
            self._dependencies = set()
        elif isinstance(dependencies, str):
            self._dependencies = {dependencies}
        elif isinstance(dependencies, Iterable):
            self._dependencies = set(dependencies)

        # List of lines of the batch file
        self._lines = []

        # Set default permissions for the batch file to 700
        self._permissions = FilePermissions('700')

    @property
    def file(self) -> Path:
        """
        Returns the batch file Path object.
        """
        return self._filename

    @property
    def shell(self) -> Shell:
        """
        Returns the underlying Shell that will be used to execute the batch file.
        """
        return self._shell

    @property
    def compute_site(self) -> ComputeSite:
        """
        Returns the ComputeSite where the batch file will be executed
        """
        return self._compute_site

    @property
    def permissions(self):
        """
        Returns the permissions that will be set for the batch file in octal format.
        """
        return self._permissions.octal

    @permissions.setter
    def permissions(self, value: int | str | FilePermissions):
        """
        Set the permissions that will be set for the batch file. Expects a 3-digit octal number.
        """
        self._permissions = FilePermissions(value)

    @property
    def dependencies(self) -> list['DependencySettings']:
        """
        Returns a list of DependencySettings that are required by this batch file.
        """
        from xtl import settings
        deps = list()

        for dname, dep in settings.dependencies:
            if dname in self._dependencies:
                deps.append(dep)
            else:
                logger.warning('Dependency %(name)s not found in xtl.settings, '
                               'skipping', {'name': dname})
        return deps

    def get_execute_command(self, arguments: list = None, as_list: bool = False) -> str:
        if self._wsl_filename:
            return self.shell.get_batch_command(batch_file=self._wsl_filename, batch_arguments=arguments, as_list=as_list)
        return self.shell.get_batch_command(batch_file=self.file, batch_arguments=arguments, as_list=as_list)

    def _add_line(self, line: str) -> None:
        """
        Add a line to the batch file.
        """
        if line:
            self._lines.append(str(line))

    def add_empty_line(self, count: int = 1) -> None:
        """
        Add an empty line to the batch file.
        """
        if not isinstance(count, int):
            raise TypeError(f'\'count\' must be an int, not {type(count)}')
        if count < 1:
            raise ValueError('\'count\' must be greater than 0')
        for _ in range(count):
            self._lines.append('')

    def add_command(self, command: str) -> None:
        """
        Add a command to the batch file
        """
        self._add_line(command)

    def add_commands(self, *commands: str | Iterable[str]) -> None:
        """
        Add multiple commands to the batch file all at once.
        """
        if len(commands) == 1 and isinstance(commands[0], Iterable):  # unpack a list or tuple of commands
            commands = commands[0]
        for command in commands:
            if not isinstance(command, str):
                raise TypeError(f'\'command\' must be a str, not {type(command)}')
            self.add_command(command)

    def add_comment(self, comment):
        """
        Add a comment to the batch file.
        """
        self._add_line(f"{self.shell.comment_char} {comment}")

    def load_modules(self, modules: str | Iterable[str]):
        """
        Add command for loading one or more modules on the compute site.
        """
        # TODO: Remove when refactoring AutoPROCJob
        self.add_command(self.compute_site.load_modules(modules))

    def purge_modules(self):
        """
        Add command for purging all loaded modules on the compute site.
        """
        # TODO: Remove when refactoring AutoPROCJob
        self.add_command(self.compute_site.purge_modules())

    def get_preamble(self) -> list[str]:
        return self.compute_site.get_preamble(dependencies=self.dependencies,
                                              shell=self.shell)

    def assign_variable(self, variable: str, value: Any):
        """
        Assign a value to a variable in the batch file.
        """
        self.add_command(str(variable) + '=' + str(value))

    def purge_file(self) -> None:
        """
        Delete the contents of the file.
        """
        self._lines = []

    def save(self, change_permissions: bool = True):
        """
        Save the batch file to disk and optionally change its permissions.
        """
        # Delete the file if it already exists
        self._filename.unlink(missing_ok=True)

        # Write contents to file
        nl = self.shell.new_line_char
        preamble = self.get_preamble()
        text = self.shell.shebang + nl if self.shell.shebang else ''
        if preamble:
            text += nl.join(self.get_preamble()) + nl
        text += nl.join(self._lines)
        self._filename.write_text(text, encoding='utf-8', newline=nl)

        # Update permissions (user: read, write, execute; group: read, write)
        if change_permissions and OS_POSIX:
            self._filename.chmod(self._permissions.decimal)