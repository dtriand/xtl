"""
.. |BatchFile| replace:: :class:`BatchFile <xtl.jobs.batchfiles.BatchFile>`
.. |BaseShell| replace:: :class:`BaseShell <xtl.jobs.shells.BaseShell>`
.. |Shell| replace:: :class:`Shell <xtl.jobs.shells.Shell>`
.. |DefaultShell| replace:: :class:`DefaultShell <xtl.jobs.shells.DefaultShell>`
.. |BashShell| replace:: :class:`BashShell <xtl.jobs.shells.BashShell>`
.. |CmdShell| replace:: :class:`CmdShell <xtl.jobs.shells.CmdShell>`
.. |PowerShell| replace:: :class:`PowerShell <xtl.jobs.shells.PowerShell>`
.. |ShellType| replace:: :class:`ShellType <xtl.jobs.shells.ShellType>`

Configuration for different shells used for |BatchFile| generation and execution.

This module provides a set of predefined shell configurations, including:

* |BashShell|: Configuration for the Bash shell.
* |CmdShell|: Configuration for the Windows Command Prompt shell
* |PowerShell|: Configuration for the Windows PowerShell shell.

Additionally, it defines a |DefaultShell| that selects the appropriate shell based on
the operating system.

New shell configurations can be created by subclassing the |BaseShell| dataclass and
overriding the default parameters. Type checking is facilitated through the |ShellType|
type alias.

Lastly, the |Shell| enum provides a convenient way to reference and retrieve instances
of the supported shell configurations.

.. code-block:: python

    from xtl.jobs.shells import Shell, ShellType

    bash = Shell.BASH.get()  # Returns an instance of BashShell
    assert isinstance(bash, ShellType)  # True

Shell instances are implemented as frozen dataclass singletons to ensure that only one
immutable instance is created per interpreter session. Comparison can be performed
between |Shell| enum members and |BaseShell| instances directly.

.. code-block:: python

    from xtl.jobs.shells import Shell, BashShell

    assert BashShell() is BashShell()  # True
    assert BashShell() == Shell.BASH  # True

Implementing a new shell configuration involves subclassing |BaseShell| and overriding
the default class variables to match the desired shell's properties.

.. code-block:: python

    from xtl.jobs.shells import BaseShell

    class CshShell(BaseShell):
        name: str = 'csh'
        executable: str = '/bin/csh'
        is_posix: bool = True
        shebang: str = '#!/bin/csh'
        comment_char: str = '#'
        new_line_char: str = '\\n'
        batch_extension: str = '.csh'
        batch_command: str = '{executable} {batch_file} {batch_arguments}'

    csh = CshShell()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import shlex
from typing import Iterable, Optional

from xtl.common.compatibility import OS_WINDOWS, PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


__all__ = ['Shell', 'ShellType', 'BaseShell', 'BashShell', 'CmdShell', 'PowerShell',
           'DefaultShell']


class Shell(StrEnum):
    """
    Enumerator of supported shells for batch file generation and execution.
    """

    DEFAULT = 'default'
    """The default shell for the current operating system 
    (see: :class:`DefaultShell <xtl.jobs.shells.DefaultShell>`)"""
    BASH = 'bash'
    """The Bash shell 
    (see: :class:`BashShell <xtl.jobs.shells.BashShell>`)"""
    CMD = 'cmd'
    """The Windows Command Prompt shell 
    (see: :class:`CmdShell <xtl.jobs.shells.CmdShell>`)"""
    POWERSHELL = 'powershell'
    """The Windows PowerShell shell 
    (see: :class:`PowerShell <xtl.jobs.shells.PowerShell>`)"""

    def get(self) -> ShellType:
        """
        Get an instance of the shell configuration corresponding to this enum member.
        """
        mapping = {
            Shell.DEFAULT: DefaultShell,
            Shell.BASH: BashShell,
            Shell.CMD: CmdShell,
            Shell.POWERSHELL: PowerShell,
        }
        return mapping[self]()

    def __eq__(self, other):
        if isinstance(other, ShellType):
            return self.get() == other
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


@dataclass(frozen=True)
class BaseShell:
    """
    Data container for different shell configurations. This dataclass is mainly used
    for configuring the generation of |BatchFile| and their execution.

    This class should always be subclassed by overriding the default class variables.

    :raises ValueError: If the `batch_command` f-string is invalid
    :raises TypeError: If attempting to instantiate the base class directly
    """
    name: str = field(init=False)
    """The name of the shell"""
    executable: str = field(init=False)
    """The path to the shell executable"""
    is_posix: bool = field(init=False)
    """Whether the shell is POSIX compliant"""

    # Shell syntax properties
    shebang: str = field(init=False, repr=False)
    """The shebang line for the shell 
    (see `here <https://en.wikipedia.org/wiki/Shebang_(Unix)>`_)"""
    comment_char: str = field(init=False, repr=False)
    """The character used to denote comments in the shell"""
    new_line_char: str = field(init=False, repr=False)
    """The character used to denote new lines in the shell"""
    batch_extension: str = field(init=False, repr=False)
    """The file extension for scripts"""

    # Commands
    batch_command: str = field(init=False, repr=False)
    """The command used to execute the batch file. This is an f-string that should 
    contain the keys ``executable``, ``batch_file`` and ``batch_arguments``."""
    _batch_command_fstring_keys: frozenset[str] = \
        field(init=False, repr=False, default_factory=lambda: frozenset(
            {'executable', 'batch_file', 'batch_arguments'}
        ))

    dependency_command: str = field(init=False, repr=False)
    """The command used to resolve dependencies in the shell. This is an f-string that
    should contain the key ``dependency``. This command should 
    return an exit code of 0 if the executable is found, and non-zero otherwise."""
    _dependency_command_fstring_keys: frozenset[str] = \
        field(init=False, repr=False, default_factory=lambda: frozenset(
            {'dependency'}
        ))

    execute_command: str = field(init=False, repr=False)
    """The command used to execute a command directly in the shell. This is an f-string
    that should contain the keys ``executable`` and ``command``."""
    _execute_command_fstring_keys: frozenset[str] = \
        field(init=False, repr=False, default_factory=lambda: frozenset(
            {'executable', 'command'}
        ))

    def __post_init__(self):
        if not self.batch_extension.startswith('.'):
            # Workaround for frozen dataclass
            object.__setattr__(self, 'batch_extension', f'.{self.batch_extension}')
        # Validate the f-strings
        self._validate_fstring(fstring=self.batch_command, name='batch_command',
                               required_keys=self._batch_command_fstring_keys)
        self._validate_fstring(fstring=self.dependency_command, name='dependency_command',
                               required_keys=self._dependency_command_fstring_keys)
        self._validate_fstring(fstring=self.execute_command, name='execute_command',
                               required_keys=self._execute_command_fstring_keys)

    @staticmethod
    def _validate_fstring(fstring: str, name: str, required_keys: Iterable[str]) -> None:
        """
        Check if the f-string is valid by ensuring it contains the required keys

        :param fstring: The f-string to validate
        :param name: The name of the f-string (for error messages)
        :param required_keys: The keys that must be present in the f-string
        :raises ValueError: If the f-string is missing required keys or has extra keys
        """
        # Check that all required keys are present in the f-string
        for key in required_keys:
            if f'{{{key}}}' not in fstring:
                raise ValueError(f'Invalid f-string for `{name}`: {fstring}. '
                                 f'Missing key: {key}')

        # Check that there are no extra keys in the f-string
        all_keys = set(re.findall(r'{(.*?)}', fstring))
        for key in all_keys:
            if key not in required_keys:
                raise ValueError(f'Invalid f-string for `{name}`: {fstring}. '
                                 f'Unexpected key: {key}')

    # Singleton instance
    #  NB: Excluded from __hash__ to prevent infinite recursion
    __singleton: 'BaseShell' = field(default=None, init=False, repr=False, hash=False)

    def __new__(cls, *args, **kwargs):
        # Prevent direct instantiation of the base class, which would interfere with the
        #  singleton pattern
        if cls is BaseShell:
            raise TypeError(f'Cannot instantiate class {BaseShell.__name__} directly. '
                            f'Subclass it instead.')

        # Singleton pattern to ensure only one instance per shell type is created
        if cls.__singleton is None:
            cls.__singleton = super().__new__(cls)
        return cls.__singleton

    def get_execute_batch_command(self, file: str | Path,
                                  arguments: Optional[Iterable[str]] = None,
                                  as_list: bool = False) -> str | list[str]:
        """
        Get the command required to execute a batch file using this shell.

        :param file: The path to the batch file
        :param arguments: Optional list of arguments to pass to the batch file
        :param as_list: Whether to return the command as a list of strings
        :return: The command as a single string or list of strings
        """
        # Prepare the batch arguments string
        if arguments is None:
            arguments = []

        batch_arguments = ''
        for argument in arguments:
            # Map each argument to a string
            arg = str(argument)
            # Escape arguments with spaces
            if ' ' in arg:
                arg = f'\'{arg}\''
            # Append to the batch arguments string
            if arg:
                batch_arguments += f'{arg} '
        # Trim trailing whitespace
        batch_arguments = batch_arguments.rstrip()

        # Substitute values into the batch command f-string
        command = self.batch_command.format(executable=self.executable,
                                            batch_file=str(file),
                                            batch_arguments=batch_arguments)
        # Remove trailing space when there are no arguments
        command = command.rstrip()

        # Return as list or string
        if as_list:
            bits = shlex.split(command, posix=self.is_posix)
            return bits
        return command

    def get_dependency_resolution_command(self, dependency: str,
                                          as_list: bool = False) -> str | list[str]:
        """
        Get the command required to resolve dependencies using this shell.

        :param dependency: The name of the dependency/command to resolve
        :param as_list: Whether to return the command as a list of strings
        :return: The command as a single string or list of strings
        """
        # Sanitize dependency name
        dependency = str(dependency).strip()
        if not self._is_dependency_name_safe(dependency):
            raise ValueError(f'Invalid dependency name: {dependency!r}. '
                             f'Dependencies may only contain alphanumeric '
                             f'characters, dashes, underscores, and dots.')

        # Substitute values into the dependency command f-string
        dep_command = self.dependency_command.format(dependency=dependency)
        command = self.execute_command.format(executable=self.executable,
                                              command=dep_command)

        # Return as list or string
        if as_list:
            bits = shlex.split(command, posix=self.is_posix)
            return bits
        return command

    @staticmethod
    def _is_dependency_name_safe(dependency: str) -> bool:
        """
        Check if the dependency name is safe to use in shell commands. This method tries
        to mitigate the risk of command injection by enforcing strict naming rules for
        dependency names. Only alphanumeric characters, dashes, underscores, and dots
        are accepted.

        :param dependency: The name of the dependency/command to check
        :return: True if the dependency name is safe, False otherwise
        """
        # Check length constraints
        if len(dependency) == 0:
            return False
        elif len(dependency) > 255:
            # Prevent potential buffer overflow attacks
            return False
        # Pattern for a safe dependency name (dot separated segments of alphanumeric,
        #  dash, and underscore characters)
        pattern = r'^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$'
        return re.match(pattern, dependency) is not None

    def get_execute_command(self, command: str,
                            as_list: bool = False) -> str | list[str]:
        """
        Get the command required to execute a command directly in this shell.

        :param command: The command to execute
        :param as_list: Whether to return the command as a list of strings
        :return: The command as a single string or list of strings
        """
        # Substitute values into the execute command f-string
        full_command = self.execute_command.format(executable=self.executable,
                                                   command=command)

        # Return as list or string
        if as_list:
            bits = shlex.split(full_command, posix=self.is_posix)
            return bits
        return full_command

    def sanitize_value(self, value):
        match value:
            case Path():
                path_str = str(value)
                if self.is_posix:
                    # Use shlex.quote to properly escape the path for POSIX shells
                    return shlex.quote(path_str)
                elif self == Shell.CMD:  # CMD requires double quotes for escaping
                    # Escape internal quotes by doubling them
                    path_str = path_str.replace('"', '""')
                    return f'"{path_str}"'
                elif self == Shell.POWERSHELL:  # PWSH requires single quotes for escaping
                    # Escape internal quotes by doubling them
                    path_str = path_str.replace('\'', '\'\'')
                    return f'\'{path_str}\''
                else:
                    raise ValueError(f'Unsupported shell for path sanitization: {self.__class__.__name__}')
            case _:
                return value


class BashShell(BaseShell):
    """
    Configuration for the `Bash <https://en.wikipedia.org/wiki/Bash_(Unix_shell)>`_
    shell.
    """
    name = 'bash'
    executable = '/bin/bash'
    is_posix = True
    shebang = '#!/bin/bash'
    comment_char = '#'
    new_line_char = '\n'
    batch_extension = '.sh'
    batch_command = '{executable} {batch_file} {batch_arguments}'
    dependency_command = 'command -v {dependency} > /dev/null 2>&1; echo \$?'
    execute_command = '{executable} --noprofile --norc -c "{command}"'


class CmdShell(BaseShell):
    """
    Configuration for the `Windows Command Prompt
    <https://en.wikipedia.org/wiki/Cmd.exe>`_ shell.
    """
    name = 'cmd'
    executable = r'C:\Windows\System32\cmd.exe'
    is_posix = False
    shebang = ''
    comment_char = 'rem'
    new_line_char = '\n'
    batch_extension = '.bat'
    batch_command = '{executable} /Q /C {batch_file} {batch_arguments}'
    dependency_command = 'where {dependency} > nul 2>&1 && echo 0 || echo 1'
    execute_command = '{executable} /C {command}'


class PowerShell(BaseShell):
    """
    Configuration for the `Windows PowerShell
    <https://en.wikipedia.org/wiki/PowerShell>`_ shell.
    """
    name = 'powershell'
    executable = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    is_posix = False
    shebang = ''
    comment_char = '#'
    new_line_char = '\n'
    batch_extension = '.ps1'
    batch_command = '{executable} -File {batch_file} {batch_arguments}'
    dependency_command = ('Get-Command {dependency} -ErrorAction SilentlyContinue | '
                          'Out-Null; [int](-not $?)')
    # NB: PowerShell returns True/1 if the command is found, so we negate it to match
    #  the exit code convention.
    execute_command = '{executable} -NoProfile -NonInteractive -Command "{command}"'
    # We are also running PowerShell in -NoProfile and -NonInteractive mode to avoid
    #  loading user profiles or interactive prompts.

# Set the default shell based on the OS
DefaultShell = CmdShell if OS_WINDOWS else BashShell
"""
The default shell for the current operating system. 
:class:`CmdShell <xtl.jobs.shells.CmdShell>` for Windows, 
:class:`BashShell <xtl.jobs.shells.BashShell>` for POSIX systems.
"""

# Type alias
ShellType = BaseShell | BashShell | CmdShell | PowerShell
"""
A type alias for all supported shell types.
"""
