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


__all__ = ['Shell', 'ShellType', 'BashShell', 'CmdShell', 'PowerShell']


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

    This class should always be subclassed by overriding the default ``__init__`` method
    to provide the specific shell configuration.

    :param name: The name of the shell
    :param executable: The path to the shell executable
    :param is_posix: Whether the shell is POSIX compliant
    :param shebang: The shebang line for the shell
    :param comment_char: The character used to denote comments in the shell
    :param new_line_char: The character used to denote new lines in the shell
    :param batch_extension: The file extension for scripts
    :param batch_command: The command used to execute the batch file. This is an
        f-string that should contain the keys `executable`, `batch_file`, and
        `batch_arguments`.
    :raises ValueError: If the `batch_command` f-string is invalid
    :raises TypeError: If attempting to instantiate the base class directly
    """
    name: str
    """The name of the shell"""
    executable: str
    """The path to the shell executable"""
    is_posix: bool
    """Whether the shell is POSIX compliant"""

    # Shell syntax properties
    shebang: str = field(repr=False)
    """The shebang line for the shell 
    (see `here <https://en.wikipedia.org/wiki/Shebang_(Unix)>`_)"""
    comment_char: str = field(repr=False)
    """The character used to denote comments in the shell"""
    new_line_char: str = field(repr=False)
    """The character used to denote new lines in the shell"""

    # Batch file properties
    batch_extension: str = field(repr=False)
    """The file extension for scripts"""
    batch_command: str = field(repr=False)
    """The command used to execute the batch file"""
    _batch_command_fstring_keys: frozenset[str] = \
        field(repr=False, default_factory=lambda: frozenset(
            {'executable', 'batch_file', 'batch_arguments'}
        ))

    def __post_init__(self):
        if not self.batch_extension.startswith('.'):
            # Workaround for frozen dataclass
            object.__setattr__(self, 'batch_extension', f'.{self.batch_extension}')
        self._validate_batch_command_fstring()

    def _validate_batch_command_fstring(self):
        """
        Check if the `batch_command` f-string is valid by ensuring it contains the
        required keys
        """
        # Check that all required keys are present in the f-string
        for key in self._batch_command_fstring_keys:
            if f'{{{key}}}' not in self.batch_command:
                raise ValueError(f'Invalid fstring for `batch_command`: '
                                 f'{self.batch_command}. Missing key: {key}')

        # Check that there are no extra keys in the f-string
        all_keys = set(re.findall(r'{(.*?)}', self.batch_command))
        for key in all_keys:
            if key not in self._batch_command_fstring_keys:
                raise ValueError(f'Invalid f-string for `batch_command`: '
                                 f'{self.batch_command}. Unexpected key: {key}')

    # Singleton instance
    __singleton: 'BaseShell' = field(default=None, init=False, repr=False)

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


class BashShell(BaseShell):
    """
    Configuration for the `Bash <https://en.wikipedia.org/wiki/Bash_(Unix_shell)>`_
    shell.
    """
    name: str = 'bash'
    executable: str = '/bin/bash'
    is_posix: bool = True
    shebang: str = '#!/bin/bash'
    comment_char: str = '#'
    new_line_char: str = '\n'
    batch_extension: str = '.sh'
    batch_command: str = '{executable} {batch_file} {batch_arguments}'

    def __init__(self):
        super().__init__(
            name=self.name,
            executable=self.executable,
            is_posix=self.is_posix,
            shebang=self.shebang,
            comment_char=self.comment_char,
            new_line_char=self.new_line_char,
            batch_extension=self.batch_extension,
            batch_command=self.batch_command
        )


class CmdShell(BaseShell):
    """
    Configuration for the `Windows Command Prompt
    <https://en.wikipedia.org/wiki/Cmd.exe>`_ shell.
    """
    name: str = 'cmd'
    executable: str = r'C:\Windows\System32\cmd.exe'
    is_posix: bool = False
    shebang: str = ''
    comment_char: str = '#'
    new_line_char: str = '\n'
    batch_extension: str = '.bat'
    batch_command: str = r'{executable} /Q /C {batch_file} {batch_arguments}'

    def __init__(self):
        super().__init__(
            name=self.name,
            executable=self.executable,
            is_posix=self.is_posix,
            shebang=self.shebang,
            comment_char=self.comment_char,
            new_line_char=self.new_line_char,
            batch_extension=self.batch_extension,
            batch_command=self.batch_command
        )


class PowerShell(BaseShell):
    """
    Configuration for the `Windows PowerShell
    <https://en.wikipedia.org/wiki/PowerShell>`_ shell.
    """
    name: str = 'powershell'
    executable: str = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    is_posix: bool = False
    shebang: str = ''
    comment_char: str = '#'
    new_line_char: str = '\n'
    batch_extension: str = '.ps1'
    batch_command: str = '{executable} -File {batch_file} {batch_arguments}'

    def __init__(self):
        super().__init__(
            name=self.name,
            executable=self.executable,
            is_posix=self.is_posix,
            shebang=self.shebang,
            comment_char=self.comment_char,
            new_line_char=self.new_line_char,
            batch_extension=self.batch_extension,
            batch_command=self.batch_command
        )


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
