__all__ = ['Shell', 'DefaultShell', 'BashShell', 'CmdShell', 'PowerShell']

import copy
from dataclasses import dataclass
import re
import os
from pathlib import Path
import shlex
from typing import Any, Sequence


@dataclass
class Shell:
    """
    Contains configuration for different shells that is used for generating and running batch files.
    The parameter `batch_command` is an f-string that is used to execute the batch file. The f-string should contain
    the keys `executable` and `batchfile` which will be replaced with the path to the shell executable and the path to
    the batch file, respectively.

    :param name: The name of the shell
    :param shebang: The shebang line for the shell
    :param file_ext: The file extension for the batch file
    :param is_posix: Whether the shell is POSIX compliant
    :param executable: The path to the shell executable
    :param batch_command: The command used to execute the batch file
    :param comment_char: The character used to denote comments in the shell
    :param new_line_char: The character used to denote new lines in the shell
    """
    name: str
    shebang: str
    file_ext: str
    is_posix: bool
    executable: str
    batch_command: str
    comment_char: str = '#'
    new_line_char: str = '\n'
    _batch_fstring_keys = ['executable', 'batch_file', 'batch_arguments']

    def __post_init__(self):
        if not self.file_ext.startswith('.'):
            self.file_ext = '.' + self.file_ext
        self._validate_batch_fstring()

    def _validate_batch_fstring(self):
        """
        Validate the `batch_command` f-string to ensure that it contains the required keys and no extra keys.
        """
        # Check that all required keys are present in the fstring
        for key in self._batch_fstring_keys:
            if f'{{{key}}}' not in self.batch_command:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Missing key: {key}")

        # Check that there are no extra keys in the fstring
        all_keys = re.findall(r'{(.*?)}', self.batch_command)
        for key in all_keys:
            if key not in self._batch_fstring_keys:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Unexpected key: {key}")

    def get_batch_command(self, batch_file: str | Path, batch_arguments: Sequence[Any] = None,
                          as_list: bool = False) -> str:
        """
        Return the command used to execute the `batch_file`.
        :param batch_file: The path to the batch file
        :param batch_arguments: A list of arguments to pass to the batch file
        :param as_list: Whether to return the command as a list of strings
        """
        # Convert arguments to strings and quote them if they contain spaces
        batch_arguments = map(str, batch_arguments) if batch_arguments else []
        batch_arguments = [f'\'{arg}\'' if ' ' in arg else arg for arg in batch_arguments] if batch_arguments else []
        # Join the arguments into a single string
        batch_arguments = ' '.join(batch_arguments) if batch_arguments else ''

        # Format the batch command
        command = self.batch_command.format(executable=self.executable, batch_file=str(batch_file),
                                            batch_arguments=batch_arguments)

        # Remove trailing space when there are no arguments
        if command.endswith(' '):
            command = command[:-1]

        # Return the command as list or string
        if as_list:
            bits = shlex.split(command, posix=self.is_posix)
            return bits
        return command


@dataclass
class WslShell:
    distro: str
    shell: Shell
    executable: str = r'C:\Windows\System32\wsl.exe'
    batch_command: str = '{wsl_executable} -d {distro} -- {batch_command}'
    _batch_fstring_keys = ['wsl_executable', 'distro', 'batch_command']

    def __post_init__(self):
        self.name = f'wsl-{self.distro.lower().replace("-", "").replace(".", "")}-{self.shell.name}'

        # Mock the shell attributes to match the underlying executing Shell
        self.shebang = self.shell.shebang
        self.file_ext = self.shell.file_ext
        self.is_posix = False
        self.comment_char = self.shell.comment_char
        self.new_line_char = self.shell.new_line_char

        self._validate_batch_fstring()
        self._patch_shell()

    def _validate_batch_fstring(self):
        """
        Validate the `batch_command` f-string to ensure that it contains the required keys and no extra keys.
        """
        # Check that all required keys are present in the fstring
        for key in self._batch_fstring_keys:
            if f'{{{key}}}' not in self.batch_command:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Missing key: {key}")

        # Check that there are no extra keys in the fstring
        all_keys = re.findall(r'{(.*?)}', self.batch_command)
        for key in all_keys:
            if key not in self._batch_fstring_keys:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Unexpected key: {key}")

    def _patch_shell(self):
        self._shell = copy.deepcopy(self.shell)  # prevent modifying the original shell

        # Replace the batch_command with the WSL command
        batch_command = self.batch_command.replace('{batch_command}', self.shell.batch_command)
        batch_command = batch_command.replace('{wsl_executable}', self.executable)
        batch_command = batch_command.replace('{distro}', self.distro)
        self._shell.batch_command = batch_command
        self._shell._validate_batch_fstring()
        self._shell.batch_command = self._shell.batch_command.replace('{batch_file} {batch_arguments}', '"{batch_file} {batch_arguments}"')

        # Replace self.shell with the patched shell
        self.shell = self._shell

    def get_batch_command(self, batch_file: str | Path, batch_arguments: Sequence[Any] = None,
                          as_list: bool = False) -> str | list[str]:
        """
        Return the command used to execute the `batch_file`.
        :param batch_file: The path to the batch file
        :param batch_arguments: A list of arguments to pass to the batch file
        :param as_list: Whether to return the command as a list of strings
        """
        command = self.shell.get_batch_command(batch_file=batch_file, batch_arguments=batch_arguments, as_list=False)

        # Remove trailing space when there are no arguments
        if command.endswith(' "'):
            command = command[:-2] + '"'

        if as_list:
            bits = shlex.split(command, posix=self.is_posix)
            args = bits[-1]
            if args.startswith('"') and args.endswith('"'):
                args = args[1:-1]
                bits[-1] = args
            return bits
        return command


# Definitions for common shells
BashShell = Shell(name='bash',
                  shebang='#!/bin/bash',
                  file_ext='.sh',
                  is_posix=True,
                  executable='/bin/bash',
                  batch_command='{executable} -c {batch_file} {batch_arguments}')

CmdShell = Shell(name='cmd',
                 shebang='',
                 file_ext='.bat',
                 is_posix=False,
                 executable=r'C:\Windows\System32\cmd.exe',
                 batch_command=r'{executable} /Q /C {batch_file} {batch_arguments}')

PowerShell = Shell(name='powershell',
                   shebang='',
                   file_ext='.ps1',
                   is_posix=False,
                   executable=r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
                   batch_command='{executable} -File {batch_file} {batch_arguments}')


# Set the default shell based on the OS
DefaultShell = CmdShell if os.name == 'nt' else BashShell
