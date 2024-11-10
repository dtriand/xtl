__all__ = ['Shell', 'DefaultShell', 'BashShell', 'CmdShell', 'PowerShell']

from dataclasses import dataclass
import re
import os
from pathlib import Path


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

    def __post_init__(self):
        if not self.file_ext.startswith('.'):
            self.file_ext = '.' + self.file_ext
        self._validate_batch_fstring()

    def _validate_batch_fstring(self):
        """
        Validate the `batch_command` f-string to ensure that it contains the required keys and no extra keys.
        """
        keys = ['executable', 'batchfile']

        # Check that all required keys are present in the fstring
        for key in keys:
            if f'{{{key}}}' not in self.batch_command:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Missing key: {key}")

        # Check that there are no extra keys in the fstring
        all_keys = re.findall(r'{(.*?)}', self.batch_command)
        for key in all_keys:
            if key not in keys:
                raise ValueError(f"Invalid fstring for `batch_command`: {self.batch_command}. Unexpected key: {key}")

    def get_batch_command(self, batch_file: str | Path) -> str:
        """
        Return the command used to execute the `batch_file`.
        """
        return self.batch_command.format(executable=self.executable, batchfile=str(batch_file))


# Definitions for common shells
BashShell = Shell(name='bash',
                  shebang='#!/bin/bash',
                  file_ext='.sh',
                  is_posix=True,
                  executable='/bin/bash',
                  batch_command='{executable} -c {batchfile}')

CmdShell = Shell(name='cmd',
                 shebang='',
                 file_ext='.bat',
                 is_posix=False,
                 executable=r'C:\Windows\System32\cmd.exe',
                 batch_command=r'{executable} /Q /C {batchfile}')

PowerShell = Shell(name='powershell',
                   shebang='',
                   file_ext='.ps1',
                   is_posix=False,
                   executable=r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
                   batch_command='{executable} -File {batchfile}')


# Set the default shell based on the OS
DefaultShell = CmdShell if os.name == 'nt' else BashShell
