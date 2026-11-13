import asyncio
import shlex

import pytest

from tests.conftest import skipif_not_windows, skipif_not_linux
from xtl.jobs.shells import Shell, BaseShell, BashShell, CmdShell, PowerShell, DefaultShell


class TestBaseShell:
    def test_subclass_enforcement(self):
        with pytest.raises(TypeError, match='Cannot instantiate class'):
            BaseShell(
                name='bash',
                executable='/bin/bash',
                is_posix=True,
                shebang='#!/bin/bash',
                comment_char='#',
                new_line_char='\n',
                batch_extension='.sh',
                batch_command='{executable} {batch_file} {batch_arguments}',
            )

    def test_post_init(self):
        class CustomShell(BaseShell):
            name = 'bash'
            executable = '/bin/bash'
            is_posix = True
            shebang = '#!/bin/bash'
            comment_char = '#'
            new_line_char = '\n'
            batch_extension = 'sh'  # missing dot
            batch_command = '{executable} {batch_file} {batch_arguments}'
            dependency_command = 'which {dependency}'
            execute_command = '{executable} -c {command}'

        shell = CustomShell()
        assert shell.batch_extension == '.sh'

        class WrongShell1(BaseShell):
            name = 'bash'
            executable = '/bin/bash'
            is_posix = True
            shebang = '#!/bin/bash'
            comment_char = '#'
            new_line_char = '\n'
            batch_extension = '.sh'
            batch_command = '{executable} {batch_file}'  # missing {batch_arguments}
            dependency_command = 'which {dependency}'
            execute_command = '{executable} -c {command}'

        with pytest.raises(ValueError, match='Missing key'):
            WrongShell1()

        class WrongShell2(BaseShell):
            name: str = 'bash'
            executable: str = '/bin/bash'
            is_posix: bool = True
            shebang: str = '#!/bin/bash'
            comment_char: str = '#'
            new_line_char: str = '\n'
            batch_extension: str = '.sh'
            batch_command: str = '{executable} {batch_file} {batch_arguments} {whatever}'  # extra key
            dependency_command = 'which {dependency}'
            execute_command = '{executable} -c {command}'

        with pytest.raises(ValueError, match='Unexpected key'):
            WrongShell2()

    def test_get_execute_batch_command(self):
        bash = BashShell()
        assert bash.get_execute_batch_command('script.sh') == '/bin/bash script.sh'
        assert bash.get_execute_batch_command('script.sh', ['arg1', 'arg2']) == '/bin/bash script.sh arg1 arg2'
        assert bash.get_execute_batch_command('script.sh', ['arg1', 'arg2'], as_list=True) == [
            '/bin/bash',
            'script.sh',
            'arg1',
            'arg2',
        ]

        cmd = CmdShell()
        assert cmd.get_execute_batch_command('script.bat') == r'C:\Windows\System32\cmd.exe /Q /C call script.bat'
        assert (
            cmd.get_execute_batch_command('script.bat', ['arg1', 'arg2'])
            == r'C:\Windows\System32\cmd.exe /Q /C call script.bat arg1 arg2'
        )
        assert cmd.get_execute_batch_command('script.bat', ['arg1', 'arg2'], as_list=True) == [
            r'C:\Windows\System32\cmd.exe',
            '/Q',
            '/C',
            'call',
            'script.bat',
            'arg1',
            'arg2',
        ]

        pwsh = PowerShell()
        assert (
            pwsh.get_execute_batch_command('script.ps1')
            == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File script.ps1'
        )
        assert (
            pwsh.get_execute_batch_command('script.ps1', ['arg1', 'arg2'])
            == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File script.ps1 arg1 arg2'
        )
        assert pwsh.get_execute_batch_command('script.ps1', ['arg1', 'arg2'], as_list=True) == [
            r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
            '-File',
            'script.ps1',
            'arg1',
            'arg2',
        ]

    @pytest.mark.parametrize('shell', [BashShell(), CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_get_execute_batch_command_spaces_and_quotes(self, shell):
        file = '/tmp/my dir/x' + shell.batch_extension
        arguments = ['x y', "it's", 'say "hi"', '', 3]
        bits = shell.get_execute_batch_command(file, arguments, as_list=True)
        assert bits[0] == shell.executable
        assert bits[-6:] == [file, 'x y', "it's", 'say "hi"', '', '3']

        command = shell.get_execute_batch_command(file, arguments)
        if shell.is_posix:
            assert shlex.split(command) == bits

    def test_batch_arguments_word_of_its_own(self):
        with pytest.raises(ValueError, match='word of its own'):

            class GluedShell(BaseShell):
                name = 'glued'
                executable = '/bin/glued'
                is_posix = True
                shebang = '#!/bin/glued'
                comment_char = '#'
                new_line_char = '\n'
                batch_extension = '.sh'
                batch_command = '{executable} {batch_file} --args={batch_arguments}'
                dependency_command = 'which {dependency}'
                execute_command = '{executable} -c {command}'

            GluedShell()

    @skipif_not_linux
    @pytest.mark.asyncio
    async def test_bash_runs_batch_file_in_path_with_spaces(self, tmp_path):
        directory = tmp_path / 'my dir'
        directory.mkdir()
        file = directory / 'x.sh'
        file.write_bytes(b'echo "[$1] [$2]"\nexit 3\n')

        bits = BashShell().get_execute_batch_command(file, ['x y', "it's"], as_list=True)
        process = await asyncio.create_subprocess_exec(*bits, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await process.communicate()
        assert process.returncode == 3
        assert stdout.decode().strip() == "[x y] [it's]"

    @skipif_not_windows
    @pytest.mark.asyncio
    async def test_cmd_runs_batch_file_in_path_with_spaces(self, tmp_path):
        directory = tmp_path / 'my dir'
        directory.mkdir()
        file = directory / 'x.bat'
        file.write_bytes(b'@echo off\r\necho [%~1] [%~2]\r\nexit /b 3\r\n')

        bits = CmdShell().get_execute_batch_command(file, ['x y', "it's"], as_list=True)
        process = await asyncio.create_subprocess_exec(*bits, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await process.communicate()
        assert process.returncode == 3
        assert stdout.decode().strip() == "[x y] [it's]"

    @skipif_not_linux
    def test_default_shell_linux(self):
        assert DefaultShell() == BashShell()

    @skipif_not_windows
    def test_default_shell_windows(self):
        assert DefaultShell() == CmdShell()


class TestShellEnum:
    def test_equality(self):
        assert Shell.BASH == BashShell()
        assert Shell.CMD == CmdShell()
        assert Shell.POWERSHELL == PowerShell()

        assert BashShell() == Shell.BASH
        assert CmdShell() == Shell.CMD
        assert PowerShell() == Shell.POWERSHELL

    def test_singleton(self):
        assert BashShell() == BashShell()
        assert CmdShell() is CmdShell()
