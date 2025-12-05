import pytest

from tests.conftest import skipif_not_windows, skipif_not_linux
from xtl.jobs.shells import Shell, BaseShell, BashShell, CmdShell, PowerShell, \
    DefaultShell


class TestBaseShell:

    def test_subclass_enforcement(self):
        with pytest.raises(TypeError, match='Cannot instantiate class'):
            BaseShell(name='bash', executable='/bin/bash', is_posix=True,
                      shebang='#!/bin/bash', comment_char='#', new_line_char='\n',
                      batch_extension='.sh',
                      batch_command='{executable} {batch_file} {batch_arguments}')

    def test_post_init(self):
        class CustomShell(BaseShell):
            name: str = 'bash'
            executable: str = '/bin/bash'
            is_posix: bool = True
            shebang: str = '#!/bin/bash'
            comment_char: str = '#'
            new_line_char: str = '\n'
            batch_extension: str = 'sh'  # missing dot
            batch_command: str = '{executable} {batch_file} {batch_arguments}'

        shell = CustomShell()
        assert shell.batch_extension == '.sh'

        class WrongShell1(BaseShell):
            name: str = 'bash'
            executable: str = '/bin/bash'
            is_posix: bool = True
            shebang: str = '#!/bin/bash'
            comment_char: str = '#'
            new_line_char: str = '\n'
            batch_extension: str = '.sh'
            batch_command: str = '{executable} {batch_file}'  # missing {batch_arguments}

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
            batch_command: str = ('{executable} {batch_file} '
                                  '{batch_arguments} {whatever}')  # extra key

        with pytest.raises(ValueError, match='Unexpected key'):
            WrongShell2()

    def test_get_execute_batch_command(self):
        bash = BashShell()
        assert bash.get_execute_batch_command('script.sh') == '/bin/bash script.sh'
        assert bash.get_execute_batch_command('script.sh', ['arg1', 'arg2']) \
               == '/bin/bash script.sh arg1 arg2'
        assert bash.get_execute_batch_command('script.sh', ['arg1', 'arg2'], as_list=True) \
               == ['/bin/bash', 'script.sh', 'arg1', 'arg2']

        cmd = CmdShell()
        assert cmd.get_execute_batch_command('script.bat') \
               == r'C:\Windows\System32\cmd.exe /Q /C script.bat'
        assert cmd.get_execute_batch_command('script.bat', ['arg1', 'arg2']) \
               == r'C:\Windows\System32\cmd.exe /Q /C script.bat arg1 arg2'
        assert cmd.get_execute_batch_command('script.bat', ['arg1', 'arg2'], as_list=True) \
               == [r'C:\Windows\System32\cmd.exe', '/Q', '/C', 'script.bat', 'arg1', 'arg2']

        pwsh = PowerShell()
        assert pwsh.get_execute_batch_command('script.ps1') \
               == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File script.ps1'
        assert pwsh.get_execute_batch_command('script.ps1', ['arg1', 'arg2']) \
               == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File script.ps1 arg1 arg2'
        assert pwsh.get_execute_batch_command('script.ps1', ['arg1', 'arg2'], as_list=True) \
               == [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '-File', 'script.ps1', 'arg1', 'arg2']

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
