import pytest

import os
from pathlib import Path
import stat

from tests.conftest import skipif_not_windows, skipif_not_wsl, supported_distros, wsl_distro_exists
from xtl.automate.sites import LocalSite, BiotixHPC
from xtl.automate.batchfile import BatchFile, DefaultShell
from xtl.automate.shells import Shell, BashShell, CmdShell, PowerShell, WslShell


class TestBatchFile:

    def test_filename(self):
        b = BatchFile(filename='test', shell=BashShell)
        assert b.file.name == 'test.sh'

        b = BatchFile(filename='test.csh', shell=BashShell)
        assert b.file.name == 'test.sh'

    def test_shell(self):
        shell = CmdShell if os.name == 'nt' else BashShell

        b = BatchFile(filename='test')
        assert b.shell == shell

        custom_shell = shell=Shell(name='csh',
                                   shebang='#!/bin/csh',
                                   file_ext='.csh',
                                   is_posix=True,
                                   executable='/bin/csh',
                                   batch_command='{executable} -c {batch_file} {batch_arguments}')

        b = BatchFile(filename='test', shell=custom_shell)
        assert b.shell.name == 'csh'
        assert b.shell.shebang == '#!/bin/csh'
        assert b.shell.file_ext == '.csh'
        assert b.shell.executable == '/bin/csh'
        assert b.shell.comment_char == '#'
        assert b.shell.new_line_char == '\n'

    @skipif_not_windows
    @skipif_not_wsl
    @pytest.mark.parametrize('distro', supported_distros)
    def test_wsl_shell(self, distro):
        if not wsl_distro_exists(distro):
            pytest.skip(f'{distro} distro not installed')

        # Create a test script
        script = Path(fr'\\wsl.localhost\{distro}\tmp\pytest\test.sh')
        script.parent.mkdir(parents=True, exist_ok=True)
        script.touch()
        assert script.exists()

        # Create batch file
        b = BatchFile(filename=script, shell=WslShell(distro=distro, shell=BashShell))
        assert b.get_execute_command() == f'C:\\Windows\\System32\\wsl.exe -d {distro} -- /bin/bash -c "/tmp/pytest/test.sh"'

        # Delete test script
        script.unlink()
        assert not script.exists()

    def test_compute_site(self):
        b = BatchFile(filename='test')
        assert isinstance(b.compute_site, LocalSite)

        b = BatchFile(filename='test', compute_site=BiotixHPC())
        assert isinstance(b.compute_site, BiotixHPC)

        with pytest.raises(TypeError):
            b = BatchFile(filename='test', compute_site='asdf')

    def test_execute_command(self):
        b = BatchFile(filename='test', shell=BashShell)
        assert b.file.suffix == '.sh'
        assert b.get_execute_command() == r'/bin/bash -c test.sh'

        b = BatchFile(filename='test', shell=CmdShell)
        assert b.file.suffix == '.bat'
        assert b.get_execute_command() == r'C:\Windows\System32\cmd.exe /Q /C test.bat'

        b = BatchFile(filename='test', shell=PowerShell)
        assert b.file.suffix == '.ps1'
        assert b.get_execute_command() == r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File test.ps1'

    def test_add_line(self):
        b = BatchFile(filename='test')
        b._add_line('echo "Hello, World!"')
        assert b._lines == ['echo "Hello, World!"']

    def test_add_empty_line(self):
        b = BatchFile(filename='test')
        b.add_empty_line()
        assert b._lines == ['']

    def test_add_command(self):
        b = BatchFile(filename='test')
        b.add_command('echo "Hello, World!"')
        assert b._lines == ['echo "Hello, World!"']

    def test_add_commands(self):
        b = BatchFile(filename='test')
        b.add_commands(['echo "Hello, World!"', 'echo "Goodbye, World!"'])
        assert b._lines == ['echo "Hello, World!"', 'echo "Goodbye, World!"']

    def test_add_comment(self):
        b = BatchFile(filename='test')
        b.add_comment('This is a comment')
        assert b._lines == ['# This is a comment']

    def test_load_modules(self):
        b = BatchFile(filename='test', compute_site=BiotixHPC())
        b.load_modules('gcc')
        assert b._lines == ['module load gcc']

        b.purge_file()
        b.load_modules(['gcc', 'openmpi'])
        assert b._lines == ['module load gcc openmpi']

        b.purge_file()
        b.load_modules('gcc openmpi')
        assert b._lines == ['module load gcc openmpi']

    def test_purge_modules(self):
        b = BatchFile(filename='test', compute_site=BiotixHPC())
        b.purge_modules()
        assert b._lines == ['module purge']

    def test_assign_variable(self):
        b = BatchFile(filename='test')
        b.assign_variable(variable='var', value='value')
        assert b._lines == ['var=value']

    @pytest.mark.make_temp_files('test' + DefaultShell.file_ext)
    def test_write_file(self, temp_files):
        b = BatchFile(filename=temp_files)
        b.add_command('echo "Hello, World!"')
        b.save(change_permissions=True)
        with open(temp_files, 'r') as f:
            text = DefaultShell.shebang + DefaultShell.new_line_char if DefaultShell.shebang else ''
            text += 'echo "Hello, World!"'
            assert f.read() == text

    # This test will fail on Windows, as Windows does not support file permissions
    #  see: https://stackoverflow.com/questions/27500067/chmod-issue-to-change-file-permission-using-python
    @pytest.mark.xfail(condition=(os.name == 'nt'), reason='Windows does not support file permissions')
    @pytest.mark.make_temp_files('test' + DefaultShell.file_ext)
    def test_write_file_permissions(self, temp_files):
        b = BatchFile(filename=temp_files)
        b.save(change_permissions=True)
        assert stat.filemode(temp_files.stat().st_mode) == 'rwxrw----'

        b.permissions = '0o777'
        b.save(change_permissions=True)
        assert stat.filemode(temp_files.stat().st_mode) == 'rwxrwxrwx'