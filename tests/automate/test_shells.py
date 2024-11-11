import pytest

from tests.conftest import skipif_not_windows, skipif_not_wsl, wsl_distro_exists, supported_distros
from xtl.automate.shells import Shell, BashShell, CmdShell, PowerShell, WslShell


class TestShell:

    @pytest.mark.parametrize(
        'shell,      batch_file, batch_arguments,   as_list, expected', [
        (BashShell,  'test.sh',  [1, 2],            False,   r'/bin/bash -c test.sh 1 2'),
        (BashShell,  'test.sh',  [1, 2],            True,    ['/bin/bash', '-c', 'test.sh', '1', '2']),
        (CmdShell,   'test.bat', [1, 2],            False,   r'C:\Windows\System32\cmd.exe /Q /C test.bat 1 2'),
        (CmdShell,   'test.bat', [1, 2],            True,    [r'C:\Windows\System32\cmd.exe', '/Q', '/C', 'test.bat', '1', '2']),
        (PowerShell, 'test.ps1', [1, 2],            False,   r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -File test.ps1 1 2'),
        (PowerShell, 'test.ps1', [1, 2],            True,    [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '-File', 'test.ps1', '1', '2']),
        (BashShell,  'test.sh',  None,              False,   r'/bin/bash -c test.sh'),
        (CmdShell,   'test.bat', ['a test string'], False,   r"C:\Windows\System32\cmd.exe /Q /C test.bat 'a test string'"),
        (PowerShell, 'test.ps1', ['a test string'], True,    [r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '-File', 'test.ps1', "'a test string'"]),
        ]
    )
    def test_get_batch_command(self, shell: Shell, batch_file, batch_arguments, as_list, expected):
        assert shell.get_batch_command(batch_file=batch_file, batch_arguments=batch_arguments, as_list=as_list) == expected


_distros = ['Ubuntu-18.04', 'Ubuntu-22.04']

@skipif_not_windows
@skipif_not_wsl
class TestWslShell:

    @pytest.mark.parametrize('distro', supported_distros)
    def test_init(self, distro):
        if not wsl_distro_exists(distro):
            pytest.skip(f'{distro} distro not installed')
        wsl = WslShell(distro=distro, shell=BashShell)
        assert wsl.distro == distro

        assert wsl.shell.name == BashShell.name
        assert wsl.shell.executable == BashShell.executable
        assert wsl.shell.shebang == BashShell.shebang
        assert wsl.shell.file_ext == BashShell.file_ext
        assert wsl.shell.is_posix == BashShell.is_posix
        assert wsl.shell.comment_char == BashShell.comment_char
        assert wsl.shell.new_line_char == BashShell.new_line_char

        assert wsl.name == f'wsl-{distro.lower().replace("-", "").replace(".", "")}-bash'
        assert wsl.shebang == BashShell.shebang
        assert wsl.file_ext == BashShell.file_ext
        assert wsl.is_posix == False
        assert wsl.comment_char == BashShell.comment_char
        assert wsl.new_line_char == BashShell.new_line_char
        assert wsl.executable == r'C:\Windows\System32\wsl.exe'
        assert wsl.batch_command == r'{wsl_executable} -d {distro} -- {batch_command}'
        assert wsl.shell.batch_command == (fr'C:\Windows\System32\wsl.exe -d {distro} -- '
                                           fr'{{executable}} -c "{{batch_file}} {{batch_arguments}}"')

        assert (wsl.get_batch_command(batch_file='test.sh', batch_arguments=[1, 2]) ==
                fr'C:\Windows\System32\wsl.exe -d {distro} -- /bin/bash -c "test.sh 1 2"')

    @pytest.mark.parametrize('distro', supported_distros)
    @pytest.mark.parametrize(
        'batch_file, batch_arguments,   as_list, expected', [
        ('test.sh',  [1, 2],            False,   r'C:\Windows\System32\wsl.exe -d {distro} -- /bin/bash -c "test.sh 1 2"'),
        ('test.sh',  [1, 2],            True,    [r'C:\Windows\System32\wsl.exe', '-d', '{distro}', '--', '/bin/bash', '-c', 'test.sh 1 2']),
        ('test.sh',  None,              False,   r'C:\Windows\System32\wsl.exe -d {distro} -- /bin/bash -c "test.sh"'),
        ('test.bat', ['a test string'], False,   r'''C:\Windows\System32\wsl.exe -d {distro} -- /bin/bash -c "test.bat 'a test string'"'''),
        ('test.ps1', ['a test string'], True,    [r'C:\Windows\System32\wsl.exe', '-d', '{distro}', '--', '/bin/bash', '-c', "test.ps1 'a test string'"]),
    ])
    def test_get_batch_command(self, distro, batch_file, batch_arguments, as_list, expected):
        if not wsl_distro_exists(distro):
            pytest.skip(f'{distro} distro not installed')
        wsl = WslShell(distro=distro, shell=BashShell)
        if isinstance(expected, str):
            expected = expected.format(distro=distro)
        else:
            expected = [e.format(distro=distro) for e in expected]
        assert wsl.get_batch_command(batch_file=batch_file, batch_arguments=batch_arguments, as_list=as_list) == expected
