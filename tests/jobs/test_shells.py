import asyncio
from pathlib import Path
import shlex
import subprocess

import pytest

from tests.conftest import skipif_not_windows, skipif_not_linux
from xtl.jobs.shells import Shell, BaseShell, BashShell, CmdShell, PowerShell, DefaultShell
from xtl_core.system.compatibility import OS_WINDOWS


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

    def test_get(self):
        assert Shell.BASH.get() is BashShell()
        assert Shell.CMD.get() is CmdShell()
        assert Shell.POWERSHELL.get() is PowerShell()
        assert Shell.DEFAULT.get() is DefaultShell()

    def test_strings_and_hash(self):
        assert Shell('bash') is Shell.BASH
        assert Shell.BASH == 'bash'
        assert {Shell.BASH: 1}['bash'] == 1


SHELLS = [BashShell(), CmdShell(), PowerShell()]


def installed(shell: BaseShell) -> bool:
    return Path(shell.executable).exists()


class TestCommands:
    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    def test_execute_command_is_one_argument(self, shell):
        command = 'echo "a b" \'c\''
        bits = shell.get_execute_command(command, as_list=True)
        assert bits[0] == shell.executable
        assert bits[-1] == command  # without the quotes around the placeholder
        assert shell.get_execute_command(command) == shell.join(bits)

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    @pytest.mark.parametrize('name', ['', 'a b', 'a;rm', 'a$(x)', '../a', 'a' * 256])
    def test_unsafe_dependency_names(self, shell, name):
        with pytest.raises(ValueError, match='Invalid dependency name'):
            shell.get_dependency_resolution_command(name)

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    def test_dependency_command_runs_the_check(self, shell):
        bits = shell.get_dependency_resolution_command('my-tool.2', as_list=True)
        assert bits[-1] == shell.dependency_command.format(dependency='my-tool.2')

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    @pytest.mark.parametrize('exists', [True, False])
    def test_dependency_found_or_missing(self, shell, exists):
        if not installed(shell):
            pytest.skip(f'{shell.executable} not found')
        name = ('cmd' if OS_WINDOWS else 'ls') if exists else 'no-such-program-xtl'
        bits = shell.get_dependency_resolution_command(name, as_list=True)
        # NB: Windows PowerShell's `Get-Command` searches every installed module before it gives up on a name, which
        #  took over a minute on a GitHub runner
        result = subprocess.run(bits, capture_output=True, text=True, timeout=300)
        assert result.stdout.strip() == ('0' if exists else '1')

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    def test_execute_command_runs(self, shell):
        if not installed(shell):
            pytest.skip(f'{shell.executable} not found')
        # NB: cmd.exe doesn't understand the \" that a list's quotes are escaped with, so no quotes for it
        command = {'bash': 'echo "a  b"', 'cmd': 'echo a  b', 'powershell': 'Write-Output "a  b"'}[shell.name]
        result = subprocess.run(shell.get_execute_command(command, as_list=True), capture_output=True, text=True)
        assert result.stdout.strip() == 'a  b'


def run_batch(shell: BaseShell, folder: Path, lines: list[str]) -> subprocess.CompletedProcess:
    """Run lines as a batch file of the shell, as a job's would run."""
    batch = folder / f'run{shell.batch_extension}'
    text = (f'{shell.shebang}{shell.new_line_char}' if shell.shebang else '') + ''.join(
        line + shell.new_line_char for line in lines
    )
    batch.write_text(text, encoding='utf-8', newline='')
    return subprocess.run(shell.get_execute_batch_command(batch, as_list=True), capture_output=True, text=True)


def fake_program(folder: Path, name: str = 'xtl-tool') -> Path:
    """A program that only exists in `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    if OS_WINDOWS:
        (folder / f'{name}.bat').write_text('@echo ran\r\n')
    else:
        program = folder / name
        program.write_text('#!/bin/sh\necho ran\n')
        program.chmod(0o755)
    return folder


class TestDependencyCommands:
    def test_bash(self):
        shell = BashShell()
        assert shell.get_path_command('/opt/my tools') == 'export PATH=\'/opt/my tools\':"$PATH"'
        assert shell.get_module_command('purge') == 'module purge'
        assert shell.get_load_command(['gcc/12', 'xds']) == 'module load gcc/12 xds'
        assert shell.get_load_report_command('gcc/12') == (
            "module load gcc/12 && echo 'xtl-check module gcc/12 0' || echo 'xtl-check module gcc/12 1'"
        )
        assert shell.get_found_report_command('datcmp') == (
            "command -v datcmp > /dev/null 2>&1 && echo 'xtl-check executable datcmp 0' "
            "|| echo 'xtl-check executable datcmp 1'"
        )
        assert shell.get_guard_command('datcmp') == (
            "command -v datcmp > /dev/null 2>&1 || { echo 'xtl: datcmp not found' >&2; exit 127; }"
        )
        assert 'MODULESHOME' in shell.module_setup

    def test_cmd(self):
        shell = CmdShell()
        assert shell.get_path_command('C:\\my tools\\100%') == 'set "PATH=C:\\my tools\\100%%;%PATH%"'
        assert shell.get_module_command('purge') == 'call module purge'
        assert shell.get_load_command(['gcc/12']) == 'call module load gcc/12'
        assert shell.get_load_report_command('gcc/12') == (
            'call module load gcc/12 && echo xtl-check module gcc/12 0 || echo xtl-check module gcc/12 1'
        )
        assert shell.get_found_report_command('datcmp') == (
            'where datcmp > nul 2>&1 && echo xtl-check executable datcmp 0 || echo xtl-check executable datcmp 1'
        )
        assert shell.get_guard_command('datcmp') == (
            'where datcmp > nul 2>&1 || (echo xtl: datcmp not found 1>&2 & exit /b 127)'
        )
        # `module.cmd` sets itself up
        assert shell.module_setup == ''

    def test_cmd_path_with_a_quote(self):
        with pytest.raises(ValueError, match='Invalid directory'):
            CmdShell().get_path_command('C:\\a"b')

    def test_powershell(self):
        shell = PowerShell()
        assert shell.get_path_command(Path("C:/it's")) == (
            f"$env:PATH = '{Path('C:/it' + chr(39) * 2 + 's')}' + [IO.Path]::PathSeparator + $env:PATH"
        )
        # `module` is PowerShell's own command
        assert shell.get_module_command('purge') == 'envmodule purge'
        assert shell.get_load_report_command('gcc/12') == (
            "envmodule load gcc/12; if ($?) { 'xtl-check module gcc/12 0' } else { 'xtl-check module gcc/12 1' }"
        )
        assert shell.get_found_report_command('datcmp') == (
            "if (Get-Command datcmp -ErrorAction SilentlyContinue) { 'xtl-check executable datcmp 0' } "
            "else { 'xtl-check executable datcmp 1' }"
        )
        assert shell.get_guard_command('datcmp') == (
            'if (-not (Get-Command datcmp -ErrorAction SilentlyContinue)) '
            "{ [Console]::Error.WriteLine('xtl: datcmp not found'); exit 127 }"
        )
        # `envmodule` comes from the profile
        assert shell.module_setup == ''

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    @pytest.mark.parametrize('name', ['', 'a b', 'a;rm', 'a$(x)', '../a', 'a' * 256])
    def test_unsafe_executable_names(self, shell, name):
        for method in (shell.get_found_report_command, shell.get_guard_command):
            with pytest.raises(ValueError, match='Invalid executable name'):
                method(name)

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    @pytest.mark.parametrize('name', ['gcc;ls', '-x', '$(ls)'])
    def test_unsafe_module_names_are_refused(self, shell, name):
        with pytest.raises(ValueError, match='Invalid module name'):
            shell.get_load_report_command(name)
        with pytest.raises(ValueError, match='Invalid module name'):
            shell.get_load_command(['gcc/12', name])

    @pytest.mark.parametrize(
        'name',
        ['gcc', 'gcc/12', 'gcc/12.2.0', 'python/3.11', 'openmpi/4.1.5-gcc_12', '.hidden/1', 'a/b/c', 'gcc\\12'],
    )
    def test_safe_module_names(self, name):
        assert BaseShell._is_module_name_safe(name)

    @pytest.mark.parametrize(
        'name',
        ['', 'a' * 256, '-x', '--help', 'gcc/-x', 'gcc/', '/gcc', 'gcc 12', 'gcc;ls', 'gcc\n', '$(ls)', 'gcc//12'],
    )
    def test_unsafe_module_names(self, name):
        assert not BaseShell._is_module_name_safe(name)

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    def test_found_after_the_path(self, shell, tmp_path):
        if not installed(shell):
            pytest.skip(f'{shell.executable} not found')
        folder = fake_program(tmp_path / 'my bin')
        result = run_batch(
            shell,
            tmp_path,
            [
                shell.get_found_report_command('xtl-tool'),
                shell.get_path_command(folder),
                shell.get_found_report_command('xtl-tool'),
            ],
        )
        # NB: cmd's `echo` keeps the space before `||`
        reported = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert reported == ['xtl-check executable xtl-tool 1', 'xtl-check executable xtl-tool 0'], result.stderr

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    @pytest.mark.parametrize('found', [True, False])
    def test_guard(self, shell, tmp_path, found):
        if not installed(shell):
            pytest.skip(f'{shell.executable} not found')
        lines = [shell.get_path_command(fake_program(tmp_path / 'bin'))] if found else []
        result = run_batch(shell, tmp_path, [*lines, shell.get_guard_command('xtl-tool'), 'echo after'])
        if found:
            assert (result.returncode, result.stdout.strip()) == (0, 'after'), result.stderr
        else:
            assert (result.returncode, result.stdout.strip()) == (127, '')
            assert result.stderr.strip() == 'xtl: xtl-tool not found'


class TestSanitizeValue:
    def test_bash(self):
        path = Path("/my dir/it's")
        assert BashShell().sanitize_value(path) == shlex.quote(str(path))

    def test_cmd(self):
        # Quotes doubled, and % too, which a batch file expands even inside quotes
        assert CmdShell().sanitize_value(Path('C:/my dir/100%')) == f'"{Path("C:/my dir/100%%")}"'

    def test_powershell(self):
        assert PowerShell().sanitize_value(Path("C:/it's")) == f"'{Path('C:/it' + chr(39) * 2 + 's')}'"

    @pytest.mark.parametrize('shell', SHELLS, ids=lambda s: s.name)
    def test_other_values_unchanged(self, shell):
        assert shell.sanitize_value('a b; c') == 'a b; c'
        assert shell.sanitize_value(3) == 3

    @skipif_not_windows
    def test_cmd_batch_file_reads_a_path_with_percent(self, tmp_path):
        folder = tmp_path / '100%done'
        folder.mkdir()
        (folder / 'data.txt').write_text('ok')
        batch = tmp_path / 'x.bat'
        batch.write_text(f'@echo off\r\ntype {CmdShell().sanitize_value(folder / "data.txt")}\r\n')
        result = subprocess.run(
            CmdShell().get_execute_batch_command(batch, as_list=True), capture_output=True, text=True
        )
        assert result.stdout.strip() == 'ok', result.stderr
