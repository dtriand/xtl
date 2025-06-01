import pytest
import pytest_asyncio

from pathlib import Path
import subprocess

from tests.conftest import skipif_not_linux, skipif_not_windows, skipif_not_wsl, supported_distros, wsl_distro_exists
from xtl.automate.sites import LocalSite, BiotixHPC
from xtl.automate.shells import Shell, DefaultShell, BashShell, PowerShell, CmdShell, WslShell
from xtl.automate.batchfile import BatchFile
from xtl.automate.jobs import Job, limited_concurrency
from xtl.common.compatibility import OS_WINDOWS
from xtl.exceptions.warnings import IncompatibleShellWarning


class TestLimitedConcurrency:

    @pytest.mark.asyncio
    async def test_decoration(self):

        def f(): pass
        f = limited_concurrency(10)(f)

        assert f._is_semaphore_limited == True


class TestJob:

    def test_init(self):
        job = Job('test_job')
        assert job._name == 'test_job'
        assert job._shell == job._default_shell
        assert isinstance(job._compute_site, LocalSite)
        assert job._stdout.name == 'test_job.stdout.log'
        assert job._stderr.name == 'test_job.stderr.log'

        custom_shell = Shell(name='csh',
                             shebang='#!/bin/csh',
                             file_ext='.csh',
                             is_posix=True,
                             executable='/bin/csh',
                             batch_command='{executable} -c {batch_file} {batch_arguments}')

        job = Job('test_job', stdout_log='test_stdout.log', stderr_log='test_stderr.log',
                  compute_site=BiotixHPC(), shell=custom_shell)
        assert job._shell == custom_shell
        assert isinstance(job._compute_site, BiotixHPC)
        assert job._stdout.name == 'test_stdout.log'
        assert job._stderr.name == 'test_stderr.log'

    def test_determine_shell_and_site(self):
        j = Job('test_job')
        # No shell or compute_site -> job defaults
        assert j._determine_shell_and_site(shell=None, compute_site=None) == (j._default_shell, LocalSite())
        # Shell specified, no compute_site requirements -> shell defaults
        assert j._determine_shell_and_site(shell=CmdShell, compute_site=None) == (CmdShell, LocalSite())
        # No shell, no compute_site requirements -> job defaults
        assert j._determine_shell_and_site(shell=None, compute_site=LocalSite()) == (j._default_shell, LocalSite())
        # Shell but not compute_site requirements -> shell
        assert j._determine_shell_and_site(shell=PowerShell, compute_site=LocalSite()) == (PowerShell, LocalSite())
        # No shell but compute_site requirements -> compute_site defaults
        assert j._determine_shell_and_site(shell=None, compute_site=BiotixHPC()) == (BashShell, BiotixHPC())
        # Shell incompatible with compute_site -> shell but raise warning
        with pytest.warns(IncompatibleShellWarning, match='Shell \'powershell\' is not compatible with compute_site \'BiotixHPC\''):
            assert j._determine_shell_and_site(shell=PowerShell, compute_site=BiotixHPC()) == (PowerShell, BiotixHPC())
        # Shell incompatible with job -> shell but raise warning
        j._supported_shells = [CmdShell, BashShell]
        with pytest.warns(IncompatibleShellWarning, match='Shell \'powershell\' is not compatible with job \'Job\''):
            assert j._determine_shell_and_site(shell=PowerShell, compute_site=LocalSite()) == (PowerShell, LocalSite())
        # No shell but both job and compute_site requirements -> multiple compatible shells -> job default
        cs = LocalSite()
        cs._supported_shells = [CmdShell, BashShell]
        assert j._determine_shell_and_site(shell=None, compute_site=cs) == (BashShell, cs)
        # No shell but both job and compute_site requirements -> multiple compatible shell -> choose first
        j._default_shell = None
        assert j._determine_shell_and_site(shell=None, compute_site=cs) == (CmdShell, cs)
        # All above failed -> default shell
        j._default_shell = None
        j._supported_shells = []
        cs._default_shell = None
        cs._supported_shells = []
        assert j._determine_shell_and_site(shell=None, compute_site=cs) == (DefaultShell, cs)

    def test_echo(self, capsys):
        job = Job('test_job')
        job.echo('test')
        captured = capsys.readouterr()
        assert captured.out == '[test_job] test\n'
        assert captured.err == ''

    @pytest.mark.make_temp_files('test_job' + DefaultShell.file_ext)
    def test_create_batch(self, temp_files):
        job = Job('test_job', shell=DefaultShell)
        batch = job.create_batch(filename=temp_files, cmds=['echo "Hello, World!"'])
        assert isinstance(batch, BatchFile)
        assert batch.file.name == 'test_job' + DefaultShell.file_ext
        assert batch.file.exists()
        text = DefaultShell.shebang + DefaultShell.new_line_char if DefaultShell.shebang else ''
        text += 'echo "Hello, World!"'
        assert batch.file.read_text() == text

    @pytest.mark.make_temp_files('test_job' + DefaultShell.file_ext, 'test_job.stdout.log', 'test_job.stderr.log')
    @pytest.mark.asyncio
    async def test_run(self, temp_files):
        batch, stdout, stderr = temp_files
        job = Job('test_job', shell=DefaultShell)
        batch = job.create_batch(filename=batch, cmds=['echo "Hello, World!"'])
        assert batch.file.exists()

        await job.run_batch(batchfile=batch, stdout_log=stdout, stderr_log=stderr)

        assert stdout.exists()
        if job._shell == CmdShell:
            assert stdout.read_text() == '"Hello, World!"\n'
        else:
            assert stdout.read_text() == 'Hello, World!\n'

        assert stderr.exists()
        assert stderr.read_text() == ''

    @skipif_not_windows
    @pytest.mark.make_temp_files('test_dir')
    @pytest.mark.parametrize(
        'shell,      cmd,             expected', [
        (CmdShell,   'echo "%2"',     '"World!"\n'),
        (PowerShell, 'echo $args[1]', 'World!\n')
    ], ids=['cmd', 'powershell'])
    @pytest.mark.asyncio
    async def test_run_args_win(self, temp_files, shell, cmd, expected):
        job = Job('test_job', shell=shell)
        batch = job.create_batch(filename=(temp_files / 'batch').with_suffix(shell.file_ext), cmds=[cmd])
        assert batch.file.exists()

        stdout = temp_files / 'stdout.log'
        stderr = temp_files / 'stderr.log'
        await job.run_batch(batchfile=batch, arguments=['Hello', 'World!'], stdout_log=stdout, stderr_log=stderr)

        assert stdout.exists()
        assert stdout.read_text() == expected
        assert stderr.exists()
        assert stderr.read_text() == ''

    @skipif_not_windows
    @skipif_not_wsl
    @pytest.mark.parametrize('distro', supported_distros)
    @pytest.mark.asyncio
    async def test_run_args_wsl(self, distro):
        # TEMP: Mocking execution within WSL. This needs to be properly implemented in the future.

        if not wsl_distro_exists(distro):
            pytest.skip(f'{distro} distro not installed')

        # Create a test script
        script = Path(fr'\\wsl.localhost\{distro}\tmp\pytest\test.sh')
        script.parent.mkdir(parents=True, exist_ok=True)
        script.touch()
        assert script.exists()

        # Create shell
        shell = WslShell(distro=distro, shell=BashShell)
        # HACK: Fix for passing arguments to batch files through WSL
        shell.shell.batch_command = '{executable} -c {batch_file} {batch_arguments}'
        shell._patch_shell()

        # Create job
        job = Job('test_job', shell=shell)
        batch = job.create_batch(filename=script, cmds=['echo "Hello $2"'])
        subprocess.run(f'wsl -d {distro} -e chmod u+x {batch._wsl_filename}')  # default chmod doesn't work through WSL
        assert batch.file.exists()

        stdout = script.parent / 'stdout.log'
        stderr = script.parent / 'stderr.log'
        await job.run_batch(batchfile=batch, arguments=['Hello', 'to the World!'], # test for spaces in arguments
                            stdout_log=stdout, stderr_log=stderr)

        assert stdout.exists()
        assert stdout.read_text() == 'Hello to the World!\n'
        assert stderr.exists()
        assert stderr.read_text() == ''

        # Delete test script
        for f in [script, stdout, stderr]:
            f.unlink()
            assert not f.exists()

    @skipif_not_linux
    @pytest.mark.make_temp_files('test_job', 'test_job/test_job.stdout.log', 'test_job/test_job.stderr.log')
    @pytest.mark.asyncio
    async def test_run_all_shells_lin(self, temp_files):
        batch_dir, stdout, stderr = temp_files

        for shell in [BashShell]:
            job = Job('test_job', shell=shell)
            batch = job.create_batch(filename=(batch_dir / 'job').with_suffix(shell.file_ext),
                                     cmds=['echo "Hello, World!"'])
            assert batch.file.exists()

            await job.run_batch(batchfile=batch, stdout_log=stdout, stderr_log=stderr)

            assert stdout.exists()
            assert stdout.read_text() == 'Hello, World!\n'
            assert stderr.exists()
            assert stderr.read_text() == ''

    @skipif_not_linux
    @pytest.mark.make_temp_files('test_dir')
    @pytest.mark.parametrize(
        'shell,     cmd,         expected', [
        (BashShell, 'echo "$2"', 'World!\n')
        ], ids=['bash'])
    @pytest.mark.asyncio
    async def test_run_args_lin(self, temp_files, shell, cmd, expected):
        job = Job('test_job', shell=shell)
        batch = job.create_batch(filename=(temp_files / 'batch').with_suffix(shell.file_ext), cmds=[cmd])
        assert batch.file.exists()

        stdout = temp_files / 'stdout.log'
        stderr = temp_files / 'stderr.log'
        await job.run_batch(batchfile=batch, arguments=['Hello', 'World!'], stdout_log=stdout, stderr_log=stderr)

        assert stdout.exists()
        assert stdout.read_text() == expected
        assert stderr.exists()
        assert stderr.read_text() == ''

    @pytest.mark.make_temp_files('test_job' + DefaultShell.file_ext)
    @pytest.mark.asyncio
    async def test_run_missing_batchfile(self, temp_files):
        job = Job('test_job', shell=DefaultShell)
        batch = job.create_batch(filename=temp_files, cmds=['echo "Hello, World!"'])
        assert batch.file.exists()
        batch.file.unlink()
        assert not batch.file.exists()

        stdout, stderr = temp_files.with_suffix('.stdout.log'), temp_files.with_suffix('.stderr.log')
        await job.run_batch(batchfile=batch, stdout_log=stdout, stderr_log=stderr)
        if OS_WINDOWS:
            assert ('not recognized as an internal or external command, operable program or batch file' in
                    stderr.read_text().replace('\n', ' '))
        else:
            assert 'No such file or directory' in stderr.read_text().replace('\n', ' ')

    def test_update_concurrency_limit(self):
        j = Job('test_job')
        assert j._no_parallel_jobs == 10
        assert j._is_semaphore_modified == False

        SubJob = Job.update_concurrency_limit(5)
        j = SubJob('test_subjob')
        assert j._no_parallel_jobs == 5
        assert j._is_semaphore_modified == True
        assert SubJob.__name__ == 'ModifiedJob'
