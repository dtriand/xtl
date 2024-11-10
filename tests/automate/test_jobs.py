import pytest
import pytest_asyncio

import os

from xtl.automate.sites import LocalSite, BiotixHPC
from xtl.automate.shells import Shell, DefaultShell, BashShell, PowerShell, CmdShell
from xtl.automate.batchfile import BatchFile
from xtl.automate.jobs import Job, limited_concurrency


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
        assert job._shell == DefaultShell
        assert isinstance(job._compute_site, LocalSite)
        assert job._stdout.name == 'test_job.stdout.log'
        assert job._stderr.name == 'test_job.stderr.log'

        custom_shell = Shell(name='csh',
                             shebang='#!/bin/csh',
                             file_ext='.csh',
                             is_posix=True,
                             executable='/bin/csh',
                             batch_command='{executable} -c {batchfile}')

        job = Job('test_job', stdout_log='test_stdout.log', stderr_log='test_stderr.log',
                  compute_site=BiotixHPC(), shell=custom_shell)
        assert job._shell == custom_shell
        assert isinstance(job._compute_site, BiotixHPC)
        assert job._stdout.name == 'test_stdout.log'
        assert job._stderr.name == 'test_stderr.log'

    def test_determine_shell_and_site(self):
        assert Job._determine_shell_and_site(shell=None, compute_site=None) == (DefaultShell, LocalSite())
        assert Job._determine_shell_and_site(shell=CmdShell, compute_site=None) == (CmdShell, LocalSite())
        assert Job._determine_shell_and_site(shell=None, compute_site=LocalSite()) == (DefaultShell, LocalSite())
        assert Job._determine_shell_and_site(shell=PowerShell, compute_site=LocalSite()) == (PowerShell, LocalSite())
        assert Job._determine_shell_and_site(shell=None, compute_site=BiotixHPC()) == (BashShell, BiotixHPC())
        with pytest.warns(UserWarning, match='Shell \'powershell\' is not compatible with compute_site \'BiotixHPC\''):
            assert Job._determine_shell_and_site(shell=PowerShell, compute_site=BiotixHPC()) == (PowerShell, BiotixHPC())

    def test_echo(self, capsys):
        job = Job('test_job')
        job.echo('test')
        captured = capsys.readouterr()
        assert captured.out == '[test_job] test\n'
        assert captured.err == ''

    @pytest.mark.make_temp_files('test_job' + DefaultShell.file_ext)
    def test_create_batch(self, temp_files):
        job = Job('test_job')
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
        job = Job('test_job')
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

    @pytest.mark.skipif(os.name != 'nt', reason='Test only for Windows')
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

    @pytest.mark.skipif(os.name != 'posix', reason='Test only for POSIX systems')
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

    @pytest.mark.skipif(os.name != 'posix', reason='Test only for POSIX systems')
    @pytest.mark.make_temp_files('test_dir')
    @pytest.mark.parametrize(
        'shell,     cmd,         expected', [
        (BashShell, 'echo "$2"', '"World!"\n')
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
        job = Job('test_job')
        batch = job.create_batch(filename=temp_files, cmds=['echo "Hello, World!"'])
        assert batch.file.exists()
        batch.file.unlink()
        assert not batch.file.exists()

        stdout, stderr = temp_files.with_suffix('.stdout.log'), temp_files.with_suffix('.stderr.log')
        await job.run_batch(batchfile=batch, stdout_log=stdout, stderr_log=stderr)
        assert ('not recognized as an internal or external command, operable program or batch file' in
                stderr.read_text().replace('\n', ' '))

    def test_update_concurrency_limit(self):
        j = Job('test_job')
        assert j._no_parallel_jobs == 10
        assert j._is_semaphore_modified == False

        SubJob = Job.update_concurrency_limit(5)
        j = SubJob('test_subjob')
        assert j._no_parallel_jobs == 5
        assert j._is_semaphore_modified == True
        assert SubJob.__name__ == 'ModifiedJob'