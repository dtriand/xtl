import asyncio
import os
import stat
import time

import pytest

from tests.conftest import skipif_not_linux
from xtl.config.settings import DependencySettings
from xtl.jobs.batchfiles import BatchFile, BatchFileStatus, BatchTemplate
from xtl.jobs.config import BatchJobConfig
from xtl.jobs.policies import NiceCommandPolicy
from xtl.jobs.shells import BashShell, CmdShell, DefaultShell, PowerShell, Shell
from xtl.jobs.sites import LocalSite
from xtl_core.system.os import FilePermissions
from xtl_core.system.compatibility import OS_WINDOWS


EXIT_3 = 'exit /b 3' if OS_WINDOWS else 'exit 3'
TO_STDERR = 'echo oops 1>&2' if OS_WINDOWS else 'echo oops >&2'
SLEEP_2 = 'ping -n 3 127.0.0.1 > nul' if OS_WINDOWS else 'sleep 2'


class NiceSite(LocalSite):
    _policy = NiceCommandPolicy()


class TestBatchFile:
    def test_defaults(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        assert batch.file == tmp_path / f'job{DefaultShell().batch_extension}'
        assert batch.shell is DefaultShell()
        assert isinstance(batch.compute_site, LocalSite)
        assert batch.dependencies == []
        assert batch.status == BatchFileStatus.IDLE
        assert batch.process is None
        assert batch.logger.name == 'xtl.jobs.batchfiles'
        assert repr(batch) == f'BatchFile({batch.file})'

    @pytest.mark.parametrize('shell', [BashShell(), CmdShell(), PowerShell()], ids=lambda s: s.name)
    def test_extension_follows_the_shell(self, tmp_path, shell):
        assert BatchFile(tmp_path / 'job.txt', shell=shell).file.name == f'job{shell.batch_extension}'

    def test_wrong_types(self, tmp_path):
        with pytest.raises(TypeError, match='compute_site'):
            BatchFile(tmp_path / 'job', compute_site='local')
        with pytest.raises(TypeError, match='shell'):
            BatchFile(tmp_path / 'job', shell=Shell.BASH)  # a member, not a shell

    def test_permissions(self, tmp_path):
        assert BatchFile(tmp_path / 'job', permissions='750').permissions == FilePermissions('750')
        with pytest.raises(ValueError, match='owner execute'):
            BatchFile(tmp_path / 'job', permissions='640')

    def test_dependencies_resolved(self, tmp_path):
        dependency = DependencySettings(name='mytool', provides={'mytool'})
        batch = BatchFile(tmp_path / 'job', dependencies=[dependency])
        assert [d.name for d in batch.dependencies] == ['mytool']

    def test_logger(self, tmp_path):
        import logging

        logger = logging.getLogger('batch-test')
        batch = BatchFile(tmp_path / 'job', logger=logger)
        assert batch.logger is logger
        batch.logger = logging.getLogger('other')
        assert batch.logger.name == 'other'


class TestContent:
    def test_lines_and_comments(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', shell=BashShell())
        batch.add_line('echo a')
        batch.add_lines(['echo b', 'echo c'])
        batch.add_lines('echo d')  # a single string is one line
        batch.add_comment('note')
        batch.add_comment('# already a comment')
        assert batch.get_content() == 'echo a\necho b\necho c\necho d\n# note\n# already a comment'

    def test_cmd_comments(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', shell=CmdShell())
        batch.add_comment('note')
        assert batch.get_content() == 'rem note'

    def test_lines_must_be_strings(self, tmp_path):
        with pytest.raises(TypeError, match='list of strings'):
            BatchFile(tmp_path / 'job').add_lines(['echo a', 3])

    def test_policy_applies_to_each_line(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', shell=BashShell(), compute_site=NiceSite())
        batch.add_lines(['xds_par', 'echo done'])
        assert batch.get_content() == 'nice -n 10 xds_par\nnice -n 10 echo done'
        assert batch.get_preamble() == '#!/bin/bash\n'  # the preamble isn't a command

    def test_save(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', shell=BashShell())
        batch.add_line('echo é')
        batch.save()
        assert batch.file.read_bytes() == '#!/bin/bash\n\necho é\n'.encode('utf-8')

    def test_save_cmd(self, tmp_path):
        # UTF-8 first, and CRLF line endings
        batch = BatchFile(tmp_path / 'job', shell=CmdShell())
        batch.add_lines(['echo a', 'echo b'])
        batch.save()
        assert batch.file.read_bytes() == b'chcp 65001 > nul\r\n\r\necho a\r\necho b\r\n'

    def test_blank_lines_kept(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', shell=BashShell(), compute_site=NiceSite())
        batch.add_lines(['cat <<EOF', 'a', '', 'b', 'EOF'])
        batch.add_line('')
        # Blank lines aren't commands: the policy leaves them alone
        assert batch.get_content() == 'nice -n 10 cat <<EOF\nnice -n 10 a\n\nnice -n 10 b\nnice -n 10 EOF\n'

    def test_save_does_not_overwrite(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line('echo a')
        batch.save()
        with pytest.raises(FileExistsError):
            batch.save()
        batch.add_line('echo b')
        batch.save(overwrite=True)
        assert 'echo b' in batch.file.read_text()

    @skipif_not_linux
    def test_save_sets_permissions(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', permissions='750')
        batch.save()
        assert stat.S_IMODE(os.stat(batch.file).st_mode) == 0o750

    @skipif_not_linux
    def test_save_without_permissions(self, tmp_path):
        batch = BatchFile(tmp_path / 'job', permissions='700')
        batch.save(update_permissions=False)
        assert stat.S_IMODE(os.stat(batch.file).st_mode) != 0o700


class TestTemplates:
    def render(self, template, context=None, shell=None):
        return BatchFile('job', shell=shell or BashShell())._render_template(template, context)

    def test_placeholders(self):
        assert self.render('run __PROGRAM__ --in=__INPUT_FILE__', {'PROGRAM': 'xds', 'INPUT_FILE': 'a.h5'}) == (
            'run xds --in=a.h5'
        )

    def test_magic_placeholders(self):
        rendered = self.render('__XTL_COMMENT__ __XTL_DOCSTRING____XTL_NL__echo', shell=CmdShell())
        assert rendered.startswith('rem Generated by xtl ')
        assert rendered.endswith('\necho')
        assert set(BatchTemplate.__MAGIC_VARS__) == {'XTL_COMMENT', 'XTL_NL', 'XTL_DOCSTRING'}

    def test_escaped_underscores(self):
        # A literal `__` is written `____`; unescaped, a dunder reads as a placeholder
        assert self.render('python -c "print(x.____version____)"') == 'python -c "print(x.__version__)"'
        with pytest.raises(KeyError, match='version'):
            self.render('python -c "print(x.__version__)"')

    def test_missing_value(self):
        with pytest.raises(KeyError, match='Missing required key for BatchFile template: PROGRAM'):
            self.render('run __PROGRAM__')

    def test_paths_quoted_other_values_not(self, tmp_path):
        rendered = self.render('__FILE__ __ARGS__', {'FILE': tmp_path / 'my dir', 'ARGS': '-a -b'})
        assert rendered == f"'{tmp_path / 'my dir'}' -a -b"

    def test_from_config(self, tmp_path):
        shell = DefaultShell()
        config = BatchJobConfig(
            job_directory=tmp_path,
            filename='run',
            permissions='750',
            templates={Shell(shell.name): 'echo __WHO__'},
        )
        batch = BatchFile.from_config(config, context={'WHO': 'me'})
        assert batch.file == tmp_path / f'run{shell.batch_extension}'
        assert batch.permissions == FilePermissions('750')
        assert batch.get_content() == 'echo me'

    def test_template_blank_lines_kept(self, tmp_path):
        shell = DefaultShell()
        config = BatchJobConfig(job_directory=tmp_path, templates={Shell(shell.name): 'echo a\n\necho b\n'})
        assert BatchFile.from_config(config).get_content() == shell.new_line_char.join(['echo a', '', 'echo b'])

    def test_from_dict_without_template(self, tmp_path):
        batch = BatchFile.from_config({'job_directory': tmp_path})
        assert batch.get_content() == ''


class TestExecute:
    @pytest.mark.asyncio
    async def test_must_be_saved(self, tmp_path):
        with pytest.raises(RuntimeError, match='save'):
            await BatchFile(tmp_path / 'job').execute()

    @pytest.mark.asyncio
    async def test_streams_and_exit_code(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_lines(['echo hello', TO_STDERR, EXIT_3])
        batch.save()
        await batch.execute(stdout=tmp_path / 'out.log', stderr=tmp_path / 'err.log')
        assert (tmp_path / 'out.log').read_text().strip() == 'hello'
        assert (tmp_path / 'err.log').read_text().strip() == 'oops'
        assert batch.process.returncode == 3
        assert batch.status == BatchFileStatus.FAILED

    @pytest.mark.asyncio
    async def test_completed(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line('echo fine')
        batch.save()
        await batch.execute()
        assert batch.process.returncode == 0
        assert batch.status == BatchFileStatus.COMPLETED

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', ['café', 'Δοκιμή'])
    async def test_non_ascii_text(self, tmp_path, name):
        # A path and output outside the console's code page (cmd reads a batch file as UTF-8 only after chcp)
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'data.txt').write_text('ok\n', encoding='utf-8')
        batch = BatchFile(tmp_path / 'job')
        command = 'type' if OS_WINDOWS else 'cat'
        batch.add_lines([f'{command} {batch.shell.sanitize_value(folder / "data.txt")}', f'echo {name}'])
        batch.save()
        await batch.execute(stdout=tmp_path / 'out.log', stderr=tmp_path / 'err.log')
        assert batch.status == BatchFileStatus.COMPLETED, (tmp_path / 'err.log').read_text(errors='replace')
        assert (tmp_path / 'out.log').read_text(encoding='utf-8').split() == ['ok', name]

    @pytest.mark.asyncio
    async def test_arguments(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line('echo [%~1]' if OS_WINDOWS else 'echo "[$1]"')
        batch.save()
        await batch.execute(stdout=tmp_path / 'out.log', batch_args=['a b'])
        assert (tmp_path / 'out.log').read_text().strip() == '[a b]'

    @pytest.mark.asyncio
    async def test_schedule_on_a_local_site_executes(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line('echo scheduled')
        batch.save()
        await batch.execute(schedule=True, stdout=tmp_path / 'out.log')
        assert (tmp_path / 'out.log').read_text().strip() == 'scheduled'

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line(SLEEP_2)
        batch.save()
        start = time.monotonic()
        running = asyncio.create_task(batch.execute())
        while batch.status != BatchFileStatus.RUNNING:
            await asyncio.sleep(0.01)
        await batch.cancel()
        await asyncio.wait_for(running, timeout=30)
        assert batch.status == BatchFileStatus.CANCELLED
        assert batch.process.returncode is not None
        assert time.monotonic() - start < 30

    @pytest.mark.asyncio
    async def test_cancel_when_not_running(self, tmp_path, caplog):
        batch = BatchFile(tmp_path / 'job')
        await batch.cancel()
        assert batch.status == BatchFileStatus.IDLE

    @pytest.mark.asyncio
    async def test_task_cancelled(self, tmp_path):
        batch = BatchFile(tmp_path / 'job')
        batch.add_line(SLEEP_2)
        batch.save()
        running = asyncio.create_task(batch.execute())
        while batch.status != BatchFileStatus.RUNNING:
            await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert batch.status == BatchFileStatus.CANCELLED
        assert batch.process.returncode is not None
