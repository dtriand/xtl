import logging
import tempfile
from pathlib import Path

import pytest

from tests.conftest import skipif_not_windows
from xtl import settings
from xtl.jobs.batchfiles import BatchFile
from xtl.jobs.config import BatchJobConfig, JobConfig, SiteResources, SteppedJobConfig
from xtl.jobs.logging import StreamHandlerConfig
from xtl.jobs.resources import Resources
from xtl.jobs.shells import DefaultShell, Shell
from xtl.jobs.sites import ComputeSite, ModulesSite
from xtl_core.system.os import FilePermissions

SYSTEM_SHELL = Shell(DefaultShell().name)


class TestJobConfig:
    def test_defaults(self):
        config = JobConfig()
        assert config.job_directory is None
        assert config.resources is None
        assert config.log_handlers == {}

    def test_resources(self):
        assert JobConfig(resources=Resources(jobs=2)).resources == Resources(jobs=2, threads=1, processes=1)

    def test_log_handlers_round_trip(self):
        config = JobConfig(log_handlers={'console': StreamHandlerConfig(stream='stderr'), 'file': None})
        assert config.to_dict()['log_handlers']['file'] is None
        for restored in (JobConfig.from_dict(config.to_dict()), JobConfig.from_json(config.to_json())):
            restored = restored.log_handlers
            assert isinstance(restored['console'], StreamHandlerConfig)
            assert restored['console'].stream == 'stderr'
            assert restored['file'] is None


class TestSiteResources:
    def test_defaults(self):
        resources = SiteResources()
        assert (resources.cpus, resources.memory, resources.gpus, resources.timeout) == (1, 1.0, 0, None)
        assert (resources.no_tasks, resources.no_nodes) == (1, 1)

    @pytest.mark.parametrize(
        'field, value', [('cpus', 0), ('memory', -1), ('gpus', -1), ('no_tasks', 0), ('no_nodes', 0)]
    )
    def test_bounds(self, field, value):
        with pytest.raises(Exception):
            SiteResources(**{field: value})


class TestBatchJobConfig:
    def test_job_directory_default(self):
        first, second = BatchJobConfig(), BatchJobConfig()
        for config in (first, second):
            assert config.job_directory.parent == Path(tempfile.gettempdir())
            assert config.job_directory.name.startswith('xtl_batch_')
        assert first.job_directory != second.job_directory

    def test_paths(self, tmp_path):
        config = BatchJobConfig(job_directory=str(tmp_path), filename='run')
        extension = config.shell.get().batch_extension
        assert config.job_directory == tmp_path
        assert config.file == tmp_path / f'run{extension}'
        assert config.stdout == tmp_path / 'run.stdout.log'
        assert config.stderr == tmp_path / 'run.stderr.log'

    def test_permissions(self, monkeypatch):
        assert BatchJobConfig(permissions='750').permissions == FilePermissions('750')
        assert BatchJobConfig(permissions=0o750).permissions == FilePermissions('750')
        assert BatchJobConfig(permissions='750').to_dict()['permissions'] == '750'
        # The default follows the settings when the config is created
        monkeypatch.setattr(settings.jobs.permissions, 'scripts', FilePermissions('711'))
        assert BatchJobConfig().permissions == FilePermissions('711')

    def test_compute_site(self, monkeypatch):
        assert BatchJobConfig(compute_site='modules').compute_site == ComputeSite.MODULES
        monkeypatch.setattr(settings.jobs, 'compute_site', 'modules')
        assert BatchJobConfig().compute_site == ComputeSite.MODULES

    def test_templates(self, tmp_path):
        config = BatchJobConfig(job_directory=tmp_path, templates={SYSTEM_SHELL: 'echo __WHO__'})
        assert config.compatible_shells == {SYSTEM_SHELL}
        assert config.get_template() == 'echo __WHO__'
        assert config.get_context() == {}
        assert BatchJobConfig(job_directory=tmp_path).get_template() is None

    def test_get_batch(self, tmp_path):
        config = BatchJobConfig(job_directory=tmp_path, templates={SYSTEM_SHELL: 'echo __WHO__'})
        batch = config.get_batch(context={'WHO': 'me'})
        assert isinstance(batch, BatchFile)
        assert batch.file == config.file
        assert batch.get_content() == 'echo me'

    def test_round_trip(self, tmp_path):
        config = BatchJobConfig(
            job_directory=tmp_path, default_shell=Shell.POWERSHELL, templates={Shell.POWERSHELL: 'x'}
        )
        data = config.to_dict()
        assert data['shell'] == 'powershell'
        restored = BatchJobConfig.from_dict(data)
        assert restored.shell == Shell.POWERSHELL
        assert restored.file == config.file

    def test_to_slurm_names_and_logs(self, tmp_path):
        config = BatchJobConfig(job_directory=tmp_path, name='fit', description='a fit')
        args = config.to_slurm()
        assert args[:2] == ['--job-name=fit', '--comment=a fit']
        assert args[-2:] == [f'--output={config.stdout}', f'--error={config.stderr}']
        assert BatchJobConfig(job_directory=tmp_path).to_slurm()[0].startswith('--cpus=')


class TestShellChoice:
    """The shell is chosen when the configuration is created, from the compute site and the templates."""

    def test_site_without_restrictions_uses_default_shell(self):
        config = BatchJobConfig(compute_site='local', default_shell=Shell.POWERSHELL)
        assert config.shell == Shell.POWERSHELL

    def test_otherwise_the_systems_shell(self):
        assert BatchJobConfig(compute_site='local').shell == SYSTEM_SHELL

    def test_shell_without_a_template_raises(self):
        with pytest.raises(RuntimeError, match='Incompatible shell'):
            BatchJobConfig(compute_site='local', default_shell=Shell.CMD, templates={Shell.BASH: 'x'})

    @pytest.fixture
    def bash_only_site(self, monkeypatch):
        monkeypatch.setattr(ModulesSite, '_supported_shells', frozenset({Shell.BASH}))
        monkeypatch.setattr(ModulesSite, '_default_shell', Shell.BASH)
        return 'modules'

    def test_restricting_site_without_templates_uses_its_shell(self, bash_only_site):
        assert BatchJobConfig(compute_site=bash_only_site, default_shell=Shell.CMD).shell == Shell.BASH

    def test_restricting_site_takes_a_shared_shell(self, bash_only_site):
        templates = {Shell.BASH: 'x', Shell.CMD: 'y'}
        config = BatchJobConfig(compute_site=bash_only_site, default_shell=Shell.CMD, templates=templates)
        assert config.shell == Shell.BASH

    @skipif_not_windows
    def test_no_shared_shell_falls_back_to_the_system_with_a_warning(self, bash_only_site, caplog):
        # A cmd template only: nothing in common with a bash-only site, so the system's shell, with a warning
        config_logger = logging.getLogger('xtl.jobs.config')
        config_logger.addHandler(caplog.handler)
        try:
            config = BatchJobConfig(compute_site=bash_only_site, templates={Shell.CMD: 'x'})
        finally:
            config_logger.removeHandler(caplog.handler)
        assert config.shell == Shell.CMD
        assert 'is not supported by compute site' in caplog.text


def test_stepped_job_config():
    assert SteppedJobConfig().steps == {}
    assert SteppedJobConfig(steps={'a': {'value': 1}}).steps == {'a': {'value': 1}}
