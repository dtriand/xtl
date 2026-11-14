"""
End-to-end tests: real batch scripts, stepped jobs that open pools of their own, the records of worker processes, a
timeout in the middle of a launch, and a command line tool, each through every pool type where it applies.

Jobs are defined at module level, so that worker processes can import them.
"""

import asyncio
import logging
from pathlib import Path
import time

import pytest
import typer
from typer.testing import CliRunner

from tests.jobs.helpers import POOLS, WORKER_POOLS
from xtl import settings
from xtl.cli.utilities.common import ConsoleOptions, JobOptions, get_console_options, get_job_options
from xtl.cli.utilities.decorators import attach_hook, typer_async
from xtl.config.settings import XTLSettings
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.exceptions import SubprocessError
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.logging import FileHandlerConfig, JobFileHandlerConfig, LoggerConfig, LoggingFormat
from xtl.jobs.pools import MultiprocessPool
from xtl.jobs.resources import Resources, get_rc_manager
from xtl.jobs.results import JobStatus
from xtl.jobs.shells import Shell
from xtl.jobs.steps import StepSpec
from xtl.tui.console import ConsoleIO
from xtl_core.options import Option

pool_types = pytest.mark.parametrize('pool_cls', POOLS, ids=lambda c: c.__name__)


# A batch job: writes a greeting to a file in its directory and to stdout, a line to stderr, and exits with a code


class GreetConfig(BatchJobConfig):
    name: str = Option(default='world', desc='Who to greet')
    code: int = Option(default=0, desc='Exit code of the script')
    seconds: int = Option(default=0, desc='How long the script waits before it exits')
    templates: dict[Shell, str] = Option(
        default_factory=lambda: {
            Shell.BASH: 'echo hello __NAME__ > __OUT__\necho hello __NAME__\necho to stderr 1>&2\n'
            'sleep __SECONDS__\nexit __CODE__',
            Shell.CMD: 'echo hello __NAME__> __OUT__\necho hello __NAME__\necho to stderr 1>&2\n'
            'ping -n __PINGS__ 127.0.0.1 > nul\nexit /b __CODE__',
        },
        desc='Batch file templates',
    )

    def get_context(self) -> dict:
        return {
            'NAME': self.name,
            'OUT': self.shell.get().sanitize_value(self.job_directory / 'greeting.txt'),
            'CODE': self.code,
            'SECONDS': self.seconds,
            'PINGS': self.seconds + 1,
        }


class Greet(BatchJob[GreetConfig]):
    pass


# A stepped job whose first step opens a thread pool of its own, and whose second step adds up what the first returned


class SquareConfig(JobConfig):
    x: int = Option(default=0, desc='The number to square')


class Square(Job[SquareConfig]):
    async def _execute(self):
        self.logger.debug('squaring %d', self.config.x)
        self.logger.info('squared %d', self.config.x)
        return self.config.x**2


class FanConfig(JobConfig):
    xs: list[int] = Option(default_factory=list, desc='Numbers to square, one job each')
    width: int = Option(default=8, desc='Jobs to run at once in the nested pool')


class Fan(Job[FanConfig]):
    async def _execute(self):
        async with self.get_pool('threads', max_jobs=self.config.width) as pool:
            pool.submit(Square, configs=[SquareConfig(x=x) for x in self.config.xs])
            results = await pool.launch()
            width = pool.max_jobs
        return {'squares': [r.data for r in results], 'width': width}


class SumConfig(JobConfig):
    values: list[int] = Option(default_factory=list, desc='Numbers to add up')


class Sum(Job[SumConfig]):
    async def _execute(self):
        self.logger.info('adding %d numbers', len(self.config.values))
        return sum(self.config.values)


class Pipeline(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(
            name='fan',
            job_cls=Fan,
            config_cls=FanConfig,
            defaults={'xs': [1, 2, 3]},
            post_processor=lambda results, ctx: dict(results.data),
        ),
        StepSpec(
            name='sum',
            job_cls=Sum,
            config_cls=SumConfig,
            dynamic_defaults=lambda ctx: {'values': ctx.data['squares']},
        ),
    )


class KeptPipeline(Pipeline):
    _keep_files = True


def logging_to(log: Path, level: int = logging.INFO) -> LoggerConfig:
    """A pool's logging: records named relative to the pool, to a file."""
    return LoggerConfig(
        level=level,
        handlers=[
            FileHandlerConfig(filename=str(log), format=LoggingFormat(format='%(levelname)s [%(job)s] %(message)s'))
        ],
    )


class TestBatchJobs:
    @pool_types
    @pytest.mark.asyncio
    async def test_scripts_in_every_pool(self, pool_cls, tmp_path):
        configs = [
            GreetConfig(job_directory=tmp_path / name, name=name, code=code) for name, code in [('a', 0), ('b', 3)]
        ]
        pool = pool_cls(max_jobs=2, logger_config=logging_to(tmp_path / 'pool.log'))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        async with pool:
            pool.submit(Greet, configs=configs, job_ids=['a', 'b'])
            succeeded, failed = await pool.launch()

        assert succeeded.success, succeeded.error
        assert succeeded.data.stdout.split() == ['hello', 'a']
        assert succeeded.data.stderr.split() == ['to', 'stderr']
        assert succeeded.data.return_code == 0
        assert not (tmp_path / 'a').exists()  # tidied up

        assert isinstance(failed.error, SubprocessError)
        assert 'failed with return code 3' in str(failed.error)
        failed_dir = tmp_path / 'b'
        assert (failed_dir / 'greeting.txt').read_text().split() == ['hello', 'b']
        assert (failed_dir / 'batch_job.stderr.log').read_text().split() == ['to', 'stderr']
        assert 'failed with return code 3' in (failed_dir / 'xtl.log').read_text(encoding='utf-8')

        assert pool.status == {'a': JobStatus.DONE, 'b': JobStatus.FAILED}
        assert 'ERROR [b] Job failed with subprocess error' in (tmp_path / 'pool.log').read_text(encoding='utf-8')


class TestSteppedJobs:
    @pool_types
    @pytest.mark.asyncio
    async def test_a_step_opens_a_pool_of_its_own(self, pool_cls, tmp_path):
        pool = pool_cls(
            max_jobs=2,
            resources=Resources(jobs=2, threads=2, processes=2),
            logger_config=logging_to(tmp_path / 'pool.log'),
        )
        async with pool:
            pool.submit(Pipeline, configs=[SteppedJobConfig(steps={'fan': {'xs': [1, 2, 3, 4]}})], job_ids=['p'])
            (result,) = await pool.launch()

        assert result.success, result.error
        assert result.steps['fan'].data['squares'] == [1, 4, 9, 16]
        assert result.steps['sum'].data == 30
        assert result.data['squares'] == [1, 4, 9, 16]
        # The nested pool is capped by what the job was granted
        assert result.data['width'] <= 2

        log = (tmp_path / 'pool.log').read_text(encoding='utf-8')
        # The pool sees the steps, and the nested pool's jobs under the step that opened it
        assert 'INFO [p] Executing step 1/2: fan' in log
        assert 'INFO [p.1>Square|1] squared 1' in log
        assert 'INFO [p.2] adding 4 numbers' in log
        assert 'squaring' not in log  # DEBUG doesn't reach an INFO pool

    @pool_types
    @pytest.mark.asyncio
    async def test_a_failing_step_in_a_pool(self, pool_cls):
        async with pool_cls(max_jobs=1) as pool:
            # The second step gets strings to add up: its configuration can't be built
            pool.submit(Pipeline, configs=[SteppedJobConfig(steps={'sum': {'values': ['x']}})])
            (result,) = await pool.launch()
        assert not result.success
        assert result.steps['fan'].data['squares'] == [1, 4, 9]
        assert 'sum' not in result.steps


class TestWorkerLogs:
    @pytest.mark.asyncio
    async def test_records_of_worker_processes_go_where_they_belong(self, tmp_path):
        pool = MultiprocessPool(max_jobs=2, logger_config=logging_to(tmp_path / 'pool.log'))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        async with pool:
            pool.submit(KeptPipeline, configs=[SteppedJobConfig(job_directory=tmp_path / 'job')], job_ids=['p'])
            (result,) = await pool.launch()
        assert result.success, result.error

        # The job's file: every level, from the job, its steps and the nested pool's jobs, all from the worker
        job_log = (tmp_path / 'job' / 'xtl.log').read_text(encoding='utf-8')
        for line in ('DEBUG', 'squaring 3', 'squared 3', 'adding 3 numbers', 'All steps completed'):
            assert line in job_log
        # The pool's file: INFO and above only, relayed to the parent
        pool_log = (tmp_path / 'pool.log').read_text(encoding='utf-8')
        assert 'INFO [p.1>Square|3] squared 3' in pool_log
        assert 'DEBUG' not in pool_log


class TestTimeouts:
    @pool_types
    @pytest.mark.asyncio
    async def test_timeout_in_the_middle_of_a_launch(self, pool_cls, tmp_path):
        available = await get_rc_manager().available()
        configs = [GreetConfig(job_directory=tmp_path / str(i), name=str(i), seconds=2) for i in range(3)]
        pool = pool_cls(max_jobs=2)

        async def run():
            async with pool:
                pool.submit(Greet, configs=configs, job_ids=['0', '1', '2'])
                await pool.launch()

        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(run(), timeout=1)
        elapsed = time.monotonic() - start

        # Jobs that ran or waited are cancelled, the resources are back and the pool's loggers are gone
        assert set(pool.status.values()) == {JobStatus.CANCELLED}
        assert await get_rc_manager().available() == available
        assert not [name for name in logging.root.manager.loggerDict if name.startswith(pool.logger.name)]
        if pool_cls in WORKER_POOLS:
            # A job in a worker can't be stopped: the exit waits for the scripts that run (2 s, a process pool also
            #  starts its workers)
            assert elapsed >= 1.5
        else:
            # In the event loop, the scripts are stopped
            assert elapsed < 2


# A command line tool, written like xtl's: options from the hooks, a live pool from the console

app = typer.Typer()


@app.command()
@attach_hook(func=get_console_options, hook_output='console_options')
@attach_hook(func=get_job_options, hook_output='job_options')
@typer_async
async def greet(
    names: list[str] = typer.Argument(..., help='Who to greet; `fail` makes a job fail'),
    root: Path = typer.Option(..., '--root', help='Where the jobs work'),
    job_options: JobOptions = typer.Option(),
    console_options: ConsoleOptions = typer.Option(),
):
    console = ConsoleIO(verbose=console_options.verbose, debug=console_options.debug)
    console.apply_job_options(job_options)
    configs = [GreetConfig(job_directory=root / name, name=name, code=3 if name == 'fail' else 0) for name in names]
    async with console.get_pool('async', max_jobs=2) as pool:
        pool.submit(Greet, configs=configs, job_ids=names)
        results = await pool.launch()
    for result in results:
        if result.success:
            console.print(f'{result.job_id}: {result.data.stdout.strip()}')
    if not all(r.success for r in results):
        raise typer.Exit(code=1)


def run(args: list[str]):
    return CliRunner().invoke(app, args, env={'COLUMNS': '200'})


def text(result) -> str:
    """The output with its whitespace collapsed, so that where rich wraps a line (long paths) doesn't matter."""
    return ' '.join(result.output.split())


@pytest.fixture
def restore_settings():
    """Put back the settings that the command changes."""
    before = settings.to_dict()
    yield
    settings.update_from(XTLSettings.from_dict(before))


class TestCommandLine:
    def test_jobs_run_through_the_console(self, tmp_path, restore_settings):
        result = run(['a', 'b', '--root', str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert 'a: hello a' in text(result)
        assert 'b: hello b' in text(result)
        assert list(tmp_path.iterdir()) == []

    def test_a_failing_job_shows_the_logs(self, tmp_path, restore_settings):
        result = run(['a', 'fail', '--root', str(tmp_path)])
        assert result.exit_code == 1
        assert 'a: hello a' in text(result)
        assert 'An error occurred while executing jobs' in text(result)
        assert 'failed with return code 3' in text(result)
        assert [p.name for p in tmp_path.iterdir()] == ['fail']

    def test_job_options_apply(self, tmp_path, restore_settings):
        result = run(['a', '--root', str(tmp_path), '--keep-temp'])
        assert result.exit_code == 0, result.output
        # Kept, with its log file
        assert (tmp_path / 'a' / 'greeting.txt').read_text().split() == ['hello', 'a']
        assert (tmp_path / 'a' / 'xtl.log').exists()
