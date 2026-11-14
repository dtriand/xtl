"""
Tests for the pool with a live display: its progress bar, its log panel, and how it wraps a pool. How it handles
`logging` is in ``test_logging.py``.
"""

import io
import logging

import pytest
import rich.progress
from typer import BadParameter

from tests.jobs.helpers import Failing, Naps, Talker
from xtl import settings
from xtl.jobs.pools import AsyncPool, MultiprocessPool, SimplePool
from xtl.jobs.results import JobStatus
from xtl.tui.console import ConsoleIO
from xtl.tui.live_pool import JobOverviewColumn, LivePool, LogPanel, PoolProgress
from xtl.tui.logging.handlers import BufferingHandler


def console(**kwargs) -> ConsoleIO:
    return ConsoleIO(file=io.StringIO(), width=100, **kwargs)


def live(pool_type: str = 'async', **kwargs) -> LivePool:
    return LivePool(console=kwargs.pop('console', None) or console(), pool_type=pool_type, max_jobs=2, **kwargs)


def output(c: ConsoleIO) -> str:
    return c.file.getvalue()


class TestJobOverviewColumn:
    @staticmethod
    def render(**fields) -> str:
        total = fields.pop('total', 4)
        completed = fields.pop('completed', 0)
        task = rich.progress.Task(
            id=0, description='', total=total, completed=completed, _get_time=lambda: 0.0, fields=fields
        )
        return JobOverviewColumn().render(task).plain

    def test_counts(self):
        assert self.render(completed=3, success=2, failed=1) == '2/1/4'
        assert self.render(completed=2, success=2, failed=0) == '2/4'
        assert self.render(completed=1, success=0, failed=1, running=2) == '(2 running) 1/4'

    def test_counts_that_do_not_add_up(self):
        assert self.render(completed=3, success=1, failed=0) == '3/4'
        assert self.render(completed=0) == '0/4'
        assert self.render(completed=0, total=None) == '0/?'


class TestPoolProgress:
    @pytest.mark.asyncio
    async def test_sync(self):
        pool = AsyncPool(max_jobs=2)
        progress = PoolProgress(console=console(), pool=pool)
        progress.sync()
        assert progress._job_task._task.total is None
        async with pool:
            pool.submit(Naps, job_ids=['a'])
            pool.submit(Failing, job_ids=['b'])
            await pool.launch()
        progress.sync()
        task = progress._job_task._task
        assert (task.total, task.completed) == (2, 2)
        assert (task.fields['success'], task.fields['failed'], task.fields['running']) == (1, 1, 0)

    def test_cancelled_jobs_count_as_failed(self):
        pool = AsyncPool()
        pool._set_status('a', JobStatus.CANCELLED)
        progress = PoolProgress(console=console(), pool=pool)
        progress.sync()
        assert progress._job_task._task.fields['failed'] == 1

    def test_description(self):
        progress = PoolProgress(console=console(), pool=AsyncPool())
        assert progress.description == 'Waiting for jobs...'
        progress.start()
        assert progress.description == 'Running jobs'
        progress.description = 'Almost there'
        assert progress.description == 'Almost there'


class TestLogPanel:
    def test_waiting_for_logs(self):
        c = console()
        c.print(LogPanel(handler=BufferingHandler(), console=c, title='Job logs'))
        assert 'Waiting for logs...' in output(c)

    def test_shows_the_latest_records(self):
        c, handler = console(), BufferingHandler(tail_size=2)
        for i in range(3):
            handler.handle(logging.makeLogRecord({'msg': f'record {i}', 'job': 'Talker|1', 'levelno': 20}))
        c.print(LogPanel(handler=handler, console=c, log_fmt='[%(job)s] %(message)s'))
        assert 'record 0' not in output(c)
        assert 'record 1' in output(c) and 'record 2' in output(c)

    def test_replay(self, monkeypatch):
        c, handler = console(), BufferingHandler()
        monkeypatch.setattr(handler, '_records', type(handler._records)(maxlen=2))
        for i in range(3):
            handler.handle(logging.makeLogRecord({'msg': f'record {i}', 'job': 'Talker|1', 'levelno': 20}))
        LogPanel(handler=handler, console=c, log_fmt='[%(job)s] %(message)s').replay()
        assert 'Displaying the latest 2 records' in output(c)
        assert 'record 0' not in output(c)
        assert 'record 2' in output(c)


class TestLivePool:
    def test_pool(self, monkeypatch):
        assert type(LivePool(console=console())._pool) is SimplePool
        assert type(live('processes')._pool) is MultiprocessPool
        assert live()._pool._logger_config.level == logging.INFO
        assert live(console=console(debug=True))._pool._logger_config.level == logging.DEBUG
        assert live(progress=False)._progress is None

    @pytest.mark.parametrize(
        'verbose, threading_debug, fmt',
        [
            (0, False, '[%(job)s] %(message)s'),
            (2, False, '[%(name)s] %(message)s'),
            (0, True, '[%(name)s<%(processName)s:%(threadName)s>] %(message)s'),
        ],
    )
    def test_log_format(self, monkeypatch, verbose, threading_debug, fmt):
        monkeypatch.setattr(settings.jobs, 'threading_debug', threading_debug)
        assert live(console=console(verbose=verbose))._log_panel._renderer._formatter._fmt == fmt

    @pytest.mark.asyncio
    async def test_results_like_the_pool(self):
        async with live() as pool:
            pool.submit(Naps, configs=[None, None], job_ids=['slow', 'fast'])
            first = await pool.launch()
            pool.submit(Naps, job_ids=['c'])
            second = await pool.launch()
            streamed = [r async for r in await pool.launch('stream')]
        # Submission order, and only the jobs of each launch
        assert [r.data for r in first] == ['slow', 'fast']
        assert [r.data for r in second] == ['c']
        assert streamed == []

    @pytest.mark.asyncio
    async def test_launch_mode(self):
        async with live() as pool:
            with pytest.raises(BadParameter, match="`mode` must be one of: 'all', 'stream'"):
                await pool.launch('some')

    @pytest.mark.asyncio
    async def test_progress_follows_the_status(self):
        async with live() as pool:
            pool.submit(Naps, job_ids=['a'])
            pool.submit(Failing, job_ids=['b'])
            await pool.launch()
            pool._progress.sync()
            task = pool._progress._job_task._task
            assert (task.total, task.completed) == (2, 2)
            assert (task.fields['success'], task.fields['failed'], task.fields['running']) == (1, 1, 0)

    @pytest.mark.asyncio
    async def test_primitives_come_from_the_pool(self):
        async with live() as pool:
            assert pool.get_lock() is pool._pool.get_lock()
            assert pool.get_queue('q') is pool._pool.get_queue('q')
            assert pool.get_state('s') is pool._pool.get_state('s')

    @pytest.mark.asyncio
    async def test_resources_in_the_subtitle(self):
        async with live(console=console(verbose=1)) as pool:
            assert pool._log_panel._subtitle.startswith('J:2|T:')
        async with live() as pool:
            assert pool._log_panel._subtitle is None

    @pytest.mark.asyncio
    async def test_failed_jobs_replay_the_logs(self):
        c = console()
        async with live(console=c) as pool:
            pool.submit(Failing, job_ids=['b'])
            await pool.launch()
        assert 'An error occurred while executing jobs' in output(c)
        assert '--- Job logs ---' in output(c)
        assert 'Job failed due to an exception: boom' in output(c)

    @pytest.mark.asyncio
    async def test_no_replay_without_records(self):
        c = console()
        with pytest.raises(ValueError):
            async with live(console=c):
                raise ValueError
        assert 'An error occurred while executing jobs' in output(c)
        assert '--- Job logs ---' not in output(c)

    @pytest.mark.asyncio
    async def test_interrupt(self):
        c = console()
        with pytest.raises(KeyboardInterrupt):
            async with live(console=c):
                raise KeyboardInterrupt
        assert 'User cancelled the job execution.' in output(c)

    @pytest.mark.asyncio
    async def test_pool_that_cannot_start(self):
        c = console()
        pool = live(console=c)
        root_handlers = list(logging.root.handlers)
        themes = len(c._theme_stack._entries)
        async with pool:
            pool.submit(Talker)
            await pool.launch()
        with pytest.raises(RuntimeError, match='single-use'):
            async with pool:
                pass
        # The display, the buffer and the theme are gone
        assert not pool._live.is_started
        assert logging.root.handlers == root_handlers
        assert len(c._theme_stack._entries) == themes
