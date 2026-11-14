"""
Tests for the logging of jobs and pools: first the classes and functions of ``xtl.jobs.logging``, then how jobs, pools
and the TUI use them.

Records are captured with :class:`Capture`, an in-process handler, instead of caplog,
because job and pool loggers don't propagate to the root logger. Jobs are defined at
module level, so that worker processes can import them. ``fresh_logging`` (in
``conftest.py``) undoes what each test leaves in `logging`.
"""

import ast
import asyncio
from collections import defaultdict
import io
import json
import logging
import logging.handlers
from pathlib import Path
import queue
import subprocess
import sys

import pytest
import rich.traceback

import xtl
import xtl_core
from xtl.jobs import logging as xtl_logging
from xtl.jobs.batchfiles import BatchFile
from xtl.jobs.config import BatchJobConfig, JobConfig, SteppedJobConfig
from xtl.jobs.jobs import BatchJob, Job, SteppedJob
from xtl.jobs.pools import BasePool, MultiprocessPool, SimplePool
from tests.jobs.helpers import POOLS, WORKER_POOLS, Failing, Talker
from xtl.jobs.sites import LocalSite
from xtl.jobs.steps import StepSpec
from xtl.jobs.submissions import JobSubmission
from xtl.jobs.logging import (
    FileHandlerConfig,
    HandlerConfig,
    JobFileHandlerConfig,
    JobNameFilter,
    LoggerConfig,
    LoggingFormat,
    MissingPlaceholderError,
    RelayHandler,
    StreamHandlerConfig,
    get_logger_config,
    pass_through,
    pool_label,
    register_pool_class,
)
from xtl.tui.console import ConsoleIO
from xtl.tui.live_pool import LivePool
from xtl.tui.renderers import LogRenderer
from xtl_core.options import Option, OptionsValidationError

RECORDS: dict[str, list[logging.LogRecord]] = defaultdict(list)
"""Records captured by :class:`Capture`, per key. Only filled in the process that emits them."""


class Capture(logging.Handler):
    """Collects records in :data:`RECORDS`; a message that can't be formatted is stored as an error."""

    def __init__(self, key: str = 'default', level: int = logging.NOTSET):
        super().__init__(level)
        self.key = key

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record.captured = record.getMessage()
        except Exception as e:
            record.captured = f'<format error: {type(e).__name__}: {e}>'
        RECORDS[self.key].append(record)


def capture_config(key: str, level: int = logging.DEBUG) -> LoggerConfig:
    return LoggerConfig(level=level, propagate=False, handlers=[HandlerConfig(handler=Capture, options={'key': key})])


def messages(key: str) -> list[str]:
    return [r.captured for r in RECORDS[key]]


def names(key: str) -> set[str]:
    return {r.name for r in RECORDS[key]}


class TaggedHandler(logging.Handler):
    """A user-defined handler class, importable from this module."""


class TaggedHandlerConfig(HandlerConfig):
    """A user-defined handler config, importable from this module."""

    handler: type(logging.Handler) = Option(default=TaggedHandler, desc='Handler class')
    tag: str = Option(default='', desc='A field of its own')


class TaggedFileHandler(logging.FileHandler):
    """A user-defined handler class, importable by worker processes."""


class Stepped(SteppedJob[SteppedJobConfig]):
    _steps = (
        StepSpec(name='one', job_cls=Talker, config_cls=JobConfig),
        StepSpec(name='two', job_cls=Talker, config_cls=JobConfig),
    )


class Nesting(Job[JobConfig]):
    async def _execute(self):
        self.logger.info('outer from %s', self.job_id)
        async with self.get_pool('threads', max_jobs=1) as pool:
            pool.submit(Talker, configs=[JobConfig()])
            results = await pool.launch()
        return all(r.success for r in results)


class KeptTalker(Talker):
    _keep_files = True


class KeptStepped(Stepped):
    _keep_files = True


class SteppedRaising(SteppedJob[SteppedJobConfig]):
    async def _execute(self):
        raise ValueError('stepped boom')


@pytest.fixture(autouse=True)
def fresh_records():
    RECORDS.clear()
    yield
    RECORDS.clear()


async def run_pool(pool: BasePool, job_cls: type[Job], job_ids: list[str], config: JobConfig | None = None):
    async with pool:
        jobs = pool.submit(job_cls, configs=[config for _ in job_ids], job_ids=job_ids)
        results = await pool.launch()
    return jobs, results


def make_record(name: str, level: int = logging.INFO, msg: str = 'message', args: tuple = (), **kwargs):
    return logging.getLogger(name).makeRecord(name, level, __file__, 1, msg, args, None, **kwargs)


# The classes and functions of xtl.jobs.logging


class TestLoggingFormat:
    def test_defaults(self):
        fmt = LoggingFormat()
        assert fmt.format == '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s'
        assert fmt.datefmt == '%H:%M:%S'

    def test_get_formatter(self):
        formatter = LoggingFormat(format='%(levelname)s: %(message)s', datefmt='%H:%M').get_formatter()
        assert isinstance(formatter, logging.Formatter)
        assert formatter._fmt == '%(levelname)s: %(message)s'
        assert formatter.datefmt == '%H:%M'


class TestHandlerConfig:
    def test_handler_from_its_import_path(self):
        assert HandlerConfig(handler='logging.NullHandler').handler is logging.NullHandler

    def test_get_handler(self):
        handler = HandlerConfig(
            handler=Capture, options={'key': 'k'}, format=LoggingFormat(format='%(message)s')
        ).get_handler()
        assert isinstance(handler, Capture)
        assert handler.key == 'k'
        assert handler.formatter._fmt == '%(message)s'
        assert handler.level == logging.NOTSET

    def test_get_handler_level(self):
        assert HandlerConfig(handler=logging.NullHandler, level=logging.WARNING).get_handler().level == logging.WARNING

    def test_get_handler_with_a_placeholder_left(self):
        with pytest.raises(MissingPlaceholderError, match='job_id'):
            HandlerConfig(handler=Capture, options={'key': '{job_id}'}).get_handler()

    def test_to_spec(self):
        spec = TaggedHandlerConfig(tag='mine').to_spec()
        assert spec['config_cls'] == 'tests.jobs.test_logging.TaggedHandlerConfig'
        assert spec['handler'] is TaggedHandler
        assert TaggedHandlerConfig().to_spec(mode='json')['handler'] == 'tests.jobs.test_logging.TaggedHandler'

    def test_from_spec(self):
        restored = HandlerConfig.from_spec(TaggedHandlerConfig(tag='mine').to_spec(mode='json'))
        assert isinstance(restored, TaggedHandlerConfig)
        assert (restored.tag, restored.handler) == ('mine', TaggedHandler)

    def test_from_spec_without_config_class(self):
        restored = HandlerConfig.from_spec({'handler': 'logging.NullHandler'})
        assert type(restored) is HandlerConfig
        assert restored.handler is logging.NullHandler

    def test_from_spec_leaves_the_spec_alone(self):
        spec = {'config_cls': 'xtl.jobs.logging.HandlerConfig'}
        HandlerConfig.from_spec(spec)
        assert spec == {'config_cls': 'xtl.jobs.logging.HandlerConfig'}

    def test_from_spec_with_a_class_that_is_not_a_config(self):
        with pytest.raises(TypeError, match='not a HandlerConfig subclass'):
            HandlerConfig.from_spec({'config_cls': 'logging.NullHandler'})

    def test_fill(self):
        spec = HandlerConfig(
            handler=logging.NullHandler,
            format=LoggingFormat(format='{job_id} %(message)s'),
            options={'tag': '{pool}:{job_id}', 'other': '{unknown}', 'count': 3},
        )
        filled = spec.fill({'job_directory': None, 'job_id': 'T_1', 'job_class': 'T', 'pool': 'SimplePool'})
        assert filled.options == {'tag': 'SimplePool:T_1', 'other': '{unknown}', 'count': 3}
        # The format is left alone, and so is the original
        assert filled.format.format == '{job_id} %(message)s'
        assert spec.options['tag'] == '{pool}:{job_id}'

    def test_fill_string_fields(self):
        filled = TaggedHandlerConfig(tag='{job_class}').fill({'job_class': 'Talker'})
        assert isinstance(filled, TaggedHandlerConfig)
        assert filled.tag == 'Talker'

    def test_fill_without_a_value(self):
        with pytest.raises(MissingPlaceholderError, match='job_directory'):
            HandlerConfig(options={'filename': '{job_directory}/x.log'}).fill({'job_directory': None})


class TestStreamHandlerConfig:
    def test_get_handler(self):
        stream = io.StringIO()
        handler = StreamHandlerConfig(stream=stream, format=LoggingFormat(format='%(message)s')).get_handler()
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is stream
        handler.handle(make_record('stream'))
        assert stream.getvalue() == 'message\n'

    def test_named_streams_are_resolved_when_the_handler_is_made(self, monkeypatch):
        stream = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', stream)
        assert StreamHandlerConfig().get_handler().stream is stream
        assert StreamHandlerConfig(stream='stderr').get_handler().stream is sys.stderr

    def test_unknown_stream_name(self):
        with pytest.raises(ValueError, match="Unknown stream 'stdin'"):
            StreamHandlerConfig(stream='stdin').get_handler()

    def test_get_handler_level(self):
        assert StreamHandlerConfig(level=logging.ERROR).get_handler().level == logging.ERROR

    def test_get_handler_options(self):
        class Tagged(logging.StreamHandler):
            def __init__(self, stream=None, tag=''):
                super().__init__(stream)
                self.tag = tag

        assert StreamHandlerConfig(handler=Tagged, options={'tag': 'mine'}).get_handler().tag == 'mine'

    def test_json_names_the_stream(self):
        assert json.loads(StreamHandlerConfig(stream='stderr').to_json())['stream'] == 'stderr'
        assert json.loads(StreamHandlerConfig(stream=sys.stdout).to_json())['stream'] == 'stdout'
        assert json.loads(StreamHandlerConfig(stream=sys.stderr).to_json())['stream'] == 'stderr'

    def test_stream_object_cannot_go_to_json(self):
        with pytest.raises(Exception, match="stream object can't be serialized"):
            StreamHandlerConfig(stream=io.StringIO()).to_json()


class TestFileHandlerConfig:
    def test_defaults(self):
        config = FileHandlerConfig()
        assert (config.handler, config.filename, config.mode, config.encoding) == (
            logging.FileHandler,
            'job.log',
            'a',
            'utf-8',
        )

    def test_get_handler(self, tmp_path):
        log = tmp_path / 'test.log'
        log.write_text('before\n', encoding='utf-8')
        handler = FileHandlerConfig(
            filename=str(log), level=logging.INFO, format=LoggingFormat(format='%(message)s')
        ).get_handler()
        assert isinstance(handler, logging.FileHandler)
        assert (handler.baseFilename, handler.mode, handler.encoding) == (str(log), 'a', 'utf-8')
        assert handler.level == logging.INFO
        handler.handle(make_record('file', msg='ünïcode'))
        handler.close()
        assert log.read_text(encoding='utf-8') == 'before\nünïcode\n'

    def test_get_handler_options(self, tmp_path):
        config = FileHandlerConfig(
            filename=str(tmp_path / 'test.log'),
            handler=logging.handlers.RotatingFileHandler,
            options={'maxBytes': 100, 'backupCount': 2},
        )
        handler = config.get_handler()
        handler.close()
        assert (handler.maxBytes, handler.backupCount) == (100, 2)


class TestJobFileHandlerConfig:
    def test_defaults(self):
        config = JobFileHandlerConfig()
        assert (config.filename, config.mode, config.level) == ('{job_directory}/xtl.log', 'w', logging.DEBUG)
        assert config.format.datefmt == '%Y-%m-%d %H:%M:%S'

    def test_get_handler_creates_the_directory(self, tmp_path):
        handler = JobFileHandlerConfig().fill({'job_directory': str(tmp_path / 'new')}).get_handler()
        handler.close()
        assert Path(handler.baseFilename) == tmp_path / 'new' / 'xtl.log'

    def test_get_handler_without_a_job(self, tmp_path, monkeypatch):
        # Outside a job, e.g. among a pool's handlers, there's no directory to create
        monkeypatch.chdir(tmp_path)
        with pytest.raises(MissingPlaceholderError, match='job_directory'):
            JobFileHandlerConfig().get_handler()
        assert list(tmp_path.iterdir()) == []


class TestLoggerConfig:
    def test_defaults(self):
        config = LoggerConfig()
        assert config.level == logging.INFO
        assert config.propagate is False
        assert len(config.handlers) == 1
        assert isinstance(config.handlers[0], StreamHandlerConfig)
        assert LoggerConfig().handlers[0] is not config.handlers[0]

    def test_handlers_from_specs(self):
        specs = (FileHandlerConfig(filename='a.log').to_spec(), {'handler': 'logging.NullHandler'})
        config = LoggerConfig(handlers=specs)
        assert [type(h) for h in config.handlers] == [FileHandlerConfig, HandlerConfig]
        with pytest.raises(OptionsValidationError):
            LoggerConfig(handlers=FileHandlerConfig())

    def test_dict_round_trip(self):
        config = LoggerConfig(handlers=[FileHandlerConfig(filename='job.log', mode='w')])
        restored = LoggerConfig.from_dict(config.to_dict()).handlers[0]
        assert isinstance(restored, FileHandlerConfig)
        assert (restored.filename, restored.mode) == ('job.log', 'w')

    def test_json_round_trip(self):
        config = LoggerConfig(handlers=[FileHandlerConfig(filename='job.log', mode='w'), JobFileHandlerConfig()])
        restored = LoggerConfig.from_json(config.to_json()).handlers
        assert [type(h) for h in restored] == [FileHandlerConfig, JobFileHandlerConfig]
        assert (restored[0].filename, restored[0].mode) == ('job.log', 'w')

    def test_json_names_classes_by_import_path(self):
        spec = json.loads(LoggerConfig(handlers=[FileHandlerConfig()]).to_json())['handlers'][0]
        assert spec['handler'] == 'logging.FileHandler'
        assert spec['config_cls'] == 'xtl.jobs.logging.FileHandlerConfig'

    def test_user_defined_config_survives_json(self):
        config = LoggerConfig(handlers=[TaggedHandlerConfig(tag='mine', options={'level': logging.WARNING})])
        restored = LoggerConfig.from_json(config.to_json()).handlers[0]
        assert isinstance(restored, TaggedHandlerConfig)
        assert restored.tag == 'mine'
        assert restored.handler is TaggedHandler
        assert restored.get_handler().level == logging.WARNING

    def test_configure(self):
        stream = io.StringIO()
        config = LoggerConfig(
            level=logging.DEBUG,
            handlers=[StreamHandlerConfig(stream=stream, format=LoggingFormat(format='%(levelname)s: %(message)s'))],
        )
        logger = logging.getLogger('configure')
        logger.propagate = True
        config.configure(logger)
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        [handler] = logger.handlers
        assert handler._xtl_owned
        logger.debug('Debug message')
        assert stream.getvalue() == 'DEBUG: Debug message\n'

    def test_configure_several_handlers(self, tmp_path):
        log, stream = tmp_path / 'configure.log', io.StringIO()
        config = LoggerConfig(
            level=logging.DEBUG,
            handlers=[
                StreamHandlerConfig(stream=stream, format=LoggingFormat(format='STREAM: %(levelname)s - %(message)s')),
                FileHandlerConfig(filename=str(log), format=LoggingFormat(format='FILE: %(levelname)s - %(message)s')),
            ],
        )
        logger = logging.getLogger('configure_several')
        config.configure(logger)
        logger.debug('Debug message')
        logger.warning('Warning message')
        for handler in logger.handlers:
            handler.close()
        assert stream.getvalue() == 'STREAM: DEBUG - Debug message\nSTREAM: WARNING - Warning message\n'
        assert log.read_text(encoding='utf-8') == 'FILE: DEBUG - Debug message\nFILE: WARNING - Warning message\n'

    def test_configure_again_replaces_its_own_handlers(self, tmp_path):
        logger = logging.getLogger('configure_again')
        foreign = Capture('foreign')
        logger.addHandler(foreign)
        LoggerConfig(handlers=[FileHandlerConfig(filename=str(tmp_path / 'first.log'))]).configure(logger)
        [first] = [h for h in logger.handlers if h is not foreign]
        LoggerConfig(handlers=[HandlerConfig(handler=Capture)]).configure(logger)
        # The first config's handler is closed and gone, the foreign one stays
        assert first.stream is None
        assert first not in logger.handlers
        assert foreign in logger.handlers
        assert len(logger.handlers) == 2

    def test_configure_without_handlers(self):
        logger = logging.getLogger('configure_none')
        LoggerConfig(level=logging.WARNING, handlers=[]).configure(logger)
        [handler] = logger.handlers
        assert isinstance(handler, logging.NullHandler)
        assert logger.level == logging.WARNING

    def test_handler_levels(self):
        config = LoggerConfig(
            level=logging.INFO,
            handlers=[HandlerConfig(handler=logging.NullHandler), HandlerConfig(handler=logging.NullHandler, level=10)],
        )
        logger = logging.getLogger('handler_levels')
        config.configure(logger)
        # A handler without a level gets the config's; the logger lets through what the most verbose one wants
        assert [h.level for h in logger.handlers] == [logging.INFO, logging.DEBUG]
        assert logger.level == logging.DEBUG

    def test_name_filter(self):
        parent = logging.getLogger('named')
        LoggerConfig(handlers=[HandlerConfig(handler=Capture, options={'key': 'named'})]).configure(parent)
        child = parent.getChild('2')
        child.propagate = True
        parent.info('parent')
        child.info('child')
        assert [r.job for r in RECORDS['named']] == ['named', 'named.2']

        LoggerConfig(handlers=[HandlerConfig(handler=Capture, options={'key': 'label'})]).configure(parent, 'Label')
        child.info('child')
        parent.info('parent')
        assert [r.job for r in RECORDS['label']] == ['2', 'Label']

    def test_live_handlers(self):
        live = logging.NullHandler()
        config = LoggerConfig(handlers=[live, StreamHandlerConfig(stream=io.StringIO())])
        logger = logging.getLogger('live_handlers')
        config.configure(logger, label='Live')
        assert live in logger.handlers
        assert len(logger.handlers) == 2
        assert len(live.filters) == 1

        # Reconfiguring detaches the live handler without closing it, and doesn't add a second filter
        LoggerConfig(handlers=[]).configure(logger)
        assert live not in logger.handlers
        config.configure(logger, label='Live')
        assert len(live.filters) == 1

        assert config.to_dict()['handlers'][0] is live
        with pytest.raises(Exception, match="live handler can't be serialized"):
            config.to_json()


class TestGetLoggerConfig:
    def test_defaults(self):
        config = get_logger_config()
        assert (config.level, config.propagate) == (logging.INFO, False)
        [handler] = config.handlers
        assert isinstance(handler, StreamHandlerConfig)
        assert handler.format.format == '[%(asctime)s.%(msecs)03d:%(job)s] %(message)s'
        assert get_logger_config(level=logging.DEBUG).level == logging.DEBUG

    def test_threading_debug(self, monkeypatch):
        monkeypatch.setattr(xtl.settings.jobs, 'threading_debug', True)
        assert '%(threadName)s' in get_logger_config().handlers[0].format.format


class TestPassThrough:
    def test_pass_through(self, tmp_path):
        logger = logging.getLogger('pass_through')
        foreign = Capture('foreign')
        logger.addHandler(foreign)
        LoggerConfig(level=logging.WARNING, handlers=[FileHandlerConfig(filename=str(tmp_path / 'x.log'))]).configure(
            logger
        )
        [owned] = [h for h in logger.handlers if h is not foreign]
        pass_through(logger)
        assert logger.handlers == [foreign]
        assert owned.stream is None
        assert logger.level == logging.NOTSET
        assert logger.propagate is True


class TestPoolLabel:
    @pytest.mark.parametrize(
        'segment, label',
        [
            ('SimplePool#1', 'SimplePool'),
            ('SimplePool|refine#1', 'SimplePool|refine'),
            ('SimplePool', None),
            ('Talker|1', None),
            ('job#1', None),
            ('job|SimplePool#1', None),
        ],
    )
    def test_pool_label(self, segment, label):
        assert pool_label(segment) == label

    def test_register_pool_class(self, monkeypatch):
        monkeypatch.setattr(xtl_logging, '_POOL_CLASSES', set(xtl_logging._POOL_CLASSES))
        assert pool_label('Custom#1') is None
        register_pool_class('Custom')
        assert pool_label('Custom|x#1') == 'Custom|x'

    def test_pool_classes_register_themselves(self, monkeypatch):
        monkeypatch.setattr(xtl_logging, '_POOL_CLASSES', set(xtl_logging._POOL_CLASSES))

        class CustomPool(SimplePool):
            pass

        assert pool_label('CustomPool#1') == 'CustomPool'


class TestJobNameFilter:
    @pytest.mark.parametrize(
        'name, shown',
        [
            ('MultiprocessPool#123', 'MultiprocessPool'),
            ('MultiprocessPool#123.Talker|1', 'Talker|1'),
            ('MultiprocessPool#123.Talker|1.2', 'Talker|1.2'),
            ('MultiprocessPool#123.my.job', 'my.job'),
            ('MultiprocessPool#123.Outer|01.ThreadedPool#9.Talker|03', 'Outer|01>Talker|03'),
            ('MultiprocessPool#123.Outer|01.ThreadedPool#9', 'Outer|01>ThreadedPool'),
            ('MultiprocessPool#123.Outer|01.ThreadedPool|refine#9', 'Outer|01>ThreadedPool|refine'),
            ('MultiprocessPool#123.Outer|01.ThreadedPool|refine#9.Talker|03', 'Outer|01>Talker|03'),
            ('MultiprocessPool#1234.Talker|1', 'MultiprocessPool#1234.Talker|1'),
            ('xtl.jobs.sites', 'xtl.jobs.sites'),
        ],
    )
    def test_display_name(self, name, shown):
        name_filter = JobNameFilter(base='MultiprocessPool#123', label='MultiprocessPool')
        assert name_filter.display_name(name) == shown

    def test_label_defaults_to_the_base(self):
        assert JobNameFilter(base='base').display_name('base') == 'base'

    def test_relative_to_the_root(self):
        name_filter = JobNameFilter(base='')
        assert name_filter.display_name('job.SimplePool#7.Talker|1') == 'job>Talker|1'

    def test_for_a_standalone_job(self):
        name_filter = JobNameFilter.for_logger(logging.getLogger('standalone'))
        assert name_filter.base == ''
        assert name_filter.display_name('standalone') == 'standalone'
        assert name_filter.display_name('standalone.2') == 'standalone.2'
        assert name_filter.display_name('standalone.SimplePool#7.Talker|1') == 'standalone>Talker|1'

    def test_for_a_job_in_a_pool(self):
        logging.getLogger('SimplePool#7')
        name_filter = JobNameFilter.for_logger(logging.getLogger('SimplePool#7.Talker|1'))
        assert name_filter.base == 'SimplePool#7'
        assert name_filter.display_name('SimplePool#7.Talker|1.2') == 'Talker|1.2'

    def test_for_a_pool(self):
        name_filter = JobNameFilter.for_logger(logging.getLogger('SimplePool#7'), label='SimplePool')
        assert (name_filter.base, name_filter.label) == ('SimplePool#7', 'SimplePool')

    def test_filter(self):
        record = make_record('SimplePool#7.Talker|1')
        assert JobNameFilter(base='SimplePool#7', label='SimplePool').filter(record) is True
        assert record.job == 'Talker|1'


class TestRelayHandler:
    def test_record_to_dict(self):
        try:
            raise ValueError('boom')
        except ValueError:
            record = logging.getLogger('relay').makeRecord(
                'relay', logging.ERROR, __file__, 1, 'failed %s', ('here',), sys.exc_info(), extra={'payload': object()}
            )
        record.message = record.getMessage()  # as a formatter leaves it; left out
        data = RelayHandler.record_to_dict(record)
        assert json.loads(json.dumps(data)) == data
        assert (data['msg'], data['args'], data['exc_info']) == ('failed here', None, None)
        assert data['exc_text'].startswith('Traceback') and 'ValueError: boom' in data['exc_text']
        assert data['payload'].startswith('<object object')
        assert 'message' not in data
        assert logging.makeLogRecord(data).getMessage() == 'failed here'

    def test_record_to_dict_keeps_the_traceback_text(self):
        record = make_record('relay')
        record.exc_text = 'already formatted'
        assert RelayHandler.record_to_dict(record)['exc_text'] == 'already formatted'

    def test_percent_sign_in_the_message(self):
        data = RelayHandler.record_to_dict(make_record('relay', msg='%s done', args=('100%',)))
        assert logging.makeLogRecord(data).getMessage() == '100% done'

    def test_emit(self):
        sink = queue.Queue()
        logger = logging.getLogger('relay_emit')
        logger.addHandler(RelayHandler(sink))
        logger.warning('to the %s', 'parent')
        item = sink.get_nowait()
        assert isinstance(item, dict)
        assert (item['name'], item['levelno'], item['msg']) == ('relay_emit', logging.WARNING, 'to the parent')


# How jobs, pools and the TUI use them


class TestStandaloneJob:
    def test_default_logger(self):
        logger = Talker(job_id='a1').logger
        assert logger.name == 'a1'
        assert logger.propagate is False
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert getattr(logger.handlers[0], '_xtl_owned', False)

    @pytest.mark.asyncio
    async def test_levels(self):
        job = Talker(job_id='a2', logger=Talker.get_logger('a2', capture_config('a2', level=logging.INFO)))
        job.configure(JobConfig())
        await job.run()
        assert messages('a2') == ['info from a2', 'warning from a2']  # no debug, no lifecycle messages

    @pytest.mark.asyncio
    async def test_failing_job_logs_the_error(self):
        job = Failing(job_id='a3', logger=Failing.get_logger('a3', capture_config('a3')))
        job.configure(JobConfig())
        result = await job.run()
        assert not result.success
        assert any(m.startswith('Job failed due to an exception: boom') for m in messages('a3'))

    def test_reconfigure_keeps_foreign_handlers(self):
        logger = logging.getLogger('a4')
        foreign = Capture('a4_foreign')
        logger.addHandler(foreign)
        Talker.get_logger('a4', capture_config('a4'))
        Talker.get_logger('a4', capture_config('a4'))
        assert foreign in logger.handlers
        assert len([h for h in logger.handlers if getattr(h, '_xtl_owned', False)]) == 1

    def test_reconfigure_twice_with_two_handlers(self):
        two = LoggerConfig(handlers=[HandlerConfig(handler=Capture), HandlerConfig(handler=Capture)])
        logger = Talker.get_logger('a4b', two)
        Talker.get_logger('a4b', two)
        assert len(logger.handlers) == 2

    def test_configured_logger_is_left_alone(self):
        logger = Talker.get_logger('a5', capture_config('a5'))
        handlers = list(logger.handlers)
        assert Talker.get_logger('a5').handlers == handlers

    def test_defaults_follow_the_settings(self, monkeypatch):
        # The defaults are built when needed, so a change after import applies to new jobs and pools
        monkeypatch.setattr(xtl.settings.jobs, 'threading_debug', True)
        for logger in (Talker(job_id='a1b').logger, SimplePool(name='a1b').logger):
            assert 'processName' in logger.handlers[0].formatter._fmt

    def test_logger_passed_in_is_used(self):
        custom = logging.getLogger('a5_custom')
        assert Talker(job_id='a5b', logger=custom).logger is custom


class TestChildJobs:
    @pytest.mark.asyncio
    async def test_steps_reach_the_parent_handlers(self):
        job = Stepped(job_id='b6', logger=Stepped.get_logger('b6', capture_config('b6')))
        job.configure(SteppedJobConfig())
        result = await job.run()
        assert result.success
        assert 'info from b6.1' in messages('b6')
        assert 'info from b6.2' in messages('b6')
        assert {'b6', 'b6.1', 'b6.2'} <= names('b6')

    def test_step_loggers_use_the_parents_handlers(self):
        parent = Stepped(job_id='b6b', logger=Stepped.get_logger('b6b', capture_config('b6b')))
        child = parent._create_child(Talker, JobConfig(), suffix=1, inherit_pool=False)
        assert child.logger.name == 'b6b.1'
        assert child.logger.parent is parent.logger
        assert child.logger.handlers == []
        assert child.logger.propagate is True
        assert child.logger.level == logging.NOTSET

    def test_step_logger_config_override(self):
        parent = Stepped(job_id='b6c', logger=Stepped.get_logger('b6c', capture_config('b6c')))
        child = parent._create_child(
            Talker, JobConfig(), suffix=1, logger_config=capture_config('b6c_child'), inherit_pool=False
        )
        child.logger.info('from the child')
        assert messages('b6c_child') == ['from the child']
        assert messages('b6c') == []

    @pytest.mark.asyncio
    async def test_stepped_job_error_is_returned(self):
        job = SteppedRaising(job_id='b7', logger=SteppedRaising.get_logger('b7', capture_config('b7')))
        job.configure(SteppedJobConfig())
        result = await job.run()
        assert isinstance(result.error, ValueError)

    @pytest.mark.asyncio
    async def test_site_messages_reach_the_job(self, tmp_path, monkeypatch):
        async def cannot_start(*args, **kwargs):
            raise OSError('cannot start')

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', cannot_start)
        job = BatchJob.with_config(
            BatchJobConfig(job_directory=tmp_path / 'b8'),
            job_id='b8',
            logger=BatchJob.get_logger('b8', capture_config('b8')),
        )
        result = await job.run()
        assert not result.success
        assert any(m.startswith('Error executing batch file') for m in messages('b8'))

    def test_batch_file_default_logger(self, tmp_path):
        batch = BatchFile(tmp_path / 'b8b')
        assert batch.logger is logging.getLogger('xtl.jobs.batchfiles')
        job_logger = logging.getLogger('b8b')
        assert BatchFile(tmp_path / 'b8c', logger=job_logger).logger is job_logger


class TestPools:
    @pytest.mark.asyncio
    async def test_pool_logger(self):
        pool = SimplePool(name='c9', logger_config=capture_config('c9'))
        await run_pool(pool, Talker, ['c9_job'], JobConfig())
        assert pool.logger.name == f'SimplePool|c9#{pool.pool_id}'
        assert pool.logger.propagate is False
        assert pool.logger.name in names('c9')

    @pytest.mark.asyncio
    async def test_job_loggers_hang_under_the_pool(self):
        pool = SimplePool(name='c10', logger_config=capture_config('c10'))
        jobs, _ = await run_pool(pool, Talker, ['c10_job'], JobConfig())
        logger = jobs[0].logger
        assert logger.name == f'SimplePool|c10#{pool.pool_id}.c10_job'
        assert logger.handlers == []
        assert logger.propagate is True
        assert logger.level == logging.NOTSET
        assert 'info from c10_job' in messages('c10')

    def test_unnamed_pool_logger(self):
        pool = SimplePool()
        assert pool.logger.name == f'SimplePool#{pool.pool_id}'
        assert pool.label == 'SimplePool'
        assert SimplePool(name='refine').label == 'SimplePool|refine'

    @pytest.mark.parametrize('name', ['a.b', 'a#b'])
    def test_pool_name_without_dots_or_hashes(self, name):
        with pytest.raises(ValueError, match='dots or #'):
            SimplePool(name=name)

    @pytest.mark.asyncio
    async def test_same_named_pools_at_once(self):
        # The first pool to exit removes only its own loggers; the other's jobs still reach its handlers
        first = SimplePool(name='c10b', logger_config=capture_config('c10b_first'))
        second = SimplePool(name='c10b', logger_config=capture_config('c10b_second'))
        assert first.logger is not second.logger
        assert first.label == second.label == 'SimplePool|c10b'
        async with second:
            await run_pool(first, Talker, ['c10b_1'], JobConfig())
            second.submit(Talker, configs=[JobConfig()], job_ids=['c10b_2'])
            await second.launch()
        assert 'info from c10b_1' in messages('c10b_first')
        assert 'info from c10b_2' in messages('c10b_second')
        assert 'info from c10b_2' not in messages('c10b_first')

    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_add_handler_gets_everything(self, pool_cls):
        key = f'c15_{pool_cls.__name__}'
        pool = pool_cls(max_jobs=2, logger_config=LoggerConfig(level=logging.DEBUG, handlers=[]))
        handler = Capture(key)
        pool.add_handler(handler)
        _, results = await run_pool(pool, Stepped, [f'{key}_1'], SteppedJobConfig())
        assert all(r.success for r in results)
        shown = {r.job for r in RECORDS[key]}
        assert {pool_cls.__name__, f'{key}_1', f'{key}_1.1', f'{key}_1.2'} <= shown, shown
        assert f'info from {key}_1.2' in messages(key)
        pool.remove_handler(handler)
        assert handler not in pool.logger.handlers
        assert handler.filters == []

    def test_add_handler_levels(self):
        pool = SimplePool(logger_config=LoggerConfig(level=logging.INFO, handlers=[]))
        plain, verbose = Capture('c15b'), Capture('c15b', level=logging.DEBUG)
        pool.add_handler(plain)
        assert plain.level == logging.INFO
        assert pool.logger.level == logging.INFO
        pool.add_handler(verbose)
        assert verbose.level == logging.DEBUG
        assert pool.logger.level == logging.DEBUG

    @pytest.mark.parametrize('pool_cls', [SimplePool, MultiprocessPool])
    @pytest.mark.asyncio
    async def test_nested_pool_names(self, pool_cls):
        key = f'c16_{pool_cls.__name__}'
        pool = pool_cls(max_jobs=1, logger_config=capture_config(key))
        _, results = await run_pool(pool, Nesting, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results), [r.error for r in results]
        shown = {r.job for r in RECORDS[key]}
        outer = f'{key}_1'
        assert {pool_cls.__name__, outer, f'{outer}>ThreadedPool', f'{outer}>Talker|1'} <= shown, shown
        assert 'info from Talker|1' in messages(key)

    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_job_records_reach_the_pool(self, pool_cls):
        key = f'c11_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=capture_config(key))
        ids = [f'{key}_1', f'{key}_2']
        _, results = await run_pool(pool, Talker, ids, JobConfig())
        assert all(r.success for r in results)
        for job_id in ids:
            assert f'info from {job_id}' in messages(key)

    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_stream_handler_in_pool(self, pool_cls):
        stream = io.StringIO()
        config = LoggerConfig(level=logging.INFO, handlers=[StreamHandlerConfig(stream=stream)])
        key = f'c11b_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=config)
        _, results = await run_pool(pool, Talker, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results)
        assert f'info from {key}_1' in stream.getvalue()

    @pytest.mark.asyncio
    async def test_tui_config_in_process_pool(self):
        # The TUI's lifeline: with no handlers, records still come back to the parent
        config = LoggerConfig(level=logging.INFO, handlers=[])
        pool = MultiprocessPool(name='c12', max_jobs=2, logger_config=config)
        async with pool:
            jobs = pool.submit(Stepped, configs=[SteppedJobConfig()], job_ids=['c12_job'])
            for job in jobs:
                job.logger.addHandler(Capture('c12'))
            results = await pool.launch()
        assert all(r.success for r in results)
        # Under their own names; endswith, so that the test holds if job loggers get a prefix
        for suffix in ('c12_job', 'c12_job.1', 'c12_job.2'):
            assert any(name.endswith(suffix) for name in names('c12')), names('c12')
        assert 'info from c12_job.1' in messages('c12')
        assert all(r.processName.startswith('Process') for r in RECORDS['c12'])

    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_step_records_reach_the_pool(self, pool_cls):
        key = f'c11c_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=capture_config(key))
        _, results = await run_pool(pool, Stepped, [f'{key}_1'], SteppedJobConfig())
        assert all(r.success for r in results)
        assert f'info from {key}_1.1' in messages(key)
        assert f'info from {key}_1.2' in messages(key)

    @pytest.mark.asyncio
    async def test_no_config_leaks(self):
        pool = SimplePool(name='c13', logger_config=capture_config('c13'))
        await run_pool(pool, Talker, ['c13_job'], JobConfig())
        later_job = Talker(job_id='c13_later')
        later_pool = SimplePool(name='c13_later')
        for logger in (later_job.logger, later_pool.logger):
            assert not any(isinstance(h, Capture) for h in logger.handlers)

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS)
    @pytest.mark.asyncio
    async def test_file_handler_in_worker_pool(self, pool_cls, tmp_path):
        log = tmp_path / 'jobs.log'
        config = LoggerConfig(level=logging.INFO, handlers=[FileHandlerConfig(filename=str(log))])
        key = f'c14_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=config)
        _, results = await run_pool(pool, Talker, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results), [r.error for r in results]
        assert f'info from {key}_1' in log.read_text(encoding='utf-8')

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS)
    @pytest.mark.asyncio
    async def test_worker_records_follow_the_level(self, pool_cls):
        key = f'c14d_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=capture_config(key, level=logging.INFO))
        _, results = await run_pool(pool, Talker, [f'{key}_1'], JobConfig())
        assert all(r.success for r in results)
        assert f'info from {key}_1' in messages(key)
        assert f'debug from {key}_1' not in messages(key)

    @pytest.mark.parametrize('pool_cls', WORKER_POOLS)
    @pytest.mark.asyncio
    async def test_job_without_config_in_worker_pool(self, pool_cls):
        key = f'c14b_{pool_cls.__name__}'
        pool = pool_cls(name=key, max_jobs=2, logger_config=capture_config(key))
        _, results = await run_pool(pool, Talker, [f'{key}_1'], None)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_get_pool_warning_is_formatted(self):
        job = Talker(job_id='c14c', logger=Talker.get_logger('c14c', capture_config('c14c')))
        async with job.get_pool('simple', max_jobs=1_000_000):
            pass
        warnings = [r.captured for r in RECORDS['c14c'] if r.levelno == logging.WARNING]
        assert any('capping to' in m for m in warnings), warnings


class TestJobHandlers:
    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_job_file_holds_job_and_step_records(self, pool_cls, tmp_path):
        key = f'h1_{pool_cls.__name__}'
        pool = pool_cls(max_jobs=2, logger_config=capture_config(key, level=logging.INFO))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        jobs, results = await run_pool(pool, KeptStepped, [key], SteppedJobConfig(job_directory=tmp_path / key))
        assert all(r.success for r in results), [r.error for r in results]

        text = (tmp_path / key / 'xtl.log').read_text(encoding='utf-8')
        # The file: DEBUG, with level names and full logger names, from the job and its steps
        assert f'DEBUG {pool.logger.name}.{key}.1: debug from {key}.1' in text
        assert f'INFO {pool.logger.name}.{key}.2: info from {key}.2' in text
        # The pool's handler: only INFO and above
        assert f'info from {key}.1' in messages(key)
        assert not any(m.startswith('debug from') for m in messages(key))
        # The job's logger is left as it was
        assert jobs[0].logger.handlers == []
        assert jobs[0].logger.level == logging.NOTSET

    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_job_file_stays_on_failure_and_goes_on_success(self, pool_cls, tmp_path):
        key = f'h2_{pool_cls.__name__}'
        pool = pool_cls(max_jobs=2, logger_config=capture_config(key))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        failed, succeeded = tmp_path / 'failed', tmp_path / 'succeeded'
        async with pool:
            pool.submit(Failing, configs=[JobConfig(job_directory=failed)], job_ids=[f'{key}_f'])
            pool.submit(Talker, configs=[JobConfig(job_directory=succeeded)], job_ids=[f'{key}_s'])
            results = await pool.launch()
        assert [r.success for r in results] == [False, True]
        assert 'Job failed due to an exception: boom' in (failed / 'xtl.log').read_text(encoding='utf-8')
        # The file was closed before the directory was deleted (on Windows, an open file blocks it)
        assert not succeeded.exists()

    @pytest.mark.parametrize('pool_cls', [SimplePool, MultiprocessPool])
    @pytest.mark.asyncio
    async def test_user_defined_handler_class(self, pool_cls, tmp_path):
        key = f'h3_{pool_cls.__name__}'
        pool = pool_cls(name='h3', max_jobs=1, logger_config=capture_config(key))
        spec = HandlerConfig(
            handler=TaggedFileHandler, options={'filename': '{job_directory}/{job_class}.{job_id}.{pool}.log'}
        )
        pool.add_job_handler(spec, name='tagged')
        _, results = await run_pool(pool, KeptTalker, [f'{key}|1'], JobConfig(job_directory=tmp_path))
        assert all(r.success for r in results), [r.error for r in results]
        log = tmp_path / f'KeptTalker.{key}_1.{pool_cls.__name__}_h3.log'
        assert f'info from {key}|1' in log.read_text(encoding='utf-8')

    @pytest.mark.asyncio
    async def test_config_adds_replaces_and_switches_off(self, tmp_path):
        pool = SimplePool(max_jobs=1, logger_config=capture_config('h4'))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        configs = {
            'off': {'file': None},
            'replaced': {'file': JobFileHandlerConfig(filename='{job_directory}/other.log')},
            'added': {'extra': FileHandlerConfig(filename='{job_directory}/extra.log')},
        }
        async with pool:
            for job_id, log_handlers in configs.items():
                config = JobConfig(job_directory=tmp_path / job_id, log_handlers=log_handlers)
                pool.submit(KeptTalker, configs=[config], job_ids=[job_id])
            results = await pool.launch()
        assert all(r.success for r in results)
        files = {job_id: sorted(f.name for f in (tmp_path / job_id).glob('*.log')) for job_id in configs}
        assert files == {'off': [], 'replaced': ['other.log'], 'added': ['extra.log', 'xtl.log']}

    @pytest.mark.asyncio
    async def test_job_without_directory_skips_the_file(self):
        pool = SimplePool(max_jobs=1, logger_config=capture_config('h5'))
        pool.add_job_handler(JobFileHandlerConfig(), name='file')
        _, results = await run_pool(pool, Talker, ['h5_job'], JobConfig())
        assert all(r.success for r in results)
        # A job without a directory has nowhere to put the file: that's no reason to warn
        skipped = [r for r in RECORDS['h5'] if r.captured.startswith("Skipped log handler 'file'")]
        assert [r.levelno for r in skipped] == [logging.DEBUG], messages('h5')

    @pytest.mark.asyncio
    async def test_broken_spec_is_skipped_with_a_warning(self, tmp_path):
        pool = SimplePool(max_jobs=1, logger_config=capture_config('h5b'))
        pool.add_job_handler(HandlerConfig(handler=TaggedFileHandler, options={'no_such_option': 1}), name='broken')
        _, results = await run_pool(pool, Talker, ['h5b_job'], JobConfig(job_directory=tmp_path))
        assert all(r.success for r in results)
        skipped = [r for r in RECORDS['h5b'] if r.captured.startswith("Skipped log handler 'broken'")]
        assert [r.levelno for r in skipped] == [logging.WARNING], messages('h5b')

    def test_live_pool_adds_the_job_file(self, monkeypatch):
        live = LivePool(console=ConsoleIO(file=io.StringIO(), width=100), pool_type='simple', max_jobs=1)
        assert isinstance(live._pool._job_handlers['file'], JobFileHandlerConfig)

        monkeypatch.setattr(xtl.settings.cli, 'job_log_files', False)
        live = LivePool(console=ConsoleIO(file=io.StringIO(), width=100), pool_type='simple', max_jobs=1)
        assert live._pool._job_handlers == {}


def registered_under(name: str) -> list[str]:
    """Names of the loggers (and placeholders) registered as `name` or under it."""
    return [k for k in logging.root.manager.loggerDict if k == name or k.startswith(f'{name}.')]


class TestLoggerLifetime:
    @pytest.mark.parametrize('pool_cls', POOLS)
    @pytest.mark.asyncio
    async def test_pool_removes_its_loggers(self, pool_cls, tmp_path):
        key = f'l1_{pool_cls.__name__}'
        pool_log, live_log = tmp_path / 'pool.log', tmp_path / 'live.log'
        config = LoggerConfig(
            level=logging.INFO,
            handlers=[HandlerConfig(handler=Capture, options={'key': key}), FileHandlerConfig(filename=str(pool_log))],
        )
        pool = pool_cls(max_jobs=2, logger_config=config)
        live = logging.FileHandler(live_log, encoding='utf-8')
        pool.add_handler(live)
        built = [h for h in pool.logger.handlers if isinstance(h, logging.FileHandler) and h is not live]
        await run_pool(pool, Stepped, [f'{key}_1', f'{key}_2'], SteppedJobConfig())

        # Nothing of the pool's tree stays registered: pool, jobs, steps (in a process pool, relayed names too)
        assert registered_under(pool.logger.name) == []
        # The handlers built from the config are closed, so their files can go (on Windows too)
        assert [h.stream for h in built] == [None]
        pool_log.unlink()
        # A handler added with add_handler() is detached, without its name filter, but left open for its owner
        assert live not in pool.logger.handlers
        assert live.filters == []
        assert live.stream is not None
        live.close()
        assert f'info from {key}_2.1' in live_log.read_text(encoding='utf-8')

    @pytest.mark.asyncio
    async def test_nested_pools_go_with_the_outer_pool(self):
        pool = SimplePool(max_jobs=1, logger_config=capture_config('l2'))
        _, results = await run_pool(pool, Nesting, ['l2_job'], JobConfig())
        assert all(r.success for r in results)
        assert registered_under(pool.logger.name) == []

    @pytest.mark.asyncio
    async def test_pools_are_single_use(self):
        pool = SimplePool(max_jobs=1, logger_config=capture_config('l3'))
        await run_pool(pool, Talker, ['l3_job'], JobConfig())
        with pytest.raises(RuntimeError, match='single-use'):
            async with pool:
                pass

    @pytest.mark.asyncio
    async def test_standalone_job_keeps_its_logger(self):
        job = Talker.with_config(JobConfig(), job_id='l4', logger=Talker.get_logger('l4', capture_config('l4')))
        await job.run()
        assert registered_under('l4') == ['l4']
        assert len(job.logger.handlers) == 1


class TestSerialization:
    def test_submission_carries_the_level(self):
        job = Talker(job_id='d16', logger_config=capture_config('d16', level=logging.DEBUG))
        submission = JobSubmission.from_job(job)
        assert submission.log_level == logging.DEBUG
        assert JobSubmission.from_dict(submission.to_dict()).log_level == logging.DEBUG

    def test_handler_specs_travel_with_the_job(self):
        job = Talker(job_id='d16b')
        job._log_handlers = {'file': JobFileHandlerConfig(level=logging.INFO)}
        job._pool_label = 'SimplePool|d16b'
        restored = Job.deserialize(json.loads(json.dumps(job.serialize())))
        spec = restored._log_handlers['file']
        assert isinstance(spec, JobFileHandlerConfig)
        assert spec.level == logging.INFO
        assert restored._pool_label == 'SimplePool|d16b'

    def test_relayed_record_is_emitted_again(self):
        logger = logging.getLogger('d17')
        logger.handlers[:] = [Capture('d17')]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            raise ValueError('boom')
        except ValueError:
            record = logger.makeRecord(
                'd17', logging.ERROR, __file__, 1, 'failed %s', ('here',), sys.exc_info(), extra={'payload': object()}
            )

        MultiprocessPool._relay_log(json.loads(json.dumps(RelayHandler.record_to_dict(record))))

        [relayed] = RECORDS['d17']
        assert relayed.captured == 'failed here'
        assert relayed.payload.startswith('<object object')
        formatted = logging.Formatter('%(levelname)s %(message)s').format(relayed)
        assert formatted.startswith('ERROR failed here\nTraceback')
        assert 'ValueError: boom' in formatted

    def test_relay_respects_the_level(self):
        logger = logging.getLogger('d18')
        logger.handlers[:] = [Capture('d18')]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        MultiprocessPool._relay_log(RelayHandler.record_to_dict(make_record('d18', logging.DEBUG, 'quiet')))
        assert RECORDS['d18'] == []


def logger_states() -> dict[str, tuple]:
    """Handlers, `propagate` and level of the root and every logger."""
    loggers = [logging.root] + [lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    return {lg.name: (list(lg.handlers), lg.propagate, lg.level) for lg in loggers}


class TestTUI:
    @pytest.mark.asyncio
    async def test_live_pool_leaves_logging_as_it_found_it(self):
        original_get_logger = logging.getLogger
        before = logger_states()
        live = LivePool(console=ConsoleIO(file=io.StringIO(), width=100), pool_type='simple', max_jobs=1)
        async with live:
            # Only the root logger is swapped during the display
            assert logging.root.handlers == [live._buffer]
            assert logging.getLogger is original_get_logger
            live.submit(Talker, configs=[JobConfig()], job_ids=['e17'])
            await live.launch()
        after = logger_states()
        assert logging.getLogger is original_get_logger
        assert {name: state for name, state in after.items() if name in before} == before
        assert live._buffer not in live._pool.logger.handlers
        assert live._buffer.filters == []

    @pytest.mark.asyncio
    async def test_live_pool_shows_library_warnings(self):
        live = LivePool(console=ConsoleIO(file=io.StringIO(), width=100), pool_type='simple', max_jobs=1)
        async with live:
            logging.getLogger('e18.library').warning('careful')
            records = list(live._buffer._records)
        assert [(r.job, r.getMessage()) for r in records] == [('e18.library', 'careful')]

    def test_traceback_rendered_once(self):
        renderer = LogRenderer(console=ConsoleIO(file=io.StringIO(), width=100))
        try:
            raise ValueError('boom')
        except ValueError:
            record = logging.getLogger('e20').makeRecord(
                'e20', logging.ERROR, __file__, 1, 'failed', (), sys.exc_info()
            )

        # In-process: the text has the message only, and rich draws the traceback
        text, traceback = renderer._create_renderables(record)
        assert 'Traceback' not in text.plain
        assert isinstance(traceback, rich.traceback.Traceback)

        # Relayed from a worker: the traceback is text, the only place it shows
        relayed = logging.makeLogRecord(RelayHandler.record_to_dict(record))
        [text] = renderer._create_renderables(relayed)
        assert 'ValueError: boom' in text.plain

    @pytest.mark.parametrize('pool_type', ['simple', 'processes'])
    @pytest.mark.asyncio
    async def test_live_pool(self, pool_type):
        original_get_logger = logging.getLogger
        bystander = logging.getLogger('e19_bystander')
        handler = Capture('e19_bystander')
        bystander.addHandler(handler)

        console = ConsoleIO(file=io.StringIO(), width=100)
        live = LivePool(console=console, pool_type=pool_type, max_jobs=2)
        key = f'e19_{pool_type}'
        async with live:
            live.submit(Stepped, configs=[SteppedJobConfig()], job_ids=[key])
            results = await live.launch()
            records = list(live._buffer._records)

        assert all(r.success for r in results)
        assert f'info from {key}.1' in [r.getMessage() for r in records]
        assert {r.job for r in records} >= {key, f'{key}.1', f'{key}.2'}
        assert logging.getLogger is original_get_logger
        assert bystander.handlers == [handler]


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)


FORKED_WORKER = """
import asyncio, logging, queue, tempfile
from pathlib import Path
from xtl.jobs.config import JobConfig
from xtl.jobs.logging import JobFileHandlerConfig
from xtl.jobs.pools import MultiprocessPool, SimplePool
from xtl.jobs.resources import CURRENT_LEASE, ResourceManager, Resources
from xtl.jobs.submissions import JobSubmission
from tests.jobs.test_logging import Nesting

# What a worker started with fork inherits: a configured pool, the parent's lease and, if the application patched
#  it, a getLogger that gives each logger a handler of its own and stops it from propagating
pool = SimplePool(name='forked')
get_logger = logging.getLogger


def patched_get_logger(name=None):
    logger = get_logger(name)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


logging.getLogger = patched_get_logger
lease = asyncio.run(ResourceManager(total=Resources(jobs=1, threads=1, processes=1)).acquire(Resources(jobs=1)))
CURRENT_LEASE.set(lease)
job = Nesting(job_id='w1', logger=pool.logger.getChild('w1'))
job.configure(JobConfig(job_directory=Path(tempfile.mkdtemp())))
job._log_handlers = {'file': JobFileHandlerConfig()}  # wants DEBUG, which must not travel
payload = JobSubmission.from_job(job).to_dict()

sink = queue.Queue()
MultiprocessPool._init_worker(sink)
result = MultiprocessPool._run_in_process(payload)
assert result.success, result.error
names, levels = set(), set()
while not sink.empty():
    record = sink.get()
    names.add(record['name'])
    levels.add(record['levelno'])
assert min(levels) >= logging.INFO, levels
print(sorted(names))
"""


class TestForkedWorker:
    def test_inherited_state_is_reset(self):
        # Workers started with fork (Linux before Python 3.14) inherit the parent's logging and context; the
        #  subprocess recreates that state, so that the test also runs where fork isn't available
        result = _run_python(FORKED_WORKER)
        assert result.returncode == 0, result.stderr
        assert "'SimplePool|forked#" in result.stdout
        assert '.w1' in result.stdout
        assert 'Talker|1' in result.stdout


class TestModuleLoggers:
    def test_import_keeps_the_logger_class(self):
        result = _run_python('import logging, xtl; assert logging.getLoggerClass() is logging.Logger')
        assert result.returncode == 0, result.stderr

    def test_plain_logger_created_first(self):
        result = _run_python("import logging; logging.getLogger('xtl.jobs.jobs'); import xtl.jobs.jobs")
        assert result.returncode == 0, result.stderr

    def test_import_is_quiet(self):
        result = _run_python('import logging, xtl; assert not logging.root.handlers')
        assert result.returncode == 0, result.stderr
        assert result.stdout == ''
        assert result.stderr == ''


class _Dummy:
    """Formats as anything: %s, %d, %f, %r."""

    def __str__(self):
        return 'x'

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __index__(self):
        return 0


LOGGING_METHODS = {'debug', 'info', 'warning', 'error', 'exception', 'critical'}


def format_problems(path: Path) -> list[str]:
    """Logging calls with a literal message that `LogRecord.getMessage()` couldn't format."""
    problems = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'warnings':
            continue
        if node.func.attr in LOGGING_METHODS:
            args = node.args
        elif node.func.attr == 'log':
            args = node.args[1:]
        else:
            continue
        if len(args) < 2 or not (isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)):
            continue
        msg, rest = args[0].value, args[1:]
        if len(rest) == 1 and isinstance(rest[0], ast.Dict):
            values = {k.value: _Dummy() for k in rest[0].keys if isinstance(k, ast.Constant)}
        elif len(rest) == 1 and isinstance(rest[0], ast.Name):
            continue  # might be a dict or a single value
        else:
            values = tuple(_Dummy() for _ in rest)
        try:
            msg % values
        except Exception as e:
            problems.append(f'{path}:{node.lineno}: {type(e).__name__}: {e}: {msg!r}')
    return problems


class TestFormatsAndTimeouts:
    def test_logging_format_strings(self):
        problems = []
        for package in (xtl, xtl_core):
            for path in sorted(Path(package.__file__).parent.rglob('*.py')):
                problems.extend(format_problems(path))
        assert not problems, '\n'.join(problems)

    @pytest.mark.asyncio
    async def test_cancel_message_is_formatted(self, tmp_path):
        logger = logging.getLogger('g_cancel')
        logger.addHandler(Capture('g_cancel'))
        await LocalSite().cancel_batch(BatchFile(tmp_path / 'idle', logger=logger))
        assert any(m.startswith('Cannot cancel batch file') for m in messages('g_cancel')), messages('g_cancel')

    def test_execute_command_timeout(self):
        logger = logging.getLogger('xtl.jobs.sites')
        logger.addHandler(Capture('g_timeout'))
        try:
            result = LocalSite.execute_command([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=1)
        finally:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, Capture)]
        assert result is None
        assert any('timed out' in m for m in messages('g_timeout'))
