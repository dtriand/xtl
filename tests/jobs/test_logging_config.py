import io
import json
import logging
import sys
import pytest
import tempfile
from pathlib import Path

from xtl.jobs.logging import (
    LoggingFormat,
    HandlerConfig,
    StreamHandlerConfig,
    FileHandlerConfig,
    JobFileHandlerConfig,
    LoggerConfig,
    MissingPlaceholderError,
)
from xtl_core.options import Option


class TaggedHandler(logging.Handler):
    """A user-defined handler class, importable from this module."""


class TaggedHandlerConfig(HandlerConfig):
    """A user-defined handler config, importable from this module."""

    handler: type(logging.Handler) = Option(default=TaggedHandler, desc='Handler class')
    tag: str = Option(default='', desc='A field of its own')


class TestLoggingFormat:
    """Tests for the LoggingFormat class."""

    def test_get_formatter(self):
        """Test that a formatter is created with the correct format."""
        fmt = LoggingFormat(format='%(levelname)s: %(message)s', datefmt='%H:%M')
        formatter = fmt.get_formatter()
        assert isinstance(formatter, logging.Formatter)
        assert formatter._fmt == '%(levelname)s: %(message)s'
        assert formatter.datefmt == '%H:%M'


class TestHandlerConfig:
    """Tests for the HandlerConfig class."""

    def test_get_handler(self):
        """Test that a handler is created with the correct configuration."""
        config = HandlerConfig(handler=logging.NullHandler)
        handler = config.get_handler()
        assert isinstance(handler, logging.NullHandler)
        assert isinstance(handler.formatter, logging.Formatter)


class TestStreamHandlerConfig:
    """Tests for the StreamHandlerConfig class."""

    def test_get_handler(self):
        """Test that a stream handler is created with the correct configuration."""
        stream = io.StringIO()
        config = StreamHandlerConfig(stream=stream)
        handler = config.get_handler()
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is stream

    def test_logging_to_stream(self):
        """Test that logging works correctly with a stream handler."""
        stream = io.StringIO()
        config = StreamHandlerConfig(stream=stream, format=LoggingFormat(format='%(message)s'))
        handler = config.get_handler()

        logger = logging.getLogger('test_stream')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

        # Clear existing handlers
        for h in logger.handlers[:]:
            if h is not handler:
                logger.removeHandler(h)

        logger.info('Test message')
        assert stream.getvalue() == 'Test message\n'


class TestFileHandlerConfig:
    """Tests for the FileHandlerConfig class."""

    def test_get_handler(self):
        """Test that a file handler is created with the correct configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'test.log'
            config = FileHandlerConfig(filename=str(log_path))
            handler = config.get_handler()
            assert isinstance(handler, logging.FileHandler)
            assert handler.baseFilename == str(log_path)
            handler.close()  # release the file handler so that tempfile can be deleted

    def test_logging_to_file(self):
        """Test that logging works correctly with a file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'test.log'
            config = FileHandlerConfig(filename=str(log_path), format=LoggingFormat(format='%(message)s'))
            handler = config.get_handler()

            logger = logging.getLogger('test_file')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.propagate = False

            # Clear existing handlers
            for h in logger.handlers[:]:
                if h is not handler:
                    logger.removeHandler(h)

            logger.info('Test message')
            handler.close()  # Ensure file is written

            with open(log_path, 'r') as f:
                content = f.read()
                assert content == 'Test message\n'


class TestLoggerConfig:
    """Tests for the LoggerConfig class."""

    def test_default_logger_config(self):
        """Test that the default logger config is created correctly."""
        config = LoggerConfig()
        assert config.level == logging.INFO
        assert config.propagate is False
        assert len(config.handlers) == 1
        assert isinstance(config.handlers[0], StreamHandlerConfig)

    def test_custom_logger_config(self):
        """Test that a custom logger config can be created."""
        stream_handler = StreamHandlerConfig(stream=io.StringIO())
        file_handler = FileHandlerConfig(filename='custom.log')
        config = LoggerConfig(level=logging.DEBUG, propagate=True, handlers=[stream_handler, file_handler])
        assert config.level == logging.DEBUG
        assert config.propagate is True
        assert len(config.handlers) == 2
        assert config.handlers[0] is stream_handler
        assert config.handlers[1] is file_handler

    def test_configure_logger(self):
        """Test that a logger is configured correctly."""
        stream = io.StringIO()
        config = LoggerConfig(
            level=logging.DEBUG,
            handlers=[StreamHandlerConfig(stream=stream, format=LoggingFormat(format='%(levelname)s: %(message)s'))],
        )

        logger = logging.getLogger('test_configure')

        # Clear existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        config.configure(logger)

        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert len(logger.handlers) == 1

        logger.debug('Debug message')
        assert stream.getvalue() == 'DEBUG: Debug message\n'

    def test_configure_with_no_handlers(self):
        """Test that a NullHandler is added if no handlers are present."""
        config = LoggerConfig(handlers=[])

        logger = logging.getLogger('test_null_handler')

        # Clear existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        config.configure(logger)

        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)

    def test_dict_round_trip(self):
        """Test that handler configs keep their type and fields through a dict round trip."""
        config = LoggerConfig(handlers=[FileHandlerConfig(filename='job.log', mode='w')])
        restored = LoggerConfig.from_dict(config.to_dict())
        handler = restored.handlers[0]
        assert isinstance(handler, FileHandlerConfig)
        assert (handler.filename, handler.mode) == ('job.log', 'w')

    def test_json_round_trip(self):
        """Test that handler configs keep their type and fields through a JSON round trip."""
        config = LoggerConfig(handlers=[FileHandlerConfig(filename='job.log', mode='w')])
        restored = LoggerConfig.from_json(config.to_json())
        handler = restored.handlers[0]
        assert isinstance(handler, FileHandlerConfig)
        assert (handler.filename, handler.mode) == ('job.log', 'w')

    def test_json_names_classes_by_import_path(self):
        data = json.loads(LoggerConfig(handlers=[FileHandlerConfig()]).to_json())
        spec = data['handlers'][0]
        assert spec['handler'] == 'logging.FileHandler'
        assert spec['config_cls'] == 'xtl.jobs.logging.FileHandlerConfig'

    def test_user_defined_config_survives_json(self):
        config = LoggerConfig(handlers=[TaggedHandlerConfig(tag='mine', options={'level': logging.WARNING})])
        restored = LoggerConfig.from_json(config.to_json()).handlers[0]
        assert isinstance(restored, TaggedHandlerConfig)
        assert restored.tag == 'mine'
        assert restored.handler is TaggedHandler
        assert restored.get_handler().level == logging.WARNING

    def test_spec_without_config_class(self):
        restored = LoggerConfig.from_dict({'handlers': [{'handler': 'logging.NullHandler'}]}).handlers[0]
        assert type(restored) is HandlerConfig
        assert restored.handler is logging.NullHandler

    def test_stream_defaults_to_the_current_stdout(self, monkeypatch):
        stream = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', stream)
        assert StreamHandlerConfig().get_handler().stream is stream
        assert StreamHandlerConfig(stream='stderr').get_handler().stream is sys.stderr
        assert json.loads(StreamHandlerConfig(stream=sys.stdout).to_json())['stream'] == 'stdout'

    def test_stream_object_cannot_go_to_json(self):
        with pytest.raises(Exception, match="stream object can't be serialized"):
            LoggerConfig(handlers=[StreamHandlerConfig(stream=io.StringIO())]).to_json()

    def test_handler_levels(self):
        config = LoggerConfig(
            level=logging.INFO,
            handlers=[HandlerConfig(handler=logging.NullHandler), HandlerConfig(handler=logging.NullHandler, level=10)],
        )
        logger = logging.getLogger('test_handler_levels')
        config.configure(logger)
        assert [h.level for h in logger.handlers] == [logging.INFO, logging.DEBUG]
        assert logger.level == logging.DEBUG
        logger.handlers.clear()

    def test_fill_placeholders(self):
        spec = HandlerConfig(
            handler=logging.NullHandler,
            format=LoggingFormat(format='{job_id} %(message)s'),
            options={'tag': '{pool}:{job_id}', 'other': '{unknown}', 'count': 3},
        )
        filled = spec.fill({'job_directory': None, 'job_id': 'T_1', 'job_class': 'T', 'pool': 'SimplePool'})
        assert filled.options == {'tag': 'SimplePool:T_1', 'other': '{unknown}', 'count': 3}
        assert filled.format.format == '{job_id} %(message)s'
        assert spec.options['tag'] == '{pool}:{job_id}'

    def test_fill_without_a_value(self):
        with pytest.raises(MissingPlaceholderError, match='job_directory'):
            JobFileHandlerConfig().fill({'job_directory': None})

    def test_job_file_defaults(self):
        spec = JobFileHandlerConfig()
        assert (spec.filename, spec.mode, spec.level) == ('{job_directory}/xtl.log', 'w', logging.DEBUG)
        restored = LoggerConfig.from_json(LoggerConfig(handlers=[spec]).to_json()).handlers[0]
        assert isinstance(restored, JobFileHandlerConfig)

    def test_live_handlers(self):
        live = logging.NullHandler()
        config = LoggerConfig(handlers=[live, StreamHandlerConfig(stream=io.StringIO())])
        logger = logging.getLogger('test_live_handlers')
        config.configure(logger, label='Live')
        assert live in logger.handlers
        assert len(logger.handlers) == 2
        assert len(live.filters) == 1

        # Reconfiguring detaches the live handler without closing it, and doesn't add a second filter
        LoggerConfig(handlers=[]).configure(logger)
        assert live not in logger.handlers
        config.configure(logger, label='Live')
        assert len(live.filters) == 1
        logger.handlers.clear()

        assert config.to_dict()['handlers'][0] is live
        with pytest.raises(Exception, match="live handler can't be serialized"):
            config.to_json()


@pytest.fixture
def cleanup_logging():
    """Fixture to clean up logging configuration after tests."""
    yield
    # Reset the root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)


@pytest.mark.usefixtures('cleanup_logging')
class TestIntegration:
    """Integration tests for the logging configuration."""

    def test_full_logging_setup(self):
        """Test a complete logging setup with multiple handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'integration.log'
            stream = io.StringIO()

            # Create a complex logger configuration
            config = LoggerConfig(
                level=logging.DEBUG,
                propagate=False,
                handlers=[
                    StreamHandlerConfig(
                        stream=stream, format=LoggingFormat(format='STREAM: %(levelname)s - %(message)s')
                    ),
                    FileHandlerConfig(
                        filename=str(log_path), format=LoggingFormat(format='FILE: %(levelname)s - %(message)s')
                    ),
                ],
            )

            # Configure a logger with this setup
            logger = logging.getLogger('integration_test')

            # Clear existing handlers
            for h in logger.handlers[:]:
                logger.removeHandler(h)

            config.configure(logger)

            # Log messages at different levels
            logger.debug('Debug message')
            logger.info('Info message')
            logger.warning('Warning message')

            # Close handlers to ensure file is written
            for handler in logger.handlers:
                handler.close()

            # Check stream output
            stream_output = stream.getvalue()
            assert 'STREAM: DEBUG - Debug message' in stream_output
            assert 'STREAM: INFO - Info message' in stream_output
            assert 'STREAM: WARNING - Warning message' in stream_output

            # Check file output
            with open(log_path, 'r') as f:
                file_output = f.read()
                assert 'FILE: DEBUG - Debug message' in file_output
                assert 'FILE: INFO - Info message' in file_output
                assert 'FILE: WARNING - Warning message' in file_output
