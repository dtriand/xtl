import io
import logging
import os
import pytest
import sys
import tempfile
from pathlib import Path

from xtl.logging.config import (
    LoggingFormat, HandlerConfig, StreamHandlerConfig,
    FileHandlerConfig, LoggerConfig
)


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
        config = StreamHandlerConfig(
            stream=stream,
            format=LoggingFormat(format='%(message)s')
        )
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
            config = FileHandlerConfig(
                filename=str(log_path),
                format=LoggingFormat(format='%(message)s')
            )
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
        config = LoggerConfig(
            level=logging.DEBUG,
            propagate=True,
            handlers=[stream_handler, file_handler]
        )
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
            handlers=[
                StreamHandlerConfig(
                    stream=stream,
                    format=LoggingFormat(format='%(levelname)s: %(message)s')
                )
            ]
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
                        stream=stream,
                        format=LoggingFormat(
                            format='STREAM: %(levelname)s - %(message)s'
                        )
                    ),
                    FileHandlerConfig(
                        filename=str(log_path),
                        format=LoggingFormat(
                            format='FILE: %(levelname)s - %(message)s'
                        )
                    )
                ]
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
