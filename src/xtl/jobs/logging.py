import logging

from xtl.logging.config import LoggerConfig, StreamHandlerConfig, LoggingFormat


def get_logger_config(level: int = logging.INFO) -> LoggerConfig:
    """
    Get a LoggerConfig instance for configuring job and pool loggers.
    """
    from xtl import settings

    if settings.jobs.threading_debug:
        fmt = '[%(asctime)s.%(msecs)03d:%(processName)s:%(threadName)s:%(name)s] %(message)s'
    else:
        fmt = '[%(asctime)s.%(msecs)03d:%(name)s] %(message)s'

    config = LoggerConfig(
        level=level,
        propagate=False,
        handlers=[
            StreamHandlerConfig(
                format=LoggingFormat(
                    format=fmt,
                    datefmt='%H:%M:%S'
                )
            )
        ]
    )
    return config