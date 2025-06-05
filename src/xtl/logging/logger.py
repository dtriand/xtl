import logging


class XTLLogger(logging.Logger):
    """
    Custom logger class for XTL.
    """
    ...


# Set the default logger class to XTLLogger
logging.setLoggerClass(XTLLogger)


class LoggerFactory:
    """
    A factory for spawning `XTLLogger` instances.
    """

    def __call__(self, name) -> XTLLogger:
        # Create a new logger with the specified name
        logger = logging.getLogger(name)

        # Ensure the logger is an instance of XTLLogger
        if not isinstance(logger, XTLLogger):
            raise TypeError(f'Logger for {name!r} is not an '
                            f'{XTLLogger.__name__}. Did you forget to call '
                            f'logging.setLoggerClass({XTLLogger.__name__}) ?')

        # Add a NullHandler to silence the 'No handlers found' warning
        logger.addHandler(logging.NullHandler())

        return logger
