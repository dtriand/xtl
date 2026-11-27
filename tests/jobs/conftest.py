import logging

import pytest


@pytest.fixture(autouse=True)
def fresh_logging():
    """
    Undo what jobs, pools and the TUI leak into `logging`: `logging.getLogger`, the
    handlers, `propagate` and level of every logger, and the loggers created during the
    test.
    """
    get_logger = logging.getLogger
    manager = logging.root.manager
    existing = set(manager.loggerDict)
    # Restore every logger, so that a test that leaves one changed can't affect the next
    loggers = [logging.root] + [lg for lg in manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    states = {lg: (list(lg.handlers), lg.propagate, lg.level) for lg in loggers}

    yield

    logging.getLogger = get_logger
    for lg, (handlers, propagate, level) in states.items():
        lg.handlers[:] = handlers
        lg.propagate = propagate
        lg.setLevel(level)
    for name in set(manager.loggerDict) - existing:
        logger = manager.loggerDict.pop(name)
        for handler in getattr(logger, 'handlers', []):
            handler.close()
    manager._clear_cache()
