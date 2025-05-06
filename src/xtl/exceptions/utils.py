import traceback
from typing import Callable, Optional
import warnings


class Catcher:
    def __init__(self, silent: bool = False, error_message: str = 'Exception was raised: ',
                 echo_func: Callable = print, traceback_func: Optional[Callable] = None,
                 error_kwargs: dict = None, warning_kwargs: dict = None):
        self.silent = silent
        self.echo_func = echo_func
        self.traceback_func = traceback_func
        self._errors = []
        self.error_message = error_message
        self.error_kwargs = error_kwargs or {}
        self.warning_kwargs = warning_kwargs or {}
        self._warnings = []
        self._warnings_ctx: warnings.catch_warnings
        self.raised = False

    def __enter__(self):
        self._warnings_ctx = warnings.catch_warnings(record=True)
        self._warnings = self._warnings_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._warnings_ctx.__exit__(exc_type, exc_val, exc_tb)
        for w in self._warnings:
            if not self.silent:
                self.echo_func(f'Warning was raised by: {w.filename}:{w.lineno}', **self.warning_kwargs)
                self.echo_func(f'    {w.category.__name__}: {w.message}', **self.warning_kwargs)
        if exc_type:
            self.log_exception([exc_type, exc_val, exc_tb])
            if not self.silent:
                if self.traceback_func is not None:
                    self.traceback_func(exc_val)
                else:
                    self.echo_func(self.error_message, **self.error_kwargs)
                    for line in traceback.format_exception(exc_type, exc_val, exc_tb):
                        self.echo_func(f'    {line}', **self.error_kwargs)
        return True

    @property
    def errors(self):
        return self._errors

    @property
    def warnings(self):
        return self._warnings

    def log_exception(self, exc: object):
        self._errors.append(exc)
        self.raised = True