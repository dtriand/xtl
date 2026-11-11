from typing import Any


class Error(Exception):
    def __init__(self, message='', raiser=None):
        self.raiser = raiser
        self.message = message

    def __str__(self):
        return f'{self.raiser}: {self.message}' if self.raiser else self.message


class SubprocessError(Error):
    def __init__(self, message: str, command: Any, raiser=None):
        self.message = message
        self.command = command
        self.raiser = raiser


class StderrError(Error):
    def __init__(self, message: str, stderr: str, raiser=None):
        self.message = message
        self.stderr = stderr
        self.raiser = raiser
