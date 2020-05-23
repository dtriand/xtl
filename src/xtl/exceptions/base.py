class Error(Exception):

    def __init__(self, message='', raiser=None):
        self.raiser = raiser
        self.message = message

    def __str__(self):
        return f'{self.raiser}: {self.message}' if self.raiser else self.message


class InvalidArgument(Error):
    pass


class FileError(Error):

    def __init__(self, file, message, details=None):
        self.file = file
        self.message = message
        self.details = details

    def __str__(self):
        return f'{self.message} in {self.file}' + (f'\n{self.details}' if self.details else '')
