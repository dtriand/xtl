class Error(Exception):

    def __init__(self, message='', raiser=None):
        self.raiser = raiser
        self.message = message

    def __str__(self):
        return f'{self.raiser}: {self.message}' if self.raiser else self.message


class InvalidArgument(Error):
    pass


class DimensionalityError(Error):

    def __init__(self, raiser=None, src_units='', dst_units=''):
        self.raiser = raiser
        self.src_units = src_units
        self.dst_units = dst_units

    def __str__(self):
        msg = f"Cannot convert from '{self.src_units}' to '{self.dst_units}'."
        return f'{self.raiser}: ' + msg if self.raiser else msg


class FileError(Error):

    def __init__(self, file, message, details=None):
        self.file = file
        self.message = message
        self.details = details

    def __str__(self):
        return f'{self.message} in {self.file}' + (f'\n{self.details}' if self.details else '')


class MissingArgument(Error):

    def __init__(self, arg, message):
        self.arg = arg
        self.message = message

    def __str__(self):
        return f'{self.arg}: {self.message}'