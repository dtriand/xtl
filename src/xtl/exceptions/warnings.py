class Warning_(Warning):
    pass


class ConfigWarning(Warning_):
    pass


class ObjectInstantiationWarning(Warning_):

    def __init__(self, message='', raiser=None):
        self.raiser = raiser
        self.message = message

    def __str__(self):
        return f'{self.raiser}: {self.message}' if self.raiser else self.message
