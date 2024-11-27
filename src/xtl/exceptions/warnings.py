class Warning_(Warning):
    pass


class ConfigWarning(Warning_):
    pass


class FileNotFoundWarning(Warning_):
    pass


class HTMLUpdateWarning(Warning_):
    pass


class RegexWarning(Warning_):
    pass


class IncompatibleShellWarning(Warning_):
    pass


class ObjectInstantiationWarning(Warning_):

    def __init__(self, message='', raiser=None):
        self.raiser = raiser
        self.message = message

    def __str__(self):
        return f'{self.raiser}: {self.message}' if self.raiser else self.message


class ExistingReagentWarning(Warning_):

    def __init__(self, message='', raiser: 'xtl.crystallization.experiments.Reagent' = None):
        self.raiser = raiser

    def __str__(self):
        return f'Reagent {self.raiser.name} already in the list of reagents'
