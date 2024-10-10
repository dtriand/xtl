from dataclasses import dataclass


@dataclass
class _PrioritySystem:
    _system_type: str = None

    def prepare_command(self, command: str):
        if not isinstance(command, str):
            raise TypeError('command must be a string')
        return command


@dataclass
class DefaultPrioritySystem(_PrioritySystem):
    def __init__(self):
        super().__init__()


@dataclass
class NicePrioritySystem(_PrioritySystem):
    def __init__(self, nice_level: int = 10):
        super().__init__('nice')
        self._nice_level = nice_level

    @property
    def nice_level(self):
        return self._nice_level

    @nice_level.setter
    def nice_level(self, value):
        if not isinstance(value, int):
            raise TypeError('nice_level must be an integer')
        self._nice_level = value

    def prepare_command(self, command: str):
        if not isinstance(command, str):
            raise TypeError('command must be a string')
        return f'nice -n {self.nice_level} {command}'
