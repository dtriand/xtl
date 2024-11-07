from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrioritySystem(ABC):
    _system_type: str

    @abstractmethod
    def prepare_command(self, command: str) -> str:
        """
        Prepare a command for execution with this priority system.
        """
        pass

    @property
    def system_type(self) -> Optional[str]:
        """
        The type of the priority system.
        """
        return self._system_type


@dataclass
class DefaultPrioritySystem(PrioritySystem):

    def __init__(self, system_type: str = None):
        """
        Plain priority system with no special handling. Equivalent to not using a priority system.
        """
        super().__init__(_system_type=system_type)

    def prepare_command(self, command: str) -> str:
        if not isinstance(command, str):
            raise ValueError('\'command\' must be a string')
        return command


@dataclass
class NicePrioritySystem(DefaultPrioritySystem):

    def __init__(self, nice_level: int = 10):
        """
        Priority system that uses the 'nice' command to set the priority level of a command.
        """
        super().__init__('nice')
        self._nice_level = nice_level

    @property
    def nice_level(self) -> int:
        """
        The priority level to set with the 'nice' command.
        """
        return self._nice_level

    @nice_level.setter
    def nice_level(self, value: int):
        if not isinstance(value, int):
            raise ValueError('\'nice_level\' must be an integer')
        self._nice_level = value

    def prepare_command(self, command: str) -> str:
        command = super().prepare_command(command)
        return f'nice -n {self.nice_level} {command}'
