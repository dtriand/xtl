from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from xtl.automate.priority_system import PrioritySystem, DefaultPrioritySystem, NicePrioritySystem
from xtl.automate.shells import Shell, DefaultShell, BashShell


@dataclass
class ComputeSite(ABC):
    """
    Abstract base class for a computational site. It defines an interface for executing commands and various operations
    on a specific site.
    """

    _priority_system: Optional[PrioritySystem] = None
    _default_shell: Optional[Shell] = None
    _supported_shells: Sequence[Shell] | None = None

    @property
    def priority_system(self):
        """
        The priority system for this compute site. This is used to prepare commands for execution on the site.
        """
        return self._priority_system

    @priority_system.setter
    def priority_system(self, value):
        if not isinstance(value, PrioritySystem):
            raise TypeError('\'priority_system\' must be an instance of PrioritySystem')
        self._priority_system = value

    @property
    def default_shell(self) -> Optional[Shell]:
        """
        The default shell for this compute site. This is used to execute commands on the site.
        """
        return self._default_shell

    @property
    def supported_shells(self) -> Sequence[Shell] | None:
        """
        The supported shells for this compute site. This is used to validate the shell passed to the BatchFile.
        """
        return self._supported_shells

    def is_valid_shell(self, shell: Shell) -> bool:
        """
        Checks if the specified shell is supported by this compute site.
        """
        if self.supported_shells is None:
            return True
        return shell in self.supported_shells

    @abstractmethod
    def load_modules(self, modules: str | Sequence[str]) -> str:
        """
        Generates a command for loading the specified modules on the compute site.
        """
        pass

    @abstractmethod
    def purge_modules(self) -> str:
        """
        Generates a command for purging all loaded modules on the compute site.
        """
        pass

    def prepare_command(self, command: str) -> str:
        """
        Prepares a command for execution on the compute site using the underlying priority system.
        """
        return self.priority_system.prepare_command(command)


class LocalSite(ComputeSite):
    def __init__(self):
        """
        A compute site that represents the local machine. It does not support loading or purging modules, rather it just
        assumes that all required executables are available on PATH.
        """
        self._priority_system = DefaultPrioritySystem()

    def load_modules(self, modules) -> str:
        return ''

    def purge_modules(self) -> str:
        return ''


class BiotixHPC(ComputeSite):
    def __init__(self):
        """
        A compute site that represents the Biotix HPC cluster. It supports loading and purging modules using the
        `module` command from the Environmental Modules package (see: https://modules.readthedocs.io/en/latest/).
        The default priority system is 'nice -n 10'.
        """
        self._priority_system = NicePrioritySystem(10)
        self._default_shell = BashShell
        self._supported_shells = [BashShell]

    def load_modules(self, modules: str | Sequence[str]) -> str:
        mods = []
        if isinstance(modules, str):
            # Check for space-separated modules in a single string
            for mod in modules.split():
                mods.append(mod)
        elif isinstance(modules, Sequence):
            # Otherwise append each module in the list
            for i, mod in enumerate(modules):
                if not isinstance(mod, str):
                    raise ValueError('\'modules\' must be a string or a list of strings, while modules[{i}] '
                                     'is {type(mod)}')
                mods.append(mod)
        else:
            raise ValueError(f'\'modules\' must be a string or a list of strings, not {type(modules)}')
        if not mods:
            # Do not generate a command if no modules are provided
            return ''
        return 'module load ' + ' '.join(mods)

    def purge_modules(self) -> str:
        return 'module purge'
