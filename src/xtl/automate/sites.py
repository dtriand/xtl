from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable, TYPE_CHECKING

from xtl.automate.priority_system import PrioritySystemType, DefaultPrioritySystem, NicePrioritySystem
from xtl.automate.shells import Shell, DefaultShell, BashShell, CmdShell, WslShell

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings


@dataclass
class ComputeSite(ABC):
    """
    Abstract base class for a computational site. It defines an interface for executing commands and various operations
    on a specific site.
    """

    _priority_system: Optional[PrioritySystemType] = None
    _default_shell: Optional[Shell] = None
    _supported_shells: Iterable[Shell] | None = None

    @property
    def priority_system(self):
        """
        The priority system for this compute site. This is used to prepare commands for execution on the site.
        """
        return self._priority_system

    @priority_system.setter
    def priority_system(self, value):
        if not isinstance(value, PrioritySystemType):
            raise TypeError('\'priority_system\' must be an instance of PrioritySystem')
        self._priority_system = value

    @property
    def default_shell(self) -> Optional[Shell]:
        """
        The default shell for this compute site. This is used to execute commands on the site.
        """
        return self._default_shell

    @property
    def supported_shells(self) -> Iterable[Shell] | None:
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
    def load_modules(self, modules: str | Iterable[str]) -> str:
        """
        Generates a command for loading the specified modules on the compute site.
        """
        #  TODO: Remove when refactoring AutoPROCJob
        pass

    @abstractmethod
    def purge_modules(self) -> str:
        """
        Generates a command for purging all loaded modules on the compute site.
        """
        #  TODO: Remove when refactoring AutoPROCJob
        pass

    @abstractmethod
    def get_preamble(self, dependencies: 'DependencySettings' |
                                         Iterable['DependencySettings'],
                     shell: Shell | WslShell = None) -> list[str]:
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
        self._default_shell = DefaultShell

    def load_modules(self, modules) -> str:
        return ''

    def purge_modules(self) -> str:
        return ''

    def get_preamble(self, dependencies: 'DependencySettings' |
                     Iterable['DependencySettings'],
                     shell: Shell | WslShell = None) -> list[str]:
        # Returns an empty preamble for the local site, as it assumes that all required
        # executables are available on PATH
        return []


class ModulesSite(ComputeSite):
    def __init__(self):
        self._default_shell = DefaultShell
        self._priority_system = DefaultPrioritySystem()

    def load_modules(self, modules: str | Iterable[str]) -> str: ...

    def purge_modules(self) -> str: ...

    @staticmethod
    def _load_modules(modules: Iterable[str], shell: Shell = None) -> str:
        cmd = 'module load ' + ' '.join(modules)
        if shell is CmdShell:
            # For Windows CMD, we need to use 'call' to execute the module commands
            cmd = f'call {cmd}'
        return cmd

    @staticmethod
    def _purge_modules(shell: Shell = None) -> str:
        cmd = 'module purge'
        if shell is CmdShell:
            cmd = f'call {cmd}'
        return cmd

    def get_preamble(self, dependencies: 'DependencySettings' |
                                         Iterable['DependencySettings'],
                     shell: Shell | WslShell = None) -> list[str]:
        if shell is None:
            shell = self._default_shell
        if not isinstance(dependencies, Iterable):
            dependencies = [dependencies]

        modules = []
        for dep in dependencies:
            if dep.modules:
                modules.extend(dep.modules)

        cmds = [self._purge_modules(shell=shell)]
        if modules:
            cmds.append(self._load_modules(modules, shell=shell))
        return cmds


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

    def load_modules(self, modules: str | Iterable[str]) -> str:
        mods = []
        if isinstance(modules, str):
            # Check for space-separated modules in a single string
            for mod in modules.split():
                mods.append(mod)
        elif isinstance(modules, Iterable):
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

    def get_preamble(self, dependencies: 'DependencySettings' |
                                         Iterable['DependencySettings'],
                     shell: Shell | WslShell = None) -> list[str]:
        # TODO: Implement this
        return []


ComputeSiteType = LocalSite | ModulesSite | BiotixHPC