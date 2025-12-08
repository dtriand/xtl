"""
.. |BatchFile| replace:: :class:`BatchFile <xtl.jobs.batchfiles.BatchFile>`
.. |CommandPolicy| replace:: :class:`CommandPolicy <xtl.jobs.policies.CommandPolicy>`
.. |CommandPolicyType| replace:: :class:`CommandPolicyType <xtl.jobs.policies.CommandPolicyType>`
.. |BaseCommandPolicy| replace:: :class:`BaseCommandPolicy <xtl.jobs.policies.BaseCommandPolicy>`
.. |DefaultCommandPolicy| replace:: :class:`DefaultCommandPolicy <xtl.jobs.policies.DefaultCommandPolicy>`
.. |NiceCommandPolicy| replace:: :class:`NiceCommandPolicy <xtl.jobs.policies.NiceCommandPolicy>`

Modifiers for the contents of |BatchFile| instances.

This module defines modifiers that can be applied to the contents of the |BatchFile|
instances before they are executed on a compute site. The module defines an abstract
base class |BaseCommandPolicy| that specifies the interface for command policies.

An example implementation that checks for the presence of ``rm`` commands and raises an
exception if any are found could be as follows:

.. code-block:: python

    import shlex
    from xtl.jobs.policies import BaseCommandPolicy

    class NoRmPolicy(BaseCommandPolicy):
        name: str = 'no_rm'

        def intercept_preamble(self, preamble: str) -> str:
            return preamble

        def intercept_command(self, command: str) -> str:
            # Tokenize the input command
            tokens = shlex.split(command)
            for token in tokens:
                if token in ['rm', 'del', 'erase', 'rmdir', 'Remove-Item']:
                    # The policy is agnostic to the shell being used
                    raise ValueError("Usage of 'rm' command is not allowed.")
            return command

        def intercept_content(self, content: str) -> str:
            return content

        def intercept_postamble(self, postamble: str) -> str:
            return postamble

Two built-in command policies are provided:

- |DefaultCommandPolicy|: A policy that does not modify any part of the batch file
  contents.
- |NiceCommandPolicy|: A policy that modifies commands to be executed with the ``nice``
  command to set their priority level on Unix-like systems.

The |CommandPolicy| enumeration provides a convenient way to select and instantiate
these built-in command policies.

.. code-block:: python

    from xtl.jobs.policies import CommandPolicy

    # Instantiate the default command policy
    policy = CommandPolicy.DEFAULT.get()

    # Check policy type
    policy == CommandPolicy.DEFAULT  # True
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from xtl.common.compatibility import PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


__all__ = ['CommandPolicy', 'CommandPolicyType',
           'BaseCommandPolicy', 'DefaultCommandPolicy', 'NiceCommandPolicy']


class CommandPolicy(StrEnum):
    """
    Enumeration of supported command policies for compute sites.
    """

    DEFAULT = 'default'
    """The default command policy that does not modify commands."""
    NICE = 'nice'
    """A command policy that uses the ``nice`` command to set process priority 
    on Unix-like systems."""

    def get(self) -> CommandPolicyType:
        """
        Get an instance of the command policy corresponding to this enum member.
        """
        mapping = {
            CommandPolicy.DEFAULT: DefaultCommandPolicy,
            CommandPolicy.NICE: NiceCommandPolicy,
        }
        return mapping[self]()

    def __eq__(self, other):
        # Compare by type of the underlying command policy
        if isinstance(other, CommandPolicyType):
            return isinstance(other, self.get().__class__)
        elif isinstance(other, type):
            return other == self.get().__class__
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


@dataclass
class BaseCommandPolicy(ABC):
    name: str = field(init=False, repr=False)
    """The name of the command policy."""

    def __init__(self):
        # Concrete classes should have default values for all the fields. The __init__
        #  method is only defined here to avoid dataclass auto-generating one.
        pass

    @abstractmethod
    def intercept_preamble(self, preamble: str) -> str:
        """
        Intercepts the preamble of the batch file before any commands are added.

        :param preamble: The preamble string to be intercepted.
        :return: The modified preamble string.
        """
        ...

    @abstractmethod
    def intercept_command(self, command: str) -> str:
        """
        Intercepts a single command/line before it is added to the batch file.

        :param command: The command string to be intercepted.
        :return: The modified command string.
        """
        ...

    @abstractmethod
    def intercept_content(self, content: str) -> str:
        """
        Intercepts the entire content (*i.e.* all added commands) of the batch file
        before it is finalized.

        :param content: The content string to be intercepted.
        :return: The modified content string.
        """
        ...

    @abstractmethod
    def intercept_postamble(self, postamble: str) -> str:
        """
        Intercepts the postamble of the batch file after all commands have been added.

        :param postamble: The postamble string to be intercepted.
        :return: The modified postamble string.
        """
        ...


class DefaultCommandPolicy(BaseCommandPolicy):
    """
    A command policy that does not modify any part of the batch file contents. All
    ``intercept_*`` methods return their input unmodified.
    """

    name: str = 'default'

    def intercept_preamble(self, preamble: str) -> str:
        return preamble

    def intercept_command(self, command: str) -> str:
        return command

    def intercept_content(self, content: str) -> str:
        return content

    def intercept_postamble(self, postamble: str) -> str:
        return postamble


class NiceCommandPolicy(BaseCommandPolicy):
    """
    A command policy that modifies commands to be executed with the
    `nice <https://pubs.opengroup.org/onlinepubs/9799919799/utilities/nice.html>`_
    command to set their priority level. The niceness value can be configured via the
    ``nice_value`` attribute (default: 10).
    """

    name: str = 'nice'
    nice_value: int = 10
    """The niceness value to use with the ``nice`` command. This value should be between 
    -20 (highest priority) and 19 (lowest priority)."""

    def intercept_preamble(self, preamble: str) -> str:
        return preamble

    def intercept_command(self, command: str) -> str:
        """
        Pad commands with ``nice`` to set their priority level.

        :param command: The command string to be intercepted.
        :return: The modified command string.
        """
        return f'nice -n {self.nice_value} {command}'

    def intercept_content(self, content: str) -> str:
        return content

    def intercept_postamble(self, postamble: str) -> str:
        return postamble


CommandPolicyType = BaseCommandPolicy | DefaultCommandPolicy | NiceCommandPolicy
"""
A type alias for all supported command policy types.
"""
