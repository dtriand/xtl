from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeVar

from xtl.common.compatibility import PY310_OR_LESS

if PY310_OR_LESS:
    class StrEnum(str, Enum):
        pass
else:
    from enum import StrEnum


@dataclass(frozen=True)
class UnitsDescription:
    """
    Description of a unit, including various string representations.
    """
    name: str
    """The canonical name of the unit, used for lookups and as a default for other representations."""

    desc: str = ''
    """The human-readable description of the unit, used for documentation and tooltips."""

    repr: str = ''
    """An ASCII-safe representation of the unit"""

    pretty: str = ''
    """A prettified representation of the unit, using Unicode characters where applicable"""

    latex: str = ''
    """A LaTeX representation of the unit"""

    aliases: tuple[str, ...] = field(default_factory=tuple)
    """A list of aliases for this unit"""


_U = TypeVar('_U', bound='Units')


class Units(StrEnum):

    def __new__(cls, value: str, meta: Optional[UnitsDescription] = None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        if meta is not None and not isinstance(meta, UnitsDescription):
            raise TypeError(f'`meta` must be a {UnitsDescription.__name__}, not {meta.__class__.__name__}')
        obj._meta = meta or UnitsDescription(name=value)
        return obj

    def __setattr__(self, key, value):
        if key == '_meta' and hasattr(self, '_meta'):
            raise AttributeError(f'Cannot reassign {UnitsDescription.__name__} on an enum member')
        super().__setattr__(key, value)

    @property
    def name(self) -> str:
        """
        The name of the unit
        :return:
        """
        return self._meta.name

    @property
    def desc(self) -> str:
        """
        A human-readable description of the unit
        :return:
        """
        return self._meta.desc

    @property
    def repr(self) -> str:
        """
        An ASCII-safe representation of the unit
        :return:
        """
        return self._meta.repr or self._meta.name

    @property
    def pretty(self) -> str:
        """
        A pretty representation of the unit, including special Unicode characters
        :return:
        """
        return self._meta.pretty or self._meta.name

    @property
    def latex(self) -> str:
        """
        A LaTeX representation of the unit
        :return:
        """
        return self._meta.latex or self._meta.name

    @property
    def aliases(self) -> tuple[str, ...]:
        """
        A list of aliases for this unit
        :return:
        """
        return self._meta.aliases

    @classmethod
    def by_alias(cls: type[_U], alias: str) -> Optional[_U]:
        """
        Get a unit by its name or one of its aliases.

        :param alias: The name or alias of the unit to look up
        :return: The corresponding unit enum member, or None if no match is found
        """
        alias = alias.lower()
        for member in cls:
            if alias in member.aliases or alias == member.name:
                return member
        return None
