from enum import Enum
import platform

from xtl.common.compatibility import PY310_OR_LESS
from .sites import LocalSite, BiotixHPC
from .sites import ComputeSite as _ComputeSite

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


# Get the current hostname
_hostname = platform.node()


class ComputeSite(StrEnum):
    # General purpose compute sites
    LOCAL = 'local'
    # MODULES = 'modules'  # TODO: Enable when ModuleSite is implemented

    # Specialized compute sites (available only on specific hosts)
    if _hostname.startswith('biotix'):
        BIOTIX = 'biotix'

    def get_site(self) -> _ComputeSite:
        """
        Returns the :class:``xtl.automate.sites.ComputeSite`` object for the current
        site.

        :return: An instance of a subclass of :class:``xtl.automate.sites.ComputeSite``
        :raises ValueError: If the compute site is unknown or not implemented.
        """
        # NB: Comparison with string literals to avoid issues with conditional members
        if self == 'local':
            return LocalSite()
        elif self == 'biotix':
            return BiotixHPC()
        else:
            raise ValueError(f'Unknown compute site: {self!r}')
