from typing import NamedTuple
from dataclasses import dataclass, field

from xtl.exceptions import InvalidArgument


def _make_empty_list():
    return []


def _make_empty_list_with_empty_string():
    return ['']


@dataclass
class MapData:
    """
    Dataclass for map dictionary data located under ``phase['General']['Map']``.
    """
    maptype: str = field(default='')
    reflist: list = field(default_factory=_make_empty_list_with_empty_string)
    gridstep: float = field(default=0.25)
    showbonds: bool = field(default=True)
    rho: list = field(default_factory=_make_empty_list)
    rhomax: float = field(default=0.)
    mapsize: float = field(default=10.)
    cutoff: float = field(default=50.)
    flip: bool = field(default=False)

    @property
    def dictionary(self):
        return {
            'MapType': self.maptype,
            'RefList': self.reflist,
            'GridStep': self.gridstep,
            'Show bonds': self.showbonds,
            'rho': self.rho,
            'rhoMax': self.rhomax,
            'mapSize': self.mapsize,
            'cutOff': self.cutoff,
            'Flip': self.flip
        }


def get_map_type(map_name: str):
    class Map(NamedTuple):
        name: str
        name_pretty: str
        gsas_map_type: str

    mname = map_name.replace(' ', '').lower()
    if mname in ('fo', 'fobs'):
        return Map(name=map_name, name_pretty='Fo', gsas_map_type='Fobs')
    elif mname in ('fc', 'fcalc'):
        return Map(name=map_name, name_pretty='Fc', gsas_map_type='Fcalc')
    elif mname in ('fo-fc', 'fobs-fcalc', 'delta-f', 'delt-f', 'df'):
        return Map(name=map_name, name_pretty='Fo-Fc', gsas_map_type='delt-F')
    elif mname in ('2fo-fc', '2fobs-fcalc', '2*fo-fc', '2*fobs-fcalc'):
        return Map(name=map_name, name_pretty='2Fo-Fc', gsas_map_type='2*Fo-Fc')
    elif mname in ('patt', 'patterson', 'fo^2', 'fo2', 'fosq', 'fobs^2', 'fobs2'):
        return Map(name=map_name, name_pretty='Fo2', gsas_map_type='Patterson')
    else:
        raise InvalidArgument(raiser='map_name', message=f'{map_name}. Accepted values are: \n'
                                                         f' Fo: Fobs\n'
                                                         f' Fc: Fcalc\n'
                                                         f' Fo-Fc: Fobs - Fcalc\n'
                                                         f' 2Fo-Fc: 2 * Fobs - Fcalc\n'
                                                         f' Fo2: Patterson (Fobs^2)')
