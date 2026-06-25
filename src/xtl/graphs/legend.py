from enum import Enum

from xtl.common.options import Option
from xtl.common.compatibility import PY310_OR_LESS
from .base import BaseGraphModel

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class LegendLocation(StrEnum):
    BEST = 'best'
    TOP_RIGHT = 'top_left'
    TOP_CENTER = 'top_center'
    TOP_LEFT = 'top_left'
    MIDDLE_RIGHT = 'middle_right'
    MIDDLE_CENTER = 'middle_center'
    MIDDLE_LEFT = 'middle_left'
    BOTTOM_RIGHT = 'bottom_right'
    BOTTOM_CENTER = 'bottom_center'
    BOTTOM_LEFT = 'bottom_left'


class LegendOptions(BaseGraphModel):

    show: bool = \
        Option(
            desc='Whether to display the legend',
            default=False
        )

    location: LegendLocation = \
        Option(
            desc='The location of the legend on the plot',
            default=LegendLocation.BEST
        )
