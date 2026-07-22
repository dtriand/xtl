from enum import Enum

from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.options import Option
from .colors import Color, COLORS, Colormap

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


from .base import BaseGraphModel


class GraphStyle(BaseGraphModel):
    ...


class LineStyleType(StrEnum):
    NONE = 'none'
    SOLID = 'solid'
    DASHED = 'dashed'
    DOTTED = 'dotted'
    DASHDOTTED = 'dashdotted'


class PointStyleType(StrEnum):
    NONE = 'none'
    CIRCLE = 'circle'
    SQUARE = 'square'
    DIAMOND = 'diamond'
    TRIANGLE_UP = 'triangle_up'
    TRIANGLE_DOWN = 'triangle_down'
    TRIANGLE_LEFT = 'triangle_left'
    TRIANGLE_RIGHT = 'triangle_right'
    PENTAGON = 'pentagon'
    STAR = 'star'
    HEXAGON = 'hexagon'
    HEXAGON_UP = 'hexagon_up'
    OCTAGON = 'octagon'
    CROSS = 'cross'
    CROSS_SIDE = 'cross_side'
    TICK_VERTICAL = 'tick_vertical'
    TICK_HORIZONTAL = 'tick_horizontal'


class TraceStyle(BaseGraphModel):
    ...


class LineStyle(BaseGraphModel):

    style: LineStyleType = \
        Option(
            desc='The style of the line',
            default=LineStyleType.SOLID,
        )

    color: Color = \
        Option(
            desc='The color of the line',
            default=COLORS['default'][0]
        )

    width: float = \
        Option(
            desc='The width of the line (in pts)',
            default=1.5
        )


class PointStyle(BaseGraphModel):

    style: PointStyleType = \
        Option(
            desc='The style of the points',
            default=PointStyleType.CIRCLE
        )

    color: Color = \
        Option(
            desc='The color of the points',
            default=COLORS['default'][0]
        )

    colormap: Colormap | None = \
        Option(
            desc='The colormap of the points',
            default=None
        )


class LineTraceStyle(TraceStyle):

    line: LineStyle = \
        Option(
            desc='The style of the line',
            default_factory=LineStyle
        )

    points: PointStyle = \
        Option(
            desc='The style of the points',
            default_factory=PointStyle
        )


# class ScatterTraceStyle(TraceStyle):
#
#     points: PointStyle = \
#         Option(
#             desc='The style of the points',
#             default_factory=PointStyle
#         )
