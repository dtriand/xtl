from enum import Enum
from typing import Union, Annotated, Literal

from pydantic import Field

from xtl.common.options import Option
from xtl.common.compatibility import PY310_OR_LESS
from .base import BaseGraphModel

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class AxisScaleType(StrEnum):
    LINEAR = 'linear'
    LOG = 'log'
    SYMLOG = 'symlog'


class AxisScale(BaseGraphModel):

    kind: AxisScaleType = \
        Option(
            desc='The axis scale type'
        )


class LinearScale(AxisScale):

    kind: Literal[AxisScaleType.LINEAR] = AxisScaleType.LINEAR


class LogScale(AxisScale):

    kind: Literal[AxisScaleType.LOG] = AxisScaleType.LOG


class SymLogScale(AxisScale):

    kind: Literal[AxisScaleType.SYMLOG] = AxisScaleType.SYMLOG
    threshold: float = \
        Option(
            desc='The linear threshold for the symlog scale',
            default=1e-3,
        )


TAxisScale = Annotated[
    Union[LinearScale, LogScale, SymLogScale],
    Field(discriminator='kind')
]


class AxisLimits(BaseGraphModel):

    min: float | None = \
        Option(
            desc='The minimum limit of the axis',
            default=None,
        )

    max: float | None = \
        Option(
            desc='The maximum limit of the axis',
            default=None,
        )


class AxisOptions(BaseGraphModel):

    label: str | None = \
        Option(
            desc='The axis label',
            default=None,
        )

    scale: TAxisScale = \
        Option(
            desc='The axis scale',
            default_factory=LinearScale,
        )

    limits: AxisLimits = \
        Option(
            desc='The axis limits',
            default_factory=AxisLimits,
        )


class AxesType(StrEnum):
    CARTESIAN = 'cartesian'
    POLAR = 'polar'


class PanelAxes(BaseGraphModel):

    kind: AxesType = \
        Option(
            desc='The kind of axes in the plot'
        )


class CartesianAxes(PanelAxes):

    kind: Literal[AxesType.CARTESIAN] = AxesType.CARTESIAN

    x: AxisOptions = \
        Option(
            desc='Configuration for the x axis (horizontal)',
            default_factory=AxisOptions,
        )

    y: AxisOptions = \
        Option(
            desc='Configuration for the y axis (vertical)',
            default_factory=AxisOptions,
        )


class PolarAxes(PanelAxes):

    kind: Literal[AxesType.POLAR] = AxesType.POLAR

    r: AxisOptions = \
        Option(
            desc='Configuration for the radial axis',
            default_factory=AxisOptions,
        )

    theta: AxisOptions = \
        Option(
            desc='Configuration for the angular (theta) axis',
            default_factory=AxisOptions,
        )


TPanelAxes = Annotated[
    Union[CartesianAxes, PolarAxes],
    Field(discriminator='kind')
]
