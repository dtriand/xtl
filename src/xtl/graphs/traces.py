from enum import Enum
from typing import Union, Annotated, Literal

from pydantic import Field

from xtl.common.options import Option
from xtl.common.compatibility import PY310_OR_LESS
from .base import BaseGraphModel, random_uid
from .data import TDataSeries

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class TraceType(StrEnum):
    LINE = 'line'
    SCATTER = 'scatter'
    HEATMAP = 'heatmap'


class Trace(BaseGraphModel):

    id: str = \
        Option(
            desc='A unique identifier for the trace',
            default_factory=random_uid
        )

    kind: TraceType = \
        Option(
            desc='The type of trace (e.g. line, scatter)'
        )

    data: TDataSeries = \
        Option(
            desc='The data series to plot'
        )


class LineTrace(Trace):
    kind: Literal[TraceType.LINE] = TraceType.LINE


class ScatterTrace(Trace):
    kind: Literal[TraceType.SCATTER] = TraceType.SCATTER


class HeatmapTrace(Trace):
    kind: Literal[TraceType.HEATMAP] = TraceType.HEATMAP


TTrace = Annotated[
    Union[LineTrace, ScatterTrace, HeatmapTrace],
    Field(discriminator='kind')
]