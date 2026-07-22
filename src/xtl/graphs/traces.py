from __future__ import annotations

from enum import Enum
from typing import Any, Union, Annotated, Literal, Callable

from pydantic import Field

from xtl.common.options import Option
from xtl.common.compatibility import PY310_OR_LESS
from .base import BaseGraphModel, random_uid
from .data import TDataSeries
from .styles import TraceStyle, LineTraceStyle, PointStyle, LineStyle, LineStyleType, PointStyleType

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

    style: TraceStyle = \
        Option(
            desc='The style of the trace',
            default_factory=TraceStyle
        )


class LineTrace(Trace):
    kind: Literal[TraceType.LINE] = TraceType.LINE

    style: LineTraceStyle = Option(
        desc='The style of the trace',
        default_factory=lambda: LineTraceStyle(
            points=PointStyle(
                style=PointStyleType.NONE
            )
        )
    )


class ScatterTrace(Trace):
    kind: Literal[TraceType.SCATTER] = TraceType.SCATTER

    style: LineTraceStyle = Option(
        desc='The style of the trace',
        default_factory=lambda: LineTraceStyle(
            points=PointStyle(
                style=PointStyleType.CIRCLE
            )
        )
    )


class HeatmapTrace(Trace):
    kind: Literal[TraceType.HEATMAP] = TraceType.HEATMAP


TTrace = Annotated[
    Union[LineTrace, ScatterTrace, HeatmapTrace],
    Field(discriminator='kind')
]