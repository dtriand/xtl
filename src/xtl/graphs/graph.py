from __future__ import annotations

from typing import Optional

from pydantic import computed_field

from xtl.common.options import Option
from xtl.common.typed_vars import TypedList, TypedDict
from xtl.math.constants import MM_PER_INCH

from .base import BaseGraphModel
from .layouts import TLayout, GridLayout
from .traces import TTrace
from .panels import GraphPanel, GraphPanelLink
from .styles import GraphStyle


class GraphDimensions(BaseGraphModel):

    width: int = \
        Option(
            desc='The width of the graph (in pixels)',
            default=640
        )

    height: int = \
        Option(
            desc='The height of the graph (in pixels)',
            default=480
        )

    dpi: int = \
        Option(
            desc='The dots-per-inch (DPI) of the graph',
            default=100,
        )

    @classmethod
    def from_inches(cls, width: int, height: int, dpi: int = 100) -> GraphDimensions:
        return GraphDimensions(
            width=int(width * dpi),
            height=int(height * dpi),
            dpi=dpi
        )

    @classmethod
    def from_mm(cls, width: int, height: int, dpi: int = 100) -> GraphDimensions:
        return GraphDimensions(
            width=int(width / MM_PER_INCH * dpi),
            height=int(height / MM_PER_INCH * dpi),
            dpi=dpi
        )


class Graph(BaseGraphModel):

    title: Optional[str] = \
        Option(
            desc='The title of the graph',
            default=None,
        )

    dimensions: GraphDimensions = \
        Option(
            desc='The dimensions of the graph',
            default_factory=GraphDimensions
        )

    style: GraphStyle = \
        Option(
            desc='The style of the graph',
            default_factory=GraphStyle,
        )

    layout: TLayout = \
        Option(
            desc='The layout of each panel within the graph',
            default_factory=GridLayout
        )

    @property
    def panels(self) -> dict[str, GraphPanel]:
        return self.layout.panels

    @classmethod
    def simple(cls, *traces: TTrace) -> Graph:

        panel = GraphPanel(
            traces=traces,
        )

        graph = Graph(
            layout=GridLayout.from_panels(panel),
            panels={
                '0': panel
            }
        )

        return graph
