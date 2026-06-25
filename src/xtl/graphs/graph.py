from xtl.common.options import Option

from .base import BaseGraphModel
from .panels import GraphPanel


class Graph(BaseGraphModel):

    # layout: Layout = \
    #     Option(
    #         desc='The layout of each panel within the graph',
    #     )

    panels: dict[str, GraphPanel] = \
        Option(
            desc='A collection of panels, each with its own traces',
            default_factory=dict,
        )

    # links: list[GraphPanelLink] = \
    #     Option(
    #         desc='',
    #         default_factory=list,
    #     )
    #
    # style: GraphStyle = \
    #     Option(
    #         desc='The style of the graph',
    #         default_factory=GraphStyle,
    #     )
