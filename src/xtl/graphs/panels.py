from typing import Optional

from xtl.common.options import Option
from xtl.common.typed_vars import TypedList
from .axes import TPanelAxes, CartesianAxes
from .base import BaseGraphModel, random_uid
from .legend import LegendOptions, LegendLocation
from .traces import TTrace


class GraphPanel(BaseGraphModel):

    panel_id: str = \
        Option(
            desc='A unique identifier for the panel',
            default_factory=random_uid
        )

    traces: TypedList[TTrace] = \
        Option(
            desc='List of traces to plot within the graph',
            default_factory=list
        )

    title: Optional[str] = \
        Option(
            desc='The title of the graph',
            default=None,
        )

    axes: TPanelAxes = \
        Option(
            desc='The axes configuration',
            default_factory=CartesianAxes
        )

    legend: LegendOptions = \
        Option(
            desc='Legend configuration',
            default_factory=LegendOptions
        )
