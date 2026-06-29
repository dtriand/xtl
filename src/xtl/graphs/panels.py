from typing import Optional

from xtl.common.options import Option
from xtl.common.typed_vars import TypedList
from .axes import TPanelAxes, CartesianAxes, AxisName
from .base import BaseGraphModel, random_uid
from .legend import LegendOptions, LegendLocation
from .traces import TTrace


class GraphPanel(BaseGraphModel):

    id: str = \
        Option(
            desc='A unique identifier for the panel',
            default_factory=random_uid
        )

    title: Optional[str] = \
        Option(
            desc='The title of the panel',
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

    traces: TypedList[TTrace] = \
        Option(
            desc='List of traces to plot within the panel',
            default_factory=list
        )


class GraphPanelLink(BaseGraphModel):

    panel_ids: TypedList[str] = \
        Option(
            desc='List of panel IDs to link together',
            min_length=2
        )

    axes: TypedList[AxisName] = \
        Option(
            desc='List of linked axes',
            min_length=1
        )
