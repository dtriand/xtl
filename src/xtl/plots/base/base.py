from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, validate_call

from xtl.common.data import TData, Data0D, Data1D
from xtl.common.labels import Label, LabelFmt
from xtl.plots.config import GraphConfig
from xtl.plots.config.axis import AxisConfig


class GraphBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True,
                              extra='forbid')

    data: TData
    config: GraphConfig = Field(default_factory=GraphConfig)

    # Fix: Should axis & figure be excluded from the model serialization?
    axis: Optional[Axes] = Field(default_factory=lambda: plt.gca(), alias='ax')
    figure: Optional[Figure] = Field(default_factory=lambda: plt.gcf(), alias='fig')

    title: Optional[str] = None

    def _get_label(self, data: Data0D) -> Optional[str]:
        """
        Returns a string in the form of `quantity (units)` for the given data. The
        format is determined by the `label_fmt` attribute of the `config` attribute.
        """
        quant = getattr(data.label, self.config.label_fmt.value, None)
        units = getattr(data.units, self.config.label_fmt.value, None)

        q = quant or getattr(data.label, 'value', None)
        u = units or getattr(data.units, 'value', None)
        if self.config.label_fmt in [LabelFmt.VALUE, LabelFmt.LATEX]:
            if u:
                return f'{q} ({u})'
            return q
        elif self.config.label_fmt == LabelFmt.REPR:
            if u:
                return f'{q} [{u}]'
            return q
        return None

    def plot(self):
        raise NotImplementedError


class Graph1D(GraphBase):
    data: Data1D

    config_x: Optional[AxisConfig] = Field(default_factory=AxisConfig, repr=False, exclude=True)
    config_y: Optional[AxisConfig] = Field(default_factory=AxisConfig, repr=False, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config.axes.x = self.config_x
        self.config.axes.y = self.config_y

    @property
    def label_x(self) -> str:
        return self._get_label(self.data.x)

    @property
    def label_y(self) -> str:
        return self._get_label(self.data.y)

    @validate_call
    def plot(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
        if xlabel:
            self.data.x.label = Label(value=xlabel)
        if ylabel:
            self.data.y.label = Label(value=ylabel)

        self.axis.plot(self.data.x.data, self.data.y.data)

        self.axis.set_title(self.title)
        self.axis.set_xlabel(self.label_x)
        self.axis.set_ylabel(self.label_y)
