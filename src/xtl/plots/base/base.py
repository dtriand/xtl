from typing import Any, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, validate_call

from xtl.common.data import TData, Data0D, Data1D
from xtl.common.labels import Label, LabelFmt


class GraphConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    label_fmt: LabelFmt = LabelFmt.VALUE


class GraphBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True,
                              extra='forbid')

    data: TData
    config: GraphConfig = Field(default_factory=GraphConfig)

    # Fix: Should axis & figure be excluded from the model serialization?
    axis: Optional[Axes] = Field(default_factory=lambda: plt.gca(), alias='ax')
    figure: Optional[Figure] = Field(default_factory=lambda: plt.gcf(), alias='fig')

    title: Optional[str] = None

    def get_label(self, data: Data0D) -> Optional[str]:
        value = getattr(data.label, self.config.label_fmt.value)
        units = getattr(data.units, self.config.label_fmt.value)

        v = value or data.label.value
        u = units or data.units.value
        if self.config.label_fmt in [LabelFmt.VALUE, LabelFmt.LATEX]:
            if u:
                return f'{v} ({u})'
            return v
        elif self.config.label_fmt == LabelFmt.REPR:
            if u:
                return f'{v} [{u}]'
            return v
        return None

    def plot(self):
        raise NotImplementedError


class Graph1D(GraphBase):
    data: Data1D

    @property
    def xlabel(self) -> str:
        return self.get_label(self.data.x)

    @property
    def ylabel(self) -> str:
        return self.get_label(self.data.y)

    @validate_call
    def plot(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
        if xlabel:
            self.data.x.label = Label(value=xlabel)
        if ylabel:
            self.data.y.label = Label(value=ylabel)

        self.axis.plot(self.data.x.data, self.data.y.data)

        self.axis.set_xlabel(self.xlabel)
        self.axis.set_ylabel(self.ylabel)
