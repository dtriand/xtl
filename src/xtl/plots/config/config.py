from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xtl.common.labels import LabelFmt
from xtl.common.data import enum_validator
from xtl.plots.config.axis import AxisConfigCollection


class GraphConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    # Attributes
    axes: AxisConfigCollection | dict = Field(default_factory=AxisConfigCollection)
    label_fmt: LabelFmt | str = LabelFmt.VALUE

    # Validators
    _validate_label_fmt = enum_validator('label_fmt', LabelFmt)

    @field_validator('axes')
    @classmethod
    def _validate_axes(cls, v: Any) -> AxisConfigCollection:
        if isinstance(v, dict):
            return AxisConfigCollection(**v)
        return v
