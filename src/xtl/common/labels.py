from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LabelFmt(Enum):
    VALUE = 'value'
    DESC = 'desc'
    REPR = 'repr'
    LATEX = 'latex'


class Label(BaseModel):
    __pydantic_config__ = ConfigDict(validate_assignment=True, extra='forbid')

    value: str
    desc: Optional[str] = Field(default=None, repr=False)
    repr: Optional[str] = Field(default=None, repr=False)  # Todo: rename to ascii
    latex: Optional[str] = Field(default=None, repr=False)

    def __format__(self, format_spec):
        if format_spec in ['value', LabelFmt.VALUE]:
            return self.value
        elif format_spec in ['desc', LabelFmt.DESC]:
            return self.desc or self.value
        elif format_spec in ['repr', LabelFmt.REPR]:
            return self.repr or self.value
        elif format_spec in ['latex', LabelFmt.LATEX]:
            return self.latex or self.value
        return super().__format__(format_spec)

