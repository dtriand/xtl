from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Label(BaseModel):
    __pydantic_config__ = ConfigDict(validate_assignment=True, extra='forbid')

    value: str
    desc: Optional[str] = Field(default=None, repr=False)
    repr: Optional[str] = Field(default=None, repr=False)
    latex: Optional[str] = Field(default=None, repr=False)
