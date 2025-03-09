from dataclasses import dataclass
from typing import Optional


@dataclass
class Label:
    value: str
    description: Optional[str] = None
    repr: Optional[str] = None
    latex: Optional[str] = None