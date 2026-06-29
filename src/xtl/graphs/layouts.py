from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import math
from typing import Annotated, Literal, Union

from pydantic import Field, computed_field, field_validator, model_validator
from typing_extensions import Self

from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.options import Option
from xtl.common.typed_vars import TypedList
from .base import BaseGraphModel
from .panels import GraphPanel, GraphPanelLink

if PY310_OR_LESS:
    class StrEnum(str, Enum): ...
else:
    from enum import StrEnum


class LayoutType(StrEnum):
    GRID = 'grid'


class Layout(ABC, BaseGraphModel):

    kind: LayoutType = \
        Option(
            desc='The layout type'
        )

    links: TypedList[GraphPanelLink] = \
        Option(
            desc='Groups of panels with linked/shared axes',
            default_factory=list,
        )

    @property
    @abstractmethod
    def panels(self) -> dict[str, GraphPanel]: ...

    @model_validator(mode='after')
    def _validate_links(self) -> Self:
        for l, link in enumerate(self.links):
            for panel_id in link.panel_ids:
                if panel_id not in self.panels.keys():
                    raise ValueError(
                        f'Panel {panel_id!r} in link {l + 1} is not included in the layout'
                    )
        return self


class GridCell(BaseGraphModel):

    idx: tuple[int, int] = \
        Option(
            desc='The index of the cell in the grid layout (column, row)'
        )

    span: tuple[int, int] = \
        Option(
            desc='The number of cells to span in the grid layout (col_span, row_span)',
            default=(1, 1)
        )

    panel: GraphPanel = \
        Option(
            desc='The panel within the current cell'
        )

    @field_validator('idx', mode='after')
    @classmethod
    def _validate_idx(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value < (0, 0):
            raise ValueError('Index must be non-negative')
        return value

    @field_validator('span', mode='after')
    @classmethod
    def _validate_span(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value < (1, 1):
            raise ValueError('Span must be positive')
        return value


class GridLayout(Layout):

    kind: Literal[LayoutType.GRID] = LayoutType.GRID

    rows: int = \
        Option(
            desc='Number of rows',
            ge=1, default=1
        )

    cols: int = \
        Option(
            desc='Number of columns',
            ge=1, default=1
        )

    cells: TypedList[GridCell] = \
        Option(
            desc='List of cells in the grid layout',
            default_factory=list
        )

    @model_validator(mode='after')
    def _validate_cells(self) -> Self:
        for cell in self.cells:
            if cell.idx[0] + cell.span[0] - 1 >= self.cols:
                raise ValueError(
                    f'Cell {cell.panel.id!r} exceeds grid width: idx={cell.idx}, span={cell.span}, cols={self.cols}'
                )
            if cell.idx[1] + cell.span[1] - 1 >= self.rows:
                raise ValueError(
                    f'Cell {cell.panel.id!r} exceeds grid height: idx={cell.idx}, span={cell.span}, rows={self.rows}'
                )
        return self

    @property
    def panels(self) -> dict[str, GraphPanel]:
        """
        The panels within the grid layout.

        :return: A dictionary mapping of panel IDs to GraphPanel instances
        """
        return {cell.panel.id: cell.panel for cell in self.cells}

    @classmethod
    def from_panels(cls, *panels: Union[GraphPanel, str]) -> GridLayout:
        """
        Create a square grid layout from a list of panels

        :param panels: Panels to include in the layout
        :return: A new GridLayout instance
        """
        n = len(panels)
        dim = math.ceil(math.sqrt(n))
        cells = TypedList(item_type=GridCell)
        idxs = [(x, y) for y in range(dim) for x in range(dim)]
        for i, panel in enumerate(panels):
            cells.append(
                GridCell(
                    idx=idxs[i],
                    panel=panel if isinstance(panel, GraphPanel) else GraphPanel(id=panel),
                )
            )

        return cls(
            rows=dim,
            cols=dim,
            cells=cells,
        )


TLayout = Annotated[
    Union[GridLayout],
    Field(discriminator='kind')
]
