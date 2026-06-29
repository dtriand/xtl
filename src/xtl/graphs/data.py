from typing import Literal, Any, Union, Optional, Annotated

from typing_extensions import Self

import numpy as np
from numpydantic import NDArray, Shape
from numpydantic.dtype import Number
from pydantic import field_validator, model_validator, Field, ValidationInfo

from .base import BaseGraphModel


class DataArray(BaseGraphModel):

    data: NDArray
    label: Optional[str] = None

    @field_validator('data', mode='before')
    @classmethod
    def _coerce_to_numpy(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            return np.array(value, dtype=float)
        return value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataArray):
            return False
        return np.allclose(self.data, other.data, equal_nan=True) and self.label == other.label


class Data1D(DataArray):
    data: NDArray[tuple[Literal['*']], Number]


class Data2D(DataArray):
    data: NDArray[tuple[Literal['*, *']], Number]


class DataSeries(BaseGraphModel):
    kind: str

    def arrays(self) -> dict[str, Data1D]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, Data1D)}

    def _check_same_shape(self, *names):
        arrays = self.arrays()
        shapes = {name: arrays[name].data.shape for name in names if name in arrays}
        if len(set(shapes.values())) > 1:
            raise ValueError(f'Arrays must have the same shape: {shapes!r}')


class XYData(DataSeries):
    kind: Literal['xy'] = 'xy'
    x: Data1D
    y: Data1D

    @field_validator('x', 'y', mode='before')
    @classmethod
    def _coerce_to_data_array(cls, value: Any) -> Any:
        if not isinstance(value, (Data1D, dict)):
            return Data1D(data=value)
        return value

    @model_validator(mode='after')
    def _check_shapes(self) -> Self:
        self._check_same_shape('x', 'y')
        return self


class XYEData(DataSeries):
    kind: Literal['xye'] = 'xye'
    x: Data1D
    y: Data1D
    e: Data1D

    @field_validator('x', 'y', 'e', mode='before')
    @classmethod
    def _coerce_to_data_array(cls, value: Any) -> Any:
        if not isinstance(value, (Data1D, dict)):
            return Data1D(data=value)
        return value

    @model_validator(mode='after')
    def _check_shapes(self) -> Self:
        self._check_same_shape('x', 'y', 'e')
        return self


class XYZData(DataSeries):
    kind: Literal['xyz'] = 'xyz'
    x: Data1D
    y: Data1D
    z: Data1D

    @field_validator('x', 'y', 'z', mode='before')
    @classmethod
    def _coerce_to_data_array(cls, value: Any) -> Any:
        if not isinstance(value, (Data1D, dict)):
            return Data1D(data=value)
        return value

    @model_validator(mode='after')
    def _check_shapes(self) -> Self:
        self._check_same_shape('x', 'y', 'z')
        return self


class MeshXYData(DataSeries):
    kind: Literal['mesh'] = 'mesh_xy'
    x: Data1D
    y: Data1D
    z: Data2D

    @field_validator('x', 'y', 'z', mode='before')
    @classmethod
    def _coerce_to_data_array(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, (Data1D, Data2D, dict)):
            if info.field_name == 'z':
                return Data2D(data=value)
            else:
                return Data1D(data=value)
        return value

    @model_validator(mode='after')
    def _check_shapes(self) -> Self:
        nx, ny = self.x.data.shape[0], self.y.data.shape[0]
        if shape := self.z.data.shape != (nx, ny):
            raise ValueError(f'z.shape must be ({nx}, {ny}), got {shape}')
        return self


TDataSeries = Annotated[
    Union[XYData, XYEData, XYZData, MeshXYData],
    Field(discriminator='kind')
]
