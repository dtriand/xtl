from typing import Optional, Union
from typing_extensions import Self

from numpydantic import NDArray, Shape
from numpydantic.dtype import Number
from pydantic import BaseModel, ConfigDict, model_validator

from xtl.common.labels import Label


TData = Union['Data0D', 'Data1D', 'Data2D', 'Data3D', 'DataGrid2D']


class Data0D(BaseModel):
    """
    A class to represent 0D data, i.e. one series of values.
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    data: NDArray[Shape['*'], Number]
    label: Label
    units: Optional[Label] = None


class Data1D(BaseModel):
    """
    A class to represent 1D data, i.e. y = f(x).
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    x: Data0D
    y: Data0D
    
    @model_validator(mode='after')
    def check_array_shapes(self) -> Self:
        if self.x.data.shape != self.y.data.shape:
            raise ValueError('x and y shapes must be equal')
        return self


class Data2D(BaseModel):
    """
    A class to represent 2D data, i.e. z = f(x, y).

    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    x: Data0D
    y: Data0D
    z: Data0D

    @model_validator(mode='after')
    def check_array_shapes(self) -> Self:
        if self.x.data.shape != self.y.data.shape:
            raise ValueError('x and y shapes must be equal')
        if self.x.data.shape != self.z.data.shape:
            raise ValueError('x and z shapes must be equal')
        return self


class Data3D(BaseModel):
    """
    A class to represent 3D data, i.e. w = f(x, y, z).
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    x: Data0D
    y: Data0D
    z: Data0D
    w: Data0D

    @model_validator(mode='after')
    def check_array_shapes(self) -> Self:
        if self.x.data.shape != self.y.data.shape:
            raise ValueError('x and y shapes must be equal')
        if self.x.data.shape != self.z.data.shape:
            raise ValueError('x and z shapes must be equal')
        if self.x.data.shape != self.w.data.shape:
            raise ValueError('x and w shapes must be equal')
        return self


class DataGrid2D(BaseModel):
    """
    A class to represent 2D data in a grid format, i.e. z = f(x, y), where x and y are
    series of coordinates. Useful for calculating 2D functions on a grid.
    """
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    x: Data0D
    y: Data0D
    z: NDArray[Shape['*, *'], Number]

    @model_validator(mode='after')
    def check_array_shapes(self) -> Self:
        if self.x.data.shape[0] != self.z.shape[0]:
            raise ValueError('x and z shapes must be equal')
        if self.y.data.shape[0] != self.z.shape[1]:
            raise ValueError('y and z shapes must be equal')
        return self

