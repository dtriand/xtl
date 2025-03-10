from enum import Enum
from types import SimpleNamespace
from typing import Any, Optional

import pydantic
from pydantic import BaseModel
import pydantic_core
from pydantic_core import core_schema


from xtl.common.data import enum_validator


class AxisScaleType(Enum):
    LINEAR = 'linear'
    LOG = 'log'
    SYMLOG = 'symlog'
    LOGIT = 'logit'


class AxisScale(BaseModel):
    type: AxisScaleType | str = AxisScaleType.LINEAR

    _validate_type = enum_validator('type', AxisScaleType)


class AxisRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class AxisConfig(BaseModel):
    scale: AxisScale = AxisScale(type=AxisScaleType.LINEAR)
    range: AxisRange = AxisRange(min=None, max=None)

    # ToDo: Get label from data


class AxisConfigCollection(SimpleNamespace):

    def __setattr__(self, key: str, value: AxisConfig):
        if not isinstance(value, AxisConfig):
            raise TypeError(f'Value {value!r} is not an instance of {AxisConfig.__name__}')
        super().__setattr__(key, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler) \
            -> pydantic_core.CoreSchema:

        def serialize(value: Any) -> str:
            if isinstance(value, cls):
                d = {}
                for k, v in value.__dict__.items():
                    d[k] = v
                return d
            raise ValueError(f'Cannot serialize {value!r} as {cls.__name__}')

        def validate(value: Any):
            if isinstance(value, cls):
                for i, (k, v) in enumerate(value.__dict__.items()):
                    if not isinstance(v, AxisConfig):
                        raise TypeError(f'Cannot validate {value!r} as {cls.__name__} at index {i} with key {k!r}')
            raise TypeError(f'Cannot validate {value!r} as {cls.__name__}')

        schema = core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.str_schema(),
        ])

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate, schema, serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used='always'
            )
        )