from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar, Any, Callable
import warnings

from ..traces import Trace, TraceType
from ..data import TDataSeries
from ..graph import Graph


_Unset = object()


class GraphEnumMapper:

    def __init__(self, table: dict[type[Enum], dict[Enum, Any]] | None = None, warn: bool = True) -> None:
        self._table = table or dict()
        self._warn = bool(warn)

    def convert(self, value: Enum, default: Any = _Unset) -> Any:
        if value is None:
            # Handle optional values in models
            if default is not _Unset:
                return default
            raise TypeError(f'Not enum type: {value!r}')

        enum_type = type(value)
        if not issubclass(enum_type, Enum):
            if default is not _Unset:
                if self._warn:
                    warnings.warn(f'Falling back to default value for non-enum type: {enum_type.__name__!r}')
                return default
            raise TypeError(f'Not enum type: {enum_type.__name__!r}')

        mapping = self._table.get(enum_type, None)
        if mapping is None:
            if default is not _Unset:
                if self._warn:
                    warnings.warn(f'Falling back to default value for unsupported enum type: {enum_type.__name__!r}')
                return default
            raise TypeError(f'Unsupported enum type: {enum_type.__name__!r}')

        if value not in mapping:
            if default is not _Unset:
                if self._warn:
                    warnings.warn(f'Falling back to default value for unsupported enum value: '
                                  f'{enum_type.__name__}.{value.name!r}')
                return default
            raise ValueError(f'Unsupported enum value: {enum_type.__name__}.{value.name!r}')

        return mapping[value]


@dataclass(frozen=True)
class RenderContext:
    mapper: GraphEnumMapper = field(default_factory=GraphEnumMapper)

    def map(self, value: Enum, default: Any = None) -> Any:
        return self.mapper.convert(value, default)


class GraphRenderable(ABC):

    def __init__(self, graph: Graph, *, ctx: RenderContext | None = None) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f'`graph` must be a {Graph.__name__} instance, got {type(graph).__name__!r}')
        if ctx is not None:
            if not isinstance(ctx, RenderContext):
                raise TypeError(f'`ctx` must be a {RenderContext.__name__} instance, got {type(ctx).__name__!r}')

        self._graph = graph
        self._ctx = ctx or RenderContext()

    @abstractmethod
    def display(self) -> None: ...

    @abstractmethod
    def save(self, filename: str | Path) -> None: ...


R = TypeVar('R')

class GraphBackend(ABC, Generic[R]):

    @abstractmethod
    def render(self, graph: Graph) -> R: ...


