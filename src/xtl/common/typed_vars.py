from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence
from typing import get_args, Generic, TypeVar, overload

from pydantic import ValidationError, TypeAdapter
from pydantic_core import InitErrorDetails, PydanticCustomError, core_schema


T = TypeVar('T')


class TypedIterable(ABC, Generic[T]):

    _builtin_type: type
    _data: Iterable[T]

    def __init__(self, item_type: type[T], initial=None):
        self._data = self._builtin_type()
        self._item_type = item_type
        self._adapter = TypeAdapter(self._item_type)
        if initial:
            self.extend(initial)

    def _validate(self, value) -> T:
        try:
            return self._adapter.validate_python(value)
        except (ValidationError, TypeError) as e:
            raise ValidationError.from_exception_data(
                title=f'{self.__class__.__name__}<{self._item_type.__name__}>',
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError(
                            'value_error',
                            'Value error, Invalid data type for validation, ' + \
                            'expected {itemtype} got {valuetype} instead',
                            {'itemtype': self._item_type.__name__, 'valuetype': type(value)}
                        ),
                        loc=('item',),
                        input=value,
                    )
                ]
            ) from None

    def _serialize(self) -> MutableSequence[T]:
        result = self._builtin_type()
        for item in self:
            if isinstance(item, TypedIterable):
                result.append(item._serialize())
            else:
                result.append(item)
        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        (item_type,) = get_args(source_type)
        # Use list schema as the base
        list_schema = handler.generate_schema(list[item_type])

        def validate(value):
            return cls(item_type=item_type, initial=value)

        def serialize(instance):
            return instance._serialize()

        return core_schema.no_info_after_validator_function(
            validate,
            list_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                return_schema=list_schema,
            ),
        )

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item) -> bool:
        return item in self._data

    @abstractmethod
    def extend(self, values) -> None:
        ...

    def __repr__(self):
        # 'TypedList<ClassName>(values)'
        return f'{self.__class__.__name__}<{self._item_type.__name__}>({self._data})'


class TypedList(TypedIterable[T], MutableSequence[T]):

    _builtin_type = list
    _data: list[T]

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index):
        return self._data[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: int | slice, value) -> None:
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError(f'Expected an iterable for slice assignment, but got {type(value)}')
            self._data[index] = [self._validate(v) for v in value]
        else:
            self._data[index] = self._validate(value)

    def __delitem__(self, index) -> None:
        del self._data[index]

    def __contains__(self, item) -> bool:
        return item in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, TypedList):
            return self._data == other._data
        elif isinstance(other, list):
            return self._data == other
        else:
            raise NotImplementedError

    def insert(self, index, value) -> None:
        self._data.insert(index, self._validate(value))

    def append(self, value) -> None:
        self._data.append(self._validate(value))

    def extend(self, values) -> None:
        self._data.extend([self._validate(v) for v in values])

