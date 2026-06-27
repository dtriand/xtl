from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence
from typing import Any, Generic, TypeVar, Protocol, get_args, overload, runtime_checkable

from pydantic import ValidationError, TypeAdapter
from pydantic_core import InitErrorDetails, PydanticCustomError, core_schema

from xtl.exceptions.pydantic import OptionsValidationError


T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')


@runtime_checkable
class TypedMutable(Protocol):

    def _validate(self, value: Any) -> bool: ...

    def _serialize(self) -> Any: ...

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, TypedMutable):
            return value._serialize()
        return value


def _type_name(tp: Any) -> str:
    return getattr(tp, '__name__', repr(tp))


class TypedIterable(ABC, TypedMutable, Generic[T]):

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
        except (ValidationError, TypeError):
            exc = ValidationError.from_exception_data(
                title=f'{self.__class__.__name__}<{_type_name(self._item_type)}>',
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError(
                            'value_error',
                            'Value error, Invalid data type for validation, ' + \
                            'expected {itemtype} got {valuetype} instead',
                            {'itemtype': _type_name(self._item_type), 'valuetype': _type_name(type(value))}
                        ),
                        loc=('item',),
                        input=value,
                    )
                ]
            )
            raise OptionsValidationError(exc, type(self).__name__) from exc

    def _serialize(self) -> MutableSequence[T]:
        result = self._builtin_type()
        for item in self:
            result.append(self._serialize_value(item))
        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        (item_type, ) = get_args(source_type)
        list_schema = handler.generate_schema(list[item_type])

        def validate(value: TypedIterable[T]):
            return cls(item_type=item_type, initial=value)

        def serialize(instance: TypedIterable[T]):
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
    def extend(self, values) -> None: ...

    def __repr__(self):
        return f'{self.__class__.__name__}<{_type_name(self._item_type)}>({self._data})'


class TypedMapping(ABC, TypedMutable, Generic[KT, VT]):

    _builtin_type: type
    _data: Mapping[KT, VT]

    def __init__(self, key_type: type[KT], value_type: type[VT], initial=None):
        self._data = self._builtin_type()
        if not hasattr(key_type, '__hash__'):
            raise TypeError(f'Key type {key_type} is not hashable, cannot be used as a key in '
                            f'{type(self).__name__}')
        self._key_type = key_type
        self._value_type = value_type
        self._key_adapter = TypeAdapter(self._key_type)
        self._value_adapter = TypeAdapter(self._value_type)
        if initial is not None:
            self.update(initial)

    def _validate_key(self, value) -> KT:
        try:
            return self._key_adapter.validate_python(value)
        except (ValidationError, TypeError):
            from xtl.exceptions.pydantic import OptionsValidationError

            exc = ValidationError.from_exception_data(
                title=f'{self.__class__.__name__}<{_type_name(self._key_type)}, {_type_name(self._value_type)}>',
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError(
                            'value_error',
                            'Value error, Invalid data type for validation, ' + \
                            'expected {keytype} got {valuetype} instead',
                            {'keytype': _type_name(self._key_type), 'valuetype': _type_name(type(value))}
                        ),
                        loc=('key',),
                        input=value,
                    )
                ]
            )

            raise OptionsValidationError(exc, type(self).__name__) from exc

    def _validate_value(self, value) -> VT:
        try:
            return self._value_adapter.validate_python(value)
        except (ValidationError, TypeError):
            from xtl.exceptions.pydantic import OptionsValidationError

            exc = ValidationError.from_exception_data(
                title=f'{self.__class__.__name__}<{_type_name(self._key_type)}, {_type_name(self._value_type)}>',
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError(
                            'value_error',
                            'Value error, Invalid data type for validation, ' + \
                            'expected {valuetype} got {inputtype} instead',
                            {'valuetype': _type_name(self._value_type), 'inputtype': _type_name(type(value))}
                        ),
                        loc=('value',),
                        input=value,
                    )
                ]
            )
            raise OptionsValidationError(exc, type(self).__name__) from exc

    def _serialize(self) -> dict[KT, VT]:
        result = self._builtin_type()
        for key, value in self._data.items():
            result[self._serialize_value(key)] = self._serialize_value(value)
        return result

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        key_type, value_type = get_args(source_type)
        dict_schema = handler.generate_schema(dict[key_type, value_type])

        def validate(value: TypedMapping[KT, VT]):
            return cls(key_type=key_type, value_type=value_type, initial=value)

        def serialize(instance: TypedMapping[KT, VT]):
            return instance._serialize()

        return core_schema.no_info_after_validator_function(
            validate,
            dict_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                return_schema=dict_schema,
            ),
        )

    @abstractmethod
    def update(self, values: Mapping[KT, VT]) -> None: ...

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'<{_type_name(self._key_type)}, {_type_name(self._value_type)}>'
            f'({self._data})'
        )


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


class TypedDict(TypedMapping[KT, VT], MutableMapping[KT, VT]):

    _builtin_type = dict
    _data: dict[KT, VT]

    def __getitem__(self, key: KT) -> VT:
        return self._data[key]

    def __setitem__(self, key: KT, value: VT) -> None:
        self._data[self._validate_key(key)] = self._validate_value(value)

    def __delitem__(self, key: KT) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: KT) -> bool:
        return key in self._data

    def __eq__(self, other):
        if isinstance(other, TypedMapping):
            return self._data == other._data
        elif isinstance(other, Mapping):
            return self._data == dict(other)
        else:
            raise NotImplementedError

    @staticmethod
    def _iter_items(values):
        if isinstance(values, MutableMapping):
            return values.items()
        return values

    def update(self, values) -> None:
        for key, value in self._iter_items(values):
            self[key] = value
