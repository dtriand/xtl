"""
This modules defines custom validators to be used with `Pydantic` models.
"""

__all__ = [
    'LengthValidator',
    'ChoicesValidator',
    'PathExistsValidator',
    'PathIsFileValidator',
    'PathIsDirValidator',
    'PathIsAbsoluteValidator',
    'CastAsValidator',
    'CastAsNoneIfEmpty',
    'CastAsPathOrNone',
]

from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pydantic import AfterValidator, BeforeValidator


# Custom validator functions
def validate_length(value: Any, length: int) -> Any:
    """
    Check if the length of the value is equal to the provided length.
    """
    if isinstance(value, Iterable):
        if len(value) != length:
            raise ValueError(f'Expected length {length}, got {len(value)}') from None
    return value


def validate_choices(value: Any, choices: Iterable[Any]) -> Any:
    """
    Check if the value is contained in the provided choices.
    """
    if value not in choices:
        raise ValueError(f'Value is not in choices: {choices!r}') from None
    return value


def validate_path_exists(value: None | Path | Iterable[Path]) -> \
        None | Path | Iterable[Path]:
    """
    Check if the path exists.
    """
    if value is None:
        return value
    elif isinstance(value, Iterable):
        for i, v in enumerate(value):
            if not v.exists():
                raise ValueError(f'Path[{i}] does not exist: {v!r}') from None
    else:
        if not value.exists():
            raise ValueError(f'Path does not exist: {value!r}') from None
    return value


def validate_path_is_file(value: None | Path | Iterable[Path]) -> \
        None | Path | Iterable[Path]:
    """
    Check if a path is a file.
    """
    if value is None:
        return value
    elif isinstance(value, Iterable):
        for i, v in enumerate(value):
            if not v.is_file():
                raise ValueError(f'Path[{i}] is not a file: {v!r}') from None
    else:
        if not value.is_file():
            raise ValueError(f'Path is not a file: {value!r}') from None
    return value


def validate_path_is_dir(value: None | Path | Iterable[Path]) -> \
        None | Path | Iterable[Path]:
    """
    Check if the path is a directory.
    """
    if value is None:
        return value
    elif isinstance(value, Iterable):
        for i, v in enumerate(value):
            if not v.is_dir():
                raise ValueError(f'Path[{i}] is not a directory: {v!r}') from None
    else:
        if not value.is_dir():
                raise ValueError(f'Path is not a directory: {value!r}') from None
    return value


def validate_path_is_absolute(value: None | Path | Iterable[Path]) -> \
        None | Path | Iterable[Path]:
    """
    Check if the path is absolute.
    """
    if value is None:
        return value
    elif isinstance(value, Iterable):
        for i, v in enumerate(value):
            if not v.is_file():
                raise ValueError(f'Path[{i}] is not absolute: {v!r}') from None
    else:
        if not value.is_absolute():
                raise ValueError(f'Path is not absolute: {value!r}') from None
    return value


def cast_as(value: Any, type_: type | Callable) -> Any:
    """
    Cast the value to the specified type.
    """
    try:
        if isinstance(type_, type) and isinstance(value, type_):
            # Do not cast if already the right type
            return value
        return type_(value)
    except ValueError:
        raise ValueError(f'Cannot cast {value!r} as {type_.__name__}') from None


def cast_as_none_if_empty(value: Any) -> Optional[Any]:
    """
    Cast the value to ``None`` if it is empty, otherwise return the original value.
    """
    if value:
        return value
    return None


def cast_as_path_or_none(value: Any) -> Optional[Path]:
    """
    Cast the value to a :class:`Path` or return ``None`` if the value is empty.
    """
    if value:
        return Path(value)
    return None


# Custom validator classes
def LengthValidator(length: int) -> AfterValidator:
    """
    `Pydantic` validator to check if the length of a value is equal to the provided length.
    """
    return AfterValidator(partial(validate_length, length=length))


def ChoicesValidator(choices: Iterable[Any]) -> AfterValidator:
    """
    `Pydantic` validator to check if a value is in the provided choices.
    """
    return AfterValidator(partial(validate_choices, choices=choices))


def PathExistsValidator() -> AfterValidator:
    """
    `Pydantic` validator to check if a path exists.
    """
    return AfterValidator(validate_path_exists)


def PathIsFileValidator() -> AfterValidator:
    """
    `Pydantic` validator to check if a path is a file.
    """
    return AfterValidator(validate_path_is_file)


def PathIsDirValidator() -> AfterValidator:
    """
    `Pydantic` validator to check if a path is a directory.
    """
    return AfterValidator(validate_path_is_dir)


def PathIsAbsoluteValidator() -> AfterValidator:
    """
    `Pydantic` validator to check if a path is absolute.
    """
    return AfterValidator(validate_path_is_absolute)


def CastAsValidator(type_: type | Callable) -> BeforeValidator:
    """
    `Pydantic` validator to cast a value to the specified type.
    """
    return BeforeValidator(partial(cast_as, type_=type_))


def CastAsNoneIfEmpty() -> BeforeValidator:
    """
    `Pydantic` validator to cast a value to ``None`` if it is empty.
    """
    return BeforeValidator(cast_as_none_if_empty)


def CastAsPathOrNone() -> BeforeValidator:
    """
    `Pydantic` validator to cast a value to a :class:`Path` or return ``None`` if the
    value is empty.
    """
    return BeforeValidator(cast_as_path_or_none)
