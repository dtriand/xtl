__all__ = [
    'LengthValidator',
    'ChoicesValidator',
    'PathExistsValidator',
    'PathIsFileValidator',
    'PathIsDirValidator',
    'PathIsAbsoluteValidator',
    'CastAsValidator'
]

from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

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


def validate_choices(value: Any, choices: str | tuple[Any]) -> Any:
    """
    Check if the value is contained in the provided choices.
    """
    if value not in choices:
        raise ValueError(f'Value is not in choices: {choices!r}') from None
    return value


def validate_path_exists(value: Path) -> Path:
    """
    Check if the path exists.
    """
    if not value.exists():
        raise ValueError(f'Path does not exist: {value!r}') from None
    return value


def validate_path_is_file(value: Path) -> Path:
    """
    Check if a path is a file.
    """
    if not value.is_file():
        raise ValueError(f'Path is not a file: {value!r}') from None
    return value


def validate_path_is_dir(value: Path) -> Path:
    """
    Check if the path is a directory.
    """
    if not value.is_dir():
        raise ValueError(f'Path is not a directory: {value!r}') from None
    return value


def validate_path_is_absolute(value: Path) -> Path:
    """
    Check if the path is absolute.
    """
    if not value.is_absolute():
        raise ValueError(f'Path is not absolute: {value!r}') from None
    return value


def cast_as(value: Any, type_: type | Callable) -> Any:
    """
    Cast the value to the specified type.
    """
    try:
        return type_(value)
    except ValueError:
        raise ValueError(f'Cannot cast {value!r} as {type_.__name__}') from None


# Custom validator classes
def LengthValidator(length: int) -> AfterValidator:
    """
    Pydantic validator to check if the length of a value is equal to the provided length.
    """
    return AfterValidator(partial(validate_length, length=length))


def ChoicesValidator(choices: str | tuple[Any, ...]) -> AfterValidator:
    """
    Pydantic validator to check if a value is in the provided choices.
    """
    return AfterValidator(partial(validate_choices, choices=choices))


PathExistsValidator = AfterValidator(validate_path_exists)
"""Pydantic validator to check if a path exists."""


PathIsFileValidator = AfterValidator(validate_path_is_file)
"""Pydantic validator to check if a path is a file."""


PathIsDirValidator = AfterValidator(validate_path_is_dir)
"""Pydantic validator to check if a path is a directory."""


PathIsAbsoluteValidator = AfterValidator(validate_path_is_absolute)
"""Pydantic validator to check if a path is absolute."""


def CastAsValidator(type_: type | Callable) -> BeforeValidator:
    """
    Pydantic validator to cast a value to the specified type.
    """
    return BeforeValidator(partial(cast_as, type_=type_))
