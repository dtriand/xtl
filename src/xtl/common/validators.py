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
        if isinstance(type_, type) and isinstance(value, type_):
            # Do not cast if already the right type
            return value
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


def PathExistsValidator() -> AfterValidator:
    """
    Pydantic validator to check if a path exists.
    """
    return AfterValidator(validate_path_exists)


def PathIsFileValidator() -> AfterValidator:
    """
    Pydantic validator to check if a path is a file.
    """
    return AfterValidator(validate_path_is_file)


def PathIsDirValidator() -> AfterValidator:
    """
    Pydantic validator to check if a path is a directory.
    """
    return AfterValidator(validate_path_is_dir)


def PathIsAbsoluteValidator() -> AfterValidator:
    """
    Pydantic validator to check if a path is absolute.
    """
    return AfterValidator(validate_path_is_absolute)


def CastAsValidator(type_: type | Callable) -> BeforeValidator:
    """
    Pydantic validator to cast a value to the specified type.
    """
    return BeforeValidator(partial(cast_as, type_=type_))
