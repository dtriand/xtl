# ruff: noqa: F403, F405 (star imports)
"""
.. |Option| replace:: :class:`Option <xtl_core.options.base.Option>`
.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
.. |Field| replace:: :func:`pydantic.Field`
.. |BaseModel| replace:: :class:`pydantic.BaseModel`
"""

from __future__ import annotations

from annotated_types import SupportsGe, SupportsGt, SupportsLe, SupportsLt
from collections.abc import Mapping
import json
import inspect
import os
from pathlib import Path
import re
from types import EllipsisType
from typing import Any, Callable, Iterable, Optional, Union, TypeVar, cast, overload, Literal
from typing_extensions import Self

from pydantic import (
    MISSING,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
    BeforeValidator,
    AfterValidator,
    ModelWrapValidatorHandler,
    model_serializer,
    SerializerFunctionWrapHandler,
    SerializationInfo,
)
from pydantic_core import PydanticUndefined, InitErrorDetails, ValidationError
from pydantic.config import JsonDict, ExtraValues
from pydantic.fields import Deprecated, FieldInfo, ComputedFieldInfo
import tomlkit
from tomlkit.exceptions import ParseError as TomlParseError

from xtl_core.options.validators import *
from xtl_core.options.toml import toml_item, with_comment
from xtl_core.options.typed_vars import TypedIterable
from xtl_core.options.exceptions import OptionsValidationError
from xtl_core.system.compatibility.py313 import path_exists
from xtl_core.system.compatibility.py314 import sentinel


_Validator = Union[BeforeValidator, AfterValidator]
_T = TypeVar('_T')

_Unset: Any = sentinel('_Unset')
"""Default of the arguments of :func:`Option` that weren't given (typed as ``Any``, so that it fits every one)."""


# Overload signatures for different initialization scenarios
# Option(...)
@overload
def Option(
    default: EllipsisType,
    *,
    name: str | None = _Unset,
    desc: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    alias: str | None = _Unset,
    exclude: bool | None = _Unset,
    repr: bool = _Unset,
    deprecated: Deprecated | str | bool | None = _Unset,
    validate_default: bool | None = _Unset,
    choices: Iterable[Any] | None = _Unset,
    gt: SupportsGt | None = _Unset,
    ge: SupportsGe | None = _Unset,
    lt: SupportsLt | None = _Unset,
    le: SupportsLe | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    length: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    regex: str | re.Pattern[str] | None = _Unset,
    path_exists: bool | None = _Unset,
    path_is_file: bool | None = _Unset,
    path_is_dir: bool | None = _Unset,
    path_is_absolute: bool | None = _Unset,
    cast_as: type | Callable | None = _Unset,
    validator: _Validator | Iterable[_Validator] | None = _Unset,
    strict: bool | None = _Unset,
    formatter: Callable | None = _Unset,
    extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> Any: ...


# Option(default=_T)
@overload
def Option(
    default: _T,
    *,
    name: str | None = _Unset,
    desc: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    alias: str | None = _Unset,
    exclude: bool | None = _Unset,
    repr: bool = _Unset,
    deprecated: Deprecated | str | bool | None = _Unset,
    validate_default: bool | None = _Unset,
    choices: Iterable[Any] | None = _Unset,
    gt: SupportsGt | None = _Unset,
    ge: SupportsGe | None = _Unset,
    lt: SupportsLt | None = _Unset,
    le: SupportsLe | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    length: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    regex: str | re.Pattern[str] | None = _Unset,
    path_exists: bool | None = _Unset,
    path_is_file: bool | None = _Unset,
    path_is_dir: bool | None = _Unset,
    path_is_absolute: bool | None = _Unset,
    cast_as: type | Callable | None = _Unset,
    validator: _Validator | Iterable[_Validator] | None = _Unset,
    strict: bool | None = _Unset,
    formatter: Callable | None = _Unset,
    extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> _T: ...


# Option(default_factory=callable)
@overload
def Option(
    *,
    default_factory: Callable[[], _T] | Callable[[dict[str, Any]], _T],
    name: str | None = _Unset,
    desc: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    alias: str | None = _Unset,
    exclude: bool | None = _Unset,
    repr: bool = _Unset,
    deprecated: Deprecated | str | bool | None = _Unset,
    validate_default: bool | None = _Unset,
    choices: Iterable[Any] | None = _Unset,
    gt: SupportsGt | None = _Unset,
    ge: SupportsGe | None = _Unset,
    lt: SupportsLt | None = _Unset,
    le: SupportsLe | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    length: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    regex: str | re.Pattern[str] | None = _Unset,
    path_exists: bool | None = _Unset,
    path_is_file: bool | None = _Unset,
    path_is_dir: bool | None = _Unset,
    path_is_absolute: bool | None = _Unset,
    cast_as: type | Callable | None = _Unset,
    validator: _Validator | Iterable[_Validator] | None = _Unset,
    strict: bool | None = _Unset,
    formatter: Callable | None = _Unset,
    extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> _T: ...


# Option()
@overload
def Option(
    *,
    name: str | None = _Unset,
    desc: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    alias: str | None = _Unset,
    exclude: bool | None = _Unset,
    repr: bool = _Unset,
    deprecated: Deprecated | str | bool | None = _Unset,
    validate_default: bool | None = _Unset,
    choices: Iterable[Any] | None = _Unset,
    gt: SupportsGt | None = _Unset,
    ge: SupportsGe | None = _Unset,
    lt: SupportsLt | None = _Unset,
    le: SupportsLe | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    length: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    regex: str | re.Pattern[str] | None = _Unset,
    path_exists: bool | None = _Unset,
    path_is_file: bool | None = _Unset,
    path_is_dir: bool | None = _Unset,
    path_is_absolute: bool | None = _Unset,
    cast_as: type | Callable | None = _Unset,
    validator: _Validator | Iterable[_Validator] | None = _Unset,
    strict: bool | None = _Unset,
    formatter: Callable | None = _Unset,
    extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> Any: ...


def Option(
    default: Any = PydanticUndefined,
    *,
    # Default value
    default_factory: Callable[[], Any] | Callable[[dict[str, Any]], Any] | None = _Unset,
    # Metadata
    name: str | None = _Unset,
    desc: str | None = _Unset,
    examples: list[Any] | None = _Unset,
    # Serialization
    alias: str | None = _Unset,
    exclude: bool | None = _Unset,
    # Representation
    repr: bool = _Unset,  # include in repr
    deprecated: Deprecated | str | bool | None = _Unset,  # deprecation message
    # Validation
    validate_default: bool | None = _Unset,  # validate default value
    choices: Iterable[Any] | None = _Unset,  # iterable of choices
    #  for numbers
    gt: SupportsGt | None = _Unset,  # greater than
    ge: SupportsGe | None = _Unset,  # greater than or equal to
    lt: SupportsLt | None = _Unset,  # less than
    le: SupportsLe | None = _Unset,  # less than or equal to
    multiple_of: float | None = _Unset,  # multiple of
    allow_inf_nan: bool | None = _Unset,  # allow inf and nan
    #  for decimals
    max_digits: int | None = _Unset,  # maximum number of total digits
    decimal_places: int | None = _Unset,  # maximum number of decimal places
    #  for iterables
    length: int | None = _Unset,  # length of the iterable
    min_length: int | None = _Unset,  # minimum length of the iterable
    max_length: int | None = _Unset,  # maximum length of the iterable
    #  for strings
    regex: str | re.Pattern[str] | None = _Unset,  # regex pattern
    #  for paths
    path_exists: bool | None = _Unset,  # check if the path exists
    path_is_file: bool | None = _Unset,  # check if the path is a file
    path_is_dir: bool | None = _Unset,  # check if the path is a directory
    path_is_absolute: bool | None = _Unset,  # check if the path is absolute
    # Type casting
    cast_as: type | Callable | None = _Unset,
    # Additional validators
    validator: _Validator | Iterable[_Validator] | None = _Unset,
    # strict type checking (i.e., no implicit conversion)
    strict: bool | None = _Unset,
    # Serialization
    formatter: Callable | None = _Unset,
    # Extra JSON schema
    extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> Any:  # a `FieldInfo`, typed as the overloads' field value, as pydantic's `Field()`
    """
    Create a |Field| with custom validation, serialization and more intuitive metadata
    handling.

    Custom validation and serialization functions are stored in the
    ``json_schema_extra`` attribute of the field and are applied to the model if it is
    of type |Options|. If an |Option| field is used in a model that directly inherits
    from |BaseModel|, then all custom validation and serialization will simply be
    ignored.

    :param default: Default value for the field. ``pydantic.MISSING`` leaves the field unset: it is then left out of
        the output, and its validators and formatter skip it.
    :param default_factory: A callable to generate the default value.
    :param name: Name of the field (equivalent to ``title`` in |Field|).
    :param desc: Description of the field (equivalent to ``description`` in |Field|).
    :param examples: Example values for this field.
    :param alias: Alias for the field (equivalent to ``serialization_alias`` in
        |Field|).
    :param exclude: Exclude this field from serialization.
    :param repr: Include this field in the model's ``__repr__`` method.
    :param deprecated: Deprecation message for the field.
    :param validate_default: Whether to validate the default value.
    :param choices: Iterable of valid values constraint for the field.
    :param gt: Greater than constraint for numeric fields.
    :param ge: Greater than or equal to constraint for numeric fields.
    :param lt: Less than constraint for numeric fields.
    :param le: Less than or equal to constraint for numeric fields.
    :param multiple_of: Multiple of constraint for numeric fields.
    :param allow_inf_nan: Allow ``inf`` and ``nan`` values for numeric fields.
    :param max_digits: Maximum number of total digits for decimal fields.
    :param decimal_places: Maximum number of decimal places for decimal fields.
    :param length: Length constraint for iterable fields.
    :param min_length: Minimum length constraint for iterable fields.
    :param max_length: Maximum length constraint for iterable fields.
    :param regex: Regular expression pattern for string fields.
    :param path_exists: Check if the path exists for path fields.
    :param path_is_file: Check if the path is a file for path fields.
    :param path_is_dir: Check if the path is a directory for path fields.
    :param path_is_absolute: Check if the path is absolute for path fields.
    :param cast_as: Type or callable to cast the value to prior to validation.
    :param validator: Custom validator function for the field.
    :param strict: Strict type checking (i.e., no implicit conversion/type coercion).
    :param formatter: Custom serializer function for the field.
    :param extra: Extra JSON schema information (equivalent to ``json_schema_extra`` in
        |Field|).
    :return: A :class:`pydantic.FieldInfo <pydantic.fields.FieldInfo>` object with
        custom validation and metadata handling.
    :raises ValueError: If neither ``default`` nor ``default_factory`` is provided.
    :raises TypeError: If ``extra`` is a callable, but validators or a formatter are given.
    """
    # Before validators (`None` is not given, as for the other arguments)
    validators: list[_Validator] = []
    if cast_as is not _Unset and cast_as is not None:
        validators.append(CastAsValidator(cast_as))
    # After validators
    if length is not _Unset and length is not None:
        validators.append(LengthValidator(length))
    if choices is not _Unset and choices is not None:
        validators.append(ChoicesValidator(choices))
    if path_exists is not _Unset and path_exists:
        validators.append(PathExistsValidator())
    if path_is_file is not _Unset and path_is_file:
        validators.append(PathIsFileValidator())
    if path_is_dir is not _Unset and path_is_dir:
        validators.append(PathIsDirValidator())
    if path_is_absolute is not _Unset and path_is_absolute:
        validators.append(PathIsAbsoluteValidator())
    # Either before or after validators
    if validator is not _Unset and validator is not None:
        # Flatten the validators if it's an iterable
        given: list[_Validator] = list(validator) if isinstance(validator, Iterable) else [validator]

        # Type checking
        for i, v in enumerate(given):
            if not isinstance(v, _Validator):
                raise TypeError(f'Validator {i} is not a valid validator, but {type(v)} instead')

            # Inspect validator function signature
            func = v.func
            if not callable(func):
                raise TypeError(f'Validator {i} function is not a callable')
            args = inspect.signature(func).parameters.values()
            if len(args) != 1:
                raise TypeError(
                    f'Validator {i} function must take exactly 1 argument, but it takes {len(args)} instead'
                )
        validators.extend(given)

    if formatter is not _Unset and not callable(formatter):
        raise TypeError("'formatter' must be a callable")

    if extra is _Unset or extra is None:
        extra = {}
    # Store custom validators and the formatter in the `json_schema_extra`
    if validators or formatter is not _Unset:
        if not isinstance(extra, dict):
            raise TypeError("'extra' must be a dict for an option with validators or a formatter")
        # Not JSON values, so not a `JsonDict` any more
        stored: dict[str, Any] = extra
        if validators:
            # Ensure we don't overwrite existing validators
            if not isinstance(stored.get('validators'), list):
                stored['validators'] = []
            stored['validators'].extend(validators)
        if formatter is not _Unset:
            stored['serializer'] = formatter

    field_kwargs: dict[str, Any] = {
        'serialization_alias': alias,
        'title': name,
        'description': desc,
        'examples': examples,
        'exclude': exclude,
        'deprecated': deprecated,
        'json_schema_extra': extra,
        'validate_default': validate_default,
        'repr': repr,
        'pattern': regex,
        'strict': strict,
        'gt': gt,
        'ge': ge,
        'lt': lt,
        'le': le,
        'multiple_of': multiple_of,
        'allow_inf_nan': allow_inf_nan,
        'max_digits': max_digits,
        'decimal_places': decimal_places,
        'min_length': min_length,
        'max_length': max_length,
    }
    # Pass only what was given, so that pydantic applies its own defaults for the rest
    field_kwargs = {key: value for key, value in field_kwargs.items() if value is not _Unset}

    if default_factory is not _Unset:
        field_kwargs['default_factory'] = default_factory
    else:
        field_kwargs['default'] = default
    return Field(**field_kwargs)


def _schema_extra(field: FieldInfo | ComputedFieldInfo) -> dict[str, Any]:
    """The ``json_schema_extra`` of a field, where :func:`Option` keeps its validators and
    formatter; empty if it is not a dict (pydantic also takes a callable)."""
    extra = field.json_schema_extra
    return extra if isinstance(extra, dict) else {}


_TOML_SKIP = sentinel('_TOML_SKIP')
"""Marks values that are left out of a TOML file (None, since TOML has no null)."""


class Options(BaseModel):
    """
    Base class for storing various configuration options. This class provides
    validation and serialization through `Pydantic`. Custom validators and serializers
    are also supported (see |Option| for more details).

    Fields can be defined using |Option| or |Field|, but custom validation will only be
    applied if |Option| is used.

    In addition to the standard `Pydantic` serialization methods, this class can also
    be serialized to a TOML file. Deserialization is possible from Python dictionaries,
    JSON and TOML strings.

    Parsing of environment variables in string values is possible by setting
    ``parse_env=True`` in the constructor. This will replace any environment variable
    references in the format ``${VARIABLE}`` with their values. If the variable is not
    found, it will be replaced with an empty string. The default behavior is to not
    parse environment variables.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
        ser_json_inf_nan='constants',  # NaN and infinity as NaN/Infinity instead of null in JSON
    )
    _parse_env: bool = PrivateAttr(default=False)

    def __init__(self, /, **data: Any) -> None:
        # Check if the model contains any aliased keys
        aliased = [
            field.serialization_alias
            for field in self.__pydantic_fields__.values()
            if field.serialization_alias is not None
        ]
        # Check if the input data contain any aliased keys
        if any(alias in data for alias in aliased):
            for name, field in self.__pydantic_fields__.items():
                # Check if the field is aliased
                alias = field.serialization_alias
                if alias and alias in data and name not in data:
                    # Swap the alias key with the original name
                    data[name] = data.pop(alias)
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise OptionsValidationError(e, type(self).__name__) from e

    @staticmethod
    def _get_envvar(data: Any) -> Any:
        """
        Replace environment variables in the data with their values. Environment
        variables are expected to be in the format ``${VARIABLE}``. If the variable is
        not found, it will be replaced with an empty string.

        :param data: The data to parse. Only strings are checked.
        :return: The data with environment variables replaced.
        """
        if not isinstance(data, str):
            return data
        if '$' not in data:
            return data

        # Regular expression to match environment variables in the format ${VARIABLE}
        pattern = r'\${(.*?)\}'

        def replace_envvar(match: re.Match[str]) -> str:
            env_var = match.group(1)
            return os.getenv(env_var, '')

        return re.sub(pattern, replace_envvar, data)

    @classmethod
    def _get_custom_validators(cls) -> dict[str, dict[str, list]]:
        """
        Get the custom validators for the model.

        :return: A dictionary mapping in the form of
            ```
            {'field_name': {'before': [BeforeValidator], 'after': [AfterValidator]}}
            ```
        """
        custom_validators = {}
        for name, finfo in cls.__pydantic_fields__.items():
            for validator in _schema_extra(finfo).get('validators', []):
                if name not in custom_validators:
                    custom_validators[name] = {'before': [], 'after': []}
                if isinstance(validator, BeforeValidator):
                    custom_validators[name]['before'].append(validator)
                elif isinstance(validator, AfterValidator):
                    custom_validators[name]['after'].append(validator)
        return custom_validators

    @classmethod
    def _get_custom_serializers(cls) -> dict[str, Callable]:
        """
        Get the custom serializers for the model.

        :return: A dictionary mapping in the form of
            ```
            {'field_name': serializer_function}
            ```
        """
        custom_serializers = {}
        for name, finfo in cls.__pydantic_fields__.items():
            serializer = _schema_extra(finfo).get('serializer')
            if serializer:
                custom_serializers[name] = serializer
        return custom_serializers

    @staticmethod
    def _apply_validators(
        name: str,
        value: Any,
        validators: list[AfterValidator | BeforeValidator],
        errors: list[InitErrorDetails] | None = None,
    ) -> tuple[Any, list[InitErrorDetails]]:
        """
        Apply validators to the value. This is used to perform a series of validations
        on the value. If any of the validators raise a ``ValueError``, the error is
        captured and returned in the `errors` list. This is on par with the default
        pydantic behavior where all validation errors are collected and raised at once.

        :param name: Name of the field.
        :param value: Value to validate.
        :param validators: List of validators to apply.
        :param errors: List of errors to append to.
        :return: Tuple of the validated value and a list of errors.
        """
        if errors is None:
            errors = []
        new_errors = []

        # A field left unset has no value to validate
        if value is MISSING:
            return value, errors

        for validator in validators:
            # Validators of an `Option` take the value only (checked by `Option()`), not pydantic's `info` too
            func = cast(Callable[[Any], Any], validator.func)
            try:
                value = func(value)
            except ValueError as e:
                new_errors.append(InitErrorDetails(type='value_error', loc=(name,), input=value, ctx={'error': e}))

        if new_errors:
            return _Unset, errors + new_errors
        else:
            return value, errors

    @classmethod
    def _validate_before(
        cls,
        name: str,
        value: Any,
        validators: dict[str, list[BeforeValidator | AfterValidator]],
        errors: list[InitErrorDetails] | None = None,
    ) -> tuple[Any, list[InitErrorDetails]]:
        """
        Apply before validators to the value. This is used to validate the value before
        the model has been initialized.

        :param name: Name of the field.
        :param value: Value to validate.
        :param validators: Dictionary of field's validators.
        :param errors: List of errors to append to.
        :return: Tuple of the validated value and a list of errors.
        """
        if errors is None:
            errors = []

        # Apply validators
        value, new_errors = cls._apply_validators(
            name=name, value=value, validators=validators['before'], errors=errors
        )
        return value, new_errors

    @classmethod
    def _validate_after(
        cls,
        name: str,
        value: Any,
        validators: dict[str, list[BeforeValidator | AfterValidator]],
        errors: list[InitErrorDetails] | None = None,
    ) -> tuple[Any, list[InitErrorDetails]]:
        """
        Apply after validators to the value. This is used to validate the value after
        the model has been initialized and pydantic has run its own validation.

        :param name: Name of the field.
        :param value: Value to validate.
        :param validators: Dictionary of field's validators.
        :param errors: List of errors to append to.
        :return: Tuple of the validated value and a list of errors.
        """
        if errors is None:
            errors = []

        # Apply validators
        value, new_errors = cls._apply_validators(name=name, value=value, validators=validators['after'], errors=errors)
        return value, new_errors

    @model_validator(mode='wrap')
    @classmethod
    def _custom_validation(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """
        Model validator for applying custom validation functions specified in the
        `json_schema_extra` attribute of the fields.

        :param data: The data to validate.
        :param handler: The handler for the model validation.
        :return: The validated model instance.
        """
        # Get custom validators
        validators = cls._get_custom_validators()
        errors = []

        if isinstance(data, dict):
            # During model initialization data is a dict

            # Check for parsing of environment variables (kept on the instance below)
            parse_env = data.get('_parse_env', False)

            # Apply before validators for raw data only
            for name, value in data.items():
                new_errors = []

                # Check if the field contains environment variables
                if parse_env:
                    value = cls._get_envvar(value)

                if name in validators:
                    value, new_errors = cls._validate_before(
                        name=name, value=value, validators=validators[name], errors=new_errors
                    )

                # Update value (NB: This runs even if there are no before validators)
                if not new_errors:
                    data[name] = value

                errors.extend(new_errors)
            # Check and raise validation errors
            if errors:
                raise ValidationError.from_exception_data(title='before_validators', line_errors=errors)
        elif isinstance(data, cls):
            # During value assignment data is a pydantic model

            # Check for parsing of environment variables
            parse_env = data._parse_env
        else:
            # Let Pydantic's own pipeline deal with data that are neither a dict nor
            #  a model instance. This deals properly with custom before validators
            #  implemented by the models themselves
            return handler(data)

        # Generate the pydantic model
        validated_self = handler(data)
        if isinstance(data, dict):
            validated_self._parse_env = parse_env

        # Apply after validators
        for name, value in vars(validated_self).items():
            new_errors = []

            # Check for environment variables in fields that were instantiated with
            #   `default` or `default_value` and no actual value
            #   NB: this skips the before validators
            if parse_env:
                value = cls._get_envvar(value)

            if name in validators:
                value, new_errors = cls._validate_after(
                    name=name, value=value, validators=validators[name], errors=new_errors
                )
            if not new_errors:
                # Note: Workaround to prevent infinite recursion when using
                #  setattr()
                vars(validated_self)[name] = value
            errors.extend(new_errors)
        # Check and raise validation errors
        if errors:
            raise ValidationError.from_exception_data(title='after_validators', line_errors=errors)
        return validated_self

    @model_serializer(mode='wrap')
    def _custom_serialization(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        # Get custom serializers
        serializers = self._get_custom_serializers()

        # Perform standard Pydantic serialization
        data = handler(self)

        # Determine which keys to skip
        skippable = set()
        if self.__pydantic_extra__:
            skippable |= set(self.__pydantic_extra__.keys())
        if self.__pydantic_private__:
            skippable |= set(self.__pydantic_private__.keys())

        # Apply custom formatters
        poppable = set()
        aliased = set()
        for key in data:
            # Pydantic serializes any extra attributes by default
            if key in skippable:
                poppable.add(key)
                continue

            # Delay serialization of aliased fields
            if info.by_alias:
                for name, field in self.__pydantic_fields__.items():
                    if field.serialization_alias == key:
                        aliased.add((key, name))  # (alias, original)
                        break

            # Serialize non-aliased fields
            serializer = serializers.get(key, None)
            if serializer is not None:
                data[key] = serializer(getattr(self, key))

        # Serialize aliased fields
        for alias, original in aliased:
            serializer = serializers.get(original, None)
            if serializer is not None:
                data[alias] = serializer(getattr(self, original))

        # Remove skippable keys
        if poppable:
            for key in poppable:
                data.pop(key)

        return data

    def __setattr__(self, name: str, value: Any) -> None:
        # Check if the attribute is a pydantic field
        if name not in self.__pydantic_fields__:
            super().__setattr__(name, value)
            return

        # Check if the field contains environment variables
        if self._parse_env:
            value = self._get_envvar(value)

        try:
            # Apply before validators
            validators = self._get_custom_validators()
            if name in validators:
                value, errors = self._validate_before(name=name, value=value, validators=validators[name])
                if errors:
                    raise ValidationError.from_exception_data(title='before_validators', line_errors=errors)

            # Pydantic validates the new value (`validate_assignment`), then the `after`
            #  validators run in `_custom_validation()`. Pydantic sets the attribute before
            #  the after validators run, so this happens on a shallow copy, which a failure
            #  leaves behind
            trial = self.model_copy()
            super(Options, trial).__setattr__(name, value)
        except ValidationError as e:
            raise OptionsValidationError(e, type(self).__name__) from e
        # Take the validated values, including any that the validators changed
        vars(self).update(vars(trial))
        self.__pydantic_fields_set__.add(name)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ):
        try:
            return super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except ValidationError as e:
            raise OptionsValidationError(e, cls.__name__) from e

    def to_dict(self, by_alias: bool = True) -> dict[str, Any]:
        """
        Convert the |Options| to a dictionary. Nested |Options| objects will be
        converted to nested dictionaries.

        :param by_alias: Whether to use field aliases in the output dictionary.
        :return: A dictionary representation of the |Options|.
        """
        return self.model_dump(by_alias=by_alias)

    @classmethod
    def from_dict(cls, data: dict[str, Any], /) -> Self:
        """
        Create a new |Options| object from a dictionary. Nested dictionaries can be
        used to create nested |Options| objects.

        :param data: Dictionary containing the values to create the config with.
        :return: An |Options| object.
        """
        return cls.model_validate(data)

    @staticmethod
    def _read_contents(s: str | Path, mode: Literal['json', 'toml'], encoding: str = 'utf-8') -> Any:
        """
        Parse a JSON or TOML string, or the contents of a JSON or TOML file. A string is
        parsed as such first, and read as a path only if that fails.

        :param s: A string to parse, or the path to a file to read.
        :param mode: The format to parse, either ``'json'`` or ``'toml'``.
        :param encoding: The encoding of the file.
        :return: The parsed data: a dict for TOML, any JSON value for JSON.
        :raises ValueError: If ``mode`` is neither ``'json'`` nor ``'toml'``.
        :raises FileNotFoundError: If ``s`` is a :class:`~pathlib.Path` that doesn't exist.
        :raises json.JSONDecodeError: If the JSON is invalid: a string that isn't valid
            JSON nor an existing file, or a file with invalid contents.
        :raises tomlkit.exceptions.ParseError: If the TOML is invalid, likewise (also a
            :class:`ValueError`).
        """
        loaders = {
            'json': json.loads,
            # Plain Python data, without tomlkit's formatting and comments
            'toml': lambda text: tomlkit.parse(text).unwrap(),
        }
        if mode not in loaders:
            raise ValueError(f'Invalid read mode: {mode!r}. Must be one of: {", ".join(loaders.keys())}')
        loads = loaders[mode]

        if isinstance(s, str):
            # Try parsing as a string first
            try:
                return loads(s)
            except (json.JSONDecodeError, TomlParseError):
                # Not a valid string — check if it is a file
                if not path_exists(s):
                    # It was probably an invalid string
                    raise

        # Try parsing as a file path
        path = Path(s)
        if not path_exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        return loads(path.read_text(encoding=encoding))

    @staticmethod
    def _write_text(
        text: str,
        *,
        filename: str | Path,
        suffix: str,
        overwrite: bool = False,
        keep_file_ext: bool = False,
        encoding: str = 'utf-8',
    ) -> Path:
        """
        Write serialized |Options| to a file, creating any missing parent directories.

        :param text: The serialized text to write to the file.
        :param filename: The path to the file to write to.
        :param suffix: The extension the file gets, e.g. ``'.json'``.
        :param overwrite: Whether to overwrite the file if it already exists.
        :param keep_file_ext: Keep the extension of ``filename`` instead of replacing it
            with ``suffix``.
        :param encoding: The encoding of the file.
        :return: The path to the file that was written.
        :raises FileExistsError: If the file (after fixing its extension) already exists
            and ``overwrite`` is False.
        """
        path = Path(filename)
        if not path.name.endswith(suffix) and not keep_file_ext:
            path = path.with_suffix(suffix)
        if path.exists() and not overwrite:
            raise FileExistsError(f'File {path} already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
        return path

    @overload
    def to_json(
        self,
        filename: None = None,
        /,
        *,
        overwrite: bool = ...,
        keep_file_ext: bool = ...,
        indent: Optional[int] = ...,
        by_alias: bool = ...,
        encoding: str = ...,
    ) -> str: ...

    @overload
    def to_json(
        self,
        filename: str | Path,
        /,
        *,
        overwrite: bool = ...,
        keep_file_ext: bool = ...,
        indent: Optional[int] = ...,
        by_alias: bool = ...,
        encoding: str = ...,
    ) -> Path: ...

    def to_json(
        self,
        filename: Optional[str | Path] = None,
        /,
        *,
        overwrite: bool = False,
        keep_file_ext: bool = False,
        indent: Optional[int] = 4,
        by_alias: bool = True,
        encoding: str = 'utf-8',
    ) -> str | Path:
        """
        Write the |Options| to a JSON file. If ``filename`` is not provided, then the
        JSON will be returned as a string. Missing parent directories are created.

        :param filename: Optional output path for the JSON file.
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the extension of ``filename`` instead of replacing it
            with ``.json``.
        :param indent: Indentation level for the JSON string. If ``None``, then the JSON
            will be compact.
        :param by_alias: Whether to use field aliases in the output JSON.
        :param encoding: The encoding of the file.
        :return: The JSON string if no ``filename`` is given, otherwise the path of the
            file written.
        :raises FileExistsError: If the file already exists and ``overwrite`` is False.
        """
        text = self.model_dump_json(indent=indent, by_alias=by_alias)
        return (
            text
            if filename is None
            else self._write_text(
                text,
                filename=filename,
                suffix='.json',
                overwrite=overwrite,
                keep_file_ext=keep_file_ext,
                encoding=encoding,
            )
        )

    @classmethod
    def from_json(cls, s: str | Path, /, *, encoding: str = 'utf-8') -> Self:
        """
        Create a new |Options| object from a JSON string or file.

        :param s: JSON string or path to a JSON file. A string is parsed as JSON first,
            and read as a path only if that fails.
        :param encoding: The encoding of the file.
        :return: An |Options| object.
        :raises json.JSONDecodeError: If the JSON is invalid, including a string that is
            neither JSON nor an existing file.
        :raises FileNotFoundError: If ``s`` is a :class:`~pathlib.Path` that doesn't exist.
        :raises OptionsValidationError: If the data is invalid for the |Options| class.
        """
        return cls.from_dict(cls._read_contents(s, mode='json', encoding=encoding))

    @staticmethod
    def _toml_key(name: str, field: FieldInfo | ComputedFieldInfo) -> str:
        """
        Key of a field in a TOML file: its serialization alias, if any, else its name.
        """
        if isinstance(field, FieldInfo) and field.serialization_alias:
            return field.serialization_alias
        if isinstance(field, ComputedFieldInfo) and field.alias:
            return field.alias
        return name

    @classmethod
    def _value_to_toml(
        cls, value: Any, field: FieldInfo | ComputedFieldInfo | None = None, keep_comments: bool = False, path: str = ''
    ) -> Any:
        """
        Convert a value to ``tomlkit`` items: the |Options| itself to a document, nested |Options| and mappings to
        tables, iterables to arrays (lists of tables to arrays of tables), and anything else to a value (see
        :func:`~xtl_core.options.toml.toml_item`). With ``keep_comments``, a field's description becomes the inline
        comment of its value, or of its table's header line. Fields with ``exclude=True`` are left out. Since TOML
        has no null, ``None`` fields are returned as ``_TOML_SKIP``, so that they are left out (they then load as
        the default of the field), while ``None`` inside dictionaries and iterables raises an error, since leaving
        it out would change the data. Fields left unset (``MISSING``) are left out the same way, before their
        formatter runs.

        :param value: The value to convert.
        :param field: The field of the value, for its serializer and description, or None for values inside
            dictionaries and iterables (and for the |Options| itself).
        :param keep_comments: Whether to include comments in the output.
        :param path: Location of the value in the TOML data (e.g. ``table.key[0]``), for error messages; empty for
            the |Options| itself, which then becomes a document.
        :return: A ``tomlkit`` document or item, or ``_TOML_SKIP``.
        :raises ValueError: If a dictionary or iterable contains ``None`` or ``MISSING``.
        """
        # An unset field is left out, as in `to_dict()`; its formatter never sees it
        if value is MISSING:
            if field is None:
                raise ValueError(
                    f'Cannot serialize MISSING at `{path}` to TOML, since it can only be left out, which would '
                    f'change the data.'
                )
            return _TOML_SKIP

        # Apply serialization function
        serializer = _schema_extra(field).get('serializer') if field is not None else None
        if serializer is not None:
            value = serializer(value)

        if value is None:
            if field is None:
                raise ValueError(
                    f'Cannot serialize None at `{path}` to TOML, since TOML has no null. Use a '
                    f'formatter on the field that holds it to replace None with a placeholder.'
                )
            return _TOML_SKIP

        comment = field.description if keep_comments and field is not None else None

        # Nested Options: a table with one entry per field
        if isinstance(value, Options):
            table = tomlkit.document() if field is None and not path else tomlkit.table()
            for name, vfield in {**value.__pydantic_fields__, **value.__pydantic_computed_fields__}.items():
                # Fields excluded from serialization are left out, as in `to_dict()`
                if getattr(vfield, 'exclude', False):
                    continue
                key = cls._toml_key(name, vfield)
                item = cls._value_to_toml(
                    getattr(value, name),
                    field=vfield,
                    keep_comments=keep_comments,
                    path=f'{path}.{key}' if path else key,
                )
                if item is not _TOML_SKIP:
                    table.add(key, item)
            return table if isinstance(table, tomlkit.TOMLDocument) else with_comment(table, comment)

        # Mappings (dict, TypedDict): a table with one entry per key (keys must be strings in TOML)
        if isinstance(value, Mapping):
            table = tomlkit.table()
            for key, v in value.items():
                table.add(str(key), cls._value_to_toml(v, keep_comments=keep_comments, path=f'{path}.{key}'))
            return with_comment(table, comment)

        # Iterables: cast to arrays, since TOML doesn't differentiate
        if isinstance(value, (list, tuple, set, frozenset, TypedIterable)):
            items = [
                cls._value_to_toml(v, keep_comments=keep_comments, path=f'{path}[{i}]') for i, v in enumerate(value)
            ]
            return with_comment(tomlkit.item(items), comment)

        return toml_item(value, comment)

    @overload
    def to_toml(
        self,
        filename: None = None,
        /,
        *,
        comments: bool = ...,
        overwrite: bool = ...,
        keep_file_ext: bool = ...,
        encoding: str = ...,
    ) -> str: ...

    @overload
    def to_toml(
        self,
        filename: str | Path,
        /,
        *,
        comments: bool = ...,
        overwrite: bool = ...,
        keep_file_ext: bool = ...,
        encoding: str = ...,
    ) -> Path: ...

    def to_toml(
        self,
        filename: Optional[str | Path] = None,
        /,
        *,
        comments: bool = False,
        overwrite: bool = False,
        keep_file_ext: bool = False,
        encoding: str = 'utf-8',
    ) -> str | Path:
        """
        Write the |Options| to a TOML file. If ``filename`` is not provided, then the
        TOML will be returned as a string. Missing parent directories are created. Fields
        that are None are left out, since TOML has no null; they load as the default of
        the field.

        :param filename: Optional output path for the TOML file.
        :param comments: Include comments in the TOML file.
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the extension of ``filename`` instead of replacing it
            with ``.toml``.
        :param encoding: The encoding of the file.
        :return: The TOML string if no ``filename`` is given, otherwise the path of the
            file written.
        :raises FileExistsError: If the file already exists and ``overwrite`` is False.
        """
        text = tomlkit.dumps(self._value_to_toml(self, keep_comments=comments))
        return (
            text
            if filename is None
            else self._write_text(
                text,
                filename=filename,
                suffix='.toml',
                overwrite=overwrite,
                keep_file_ext=keep_file_ext,
                encoding=encoding,
            )
        )

    @classmethod
    def from_toml(cls, s: str | Path, /, *, encoding: str = 'utf-8') -> Self:
        """
        Create a new |Options| object from a TOML string or file.

        :param s: TOML string or path to a TOML file. A string is parsed as TOML first,
            and read as a path only if that fails; an empty string loads the defaults.
        :param encoding: The encoding of the file.
        :return: An |Options| object.
        :raises tomlkit.exceptions.ParseError: If the TOML is invalid, including a string
            that is neither TOML nor an existing file (also a :class:`ValueError`).
        :raises FileNotFoundError: If ``s`` is a :class:`~pathlib.Path` that doesn't exist.
        :raises OptionsValidationError: If the data is invalid for the |Options| class.
        """
        return cls.from_dict(cls._read_contents(s, mode='toml', encoding=encoding))
