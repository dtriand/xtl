from annotated_types import SupportsGe, SupportsGt, SupportsLe, SupportsLt
from functools import partial
import os
import re
from typing import Any, Callable
from typing_extensions import Self

from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr, model_validator,
                      BeforeValidator, AfterValidator, ModelWrapValidatorHandler)
from pydantic_core import PydanticUndefined, InitErrorDetails, ValidationError
from pydantic.config import JsonDict
from pydantic.fields import _Unset, Deprecated, FieldInfo

from .validators import *


def Option(
        *,
        # Default value
        default: Any | None = PydanticUndefined,
        default_factory: Callable[[], Any] | Callable[
            [dict[str, Any]], Any] | None = _Unset,
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
        choices: str | tuple[Any, ...] | None = _Unset,  # iterable of choices
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
        strict: bool | None = _Unset,
        # strict type checking (i.e., no implicit conversion)
        cast_as: type | Callable | None = _Unset,
        # Extra JSON schema
        extra: JsonDict | Callable[[JsonDict], None] | None = _Unset,
) -> FieldInfo:
    """
    Create a field with custom validation, serialization and metadata.
    """
    if extra is _Unset:
        extra = {}

    # Store custom validators in the `json_schema_extra`
    if any([v is not _Unset for v in (length, choices, path_exists, path_is_file,
                                      path_is_dir, path_is_absolute, cast_as)]):
        extra['validators'] = []
    if length is not _Unset:
        extra['validators'].append(LengthValidator(length))
    if choices is not _Unset:
        extra['validators'].append(ChoicesValidator(choices))
    if path_exists is not _Unset:
        extra['validators'].append(PathExistsValidator)
    if path_is_file is not _Unset:
        extra['validators'].append(PathIsFileValidator)
    if path_is_dir is not _Unset:
        extra['validators'].append(PathIsDirValidator)
    if path_is_absolute is not _Unset:
        extra['validators'].append(PathIsAbsoluteValidator)
    if cast_as is not _Unset:
        extra['validators'].append(CastAsValidator(cast_as))

    partial_field = partial(
        Field, serialization_alias=alias, title=name, description=desc,
        examples=examples, exclude=exclude, deprecated=deprecated,
        json_schema_extra=extra, validate_default=validate_default, repr=repr,
        pattern=regex, strict=strict, gt=gt, ge=ge, lt=lt, le=le,
        multiple_of=multiple_of, allow_inf_nan=allow_inf_nan, max_digits=max_digits,
        decimal_places=decimal_places, min_length=min_length, max_length=max_length)

    if default_factory:
        f = partial_field(default_factory=default_factory)
    else:
        f = partial_field(default=default)

    return f


class Options(BaseModel):
    """
    Base class for configuration options with customizable validation and serialization.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True,
    )
    _parse_env: bool = PrivateAttr(default=False)

    @staticmethod
    def _get_envvar(data: Any) -> Any:
        """
        Replace environment variables in the data with their values.
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
        """
        custom_validators = {}
        for name, finfo in cls.__pydantic_fields__.items():
            # Skip if no validators are defined
            if not finfo.json_schema_extra:
                continue
            validators = finfo.json_schema_extra.get('validators', [])
            for validator in validators:
                if name not in custom_validators:
                    custom_validators[name] = {'before': [], 'after': []}
                if isinstance(validator, BeforeValidator):
                    custom_validators[name]['before'].append(validator)
                elif isinstance(validator, AfterValidator):
                    custom_validators[name]['after'].append(validator)
        return custom_validators

    @model_validator(mode='wrap')
    @classmethod
    def _custom_validation(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) \
            -> Self:
        """
        Model validator for applying custom validation functions specified in the
        `json_schema_extra` attribute of the fields.
        """
        # Get custom validators
        validators = cls._get_custom_validators()
        errors = []

        if isinstance(data, dict):
            # During model initialization data is a dict
            mode = 'dict'

            # Check for parsing of environment variables
            parse_env = data.get('parse_env', False)
            cls._parse_env = parse_env
        elif isinstance(data, cls):
            # During value assignment data is a pydantic model
            mode = 'pydantic'

            # Check for parsing of environment variables
            parse_env = data._parse_env
        else:
            raise TypeError(f'Invalid data type for validation, expected '
                            f'{dict.__class__.__name__} or {cls.__class__.__name__}'
                            f'but got: {type(data)} instead')

        # Apply before validators for raw data only
        if mode == 'dict':
            for name, value in data.items():
                if name not in validators:
                    continue
                # Replace environment variables in the value
                if parse_env:
                    value = cls._get_envvar(value)
                    # Update value
                    data[name] = value
                for validator in validators[name]['before']:
                    try:
                        # Call the validator function
                        value = validator.func(value)
                        # Update value
                        data[name] = value
                    except ValueError as e:
                        errors.append(InitErrorDetails(type='value_error', loc=(name,),
                                                       input=value, ctx={'error': e}))
            # Check and raise validation errors
            if errors:
                raise ValidationError.from_exception_data(title='before_validators',
                                                          line_errors=errors)

        # Generate the pydantic model
        validated_self = handler(data)

        # Apply after validators
        for name, value in validated_self.model_dump().items():
            if name not in validators:
                continue
            for validator in validators[name]['after']:
                try:
                    # Call the validator function
                    value = validator.func(value)
                    # Update value
                    if mode == 'dict':
                        setattr(validated_self, name, value)
                    elif mode == 'pydantic':
                        # Note: Workaround to prevent infinite recursion when using
                        #  setattr
                        validated_self.__dict__[name] = value
                except ValueError as e:
                    errors.append(InitErrorDetails(type='value_error', loc=(name,),
                                                   input=value, ctx={'error': e}))
        # Check and raise validation errors
        if errors:
            raise ValidationError.from_exception_data(title='after_validators',
                                                      line_errors=errors)
        return validated_self

    def __setattr__(self, name: str, value: Any) -> None:
        errors = []
        # Check if the attribute is a field
        if name in self.__dict__:
            if name in self._get_custom_validators():
                # Replace environment variables in the value
                if self._parse_env:
                    value = self._get_envvar(value)
                # Apply before validators
                for validator in self._get_custom_validators()[name]['before']:
                    try:
                        value = validator.func(value)
                    except ValueError as e:
                        errors.append(InitErrorDetails(type='value_error', loc=(name,),
                                                       input=value, ctx={'error': e}))
        # Check and raise validation errors
        if errors:
            raise ValidationError.from_exception_data(title='before_validators',
                                                      line_errors=errors)
        # Continue with the default behavior
        super().__setattr__(name, value)
