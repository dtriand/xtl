from annotated_types import SupportsGe, SupportsGt, SupportsLe, SupportsLt
from functools import partial
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Optional
from typing_extensions import Self

from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr, model_validator,
                      BeforeValidator, AfterValidator, ModelWrapValidatorHandler)
from pydantic_core import PydanticUndefined, InitErrorDetails, ValidationError
from pydantic.config import JsonDict
from pydantic.fields import _Unset, Deprecated, FieldInfo
import toml
from toml.decoder import CommentValue

from xtl.config.validators import *
from xtl.files.toml import ExtendedTomlEncoder


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
    Create a `pydantic.Field` with custom validation and more intuitive metadata
    handling.

    Custom validation functions are stored in the `json_schema_extra` attribute of the
    field and then applied during model validation, if the model is of type `Options`.
    If applied to a model that directly inherits from `pydantic.BaseModel`, then all
    custom validation will simply be ignored.

    :param default: Default value for the field.
    :param default_factory: A callable to generate the default value.
    :param name: Name of the field (equivalent to `title` in `pydantic`).
    :param desc: Description of the field (equivalent to `description` in `pydantic`).
    :param examples: Example values for this field.
    :param alias: Alias for the field (equivalent to `serialization_alias` in
        `pydantic`).
    :param exclude: Exclude this field from serialization.
    :param repr: Include this field in the model's `__repr__` method.
    :param deprecated: Deprecation message for the field.
    :param validate_default: Whether to validate the default value.
    :param choices: Iterable of valid values constraint for the field.
    :param gt: Greater than constraint for numeric fields.
    :param ge: Greater than or equal to constraint for numeric fields.
    :param lt: Less than constraint for numeric fields.
    :param le: Less than or equal to constraint for numeric fields.
    :param multiple_of: Multiple of constraint for numeric fields.
    :param allow_inf_nan: Allow `inf` and `nan` values for numeric fields.
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
    :param strict: Strict type checking (i.e., no implicit conversion/type coersion).
    :param cast_as: Type or callable to cast the value to prior to validation.
    :param extra: Extra JSON schema information (equivalent to `json_schema_extra` in
        `pydantic`).
    :return: A `pydantic.FieldInfo` object with custom validation and metadata handling.
    """
    if default is PydanticUndefined and default_factory is _Unset:
        raise ValueError('Either \'default\' or \'default_factory\' must be provided')

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

    if default_factory is not _Unset:
        f = partial_field(default_factory=default_factory)
    else:
        f = partial_field(default=default)
    return f


class Options(BaseModel):
    """
    Base class for storing various configuration options. This class provides
    validation and serialization through `Pydantic`. Custom validators are also
    supported (see `Option` for more details).

    Fields can be defined using `Option` or `pydantic.Field`, but custom validation
    will only be applied if `Option` is used.

    In addition to the standard `Pydantic` serialization methods, this class can also
    be serialized to a TOML file. Deserialization is possible from Python `dict`s and
    JSON and TOML strings.

    Parsing of environment variables in string values is possible by setting
    `parse_env=True` in the constructor. This will replace any environment variable
    references in the format `${VARIABLE}` with their values. If the variable is not
    found, it will be replaced with an empty string. The default behavior is to not
    parse environment variables.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        use_enum_values=True
    )
    _parse_env: bool = PrivateAttr(default=False)

    @staticmethod
    def _get_envvar(data: Any) -> Any:
        """
        Replace environment variables in the data with their values. Environment
        variables are expected to be in the format `${VARIABLE}`. If the variable is
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

    @staticmethod
    def _apply_validators(name: str, value: Any,
                          validators: list[AfterValidator | BeforeValidator],
                          errors: list[InitErrorDetails] = None) \
            -> tuple[Any, list[InitErrorDetails]]:
        """
        Apply validators to the value. This is used to perform a series of validations
        on the value. If any of the validators raise a `ValueError`, the error is
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

        for validator in validators:
            try:
                value = validator.func(value)
            except ValueError as e:
                new_errors.append(InitErrorDetails(type='value_error', loc=(name,),
                                                   input=value, ctx={'error': e}))

        if new_errors:
            return _Unset, errors + new_errors
        else:
            return value, errors

    @classmethod
    def _validate_before(cls, name: str, value: Any,
                         validators: dict[str, list[BeforeValidator | AfterValidator]],
                         errors: list[InitErrorDetails] = None,
                         parse_env: bool = False) -> tuple[Any, list[InitErrorDetails]]:
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

        if parse_env:
            value = cls._get_envvar(value)

        # Apply validators
        value, new_errors = cls._apply_validators(name=name, value=value,
                                                  validators=validators['before'],
                                                  errors=errors)
        return value, new_errors

    @classmethod
    def _validate_after(cls, name: str, value: Any,
                        validators: dict[str, list[BeforeValidator | AfterValidator]],
                        errors: list[InitErrorDetails] = None) \
            -> tuple[Any, list[InitErrorDetails]]:
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
        value, new_errors = cls._apply_validators(name=name, value=value,
                                                  validators=validators['after'],
                                                  errors=errors)
        return value, new_errors

    @model_validator(mode='wrap')
    @classmethod
    def _custom_validation(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) \
            -> Self:
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
                new_errors = []
                if name not in validators:
                    continue
                value, new_errors = cls._validate_before(name=name, value=value,
                                                         validators=validators[name],
                                                         errors=new_errors,
                                                         parse_env=parse_env)

                if not new_errors:
                    # Update value
                    data[name] = value

                errors.extend(new_errors)
            # Check and raise validation errors
            if errors:
                raise ValidationError.from_exception_data(title='before_validators',
                                                          line_errors=errors)

        # Generate the pydantic model
        validated_self = handler(data)

        # Apply after validators
        for name, value in validated_self.model_dump().items():
            new_errors = []
            if name not in validators:
                continue
            value, new_errors = cls._validate_after(name=name, value=value,
                                                    validators=validators[name],
                                                    errors=new_errors)
            if not new_errors:
                # Note: Workaround to prevent infinite recursion when using
                #  setattr()
                validated_self.__dict__[name] = value
            errors.extend(new_errors)
        # Check and raise validation errors
        if errors:
            raise ValidationError.from_exception_data(title='after_validators',
                                                      line_errors=errors)
        return validated_self

    def __setattr__(self, name: str, value: Any) -> None:
        # Check if the attribute is a pydantic field
        if name not in self.__pydantic_fields__:
            super().__setattr__(name, value)
            return

        # Apply before validators
        validators = self._get_custom_validators()
        errors = []
        if name in validators:
            value, errors = self._validate_before(name=name, value=value,
                                                  validators=validators[name],
                                                  parse_env=self._parse_env)
        # Check and raise validation errors
        if errors:
            raise ValidationError.from_exception_data(title='before_validators',
                                                      line_errors=errors)

        try:
            # Validate the model with the new value
            data = self.model_dump()
            data.update({name: value})
            self.model_validate(data)
            # Update the attribute if validation is successful
            super().__setattr__(name, value)
        except ValidationError as e:
            raise e

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Options to a dictionary. Nested Options objects will be
        converted to nested dictionaries.

        :return: A dictionary representation of the Options.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a new Options object from a dictionary. Nested dictionaries can be
        used to create nested Options objects.

        :param data: Dictionary containing the values to create the config with.
        :return: An Options object.
        """
        return cls(**data)

    def to_json(self, filename: Optional[str | Path] = None, overwrite: bool = False,
                keep_file_ext: bool = False, indent: Optional[int] = 4) -> str | Path:
        """
        Write the Options to a JSON file. If `filename` is not provided, then the JSON
        will be returned as a string.

        :param filename: Optional output path for the JSON file.
        :param comments: Include comments in the JSON file.
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the file extension if `filename` is provided.
        :param indent: Indentation level for the JSON string. If `None`, then the JSON
            will be compact.
        :return: Either the JSON string or the output path.
        :raises FileExistsError: If the file already exists and `overwrite` is False.
        """
        # If no filename is provided, then return a JSON string
        if filename is None:
            return self.model_dump_json(indent=indent)

        # If a filename is provided, then write to a file
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File {filename} already exists')
        if filename.suffix != '.json' and not keep_file_ext:
            filename = filename.with_suffix('.json')

        filename.write_text(self.model_dump_json(indent=indent))
        return filename

    @classmethod
    def from_json(cls, s: str | Path) -> Self:
        """
        Create a new Options object from a JSON string or file.

        :param s: JSON string or path to a JSON file.
        :return: An Options object.
        :raises json.JSONDecodeError: If the JSON is invalid.
        :raises FileNotFoundError: If the file does not exist.
        """
        data = {}
        if isinstance(s, str):
            # Check if the string is a valid TOML
            try:
                data = json.loads(s)
            except json.JSONDecodeError as e:
                # If not, check if it's an existing file path
                s = Path(s)
                if not s.exists():
                    # It was probably an invalid JSON string
                    raise e
                else:
                    pass
        # If data is still empty, then try interpreting s as a file
        if not data:
            s = Path(s)
            if not s.exists():
                raise FileNotFoundError(f'File not found: {s}')
            data = json.loads(s.read_text())
        return cls(**data)

    def _field_to_comment_value(self, name: str, field: FieldInfo,
                                keep_comments: bool = False) -> \
            CommentValue | dict[str, Any]:
        """
        Cast a pydantic.FieldInfo to toml.CommentValue, where the comment is set to the
        field's description

        :param name: Name of the field.
        :param field: FieldInfo object.
        :param keep_comments: Whether to include comments in the output.
        :return: A CommentValue object or a dictionary of nested CommentValue objects.
        """
        # Check if the field is a valid pydantic field
        if name not in self.__pydantic_fields__:
            raise KeyError(f'Field {name!r} not defined')

        # Check if the field is another Options object
        value = getattr(self, name)
        if isinstance(value, Options):
            # Recursively convert nested Options objects to CommentValue
            return {vname: value._field_to_comment_value(name=vname, field=vfield,
                                                         keep_comments=keep_comments)
                    for vname, vfield in value.__pydantic_fields__.items()}

        # Prepare comment
        comment = f'# {field.description}' if field.description and keep_comments else ''

        return CommentValue(val=value, comment=comment, beginline=False, _dict=dict)

    def to_toml(self, filename: Optional[str | Path] = None, comments: bool = False,
                overwrite: bool = False, keep_file_ext: bool = False) -> str | Path:
        """
        Write the Options to a TOML file. If `filename` is not provided, then the TOML
        will be returned as a string.

        :param filename: Optional output path for the TOML file.
        :param comments: Include comments in the TOML file.
        :param overwrite: Overwrite the file if it already exists.
        :param keep_file_ext: Keep the file extension if `filename` is provided.
        :return: Either the TOML string or the output path.
        :raises FileExistsError: If the file already exists and `overwrite` is False.
        """
        # Cast all fields to toml.CommentValue
        data = {
            name: self._field_to_comment_value(name=name, field=field,
                                               keep_comments=comments)
            for name, field in self.__pydantic_fields__.items()
        }

        encoder = ExtendedTomlEncoder()

        # If no filename is provided, then return a TOML string
        if filename is None:
            return toml.dumps(data, encoder=encoder)

        # If a filename is provided, then write to a file
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File already exists: {filename}')
        if filename.suffix != '.toml' and not keep_file_ext:
            filename = filename.with_suffix('.toml')

        filename.write_text(toml.dumps(data, encoder=encoder))
        return filename

    @classmethod
    def from_toml(cls, s: str | Path) -> Self:
        """
        Create a new Options object from a TOML string or file.

        :param s: TOML string or path to a TOML file.
        :return: An Options object.
        :raises toml.TomlDecodeError: If the TOML is invalid.
        :raises FileNotFoundError: If the file does not exist.
        """
        decoder = toml.TomlDecoder()
        data = {}
        if isinstance(s, str):
            # Check if the string is a valid TOML
            try:
                data = toml.loads(s, decoder=decoder)
            except toml.TomlDecodeError as e:
                # If not, check if it's an existing file path
                s = Path(s)
                if not s.exists():
                    # It was probably an invalid TOML string
                    raise e
                else:
                    pass
        # If data is still empty, then try interpreting s as a file
        if not data:
            s = Path(s)
            if not s.exists():
                raise FileNotFoundError(f'File not found: {s}')
            data = toml.loads(s.read_text(), decoder=decoder)
        return cls.from_dict(data)
