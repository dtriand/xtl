from dataclasses import dataclass, Field
from dataclasses import field as _field


@dataclass
class AnnotatedDataclass:

    def _derive_type(self, field_: Field):
        t = field_.type
        if hasattr(t, '__origin__'):
            # For parsing generic types e.g. list[str]
            return t.__origin__, t.__args__[0]
        return t, None

    def _typecheck_param(self, param: Field):
        value = getattr(self, param.name)
        if value is None:
            return
        # Derive the type and nested type of the field
        t, nt = self._derive_type(param)
        if not isinstance(value, t):
            raise TypeError(f'Invalid parameter {param.name}\nExpected type {t}, got {type(value)}')
        if nt:
            for i, v in enumerate(value):
                if not isinstance(v, nt):
                    raise TypeError(f'Invalid parameter {param.name}[{i}]\nExpected type {nt}, got {type(v)}')
        return

    def _cast_param(self, param: Field):
        value = getattr(self, param.name)
        if value is None:
            return
        # Derive the type and nested type of the field
        t, nt = self._derive_type(param)
        if not isinstance(value, t):
            return t(value)
        if nt:
            return t(nt(v) for v in value)
        return value

    def _validate_param(self, param: Field):
        value = getattr(self, param.name)
        if value is None:
            return
        if param.metadata.get('param_type') == 'compound':
            for subparam in param.metadata.get('members', []):
                if not hasattr(self, subparam):
                    raise ValueError(f'Undefined member parameter {subparam} for {param.name}')
        validator = param.metadata.get('validator', {})
        if 'choice' in validator:
            if value not in validator['choice']:
                raise ValueError(f'Invalid value for {param.name}\nExpected one of {validator["choices"]}, got {value}')
        if isinstance(value, str | list | tuple):
            if 'len' in validator:
                if len(value) != validator['len']:
                    raise ValueError(f'Invalid value for {param.name}\nExpected length {validator["len"]}, '
                                     f'got {len(value)}')
        if isinstance(value, list | tuple):
            if 'choices' in validator:
                for i, v in enumerate(value):
                    if v not in validator['choices']:
                        raise ValueError(f'Invalid value for {param.name}[{i}]\nExpected one of {validator["choices"]}, '
                                         f'got {v}')
        if isinstance(value, float | int):
            if 'gt' in validator:
                if value <= validator['gt']:
                    raise ValueError(f'Invalid value for {param.name}\nExpected value > {validator["gt"]}, got {value}')
            if 'ge' in validator:
                if value < validator['ge']:
                    raise ValueError(f'Invalid value for {param.name}\nExpected value >= {validator["ge"]}, '
                                     f'got {value}')
            if 'lt' in validator:
                if value >= validator['lt']:
                    raise ValueError(f'Invalid value for {param.name}\nExpected value < {validator["lt"]}, got {value}')
            if 'le' in validator:
                if value > validator['le']:
                    raise ValueError(f'Invalid value for {param.name}\nExpected value <= {validator["le"]}, '
                                     f'got {value}')

    def _typecheck_all(self):
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('param_type', None) == '__internal':
                continue
            self._typecheck_param(param)

    def _cast_all(self):
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('param_type', None) == '__internal':
                continue
            try:
                setattr(self, param.name, self._cast_param(param))
            except ValueError:
                raise ValueError(f'Invalid value for {param.name}: {getattr(self, param.name)}\n'
                                 f'Expected type {param.type}, got {type(getattr(self, param.name))}')

    def _validate_all(self):
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('param_type', None) == '__internal':
                continue
            self._validate_param(param)

    def __post_init__(self):
        try:
            self._typecheck_all()
        except TypeError:
            self._cast_all()

        # Synchronize compound parameters
        for param in self.__dataclass_fields__.values():
            if param.metadata.get('param_type', None) == 'compound':
                if param.metadata.get('in_sync', False):
                    for subparam in param.metadata.get('members', []):
                        setattr(self, subparam, getattr(self, param.name))

        self._validate_all()


def afield(**kwargs) -> Field:
    """
    Annotated dataclass field with additional metadata. Validation options can be passed as a `validator` dict, while
    special formatting options as a `formatter` function.
    """
    kwargs['param_type'] = kwargs.get('param_type', 'standard')
    default = kwargs.pop('default', None)
    default_factory = kwargs.pop('default_factory', None)
    if default_factory:
        return _field(default_factory=default_factory, metadata=kwargs)
    return _field(default=default, metadata=kwargs)


def cfield(**kwargs) -> Field:
    """
    Compound annotated dataclass field with links to multiple members. The members of this field can be passes as a
    `members` list. An extra `in_sync` flag can be toggled if all members should accept the parent's value.
    """
    kwargs['param_type'] = 'compound'
    return afield(**kwargs)


def pfield(**kwargs) -> Field:
    """
    Private annotated dataclass field for hidden internal attributes. This field will undergo type checking, type
    casting and validation.
    """
    kwargs['param_type'] = 'private'
    return afield(**kwargs)


def _ifield(**kwargs) -> Field:
    """
    Hidden internal fields that will skip type checking, type casting and validation.
    """
    kwargs['param_type'] = '__internal'
    return afield(**kwargs)
