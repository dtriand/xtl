.. |Option| replace:: :func:`Option() <xtl.common.options.Option>`
.. |Options| replace:: :class:`Options <xtl.common.options.Options>`
.. |pydantic| replace:: :mod:`pydantic`
.. |Field| replace:: :func:`pydantic.Field`
.. |BaseModel| replace:: :class:`pydantic.BaseModel`

Options
=======

The :mod:`xtl.common.options` module provides an API around |BaseModel| that extends the existing validation and
serialization capabilities in a user-friendly format. |Options| is a powerful data container that, in combination with
|Option|, can facilitate arbitrary validators and serializers in a slim syntax. This module is used internally to parse
configuration from the ``xtl.toml`` file and to validate any modifications on the :attr:`xtl.settings` object.

Quickstart
----------

The module provides two objects: |Option| and |Options|, which are wrappers around |Field| and |BaseModel|,
respectively. Below is a simple example that showcases what is possible with this API:

.. code-block:: python

    from pathlib import Path
    from pydantic import Field
    from xtl.common.options import Option, Options

    class Employee(Options):
        name: str = Option(
            default=None,
            formatter=lambda x: x.title()
        )
        age: int = Field(
            default=None,
            gt=0
        )
        level: str = Option(
            default=None,
            choices=['junior', 'senior'],
            alias='job_level'
        )
        contract_file: Path = Option(
            default=None,
            cast_as=Path,
            path_exists=True,
            path_is_file=True
        )

    employee = Employee(
        name='john doe',
        age=30,
        level='senior',
        contract_file='/path/to/contract.txt'
    )

Now if we call :meth:`employee.to_dict() <xtl.common.options.Options.to_dict>` (which is equivalent to
:func:`employee.model_dump() <pydantic.BaseModel.model_dump>` if you prefer the `Pydantic` API),
we get the following back:

.. code-block:: python

   >>> employee.to_dict()
   {'name': 'John Doe',
   'age': 30,
   'contract_file': Path('/path/to/contract.txt'),
   'job_level': 'senior'}

We notice a few things:

#. The ``name`` attribute got formatted in title case (due to ``formatter``), however the stored value is still
   ``'john doe'``:

   >>> employee.name
   'john doe'

   The ``formatter`` option only affects the serialization of the field.

#. The ``age`` attribute was defined as a |Field| instead and it is still compatible with our model

#. The ``level`` attribute was renamed to ``job_level`` (due to ``alias='job_level'`` which sets the
   ``serialization_alias`` on the `Pydantic` field), however the attribute is still accessible through its original name:

   >>> employee.level
   'senior'
   >>> employee.job_level
   AttributeError: 'Employee' object has no attribute 'job_level'

   The ``alias`` option affects only model serialization.

#. We checked whether the value for ``level`` is one of ``['junior', 'senior']`` (due to ``choices``). If we try to
   change the value to something not in the list of choices we get a validation error:

   >>> employee.level = 'ceo'
   pydantic_core._pydantic_core.ValidationError: 1 validation error for Employee
   level
     Value error, Value is not in choices: ['junior', 'senior'] [type=value_error, input_value='ceo', input_type=str]

   .. note::

      The |Options| class has :attr:`validate_assignment <pydantic.config.ConfigDict.validate_assignment>` set
      to ``True``.

#. The ``contract_file`` was passed along as a ``str`` but is now ``Path`` (due to ``cast_as=Path``, which performs
   type casting before `Pydantic`\'s model validation)

#. We checked whether ``contract_file`` exists and whether it is a file (due to ``path_exists=True`` and
   ``path_is_file=True``, which add custom `Pydantic` validators)

Data validation
---------------

`Pydantic` comes with a variety of prebuilt validators that can be defined on the |Field| level. |Option| provides
support for all the existing `Pydantic` validators, plus a few extra ones. The complete list of validators is as
follows:

#. Numerical fields

   * ``gt``: greater than
   * ``ge``: greater than or equal to
   * ``lt``: less than
   * ``le``: less than or equal to
   * ``multiple_of``: multiple of
   * ``allow_inf_nan``: allow infinity or NaN values

#. Decimal fields

   * ``max_digits``: maximum number of digits
   * ``decimal_places``: number of decimal places
   * all validators for numerical fields

#. Iterable fields

   * ``length``\*: number of items in the iterable
   * ``min_length``: minimum number of items in the iterable
   * ``max_length``: maximum number of items in the iterable

#. String fields

   * ``regex``: regular expression
   * all validators for iterable fields

#. Path fields

   * ``path_exists``\*: path exists
   * ``path_is_file``\*: path is a file
   * ``path_is_dir``\*: path is a directory
   * ``path_is_absolute``\*: path is absolute

#. Enumeration fields

   * ``choices``\*: list of valid values

Validators marked with an asterisk (*) are implemented by XTL directly.

Type casting
^^^^^^^^^^^^
A special case of validator is generated when using the ``cast_as`` argument. This creates a
:class:`pydantic.BeforeValidator` that is called on the raw input data before any other validation is performed, and
enables type casting.

`Pydantic` already performs type coercion on certain values by default, unless ``strict=True`` is explicitly set either
on |Field| or |BaseModel|. For example, in if a model field is defined as ``int``, then passing a integer as ``str`` or
as ``float`` will be coerced into type ``int``:

.. code-block:: python

    from pydantic import BaseModel, Field

    class CoercionModel(BaseModel):
        x: int = Field(default=None)

    model = CoercionModel(x=1.0)
    assert model.x == 1
    assert isinstance(model.x, int)

    model = CoercionModel(x='1')
    assert model.x == 1
    assert isinstance(model.x, int)

Details about `Pydantic`'s type coercion can be found in the
`documentation <https://docs.pydantic.dev/latest/concepts/conversion_table/>`_.

The ``cast_as`` argument provides some additional functionality, where it first checks if the value is of the correct
type, and only attempts to type cast if not. An exemplary use case would be when an argument needs to be defined as a
``list[Any]`` but either a list or a single value could be passed:

.. code-block:: python
   :emphasize-lines: 6

    from xtl.common.options import Option, Options

    class CoercionModel(Options):
        x: list[str] = Option(default=None, cast_as=list)

    model = CoercionModel(x='1')
    assert model.x == ['1']
    assert isinstance(model.x, list)

    model = CoercionModel(x=['1'])
    assert model.x == ['1']
    assert isinstance(model.x, list)

`Pydantic` would have failed to coerce the value into a list in the first call.

Type casting is also useful when using custom types in models:

.. code-block:: python

   from xtl.common.options import Option, Options

   class Animal:
       def __init__(self, species: str):
           self.species = species

   class Pet(Options):
       animal: Animal = Option(default=None, cast_as=Animal)
       name: str = Option(default=None)

   pet = Pet(animal='Tyrannosaurus rex', name='Otto')
   assert isinstance(pet.animal, Animal)

.. note::

   |Options| has :attr:`arbitrary_types_allowed <pydantic.config.ConfigDict.arbitrary_types_allowed>` set to ``True``.

Custom validation
^^^^^^^^^^^^^^^^^

In `Pydantic` custom field validation can be done by defining methods that have been decorated with the
:func:`@field_validator <pydantic.field_validator>` decorator. In |Options| this can additionally be achieved by passing
a custom validation function to the ``validator`` argument of |Option|.

Any custom validation function should be first wrapped in a :class:`pydantic.BeforeValidator` or
:class:`pydantic.AfterValidator`, depending on whether it should be run on the raw input data, or the already
validated data.

.. code-block:: python

   from typing import Any
   from pydantic import AfterValidator
   from xtl.common.options import Option, Options

   def is_odd(value: Any) -> Any:
       if value % 2 == 0:
           raise ValueError('Value must be odd')
       return value

   IsOddValidator = AfterValidator(is_odd)

   class OddNumber(Options):
       number: int = Option(default=None, validator=IsOddValidator)

   number = OddNumber(number=4)
   # raises pydantic_core._pydantic_core.ValidationError: 1 validation error for OddNumber
   # number
   #   Value error, Value is not odd [type=value_error, input_value=4, input_type=int]

Note the signature of the :func:`is_odd` function: It accepts a single argument, and returns it as is if the validation
is passed, otherwise it raises a :class:`ValueError`. Once all :class:`pydantic.AfterValidator` have been executed, any
:class:`ValueError` that might have been raised will get aggregated into a single
:class:`pydantic.ValidationError <pydantic_core.ValidationError>`.

A few more important notes about crafting custom validators:

* **Value immutability**: |Options| allows for validators to mutate the value. This is, for example, the case for the
  :func:`CastAsValidator <xtl.common.validators.CastAsValidator>` which performs type casting. This ensure that the
  right value is propagated to the next validator. The model value is, however, only mutated from the input value if
  all validators succeed.

  .. caution::

     In general, validators should only check the value. Be very careful when mutating the value, as this can lead to
     unexpected behavior, for example in dictionaries.

  .. seealso::

     Mutating the value affects the way it is stored internally. If you find yourself crafting validators that change
     the way the value is output upon serialization, you should be creating a ``formatter`` instead (see:
     `Data formatting`_)

* **Validation order**: Validators are applied sequentially. The order of execution is as follows:

  #. *Raw data received*
  #. :class:`pydantic.BeforeValidator` by XTL (if any)
  #. :class:`pydantic.BeforeValidator` by user (if any)
  #. `Pydantic`\'s internal validation and model initialization logic
  #. :class:`pydantic.AfterValidator` by XTL (if any)
  #. :class:`pydantic.AfterValidator` by user (if any)
  #. *Model instance is returned*

Data formatting
---------------

Once our model has been instantiated, we might want to pass all its data to another function, save it in a file, display
it to the user, *etc*. Many times the way the data is stored inside the model and the way it is presented to the outside
world have different requirements. In general, this concept is called
`serialization <https://docs.pydantic.dev/latest/concepts/serialization/>`_ in `Pydantic`, and in XTL it can be
controlled by the ``formatter`` and ``alias`` arguments in |Option|.

.. code-block:: python

   from pathlib import Path
   from xtl.common.options import Option, Options

   def PathUri(value: Path) -> str:
       return value.as_uri()

   class Config(Options):
       a: int = Option(default=None, alias='param_a')
       b: str = Option(default=None, alias='param_b', formatter=lambda x: x.upper())
       file: Path = Option(default=None, formatter=PathUri)

.. code-block:: python

   >>> c = Config(a=1, b='value', file=Path('/path/to/file'))
   >>> c.to_dict()
   {'param_a': 1,
   'param_b': 'VALUE',
   'file': 'file:///path/to/file'}

Model I/O
---------

|Options| models can be serialized to and deserialized from Python dictionaries, JSON and TOML files and/or strings.
There are aptly named methods (``from_*``/``to_*``) for each purpose.

.. code-block:: python

   from xtl.common.options import Option, Options

   class NestedModel(Options):
       a: int = Option(default=None, formatter=lambda x: 10 * x,
                       desc='This is 10 times the original')

   class ParentModel(Options):
       b: float = Option(default=None, alias='c',
                         desc='This is an aliased value')
       nested: NestedModel = Option(default_factory=NestedModel)

   p = ParentModel.from_dict({'c': 3., 'nested': {'a': 2}})
   p.to_toml('output.toml', comments=True)

This will output an ``output.toml`` file, which will include all the field descriptions as inline comments to the
values.

.. code-block:: text
   :caption: ``output.toml``

   c = 3.0 # This is an aliased value

   [nested]
   a = 20 # This is 10 times the original

Environment variables
---------------------

|Options| also support parsing of environment variables on string fields. Due to safety considerations, this behaviour
is disabled by default. To enable environment variable parsing, you need to pass ``_parse_env=True`` during model
instantiation. Any strings that contain expressions in the form of ``${VARIABLE}`` will be replaced by the value of the
respective environment variable.

.. code-block:: python

   import os
   from xtl.common.options import Option, Options

   class Settings(Options):
       a: str = Option(default=None)

.. code-block:: python
   :emphasize-lines: 4

   >>> os.environ['SECRET'] = 'alakazam'
   >>> Settings(a='secret is ${SECRET}').to_dict()
   {'a': 'secret is ${SECRET}'}
   >>> Settings(a='secret is ${SECRET}', _parse_env=True).to_dict()
   {'a': 'secret is alakazam'}

If an environment variable is not set, then the content will be replaced with an empty string. The parsing of
environment variables is a mutable operation, meaning that the value of the field is replaced with that of the
variable. Changing the environment variable after model instantiation will have no effect on the field's value, unless
the same expression is reassigned to the field, and the model undergoes validation again.

.. important::

   The ``_parse_env`` argument is only local in scope, meaning that it only applies to the current model. In case of
   nested |Options| models, environment variable parsing will **not** be propagated from the parent model to its childs.

