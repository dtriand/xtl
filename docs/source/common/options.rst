Options
=======
.. |Option| replace:: :class:`Option <xtl.common.options.Option>`
.. |Options| replace:: :class:`Options <xtl.common.options.Options>`
.. |pydantic| replace:: :mod:`pydantic`
.. |Field| replace:: :func:`pydantic.Field`
.. |BaseModel| replace:: :class:`pydantic.BaseModel`

The :mod:`xtl.common.options` module provides an API around |BaseModel| that extends the existing validation and
serialization capabilities in a user-friendly format. This module is used internally to parse configuration from the
``xtl.toml`` file and to validate any modifications on the :attr:`xtl.settings` object.

The module comes with two objects: |Option| and |Options|, which are wrappers around |Field| and |BaseModel|,
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
:func:`employee.model_dump() <pydantic.BaseModel.model_dump>` if you prefer the Pydantic API),
we get the following back:

.. parsed-literal::

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
   ``serialization_alias`` on the Pydantic field), however the attribute is still accessible through its original name:

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

   Note that the |Options| class has :attr:`validate_assignment <pydantic.config.ConfigDict.validate_assignment>` set
   to ``True``.
#. The ``contract_file`` was passed along as a ``str`` but is now ``Path`` (due to ``cast_as=Path``, which performs
   type casting before Pydantic's model validation)
#. We checked whether ``contract_file`` exists and whether it is a file (due to ``path_exists=True`` and
   ``path_is_file=True``, which add custom Pydantic validators)

