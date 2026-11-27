Using ``xtl-core``
==================

``xtl-core`` holds the foundations of XTL that don't need the scientific stack: validated options, versioned data
layouts, tables and system utilities. It depends only on pydantic, tomlkit and a few small packages, so other projects
can use it without installing ``xtl``.

Installation
------------

.. code-block:: console

   $ pip install xtl-core

The conversions of :class:`~xtl_core.containers.tables.Table` to numpy, pandas and rich need the ``extra`` extra:

.. code-block:: console

   $ pip install "xtl-core[extra]"

``xtl-core`` supports Python 3.10 or newer.

Contents
--------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Package
     - Provides
   * - :mod:`xtl_core.options`
     - :class:`~xtl_core.options.base.Options`, pydantic models with validation, casting and TOML/JSON I/O
       (:doc:`options`), and :class:`~xtl_core.options.versioned.VersionedOptions`, which upgrade saved data
   * - :mod:`xtl_core.containers`
     - :class:`~xtl_core.containers.tables.Table`, 2D data with missing values (:doc:`tables`)
   * - :mod:`xtl_core.system`
     - file permissions (:mod:`~xtl_core.system.os`), environment variables
       (:mod:`~xtl_core.system.environment`), backports for older Pythons
       (:mod:`~xtl_core.system.compatibility`)
   * - :mod:`xtl_core.version_info`
     - :class:`~xtl_core.version_info.VersionInfo`, versions that compare and convert between formats
   * - :mod:`xtl_core.errors`
     - :class:`~xtl_core.errors.XTLCoreError`, the root of the exceptions of both packages

Example
-------

Settings for your own tool, read from TOML:

.. code-block:: python

   from pathlib import Path

   from xtl_core.options import Option, Options


   class MySettings(Options):
       output: Path = Option(default=Path('results'), desc='Where results are written')
       threads: int = Option(default=1, gt=0, desc='Number of threads')


   settings = MySettings.from_toml('threads = 4')
   settings.threads  # 4
   settings.threads = 0  # raises OptionsValidationError

Logging
-------

``xtl-core`` logs to loggers under ``xtl_core`` and adds only a :class:`~logging.NullHandler`: configure handlers and
levels in your application.
