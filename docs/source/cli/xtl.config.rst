``xtl.config``
==============
The ``xtl.config`` command helps with finding, inspecting and creating the ``xtl.toml``
configuration file.

Config file locations
---------------------
XTL looks for ``xtl.toml`` in four locations, named ``env``, ``local``, ``user`` and
``global``, and reads the first one that exists. The commands below refer to them by
these names. See :ref:`config-locations` for where each one is.

Commands
--------
.. hint::
   Run ``xtl.config <command> --help`` for all the options of a command.

``where``
   List the config file locations, which one is active and whether each file is valid.

   .. code-block:: console

      $ xtl.config where

``show``
   Print the settings that XTL runs with, as TOML. Optionally, only a section or a key,
   or only the values that differ from the defaults.

   .. code-block:: console

      $ xtl.config show
      $ xtl.config show jobs.resources
      $ xtl.config show --diff

``validate``
   Check the active config file (or any other file) for errors and warnings.

   .. code-block:: console

      $ xtl.config validate
      $ xtl.config validate path/to/xtl.toml

``create``
   Create a config file in the current directory (default), or in another location with
   ``--env``, ``--user``, ``--global`` or ``--path``. The new file is a copy of the active
   settings; use ``--clean`` for the default settings.

   .. code-block:: console

      $ xtl.config create
      $ xtl.config create --user --clean
