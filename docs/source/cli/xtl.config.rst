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

``upgrade``
   Upgrade the active config file (or any other file) written by an older version of XTL to the current one, in
   place: only the settings that changed are edited, and your comments and formatting stay. ``--clean`` writes the
   file afresh instead, with XTL's descriptions as comments and without your own. A file from a newer version is
   left alone.

   .. code-block:: console

      $ xtl.config upgrade
      $ xtl.config upgrade path/to/xtl.toml --clean

``deps``
   Check which shells are installed, and whether each dependency (*e.g.* ATSAS) can be found with the compute site
   and shell that a job needing it would use. Exits with code 1 if a dependency is missing; ``-v`` shows what each
   check printed. See :doc:`/configuration/dependencies`.

   .. code-block:: console

      $ xtl.config deps
      $ xtl.config deps atsas -v
      $ xtl.config deps atsas --compute-site modules --shell powershell

``create``
   Create a config file in the current directory (default), or in another location with
   ``--env``, ``--user``, ``--global`` or ``--path``. The new file is a copy of the active
   settings; use ``--clean`` for the default settings.

   .. code-block:: console

      $ xtl.config create
      $ xtl.config create --user --clean
