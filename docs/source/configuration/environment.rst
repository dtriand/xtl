Environment variables
=====================
These environment variables change how XTL behaves. An empty variable counts as unset.

.. _env-xtl-config:

``XTL_CONFIG``
--------------
Path to the config file to use, ahead of all other config file locations (see
:ref:`config-locations`). If the file does not exist, it is created with the default
settings. Environment variables and ``~`` in the path are expanded. A path to a
directory is ignored with a warning.

Useful for keeping separate settings per project, job or test run:

.. set-env:: XTL_CONFIG /path/to/my-config.toml
   :windows: C:\path\to\my-config.toml

.. _env-xtl-compute-site:

``XTL_COMPUTE_SITE``
--------------------
Enables compute sites that only exist at specific facilities. Currently, the only
recognised value is ``BIOTIX_HPC``, which adds the ``biotix`` compute site. Set it before
starting Python, since it is read when the compute sites are first imported.

.. set-env:: XTL_COMPUTE_SITE BIOTIX_HPC

.. seealso::
   :mod:`xtl.common.environment` for the API.
