"""
#############
Configuration
#############

XTL was build with customization and ease-of-use in mind. Many functions and command-line scripts can be easily
customized, without having to write your own script. All customizable parameters are stored centrally in a global
*xtl.cfg* file, which is located inside the package's directory. This global config file is generated the first time the
package is imported and is updated throughout usage, as needed.

Programmatically, the config can be accessed via the :obj:`cfg` object:

.. code-block:: python

   from xtl import cfg


Default options
---------------
The *xtl.cfg* file is generated with the following default options:

.. code-block:: ini

   [xtl]
   version = 0.0.0

   [dependencies]
   # Path for GSAS2 installation
   gsas =

.. note::

   Values for optional dependencies, such as GSAS-II, are blank by default. These options will be populated with
   user-provided values, upon first usage of a relevant feature.

Global and local configuration
------------------------------
Whenever XTL is imported, it checks for a *xtl.cfg* file to generate it's settings. First, it seeks for a local config
file in the current working directory, and if it does not find one, then it loads the global config file. Therefore, you
can easily customize the behaviour on a per-project basis, without having to constantly tweak the global config.

If neither a local or global config file can be located, then a global config will be generated with the default options
and will immediately be loaded to :obj:`cfg`.

.. tip::

   In case your global config exhibits some issues, that prevent XTL from loading, you can delete the global config and
   generate a new, default one. Alternatively, a new *xtl.cfg* file can be generated via the command-line
   (see :ref:`cli`).

Basic usage
-----------
XTL uses `ConfigUpdater <https://configupdater.readthedocs.io/en/latest/>`_ for parsing and manipulating config files.
The :obj:`cfg` object is child of :class:`.config.Config` class, which inherits from ``configupdater.ConfigUpdater``.

Retrieving values
^^^^^^^^^^^^^^^^^

.. code-block:: python

   cfg['section']['option'].value
   cfg.get('section', 'option').value

Changing values
^^^^^^^^^^^^^^^

.. code-block:: python

   cfg['section']['option'] = new_value
   cfg.set('section', 'option', new_value)

Any changes made to values affect only the :obj:`cfg` object and not *xtl.cfg*.

Saving changes to file
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cfg.save()

Changes to the *xtl.cfg* file are also made when triggering the :meth:`.Config.upgrade_config`,
:meth:`.Config.restore_config`, :meth:`.Config.register_options` methods.

Registering new options
-----------------------

If you are writing a plugin for XTL or simply want to add your own options, save them to file and access them again
later, you can use the functionality of the :obj:`cfg`, that is already in-place. New options are deliberately added
under a new section, to prevent overwriting of default options.

A new section with its own options must be provided in the form of a dictionary:

.. code-block:: python

   cfg.register_options({'section': {'option1': 'value1', 'option2': 'value2'}})

The new options are immediately accessible by the :obj:`cfg` object, and are also added to the *xtl.cfg* file (whether
local or global).

.. code-block:: ini

   [xtl]
   version = 0.0.0

   [dependencies]
   # Path for GSAS2 installation
   gsas =

   [section]
   option1 = value1
   option2 = value2

.. note::

   The :meth:`.Config.register_options` method needs to be called only once to save the options to the config file.
   Subsequent loadings of the file, still provide access to the options. However, if you want to perform validation on
   the values, based on provided restraints (see :meth:`.Config.validate`), or retain your options after a restore
   (see :meth:`.Config.restore_config`) or upgrade operation (see :meth:`.Config.upgrade_config`), you need to register
   your options every time you load your script.

   It is, therefore, strongly advised that you keep :meth:`.Config.register_options` in your package's `__init__.py`. Do
   note that calling :meth:`.Config.register_options` repeatedly, **does not** change existing values, thus it does not
   overwrite any user values.

In order to remove all entries from the config file, you can register the same ``'section'`` with an empty dictionary.

.. code-block:: python

   cfg.register_options({'section': {}})

Values, comments and restraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of adding only options and values to the config file, XTL supports comments and can also perform validation
on user values for your options. The validation restraint and the comment are provided in the same string as the
value, and are separated from the value with a ``$`` and a ``#`` symbol.

.. code-block:: python

   content = 'value $ restraint # comment'
   # e.g. '2 $ 1 <= x <= 5 # My option'
   cfg.register_options({'section': {'option': content}})

In the example above, the value is ``'2'``, the restraint is ``'1 <= x <= 5'`` and ``'My option'`` is the comment.
Comments are added immediately before the option and are helpful for providing annotation to your configuration.

.. code-block:: ini

   [section]
   # My option
   option = 2

Each option-value string that is passed to :meth:`.Config.register_options` does not need to contain all three
arguments, i.e. a value, a restraint and a comment. Any arbitrary combination of the three is parsed correctly, e.g.
value and comment, restraint only, etc. In case all three arguments are provided, they *necessarily* need to be in
the following order: ``value $ restraint # comment``, otherwise, they will not be parsed correctly.

Restraints are expressions that evaluate to True of False in a Python console. Each restraint should contain a ``x``
variable, which is interpreted as the current value for the option. Restraint validation is performed with the
following (simplified) piece of code:

.. code-block:: python

   x = value  # e.g. x = 2
   if not eval(restraint):  # e.g. 0 <= x <= 5: True
       raise Exception

.. caution::

   It is obvious that restraint evaluation is susceptible to code injections, that could perform arbitrary
   python code. *Use at your own risk*.

.. _cli:

.. Command-line interface
.. ----------------------

API Reference
-------------
"""


from xtl import __version__
from xtl.config.config import Config
from xtl.exceptions import ConfigWarning

import os
import warnings


def _version_tuple(version_string):
    """
    Convert version string to tuple, e.g. '1.0.0' -> (1, 0, 0)

    :param str version_string: version as string (e.g. '1.0.0')
    :return: version as tuple
    :rtype: tuple
    """
    l = version_string.split('.')  # ['1', '0', '0']
    i = [int(n) for n in l]        # [1, 0, 0]
    return tuple(i)


def _check_config_version(config_to_check):
    """
    Checks the config version against XTL version. If config version is lower than XTL version, an attempt is made to
    upgrade the file. If the config is from a future version of XTL, a warning is raised.

    :param Config config_to_check:
    :return:
    """
    current_version = _version_tuple(__version__)
    config_version = _version_tuple(config_to_check['xtl']['version'].value)
    if config_version < current_version:
        warnings.warn(f'Using a config from an older version of XTL. Attempting to upgrade config...', ConfigWarning)
        config_to_check.upgrade_config()
    elif config_version > current_version:
        warnings.warn(f'Using a config from a future version of XTL. XTL might not behave as intended. '
                      f'Use at your own risk!', ConfigWarning)


def _make_cfg():
    """
    Create a :class:`Config` object from the 'xtl.cfg' config file. A global config file is generated the first time the
    package is invoked. If a local 'xtl.cfg' is present at the current working directory, this will be used instead of
    the global one.

    :rtype: Config
    """
    local_config = os.path.join(os.getcwd(), 'xtl.cfg')
    global_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../xtl.cfg'))

    if os.path.exists(local_config):
        print(f'Reading local config: {local_config}')
        _cfg = Config(local_config)
        _check_config_version(_cfg)
    else:
        print(f'Reading global config: {global_config}')
        _cfg = Config(global_config, fix=True)
        _check_config_version(_cfg)
    return _cfg


cfg = _make_cfg()
"""
The object that holds all settings loaded from *xtl.cfg*. 


XTL configuration settings. Settings are read from a global 'xtl.cfg'. If a local 'xtl.cfg' is present at the current 
working directory, this will be loaded instead of the global config.

**See:** :class:`.Config` for usage.
"""

