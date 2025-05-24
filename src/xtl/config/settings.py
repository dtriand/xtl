"""
This module defines a set of :class:`Options <xtl.common.options.Options>` `Pydantic`
models that are used to configure various aspects of XTL. The main class that holds all
settings is :class:`XTLSettings`.
"""

from pathlib import Path
from pprint import pprint
from typing import ClassVar, Optional

from pydantic import PrivateAttr

from xtl import version as current_version
from xtl.automate import ComputeSite
from xtl.config.version import version_from_str
from xtl.common.os import FilePermissions
from xtl.common.options import Option, Options
from xtl.common.serializers import PermissionOctal
from xtl.common.validators import CastAsNoneIfEmpty, CastAsPathOrNone


def _get_extra_keys(options: Options) -> dict:
    """
    Helper function to recursively collect __pydantic_extra__ keys from nested Options.
    """
    extra = options.__pydantic_extra__ or dict()
    for name in options.__pydantic_fields__:
        value = getattr(options, name)
        if isinstance(value, Options):
            sub_extra = _get_extra_keys(value)
            sub_extra.pop('_parse_env', None)  # Remove _parse_env from sub-options
            if sub_extra:
                extra.update({name: sub_extra})
    extra.pop('_parse_env', None)
    return extra


class Settings(Options):
    """
    Base class for settings in XTL.
    """

    # Update model_config
    model_config = Options.model_config | {'extra': 'allow'}

    # NB: In order to support versioning and future changes to the settings structure,
    #   all options should have a default value, ensuring that the model can be
    #   instantiated without any input file. Any unknown options will then be stored
    #   in the model's `__pydantic_extra__` attribute.


class UnitsSettings(Settings):
    """
    Physical units
    """

    # Model attributes
    temperature: str = Option(default='C', choices=('C', 'K', 'F'),
                              desc='C for Celsius, K for Kelvin, F for Fahrenheit')


class AutomatePermissionsSettings(Settings):
    """
    Permission handling when using :mod:xtl.automate
    """

    # Model attributes
    update: bool = Option(default=False,
                          desc='Update permissions of the output files '
                               'after execution of external jobs')
    files: FilePermissions = Option(default=FilePermissions(0o600),
                                    desc='Permissions octal for files',
                                    cast_as=FilePermissions,
                                    formatter=PermissionOctal)
    directories: FilePermissions = Option(default=FilePermissions(0o700),
                                          desc='Permissions octal for directories',
                                          cast_as=FilePermissions,
                                          formatter=PermissionOctal)


class AutomateSettings(Settings):
    """
    Settings for :mod:`xtl.automate`
    """

    # Model attributes
    compute_site: ComputeSite = Option(default=ComputeSite.LOCAL)
    permissions: AutomatePermissionsSettings = Option(
        default=AutomatePermissionsSettings())


class DependencySettings(Settings):
    """
    Generic external dependency settings
    """

    # Model attributes
    path: Optional[Path] = Option(default=None,
                                  desc='Directory containing binaries',
                                  validator=CastAsNoneIfEmpty())
    # TODO: Enable when ModuleSite is implemented
    modules: Optional[list[str]] = Option(default=None,
                                          desc='Modules that provide the dependency',
                                          validator=CastAsNoneIfEmpty(),
                                          exclude=True)


class DependenciesSettings(Settings):
    """
    Settings for external tools and dependencies
    """

    # Model attributes
    autoproc: DependencySettings = Option(default=DependencySettings())


class CLIAutoprocSettings(Settings):
    """
    Settings for ``xtl.autoproc`` CLI
    """

    # Model attributes
    collect_logs: bool = Option(default=False,
                                desc='Collect logs during xtl.autoproc runs')
    logs_dir: Optional[Path] = Option(default=None,
                                      desc='Directory for storing logs',
                                      validator=CastAsPathOrNone(), path_exists=True,
                                      path_is_dir=True)


class CLIConsoleFormatSettings(Settings):
    """
    CLI console formatting settings
    """

    # Model attributes
    rich: bool = Option(default=True, desc='Enable rich formatted output')
    striped_tables: bool = Option(default=True,
                                  desc='Alternating row colors in tables')


class CLISettings(Settings):
    """
    Settings for CLI tools
    """

    # Model attributes
    format: CLIConsoleFormatSettings = Option(default=CLIConsoleFormatSettings())
    autoproc: CLIAutoprocSettings = Option(default=CLIAutoprocSettings())


class XTLSettings(Settings):
    """
    Settings for XTL
    """

    # Class variables are ignored by Pydantic validation/serialization
    _toml: ClassVar[str] = 'xtl.toml'
    global_config: ClassVar[Path] = (Path(__file__).parent.parent / _toml).absolute()
    """
    Path to the global configuration file, typically located in the XTL installation 
    directory.
    
    :meta hide-value:
    """

    local_config: ClassVar[Path] = (Path.cwd() / _toml).absolute()
    """
    Path to the local configuration file, located in the current working directory.
    
    :meta hide-value:
    """

    # Private attributes are also ignored by Pydantic
    _parse_env: bool = PrivateAttr(default=False)
    _file: Optional[Path] = PrivateAttr(default=None)  # Path to the input file, if any

    # Model attributes
    version: str = Option(default=current_version.string)
    """Version string of the XTL package"""

    units: UnitsSettings = Option(default=UnitsSettings())
    """Settings for physical units used in XTL"""

    automate: AutomateSettings = Option(default=AutomateSettings())
    """Settings for the :mod:`xtl.automate` module"""

    dependencies: DependenciesSettings = Option(default=DependenciesSettings())
    """Settings for external tools and dependencies used by XTL"""

    cli: CLISettings = Option(default=CLISettings())
    """Settings for CLI tools provided by XTL"""


    @classmethod
    def initialize(cls) -> 'XTLSettings':
        """
        Initialize by reading an ``xtl.toml`` file. Determines whether to load a
        local or global configuration file, and initializes the settings accordingly.
        If neither file exists, it initializes with default values.

        Once the settings are loaded, a version compatibility check is performed.

        :return: An instance of :class:`XTLSettings`
        """
        # TODO: Change all print statements to use a proper logging system when
        #  implemented
        if cls.local_config.exists():
            print(f'Reading local config: {cls.local_config}')
            _settings = cls.from_toml(cls.local_config)
            _settings._file = cls.local_config
        elif cls.global_config.exists():
            print(f'Reading global config: {cls.global_config}')
            _settings = cls.from_toml(cls.global_config)
            _settings._file = cls.global_config
        else:
            print(f'No local or global {cls._toml} found, initializing with defaults.')
            _settings = cls()

        # Check settings version
        _v = version_from_str(_settings.version)
        if _v.tuple_safe < current_version.tuple_safe:
            print(f'Using settings from an older version of XTL ({_v.string}).')
        elif _v.tuple_safe > current_version.tuple_safe:
            print(f'Using settings from a future version of XTL ({_v.string}). '
                  f'XTL might not behave as intended. Use at your own risk!')

        # Check for extra keys in the nested Options
        extra = _settings.__pydantic_extra__ or dict()
        extra.update(_get_extra_keys(_settings))
        if extra:
            print(f'Warning: Found unknown keys in the config file:')
            pprint(extra)
            print('These keys where ignored during initialization.')

        return _settings
