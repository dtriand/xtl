"""
This module defines a set of :class:`Options <xtl_core.options.base.Options>` `Pydantic`
models that are used to configure various aspects of XTL. The main class that holds all
settings is :class:`XTLSettings`.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, ClassVar, Iterable, Optional, Literal, get_origin, TYPE_CHECKING

from pydantic import PrivateAttr, model_validator

from xtl import version as current_version
from xtl.automate import ComputeSite
from xtl.jobs.dependencies import DependencyCheck, site_key
from xtl.jobs.sites import ComputeSite as ComputeSite2
from xtl.jobs.shells import Shell
from xtl.jobs.policies import CommandPolicy
from xtl_core.version import VersionInfo
from xtl.shared.environment import XTL_CONFIG
from xtl_core.system.os import FilePermissions, CPU_CORES
from xtl_core.options import Option, Options
from xtl_core.options.serializers import PermissionOctal
from xtl_core.options.validators import CastAsNoneIfEmpty, CastAsPathOrNone
from xtl_core.options.exceptions import OptionsValidationError

if TYPE_CHECKING:
    from xtl.jobs.shells import ShellType

logger = logging.getLogger(__name__)


def _get_extra_keys(options: Options) -> dict:
    """
    Helper function to recursively collect __pydantic_extra__ keys from nested Options.
    """
    extra = dict(options.__pydantic_extra__ or {})
    for name in options.__pydantic_fields__:
        value = getattr(options, name)
        if isinstance(value, Options):
            sub_extra = _get_extra_keys(value)
            sub_extra.pop('_parse_env', None)  # Remove _parse_env from sub-options
            if sub_extra:
                extra.update({name: sub_extra})
    extra.pop('_parse_env', None)
    return extra


def _flatten_keys(data: dict, prefix: str = '') -> list[str]:
    """
    Helper function to flatten nested dictionary keys to dotted paths, *e.g.*
    ``{'units': {'colour': 'blue'}} -> ['units.colour']``.
    """
    keys = []
    for key, value in data.items():
        path = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict) and value:
            keys.extend(_flatten_keys(value, path))
        else:
            keys.append(path)
    return keys


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

    def update_from(self, other: Settings) -> None:
        """
        Take the values of `other`, in place: nested settings are updated rather than replaced, so that code holding
        this object, or one of its sections, sees the new values.

        :param other: Settings of the same type.
        """
        for name in type(self).model_fields:
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, Settings) and type(mine) is type(theirs):
                mine.update_from(theirs)
            elif mine != theirs:
                setattr(self, name, theirs)


class UnitsSettings(Settings):
    """
    Physical units
    """

    # Model attributes
    temperature: str = Option(
        default='C', choices=('C', 'K', 'F'), desc='C for Celsius, K for Kelvin, F for Fahrenheit'
    )


class AutomatePermissionsSettings(Settings):
    """
    Permission handling when using :mod:xtl.automate
    """

    # Model attributes
    update: bool = Option(default=False, desc='Update permissions of the output files after execution of external jobs')
    scripts: FilePermissions = Option(
        default=FilePermissions(0o700),
        desc='Permissions octal for scripts',
        cast_as=FilePermissions,
        formatter=PermissionOctal,
    )
    files: FilePermissions = Option(
        default=FilePermissions(0o600),
        desc='Permissions octal for files',
        cast_as=FilePermissions,
        formatter=PermissionOctal,
    )
    directories: FilePermissions = Option(
        default=FilePermissions(0o700),
        desc='Permissions octal for directories',
        cast_as=FilePermissions,
        formatter=PermissionOctal,
    )


class AutomateSettings(Settings):
    """
    Settings for :mod:`xtl.automate`
    """

    # Model attributes
    job_digits: int = Option(default=5, desc='Number of digits to use for generation of random job IDs')
    compute_site: ComputeSite = Option(default=ComputeSite.LOCAL, desc='Default compute site for job execution')
    permissions: AutomatePermissionsSettings = Option(default=AutomatePermissionsSettings())
    keep_temp: bool = Option(default=False, desc='Keep temporary files after job execution')


class ResourcesSettings(Settings):
    """
    Resource settings for jobs
    """

    # Model attributes
    max_jobs: int = Option(default=CPU_CORES, desc='Maximum number of jobs to run concurrently in a pool')
    max_threads: int = Option(default=CPU_CORES, desc='Maximum number of threads to use for job execution')
    max_processes: int = Option(default=CPU_CORES, desc='Maximum number of processes to use for job execution')


class JobsSettings(Settings):
    """
    Settings for jobs in :mod:`xtl.jobs`
    """

    # Model attributes
    job_digits: int = Option(default=5, desc='Number of digits to use for generation of random job IDs')

    compute_site: ComputeSite2 = Option(default=ComputeSite2.LOCAL, desc='Default compute site for job execution')

    keep_temp: bool = Option(default=False, desc='Keep temporary files after job execution')

    tracebacks: bool = Option(default=False, desc='Whether to include tracebacks in job logs when exceptions occur')

    threading_debug: bool = Option(
        default=False, desc='Enable thread and process names in job logs for debugging concurrency issues'
    )

    permissions: AutomatePermissionsSettings = Option(
        default=AutomatePermissionsSettings(), desc='Permissions settings for jobs'
    )

    resources: ResourcesSettings = Option(
        default=ResourcesSettings(), desc='Default resource settings for job execution'
    )


class DependencySettings(Settings):
    """
    Generic external dependency settings. Each known dependency is a subclass that
    defines the executables it provides.
    """

    # Executables provided by the dependency; defined by XTL, not configurable
    _provides: ClassVar[frozenset[str]] = frozenset()

    # Model attributes
    path: Optional[Path] = Option(default=None, desc='Directory containing binaries', validator=CastAsNoneIfEmpty())

    modules: list[str] = Option(
        default_factory=list, desc='Modules that provide the dependency, loaded in this order on a modules site'
    )

    compute_site: Optional[ComputeSite2] = Option(
        default=None, desc='Compute site that runs the dependency (default: jobs.compute_site)'
    )

    shell: Optional[Shell] = Option(default=None, desc="Shell that runs the dependency (default: the compute site's)")

    policy: Optional[CommandPolicy] = Option(
        default=None, desc="Command policy for the dependency's commands (default: the compute site's)"
    )

    # Checks made in this process, or received with the settings, by (compute site, shell name), each with the
    #  settings of the dependency when it was checked; not settings, so never written to the config file
    _checks: dict[tuple[str, str], tuple[str, DependencyCheck]] = PrivateAttr(default_factory=dict)

    @model_validator(mode='after')
    def _path_or_modules(self) -> DependencySettings:
        # A dependency is found one way: in a directory, or by loading modules
        if self.path is not None and self.modules:
            raise ValueError('Cannot specify both `path` and `modules`')
        return self

    @property
    def provides(self) -> frozenset[str]:
        """
        Executables provided by this dependency. These are defined by XTL for each
        dependency and cannot be changed in the config file.
        """
        return self._provides

    def __eq__(self, other: object) -> bool:
        # The checks are a record, not settings: equal settings with different checks are equal
        if type(other) is not type(self):
            return NotImplemented
        return self.__dict__ == other.__dict__ and self.__pydantic_extra__ == other.__pydantic_extra__

    def get_check(self, site: Any, shell: ShellType) -> Optional[DependencyCheck]:
        """
        The outcome of an earlier check of this dependency on a compute site, in a shell, or ``None`` if there is none,
        or if the settings of the dependency (*e.g.* its ``modules``) have changed since.

        :param site: The compute site that checked it.
        :param shell: The shell it was checked in.
        """
        with _checks_lock:
            record = self._checks.get((site_key(site), shell.name))
        if record is None:
            return None
        checked_settings, check = record
        return check if checked_settings == self.model_dump_json() else None

    def remember_check(self, site: Any, shell: ShellType, check: DependencyCheck) -> None:
        """
        Remember the outcome of a check of this dependency on a compute site, in a shell. Missing dependencies are
        remembered too, so that the other jobs that need them fail at once.

        :param site: The compute site that checked it.
        :param shell: The shell it was checked in.
        :param check: Its outcome.
        """
        with _checks_lock:
            self._checks[(site_key(site), shell.name)] = (self.model_dump_json(), check)

    def clear_checks(self) -> None:
        """
        Forget every check of this dependency, *e.g.* after installing it in a running session.
        """
        with _checks_lock:
            self._checks.clear()

    def checks_to_list(self) -> list[dict[str, Any]]:
        """
        The checks of this dependency as plain values, *e.g.* to send them to another process with the settings.
        """
        with _checks_lock:
            return [
                {'site': site, 'shell': shell, 'settings': checked_settings, 'check': check.to_dict()}
                for (site, shell), (checked_settings, check) in self._checks.items()
            ]

    def update_checks(self, checks: list[dict[str, Any]]) -> None:
        """
        Add checks from the output of :meth:`checks_to_list`, replacing those for the same compute site and shell.

        :param checks: Checks as plain values.
        """
        with _checks_lock:
            for record in checks:
                self._checks[(record['site'], record['shell'])] = (
                    record['settings'],
                    DependencyCheck.from_dict(record['check']),
                )


# One lock for the checks of all dependencies: checks are rare, and a pool's threads may share a dependency
_checks_lock = threading.Lock()


class AtsasSettings(DependencySettings):
    """
    Settings for `ATSAS <https://www.embl-hamburg.de/biosaxs/software.html>`_
    """

    _provides = frozenset({'datcmp'})


class AutoprocSettings(DependencySettings):
    """
    Settings for `autoPROC <https://www.globalphasing.com/autoproc/>`_
    """

    _provides = frozenset({'process', 'process_wf'})


class EasyBraggSettings(DependencySettings):
    """
    Settings for easyBragg
    """

    _provides = frozenset({'easyBragg.python'})


class DependenciesSettings(Settings):
    """
    Settings for external tools and dependencies
    """

    # Model attributes
    resolution: str = Option(
        default='loose',
        choices={'strict', 'loose'},
        desc='Dependency resolution strategy. '
        'If "strict", jobs requiring missing dependencies will fail to run. '
        'If "loose", missing dependencies will be ignored.',
    )

    path: Optional[Path] = Option(
        default=None,
        desc='Directory of binaries that every job finds, before the directories of its dependencies',
        validator=CastAsNoneIfEmpty(),
    )

    modules: list[str] = Option(
        default_factory=list,
        desc='Modules that every job loads on a modules compute site, in this order, before the modules of its '
        'dependencies',
    )

    check: bool = Option(
        default=True,
        desc='Check that the dependencies of jobs can be found before running them (once per job, pool or command)',
    )

    atsas: AtsasSettings = Option(default=AtsasSettings())
    autoproc: AutoprocSettings = Option(default=AutoprocSettings())
    easybragg: EasyBraggSettings = Option(default=EasyBraggSettings())

    @classmethod
    def names(cls) -> list[str]:
        """
        Names of the known dependencies, in definition order.
        """
        # NB: `get_origin()` leaves out generic aliases (e.g. `list[str]`), which pass `isinstance(..., type)` on
        #  Python 3.10, but not `issubclass()`
        return [
            name
            for name, field in cls.model_fields.items()
            if get_origin(field.annotation) is None
            and isinstance(field.annotation, type)
            and issubclass(field.annotation, DependencySettings)
        ]

    def __contains__(self, name: object) -> bool:
        return name in self.names()

    def get(self, name: str) -> Optional[DependencySettings]:
        """
        Get the settings of a known dependency by name.

        :param name: Name of the dependency, *e.g.* ``'atsas'``
        :return: The dependency settings, or ``None`` if there is no such dependency
        """
        return getattr(self, name) if name in self else None

    def items(self) -> list[tuple[str, DependencySettings]]:
        """
        The known dependencies as ``(name, settings)`` pairs, in definition order.
        """
        return [(name, getattr(self, name)) for name in self.names()]

    def clear_checks(self) -> None:
        """
        Forget every check of every known dependency.
        """
        for _, dependency in self.items():
            dependency.clear_checks()

    def checks_to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """
        The checks of the known dependencies as plain values, by name, *e.g.* to send them to another process with the
        settings. Dependencies without checks are left out.
        """
        return {name: checks for name, dependency in self.items() if (checks := dependency.checks_to_list())}

    def update_checks(self, checks: dict[str, list[dict[str, Any]]]) -> None:
        """
        Add checks from the output of :meth:`checks_to_dict` to the known dependencies; unknown names are skipped.

        :param checks: Checks as plain values, by name.
        """
        for name, records in checks.items():
            if (dependency := self.get(name)) is not None:
                dependency.update_checks(records)

    def resolve(
        self, dependencies: Iterable[DependencySettings | str] | DependencySettings | str | None
    ) -> list[DependencySettings]:
        """
        Resolve dependency names to their settings. Instances of
        :class:`DependencySettings` are passed through. Unknown names are skipped with a
        warning, or raise an error if :attr:`resolution` is ``'strict'``.

        :param dependencies: Dependency names or settings, or ``None``
        :return: The resolved dependency settings, in the given order and without
            duplicates
        :raises ValueError: If a name is unknown and the resolution is ``'strict'``
        """
        resolved: list[DependencySettings] = []
        if dependencies is None:
            return resolved
        if isinstance(dependencies, (str, DependencySettings)):
            dependencies = [dependencies]

        for dependency in dependencies:
            dep = self.get(dependency) if isinstance(dependency, str) else dependency
            if dep is None:
                logger.warning('Dependency `%s` not found in settings', dependency)
                if self.resolution == 'strict':
                    logger.error('Strict dependency resolution enabled; missing dependency `%s`', dependency)
                    raise ValueError(f'Missing dependency `{dependency}`')
                continue
            # Skip duplicates (by identity: equal values may be different dependencies)
            if not any(d is dep for d in resolved):
                resolved.append(dep)
        return resolved


class CLIAutoprocSettings(Settings):
    """
    Settings for ``xtl.autoproc`` CLI
    """

    # Model attributes
    collect_logs: bool = Option(default=False, desc='Collect logs during xtl.autoproc runs')
    logs_dir: Optional[Path] = Option(
        default=None,
        desc='Directory for storing logs',
        validator=CastAsPathOrNone(),
        path_exists=True,
        path_is_dir=True,
    )


class CLIConsoleFormatSettings(Settings):
    """
    CLI console formatting settings
    """

    # Model attributes
    rich: bool = Option(default=True, desc='Enable rich formatted output')
    striped_tables: bool = Option(default=True, desc='Alternating row colors in tables')


class CLISettings(Settings):
    """
    Settings for CLI tools
    """

    # Model attributes
    format: CLIConsoleFormatSettings = Option(default=CLIConsoleFormatSettings())
    autoproc: CLIAutoprocSettings = Option(default=CLIAutoprocSettings())

    max_fps: int = Option(default=10, gt=1, lt=60, desc='Maximum refreshes (per s) for live console elements')

    job_log_files: bool = Option(default=True, desc='Write a log file (xtl.log) in the directory of each job')


@dataclass(frozen=True)
class ConfigFileInfo:
    """
    A candidate location for the ``xtl.toml`` configuration file.
    """

    name: Literal['env', 'local', 'user', 'global']
    path: Path
    creatable: bool

    @property
    def exists(self) -> bool:
        return self.path.is_file()


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

    # Private attributes are also ignored by Pydantic
    _parse_env: bool = PrivateAttr(default=False)
    _source: Optional[ConfigFileInfo] = PrivateAttr(default=None)  # source of settings

    # Model attributes
    version: str = Option(default=current_version.string)
    """Version string of the XTL package"""

    units: UnitsSettings = Option(default=UnitsSettings())
    """Settings for physical units used in XTL"""

    automate: AutomateSettings = Option(default=AutomateSettings())
    """Settings for the :mod:`xtl.automate` module"""

    jobs: JobsSettings = Option(default=JobsSettings())
    """Settings for jobs in :mod:`xtl.jobs`"""

    dependencies: DependenciesSettings = Option(default=DependenciesSettings())
    """Settings for external tools and dependencies used by XTL"""

    cli: CLISettings = Option(default=CLISettings())
    """Settings for CLI tools provided by XTL"""

    @property
    def source(self) -> Optional[ConfigFileInfo]:
        """
        The source of the settings. This is either a :class:`ConfigFileInfo` object
        representing the configuration file that was loaded, or ``None`` for in-memory
        defaults.
        """
        return self._source

    @classmethod
    def _load(cls, location: ConfigFileInfo) -> XTLSettings:
        """
        Load settings from a given configuration file location. If the file cannot be
        read (missing, invalid TOML or invalid values), it logs an error and returns
        default settings.

        :param location: A :class:`ConfigFileInfo` object representing the configuration
                         file location to load.
        :return: An instance of :class:`XTLSettings`
        """
        logger.info('Reading %(name)s config: %(path)s', {'name': location.name, 'path': location.path})

        _settings, errors, warnings = cls._read(location.path)
        if _settings is None:
            # Log the error, but don't fall back to the next available location
            logger.error(
                'Failed to read %(name)s config %(path)s, using default settings instead: %(error)s',
                {'name': location.name, 'path': location.path, 'error': '; '.join(errors)},
            )
            return cls()  # Return default settings if reading fails

        # Set provenance
        _settings._source = location

        # Warn about any issues in the loaded settings
        for problem in warnings:
            logger.warning(problem)
        return _settings

    @classmethod
    def _read(cls, path: Path) -> tuple[Optional['XTLSettings'], list[str], list[str]]:
        """
        Read a configuration file without logging.

        :param path: Path to the configuration file to read.
        :return: A tuple of the loaded settings (``None`` if the file cannot be read),
                 a list of errors and a list of warnings.
        """
        try:
            _settings = cls.from_toml(path)
        except OptionsValidationError as e:
            # One error per invalid value, located without the model name
            #  (e.g. 'units.temperature' instead of 'XTLSettings.units.temperature')
            errors = [f'{error["loc"].partition(".")[2] or error["loc"]}: {error["msg"]}' for error in e.errors]
            return None, errors, []
        except (OSError, ValueError) as e:
            return None, [str(e)], []
        return _settings, [], _settings.check()

    def check(self) -> list[str]:
        """
        Check the settings for potential issues, such as version mismatches or unknown
        keys. Returns a list of warning messages for any issues found.

        :return: A list of warning messages for any issues found in the settings.
        """
        problems = []

        # Check settings version
        try:
            _v = VersionInfo.from_str(self.version)
        except ValueError:
            problems.append(f'Invalid settings version: {self.version!r}')
        else:
            if _v.tuple_safe < current_version.tuple_safe:
                problems.append(
                    f'Using settings from an older version of XTL ({_v.string}, current {current_version.string}).'
                )
            elif _v.tuple_safe > current_version.tuple_safe:
                problems.append(
                    f'Using settings from a newer version of XTL '
                    f'({_v.string}, current {current_version.string}). '
                    'XTL might not behave as intended. '
                    'Use at your own risk!'
                )

        # Check for extra keys in the nested Options
        extra = _flatten_keys(_get_extra_keys(self))
        if extra:
            problems.append(f'Found and ignored unknown keys in the config file: {", ".join(extra)}')

        return problems

    @classmethod
    def check_file(cls, path: Path) -> tuple[list[str], list[str]]:
        """
        Check a given configuration file for potential issues, such as version
        mismatches, unknown keys, or file read errors. Returns a tuple containing a list
        of errors and warnings for any issues found.

        :param path: Path to the configuration file to check.
        :return: A tuple containing a list of errors and a list of warnings for any
                 issues found in the configuration file.
        """
        _, errors, warnings = cls._read(path)
        return errors, warnings

    @classmethod
    def initialize(cls, create: bool = True) -> 'XTLSettings':
        """
        Initialize from the first existing ``xtl.toml`` in :meth:`locations`, in
        lookup order: ``XTL_CONFIG``, local, user and global. Once loaded, the
        problems found by :meth:`check` are logged as warnings.

        If no file exists and ``create`` is True, the default settings are written
        to the first creatable location that succeeds (``XTL_CONFIG``, user, global;
        never local), logging a warning for each one that fails. If all of them fail,
        or ``create`` is False, the default settings are used in-memory.

        This method never raises: a file that cannot be read is logged as an error
        and the default settings are used instead.

        :param create: Whether to create a config file if none is found
        :return: An instance of :class:`XTLSettings`
        """
        locs = cls.locations()
        # Check for the first existing config file in the lookup order
        active = cls.active(locs)
        if active is not None:
            return cls._load(active)

        # If no config file exists, initialize with defaults
        _settings = cls()
        if not create:
            logger.info('No %(toml)s found, initializing with default settings', {'toml': cls._toml})
            return _settings

        # Save config to the first creatable location
        for loc in locs:
            if not loc.creatable:
                continue
            try:
                _settings.to_toml(loc.path, comments=True, keep_file_ext=True)
            except OSError as e:
                # Another process might have created the file in the meantime
                if isinstance(e, FileExistsError) and loc.exists:
                    return cls._load(loc)
                logger.warning(
                    'Failed to create %(name)s config at %(path)s: %(error)s',
                    {'name': loc.name, 'path': loc.path, 'error': e},
                )
                continue
            logger.info('Created %(name)s config at %(path)s', {'name': loc.name, 'path': loc.path})
            _settings._source = loc
            return _settings

        logger.warning(
            'Failed to create a config file in any of the candidate locations, using default settings in-memory'
        )
        return _settings

    @classmethod
    def active(cls, locations: Optional[list[ConfigFileInfo]] = None) -> Optional[ConfigFileInfo]:
        """
        The active config file: the first existing one in :meth:`locations`, which is
        the file that :meth:`initialize` loads. Whether the file can be read is not
        checked (see :meth:`check_file`).

        :param locations: Candidate locations to search, as returned
            by :meth:`locations` (default: computed now)
        :return: A :class:`ConfigFileInfo`, or ``None`` if no config file exists
        """
        if locations is None:
            locations = cls.locations()
        return next((loc for loc in locations if loc.exists), None)

    @classmethod
    def locations(cls) -> list[ConfigFileInfo]:
        """
        Candidate config file locations in lookup order: ``XTL_CONFIG``, local, user
        and global. Computed on every call, so that changes to the environment or the
        working directory are picked up.

        :return: A list of :class:`ConfigFileInfo`, whether the files exist or not
        """
        from platformdirs import user_config_dir

        locations = []

        # Environment variable (if set)
        env_path = XTL_CONFIG.value
        if env_path is not None:
            if env_path.is_dir():
                logger.warning('Ignoring XTL_CONFIG, it points to a directory: %(path)s', {'path': env_path})
            else:
                locations.append(ConfigFileInfo(name='env', path=env_path, creatable=True))

        # Local config (never create automatically)
        locations.append(ConfigFileInfo(name='local', path=Path.cwd() / cls._toml, creatable=False))

        # User config (platform-dependent)
        locations.append(
            ConfigFileInfo(name='user', path=Path(user_config_dir('xtl', appauthor=False)) / cls._toml, creatable=True)
        )

        # Global config (installation directory)
        locations.append(ConfigFileInfo(name='global', path=cls.global_config, creatable=True))

        return locations
