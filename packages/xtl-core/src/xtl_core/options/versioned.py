"""
|Options| models whose data layout carries a version of its own.

.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Callable, TypeVar, overload

from typing_extensions import Self

from xtl_core.options.base import Options
from xtl_core.version import VersionInfo

logger = logging.getLogger(__name__)


_Data = dict[str, Any]
_Options = TypeVar('_Options', bound=Options)
_StepFuncRaw = TypeVar('_StepFuncRaw', bound=Callable[[_Data], _Data])


@dataclass(frozen=True)
class UpgradeStep:
    """
    Represents a single step in the upgrade path of a versioned model's data layout.

    :param to: The version that this step upgrades to.
    :param func: The function that performs the upgrade.
    :param model: Optional frozen |Options| model of the old layout; if given, the data is
        validated as this model first and the step gets the model instance.
    """

    to: VersionInfo
    func: Callable[[Any], _Data]
    model: type[Options] | None = None


class VersionedOptions(Options):
    _version_path: ClassVar[str | None] = None
    """The path to the version tag in the data, as a dot-separated string."""

    _current_version: ClassVar[VersionInfo | None] = None
    """The model's layout current version. If ``None``, no check is performed on the model's data."""

    _upgrades: ClassVar[str | None] = None
    """Dotted path to the module containing the upgrade steps for this model."""

    _upgrade_steps: ClassVar[dict[tuple[int, int, int], UpgradeStep]] = {}
    """A registry of upgrade steps, keyed by the version they upgrade to."""

    _upgrades_loaded: ClassVar[bool] = False
    """Whether the module containing the upgrade steps has been imported."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._upgrade_steps = {}
        cls._upgrades_loaded = False
        if cls._current_version is not None:
            cls._current_version = VersionInfo.from_value(cls._current_version)
        if (cls._current_version is not None or cls._upgrades) and not cls._version_path:
            raise TypeError(
                f'{cls.__name__} must define a `_version_path` class variable if it defines a '
                '`_current_version` or `_upgrades`.'
            )

    @classmethod
    def _get_version(cls, data: dict) -> VersionInfo | None:
        if cls._version_path is None:
            return None
        node = data
        for part in cls._version_path.split('.'):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return VersionInfo.from_value(node)

    @classmethod
    def _set_version(cls, data: dict, version: VersionInfo) -> None:
        if cls._version_path is None:
            return
        node = data
        parts = cls._version_path.split('.')
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = version.pep440

    @classmethod
    def _check_version(cls, version: VersionInfo | None) -> None:
        current = cls._current_version
        if version is None or current is None:
            return
        if version.tuple_safe > current.tuple_safe:
            logger.warning(
                "%(model_cls)s data is from version %(version)s, which is newer than the model's current version "
                '%(current)s.',
                {'model_cls': cls.__name__, 'version': version.pep440, 'current': current.pep440},
            )

    @classmethod
    def upgrade_steps(cls) -> list[UpgradeStep]:
        """
        Returns a list of all registered upgrade steps for this model, sorted by version.

        :return: A list of :class:`UpgradeStep` instances, sorted by the version.
        """
        return [cls._upgrade_steps[step] for step in sorted(cls._upgrade_steps)]

    @overload
    @classmethod
    def upgrade_step(cls, *, to: VersionInfo | str, model: None = None) -> Callable[[_StepFuncRaw], _StepFuncRaw]: ...

    @overload
    @classmethod
    def upgrade_step(
        cls, *, to: VersionInfo | str, model: type[_Options]
    ) -> Callable[[Callable[[_Options], _Data]], Callable[[_Options], _Data]]: ...

    @classmethod
    def upgrade_step(cls, *, to: VersionInfo | str, model: type[Options] | None = None) -> Callable:
        """
        Decorator to register an upgrade step for a versioned model's data layout.

        :param to: The version that this step upgrades to. It can be a :class:`VersionInfo` instance or a string that can be
            parsed into a :class:`VersionInfo`.
        :param model: Optional frozen `Options` model class to validate against.
        :return: The decorated function
        """

        if cls is VersionedOptions:
            raise TypeError(f'Upgrade steps can only be registered on subclasses of {VersionedOptions.__name__}.')
        if cls._current_version is None:
            raise TypeError(
                f'{cls.__name__} does not define a `_current_version`, so upgrade steps cannot be registered.'
            )

        to = VersionInfo.from_value(to)
        if to.tuple_safe > cls._current_version.tuple_safe:
            raise ValueError(
                f"Upgrade step to version {to.pep440} is newer than the model's "
                f'current version {cls._current_version.pep440}.'
            )
        if model is not None:
            if not (isinstance(model, type) and issubclass(model, Options)):
                raise TypeError(f'Upgrade step model must be a subclass of {Options.__name__}, got {model!r}.')
            if issubclass(model, VersionedOptions):
                raise TypeError(
                    f'Upgrade step model {model.__name__} cannot be a subclass of {VersionedOptions.__name__}.'
                )

        def decorator(func: Callable[[Any], _Data]) -> Callable[[Any], _Data]:
            if not callable(func):
                raise TypeError(f'Upgrade step function must be callable, got {type(func).__name__}.')

            try:
                # Check that the function accepts a single positional argument
                inspect.signature(func).bind(None)
            except TypeError:
                raise TypeError(
                    f'Upgrade step function {func.__qualname__} must accept a single positional argument, '
                    'the data to upgrade.'
                ) from None

            if (existing := cls._upgrade_steps.get(to.tuple_safe)) is not None:
                raise ValueError(
                    f'{cls.__name__} already has an upgrade step to version {to.pep440}: {existing.func.__qualname__}.'
                )
            cls._upgrade_steps[to.tuple_safe] = UpgradeStep(to=to, func=func, model=model)
            return func

        return decorator

    @classmethod
    def upgrade(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Upgrade data from an older version of this model to the current version. The resulting data dictionary is not
        validated against the model.

        :param data: The data to upgrade.
        :return: The upgraded data dictionary, with the version tag updated to the model's current version.
        """

        # Deep-copying allows the steps to modify the data in place
        data = copy.deepcopy(data)

        # The model contains no versioning information
        if cls._current_version is None:
            return data

        # The data contain no versioning information
        version = cls._get_version(data)
        if version is None:
            return data

        # Check if the data come from a newer version than the model's current version
        cls._check_version(version)
        current = cls._current_version
        if version.tuple_safe >= current.tuple_safe:
            return data

        # Run upgrade steps in order
        at = version
        for step in cls.upgrade_steps():
            # Only run the step if the data's version is older than the step's target version
            if step.to.tuple_safe <= at.tuple_safe:
                continue

            logger.debug(
                'Running upgrade step from version %(from)s to %(to)s: %(func)s',
                {'from': at.pep440, 'to': step.to.pep440, 'func': step.func.__qualname__},
            )

            # Validate the data against the model
            arg = step.model.from_dict(data) if step.model is not None else data

            # Run the upgrade step
            result = step.func(arg)
            if not isinstance(result, dict):
                raise TypeError(
                    f'Upgrade step {step.func.__qualname__} (to {step.to.pep440}) returned {type(result).__name__}, '
                    f'expected a dict.'
                )

            # Update the data's version
            data = result
            cls._set_version(data, step.to)
            at = step.to

        cls._set_version(data, current)
        logger.info(
            'Upgraded %(model)s data from version %(from)s to %(to)s.',
            {'model': cls.__name__, 'from': version.pep440, 'to': current.pep440},
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        cls._check_version(cls._get_version(data))
        return super().from_dict(data)

    @classmethod
    def from_json(cls, s: str | Path) -> Self:
        if isinstance(s, str):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                # Not a valid JSON string, treat as a file path
                path = Path(s)
                if not path.exists():
                    raise FileNotFoundError(f'File not found: {path}')
                data = json.loads(path.read_text())
        else:
            path = Path(s)
            if not path.exists():
                raise FileNotFoundError(f'File not found: {path}')
            data = json.loads(path.read_text())

        cls._check_version(cls._get_version(data))
        return super().from_json(s)
