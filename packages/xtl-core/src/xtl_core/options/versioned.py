"""
|Options| models whose data layout carries a version of its own.

.. |Options| replace:: :class:`Options <xtl_core.options.base.Options>`
"""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, ClassVar, Callable, TypeVar, overload

from typing_extensions import Self

from xtl_core.options.base import Options
from xtl_core.version import VersionInfo
from xtl_core.utils.docs import extend_docstring


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


def _field_keys(model: type[Options]) -> set[str]:
    """
    The top-level keys a model reads: its field names and their string aliases (``Option(alias=...)`` sets the
    serialization alias, which is the key in saved data).
    """
    keys = set()
    for name, field in model.model_fields.items():
        keys.add(name)
        for alias in (field.alias, field.validation_alias, field.serialization_alias):
            if isinstance(alias, str):
                keys.add(alias)
    return keys


_UPGRADE_NOTE = (
    'Data from an older version is upgraded to the current one first (see :meth:`upgrade`); '
    'newer data logs a warning, and data without a version loads as current.'
)
_UPGRADE_PARAM = {'upgrade': 'Upgrade older data first. If ``False``, the data are validated as they are.'}


class VersionedOptions(Options):
    """
    An |Options| model whose data carries a version tag of its own, so that data saved with an older layout is
    upgraded to the current one when loaded, one version at a time.

    A model sets ``_version_path`` (where the tag lives in the data), ``_current_version`` (its current layout) and
    ``_upgrades`` (the module with its upgrade steps, registered with :meth:`upgrade_step`). Older data is upgraded
    in :meth:`from_dict`, :meth:`from_json` and :meth:`from_toml`; subclasses run their parents' steps too. See
    :ref:`the Options guide <versioned-data>` for a full example.
    """

    _version_path: ClassVar[str | None] = None
    """The path to the version tag in the data, as a dot-separated string."""

    _current_version: ClassVar[VersionInfo | str | None] = None
    """The model's current layout version, as set by the model (a ``str`` or :class:`VersionInfo`). If ``None``,
    the data's version is neither checked nor upgraded."""

    _current: ClassVar[VersionInfo | None] = None
    """``_current_version`` as a :class:`VersionInfo`, set when the subclass is created; the methods use this one."""

    _upgrades: ClassVar[str | None] = None
    """Dotted path to the module with this class's own upgrade steps, imported the first time they're needed.
    Subclasses also run the steps of their parents."""

    _upgrade_steps: ClassVar[dict[tuple[int, int, int], UpgradeStep]] = {}
    """This class's own upgrade steps, keyed by the version they upgrade to (its parents' are in their own)."""

    _upgrades_loaded: ClassVar[bool] = False
    """Whether this class's own ``_upgrades`` module has been imported."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._upgrade_steps = {}
        cls._upgrades_loaded = False
        cls._current = None if cls._current_version is None else VersionInfo.from_value(cls._current_version)
        if (cls._current is not None or cls._upgrades) and not cls._version_path:
            raise TypeError(
                f'{cls.__name__} must define a `_version_path` class variable if it defines a '
                '`_current_version` or `_upgrades`.'
            )

        # A subclass extends its parents' version history: it runs their steps, so it can't go back to an
        #  older version or move the version tag
        for base in cls.__bases__:
            if not (issubclass(base, VersionedOptions) and base._current is not None):
                continue
            if cls._current is None or cls._current.tuple_safe < base._current.tuple_safe:
                raise TypeError(
                    f'{cls.__name__} cannot lower `_current_version` below that of {base.__name__} '
                    f'({base._current.pep440}).'
                )
            if cls._version_path != base._version_path:
                raise TypeError(
                    f'{cls.__name__} cannot change `_version_path` of {base.__name__} ({base._version_path!r}).'
                )

    @classmethod
    def _get_version(cls, data: Any) -> VersionInfo | None:
        if cls._version_path is None:
            return None
        node = data
        for part in cls._version_path.split('.'):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return VersionInfo.from_value(node)

    @classmethod
    def _set_version(cls, data: _Data, version: VersionInfo) -> None:
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
        current = cls._current
        if version is None or current is None:
            return
        if version.tuple_safe > current.tuple_safe:
            logger.warning(
                "%(model_cls)s data is from version %(version)s, which is newer than the model's current version "
                '%(current)s.',
                {'model_cls': cls.__name__, 'version': version.pep440, 'current': current.pep440},
            )

    @classmethod
    def _load_upgrades(cls) -> None:
        """
        Import this class's own ``_upgrades`` module once, which registers its steps. An
        inherited ``_upgrades`` belongs to the parent, which loads it.

        :raises ImportError: If the module can't be imported; the next call tries again.
        """
        module = cls.__dict__.get('_upgrades')
        if module is None or cls._upgrades_loaded:
            return
        importlib.import_module(module)
        cls._upgrades_loaded = True
        if not cls._upgrade_steps:
            logger.warning(
                '%(model)s: importing %(module)s registered no upgrade steps for it.',
                {'model': cls.__name__, 'module': module},
            )

    @classmethod
    def _versioned_bases(cls) -> list[type[VersionedOptions]]:
        """This class and its versioned parents, from the furthest parent to this class."""
        return [
            klass
            for klass in reversed(cls.__mro__)
            if isinstance(klass, type) and issubclass(klass, VersionedOptions) and klass is not VersionedOptions
        ]

    @classmethod
    def upgrade_steps(cls) -> list[UpgradeStep]:
        """
        All upgrade steps that apply to this model, its parents' included, in the order they
        run: by the version they upgrade to and, for the same version, a parent's before its
        subclass's. The ``_upgrades`` modules are imported first.

        :return: A list of :class:`UpgradeStep` instances, in the order they run.
        """
        steps = []
        for klass in cls._versioned_bases():
            klass._load_upgrades()
            steps.extend(klass._upgrade_steps[key] for key in sorted(klass._upgrade_steps))
        # sorted() is stable, so parents' steps stay before their subclasses' for the same version
        return sorted(steps, key=lambda step: step.to.tuple_safe)

    @overload
    @classmethod
    def upgrade_step(cls, *, to: VersionInfo | str, model: None = None) -> Callable[[_StepFuncRaw], _StepFuncRaw]: ...

    @overload
    @classmethod
    def upgrade_step(
        cls, *, to: VersionInfo | str, model: type[Options]
    ) -> Callable[[Callable[[_Options], _Data]], Callable[[_Options], _Data]]: ...

    @classmethod
    def upgrade_step(
        cls, *, to: VersionInfo | str, model: type[Options] | None = None
    ) -> Callable[[Callable[[Any], _Data]], Callable[[Any], _Data]]:
        """
        Decorator to register an upgrade step on this model. The step takes the data in the layout before ``to`` and
        returns it in the layout of ``to``; the function itself is returned unchanged.

        :param to: The version that this step upgrades to. It can be a :class:`VersionInfo` instance or a string that can be
            parsed into a :class:`VersionInfo`.
        :param model: Optional frozen |Options| model of the old layout. If given, the data is validated as this model
            first and the step gets the model instance; keys the model doesn't know are carried over into the result.
        :return: The decorator.
        :raises TypeError: If registered on :class:`VersionedOptions` itself or on a model without a
            ``_current_version``, if ``model`` isn't a plain |Options| subclass, or if the function doesn't take
            exactly one positional argument.
        :raises ValueError: If ``to`` is newer than the model's current version, or the model already has a step to
            ``to``.
        """

        if cls is VersionedOptions:
            raise TypeError(f'Upgrade steps can only be registered on subclasses of {VersionedOptions.__name__}.')
        if cls._current is None:
            raise TypeError(
                f'{cls.__name__} does not define a `_current_version`, so upgrade steps cannot be registered.'
            )

        to = VersionInfo.from_value(to)
        if to.tuple_safe > cls._current.tuple_safe:
            raise ValueError(
                f"Upgrade step to version {to.pep440} is newer than the model's current version {cls._current.pep440}."
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
    def upgrade(cls, data: _Data) -> _Data:
        """
        Upgrade data from an older version of this model to the current version, running the steps of this model and
        its parents in order. The result is not validated against the model, and ``data`` is left unchanged. Data
        without a version, from the current version or from a newer one (which logs a warning) is returned as a copy,
        without running any steps.

        :param data: The data to upgrade.
        :return: The upgraded data, with the version tag set to the model's current version.
        :raises TypeError: If a step doesn't return a dict.
        """

        # Deep-copying allows the steps to modify the data in place
        data = copy.deepcopy(data)

        # The model contains no versioning information
        if cls._current is None:
            return data

        # The data contain no versioning information
        version = cls._get_version(data)
        if version is None:
            return data

        # Check if the data come from a newer version than the model's current version
        cls._check_version(version)
        current = cls._current
        if version.tuple_safe >= current.tuple_safe:
            return data

        # Run upgrade steps in order
        at = version
        for step in cls.upgrade_steps():
            # Only run the steps newer than the data's original version; comparing with the version
            #  after the previous step would skip a subclass's step to the same version as its parent's
            if step.to.tuple_safe <= version.tuple_safe:
                continue

            logger.debug(
                'Running upgrade step from version %(from)s to %(to)s: %(func)s',
                {'from': at.pep440, 'to': step.to.pep440, 'func': step.func.__qualname__},
            )

            # Typed steps get the data validated as their frozen model, which only sees the keys it
            #  knows; the rest (e.g. fields of a subclass) are carried over into the result
            if step.model is not None:
                known = _field_keys(step.model)
                arg = step.model.from_dict({k: v for k, v in data.items() if k in known})
                unknown = {k: v for k, v in data.items() if k not in known}
            else:
                arg, unknown = data, {}

            # Run the upgrade step
            result = step.func(arg)
            if not isinstance(result, dict):
                raise TypeError(
                    f'Upgrade step {step.func.__qualname__} (to {step.to.pep440}) returned {type(result).__name__}, '
                    f'expected a dict.'
                )
            result = {**unknown, **result}

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
    @extend_docstring(Options.from_dict, note=_UPGRADE_NOTE, params=_UPGRADE_PARAM)
    def from_dict(cls, data: _Data, /, *, upgrade: bool = True) -> Self:
        if upgrade:
            data = cls.upgrade(data)
        else:
            cls._check_version(cls._get_version(data))
        return super().from_dict(data)

    @classmethod
    @extend_docstring(Options.from_json, note=_UPGRADE_NOTE, params=_UPGRADE_PARAM)
    def from_json(cls, s: str | Path, /, *, upgrade: bool = True) -> Self:
        return cls.from_dict(cls._read_contents(s, mode='json'), upgrade=upgrade)

    @classmethod
    @extend_docstring(Options.from_toml, note=_UPGRADE_NOTE, params=_UPGRADE_PARAM)
    def from_toml(cls, s: str | Path, /, *, upgrade: bool = True) -> Self:
        return cls.from_dict(cls._read_contents(s, mode='toml'), upgrade=upgrade)
