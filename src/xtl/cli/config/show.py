import difflib
import json
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Optional

import typer

from xtl.cli.utilities import epilog


app = typer.Typer()

_MISSING = object()


def _plain(data: Any) -> Any:
    """
    Strip the comments from TOML data, *e.g.* ``CommentValue(16) -> 16``.
    """
    from toml.decoder import CommentValue

    if isinstance(data, CommentValue):
        return _plain(data.val)
    if isinstance(data, dict):
        return {key: _plain(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_plain(value) for value in data]
    return data


def _get(data: Any, parts: tuple[str, ...]) -> Any:
    """
    Get a nested item from TOML data, or ``_MISSING``.
    """
    for part in parts:
        if not isinstance(data, dict) or part not in data:
            return _MISSING
        data = data[part]
    return data


def _set(data: dict, parts: tuple[str, ...], value: Any) -> None:
    """
    Set a nested item in TOML data, creating the parent tables.
    """
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def _paths(data: dict, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """
    All dotted paths (tables and keys) in TOML data.
    """
    paths = []
    for key, value in data.items():
        path = prefix + (key,)
        paths.append(path)
        if isinstance(value, dict):
            paths.extend(_paths(value, path))
    return paths


def _diff(current: Any, default: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """
    Paths of the keys whose values differ between two (plain) TOML data.
    """
    if isinstance(current, dict) and isinstance(default, dict):
        paths = []
        for key in list(current) + [k for k in default if k not in current]:
            paths.extend(_diff(current.get(key, _MISSING), default.get(key, _MISSING), prefix + (key,)))
        return paths
    return [] if current == default else [prefix]


def _resolve(settings, parts: tuple[str, ...]) -> Any:
    """
    Value of a dotted path in the settings (using TOML key names), or ``_MISSING`` if
    there is no such key. Unlike the TOML data, this includes values that are None.
    """
    from xtl_core.options import Options

    value = settings
    for part in parts:
        if isinstance(value, Options):
            fields = {**value.__pydantic_fields__, **value.__pydantic_computed_fields__}
            name = next((n for n, f in fields.items() if Options._toml_key(n, f) == part), None)
            if name is None:
                return _MISSING
            value = getattr(value, name)
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return _MISSING
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


@app.command(name='show', short_help='Show the effective settings', epilog=epilog)
def cli_config_show(
    key: Optional[str] = typer.Argument(
        None, metavar='[KEY]', help='Section or key to show, in dotted notation (e.g. jobs.resources)'
    ),
    diff: bool = typer.Option(False, '--diff', help='Only show the values that differ from the defaults'),
    file: Optional[Path] = typer.Option(
        None,
        '--file',
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help='Show the settings of this file instead of the active config file',
    ),
    comments: bool = typer.Option(
        True, '--comments/--no-comments', help='Show the field descriptions (or the defaults, with --diff) as comments'
    ),
    as_json: bool = typer.Option(False, '--json', help='Output as JSON'),
):
    """
    Show the settings that XTL runs with, as TOML. These are read from the active
    config file, or are the default settings if there is none (or if it is invalid).
    """
    import toml
    from rich.console import Console
    from rich.markup import escape
    from toml.decoder import CommentValue

    from xtl.config.settings import XTLSettings
    from xtl_core.options.toml import ExtendedTomlEncoder

    stderr = Console(stderr=True, highlight=False)

    # Load the settings
    if file is not None:
        settings, errors, _ = XTLSettings._read(file)
        if settings is None:
            stderr.print(f'[red]Error:[/] invalid config file: {escape("; ".join(errors))}')
            stderr.print('See [b]xtl.config validate[/b] for details.')
            raise typer.Exit(1)
        source = str(file)
    else:
        active = XTLSettings.active()
        settings = None
        if active is None:
            source = 'in-memory defaults'
        else:
            settings, errors, _ = XTLSettings._read(active.path)
            if settings is None:
                stderr.print(
                    f'[yellow]Warning:[/] the active [b]{active.name}[/b] '
                    f'config is invalid, showing the in-memory defaults '
                    f'instead. See [b]xtl.config validate[/b] for details.'
                )
                source = f'in-memory defaults ({active.name} config is invalid)'
            else:
                source = f'{active.name} ({active.path})'
        if settings is None:
            settings = XTLSettings()

    defaults = XTLSettings()
    data = XTLSettings._value_to_toml(settings, keep_comments=comments and not diff)
    default_data = XTLSettings._value_to_toml(defaults, keep_comments=False)

    # Resolve the requested section or key
    parts = tuple(key.split('.')) if key else ()
    value = _resolve(settings, parts)
    if value is _MISSING:
        # Suggest keys at the same depth, e.g. not the keys of a matched table
        known = ['.'.join(p) for p in _paths(_plain(data)) + _paths(_plain(default_data)) if len(p) == len(parts)]
        matches = difflib.get_close_matches(key, sorted(set(known)), n=3)
        hint = f' Did you mean: {", ".join(matches)}?' if matches else ''
        raise typer.BadParameter(f'Unknown key: {key}.{hint}', param_hint="'[KEY]'")

    # Collect the items to show
    notes = []
    if value is None:
        # TOML has no null, so unset values are left out of the TOML data
        if not diff or _get(default_data, parts) is not _MISSING:
            notes.append(f'{key} is not set')
        output = {}
    elif diff:
        output = {}
        encoder = ExtendedTomlEncoder()
        for path in _diff(_plain(data), _plain(default_data)):
            if path[: len(parts)] != parts:
                continue
            item = _get(data, path)
            default = _get(default_data, path)
            default_str = (
                'not set' if default is _MISSING else str(encoder.dump_value(default)).strip()
            )  # ints are not converted
            if item is _MISSING:
                notes.append(f'{".".join(path)} is not set (default: {default_str})')
                continue
            if comments and not isinstance(item, dict):
                item = CommentValue(val=_plain(item), comment=f'# default: {default_str}', beginline=False, _dict=dict)
            _set(output, path, item)
    else:
        item = _get(data, parts) if parts else data
        output = {}
        if parts:
            _set(output, parts, item)
        else:
            output = item

    # JSON output: the plain values, without the parent tables
    if as_json:
        result = _get(_plain(output), parts) if parts else _plain(output)
        if result is _MISSING:
            result = None if not diff else {}
        console = Console()
        if console.is_terminal:
            # Fancy colors for interactive terminals
            console.print_json(data=result, default=_json_default)
        else:
            print(json.dumps(result, default=_json_default))
        return

    # TOML output (printed without rich, so that [tables] are not read as markup)
    lines = [f'# Source: {source}']
    lines += [f'# {note}' for note in notes]
    if diff and not output and not notes:
        lines.append('# Same as the defaults')
    is_leaf = parts and not isinstance(_get(output, parts), dict)
    if output:
        if is_leaf:
            # A single key is shown without its parent tables
            leaf = {parts[-1]: _get(output, parts)}
            lines.append(toml.dumps(leaf, encoder=ExtendedTomlEncoder()).rstrip())
        else:
            lines.append(toml.dumps(output, encoder=ExtendedTomlEncoder()).rstrip())
    typer.echo('\n'.join(lines))
