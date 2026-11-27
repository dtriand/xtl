from pathlib import Path
from typing import Optional

import typer

from xtl.cli.utilities import epilog


app = typer.Typer()


@app.command(name='upgrade', short_help='Upgrade a config file to the current version of XTL', epilog=epilog)
def cli_config_upgrade(
    file: Optional[Path] = typer.Argument(
        None,
        metavar='[FILE]',
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help='Config file to upgrade (default: the active config file)',
    ),
    clean: bool = typer.Option(
        False, '--clean', help="Write the file afresh, with XTL's descriptions and without your own comments"
    ),
):
    """
    Upgrade a config file written by an older version of XTL, in place: only the
    settings that changed are edited, and your comments and formatting stay. With
    [b]--clean[/b], the file is written afresh instead.
    """
    import tomlkit
    from rich.console import Console
    from rich.markup import escape

    from xtl.config.settings import XTLSettings
    from xtl.config.version import version
    from xtl_core.options.exceptions import OptionsValidationError

    console = Console(highlight=False)

    # Default to the active config file
    if file is None:
        active = XTLSettings.active()
        if active is None:
            console.print('No config file found, nothing to upgrade. Using [b]in-memory defaults[/b].')
            return
        name = f'{active.name} config'
        file = active.path
    else:
        name = 'config'

    def fail(*errors: str) -> None:
        for error in errors:
            console.print(f'[red]Error:[/] {escape(error)}')
        console.print(f'[red]Failed to upgrade {name}:[/] {escape(str(file))}; the file was left as it was.')
        raise typer.Exit(1)

    # NB: Read and written without translating line endings, so that the file keeps its own
    try:
        with open(file, encoding='utf-8', newline='') as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        fail(str(e))

    try:
        result = XTLSettings.upgrade_toml(text, clean=clean)
    except OptionsValidationError as e:
        # One error per invalid value, located without the model name, as `xtl.config validate` does
        fail(*(f'{error["loc"].partition(".")[2] or error["loc"]}: {error["msg"]}' for error in e.errors))
    except ValueError as e:
        fail(str(e))

    if result == text:
        console.print(f'The {escape(name)} is already up to date: {escape(str(file))}')
        return

    old = XTLSettings._get_version(tomlkit.parse(text).unwrap())
    try:
        with open(file, 'w', encoding='utf-8', newline='') as f:
            f.write(result)
    except OSError as e:
        fail(str(e))
    console.print(
        f'Upgraded {escape(name)} from {old.pep440 if old else "no version"} to {version.pep440}'
        f'{" (written afresh)" if clean else ""}: {escape(str(file))}'
    )
