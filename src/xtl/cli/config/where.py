import os
from pathlib import Path

import typer

from xtl.cli.utilities import epilog


app = typer.Typer()


def _is_writable(value: Path) -> bool:
    if not value.exists():
        # Check if the nearest existing parent directory is writable
        if value.parent == value:
            # Break recursion when we reach the root directory
            return False
        return _is_writable(value.parent)
    # If the file exists, check if it is writable
    return os.access(value, os.W_OK)


def _prettify_status(status: str) -> str:
    if status == 'active':
        return f'[green]{status}[/]'
    elif status == 'shadowed':
        return f'[green dim]{status}[/]'
    elif status == 'missing':
        return f'[red]{status}[/]'
    elif status == 'will create':
        return f'[cyan]{status}[/]'
    elif status == 'not set':
        return f'[yellow]{status}[/]'
    elif status == 'ignored':
        return f'[magenta]{status}[/]'
    else:
        return status


def _valid_status(valid: bool | None, errors: list, warnings: list) -> str | None:
    if valid is None:
        return None
    elif errors:
        return '[red]No[/]'
    elif warnings:
        return f'[yellow]{len(warnings)} warning{"s" if len(warnings) > 1 else ""}[/]'
    else:
        return '[green]Yes[/]'


@app.command(
    name='where',
    short_help='Lookup locations of XTL config files',
    epilog=epilog
)
def cli_config_where(
    as_json: bool = typer.Option(False, '--json', help='Output as JSON')
):
    from rich.console import Console

    from xtl.config.settings import XTLSettings
    from xtl.common.environment import XTL_CONFIG

    console = Console()
    locations = XTLSettings.locations()

    output = {
        'active': None,
        'will_create': None,
        'locations': [
            {
                'name': loc.name,
                'path': str(loc.path),
                'exists': loc.exists,
                'creatable': loc.creatable,
                'writable': _is_writable(loc.path),
                'status': '',
                'valid': None,
                'errors': [],
                'warnings': []
            }
            for loc in locations
        ]
    }

    # Check the XTL_CONFIG env var
    if not any(loc.name == 'env' for loc in locations):
        output['locations'].insert(0, {
            'name': 'env',
            'path': None if XTL_CONFIG.raw is None else str(XTL_CONFIG.value),
            'exists': False,
            'creatable': False,
            'writable': None,
            'status': 'not set' if XTL_CONFIG.raw is None else 'ignored',
            'valid': None,
            'errors': [],
            'warnings': []
        })

    # Find the active config file (if any)
    active = next((loc for loc in output['locations'] if loc['exists']), None)

    # Determine the status of each location
    for loc in output['locations']:
        if loc['exists']:
            loc['status'] = 'active' if loc is active else 'shadowed'
        elif not loc['status']:
            loc['status'] = 'missing'

    # If no active config exists, find the first location that initialize() would create
    if active is None:
        target = next((loc for loc in output['locations']
                       if loc['creatable'] and loc['writable']), None)
        if target is not None:
            target['status'] = 'will create'
            output['will_create'] = target['name']
    else:
        output['active'] = active['name']

    # Validate each existing config file and collect errors/warnings
    for loc in output['locations']:
        if not loc['exists']:
            continue
        errors, warnings = XTLSettings.check_file(Path(loc['path']))
        loc['errors'] = errors
        loc['warnings'] = warnings
        loc['valid'] = len(errors) == 0

    # Format results for JSON output
    if as_json:
        # Fancy colors for interactive terminals
        if console.is_terminal:
            console.print_json(data=output)
        else:
            import json
            print(json.dumps(output))
        return

    # Format results for table output
    from rich.markup import escape
    from rich.table import Table

    table = Table(box=None, pad_edge=False)
    table.add_column('Name')
    table.add_column('Path', overflow='fold')
    table.add_column('Exists')
    table.add_column('Writable')
    table.add_column('Status')
    table.add_column('Valid')

    for loc in output['locations']:
        if loc['status'] == 'not set':
            table.add_row(loc['name'], '(XTL_CONFIG not set)', None, None,
                          _prettify_status(loc['status']), None)
        elif loc['status'] == 'ignored':
            table.add_row(loc['name'], f'{escape(loc["path"])} [red dim](not a file)[/]',
                          None, None, _prettify_status(loc['status']), None)
        else:
            table.add_row(
                loc['name'] if loc['status'] != 'active' else f'[b]{loc["name"]}[/b]',
                escape(loc['path']),
                'Yes' if loc['exists'] else 'No',
                'Yes' if loc['writable'] else 'No',
                _prettify_status(loc['status']),
                _valid_status(loc['valid'], loc['errors'], loc['warnings'])
            )

    console.print(table)
    if active is None:
        if output['will_create'] is None:
            console.print('No config file found and no writable location '
                          'available to create one. Will be using defaults only.')
        else:
            console.print(f'No config file found. One will be created upon first use '
                          f'at: {escape(target["path"])}')
    elif not active['valid']:
        console.print(f'Active config file [b]{active["name"]}[/b] is invalid. '
                      f'Using in-memory defaults instead. Run '
                      f'[b]xtl.config validate {escape(active["path"])}[/b] for details.')
