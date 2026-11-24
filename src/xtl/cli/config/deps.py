from typing import Optional

import typer

from xtl.cli.utilities import epilog
from xtl.jobs.shells import Shell
from xtl.jobs.sites import ComputeSite


app = typer.Typer()


def _setup(dependency, site) -> str:
    """How a site sets up a dependency: its modules (on a modules site), its directory, or neither (found on ``PATH``)."""
    from xtl.jobs.sites import ModulesSite

    if dependency.modules:
        if isinstance(site, ModulesSite):
            return f'modules {", ".join(dependency.modules)}'
        return 'PATH (modules not loaded on this site)'
    if dependency.path is not None:
        return f'path {dependency.path}'
    return 'PATH'


@app.command(name='deps', short_help='Check the shells and dependencies that jobs run with', epilog=epilog)
def cli_config_deps(
    names: Optional[list[str]] = typer.Argument(
        None, metavar='[NAME]...', help='Dependencies to check (default: all of them)', show_default=False
    ),
    compute_site: Optional[ComputeSite] = typer.Option(
        None, '--compute-site', help='Compute site to check on (default: as a job would choose)', show_default=False
    ),
    shell: Optional[Shell] = typer.Option(
        None, '--shell', help='Shell to check in (default: as a job would choose)', show_default=False
    ),
    verbose: bool = typer.Option(False, '-v', '--verbose', help='Show what each check printed'),
):
    """
    Check which shells are installed, and whether each dependency can be found with the compute site and shell that a
    job needing it would use, after setting it up (its path or modules in xtl.toml).
    """
    from pathlib import Path

    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from xtl import settings
    from xtl.jobs.dependencies import select_site_and_shell, unknown_dependencies
    from xtl.jobs.exceptions import DependencyError
    from xtl.jobs.shells import DefaultShell

    console = Console(highlight=False)

    # Shells: whether their executable exists
    system = DefaultShell().name
    table = Table(box=None, pad_edge=False, title='Shells', title_justify='left', title_style='bold')
    table.add_column('Shell')
    table.add_column('Executable', overflow='fold')
    table.add_column('Found')
    for member in Shell:
        if member == Shell.DEFAULT:
            continue
        instance = member.get()
        found = Path(instance.executable).is_file()
        table.add_row(
            f'{member.value} [dim](default)[/]' if member.value == system else member.value,
            escape(instance.executable),
            '[green]Yes[/]' if found else '[red]No[/]',
        )
    console.print(table)
    console.print()

    # Dependencies: each checked on its own, with the site and shell that a job needing only it would use
    names = list(dict.fromkeys(names)) if names else settings.dependencies.names()
    problems = [
        f'unknown dependency {name} (not in the [dependencies] of xtl.toml)' for name in unknown_dependencies(names)
    ]
    details: list[tuple[str, str]] = []
    table = Table(box=None, pad_edge=False, title='Dependencies', title_justify='left', title_style='bold')
    table.add_column('Name')
    table.add_column('Compute site')
    table.add_column('Shell')
    table.add_column('Setup', overflow='fold')
    table.add_column('Found')
    for name in (name for name in names if name in settings.dependencies):
        dependency = settings.dependencies.get(name)
        site_name = compute_site or dependency.compute_site or settings.jobs.compute_site
        try:
            site, chosen = select_site_and_shell([name], compute_site=compute_site, default_shell=shell)
        except (DependencyError, ValueError) as error:
            problems.append(f'{name} could not be checked: {error}')
            table.add_row(name, str(site_name), '', '', '[yellow]Not checked[/]')
            continue

        # A fresh check, not one remembered from earlier
        dependency.clear_checks()
        check = site.check_dependencies([name], shell=chosen)[0]
        problems.extend(check.problems())
        if not check.found and check.details:
            details.append((f'{name} ({site_name}, {chosen.name})', check.details))
        found = '[green]Yes[/]' if check.found else '[yellow]Not checked[/]' if check.error else '[red]No[/]'
        table.add_row(name, str(site_name), chosen.name, escape(_setup(dependency, site)), found)

    if table.rows:
        console.print(table)
        common = [
            *([f'path {settings.dependencies.path}'] if settings.dependencies.path is not None else []),
            *([f'modules {", ".join(settings.dependencies.modules)}'] if settings.dependencies.modules else []),
        ]
        if common:
            console.print(f'[dim]Set up for every job: {escape("; ".join(common))}[/]')
        console.print()

    if not settings.dependencies.check:
        console.print('[yellow]Checks are switched off for jobs (dependencies.check = false).[/]')

    if not problems:
        console.print('[green]All dependencies found.[/]')
        return

    console.print('Missing dependencies:', style='bold red')
    for problem in problems:
        console.print(f'- {escape(problem)}', style='red')
    if verbose:
        for label, text in details:
            console.print(f'What the check of {escape(label)} printed:', style='bold')
            console.print(escape(text), style='dim')
    elif details:
        console.print('Run with -v to see what the checks printed.')
    console.print(
        'Check [b]dependencies[/] in xtl.toml (see [b]xtl.config where[/]), or name only the dependencies you use.'
    )
    raise typer.Exit(1)
