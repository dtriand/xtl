from datetime import datetime
from pathlib import Path
import re

import f90nml
import typer

from xtl import settings
from xtl.cli.cliio import Console, epilog
from xtl.cli.utils import parser_permissions
from xtl.common.os import get_permissions_in_decimal, FilePermissions


app = typer.Typer()


@app.command('fixnml', short_help='Update paths in NML files', epilog=epilog)
def cli_autoproc_fixnml(
        nml_files: list[Path] = typer.Argument(metavar='<NML_FILES>'),
        search_str: str = typer.Option(..., '-f', '--from', help='Search for this string',
                                       rich_help_panel='Search options'),
        target_str: str = typer.Option(..., '-t', '--to', help='Replace with this string',
                                       rich_help_panel='Search options'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out_dir', help='Output path for updated NML file',
                                     rich_help_panel='Output options'),
        overwrite: bool = typer.Option(False, '--overwrite', help='Overwrite if output file already '
                                                                  'exists', rich_help_panel='Output options'),
        check: bool = typer.Option(False, '--check', help='Check if the updated path exists',
                                   rich_help_panel='Output options'),
        chmod: bool = typer.Option(settings.automate.permissions.update, '--chmod',
                                   help='Change permissions of the output directories', rich_help_panel='Localization'),
        chmod_files: FilePermissions = typer.Option(settings.automate.permissions.files.string, '--chmod-files',
                                                    parser=parser_permissions, metavar='TEXT',
                                                    help='Permissions for files',
                                                    rich_help_panel='Localization'),
        verbose: int = typer.Option(0, '-v', '--verbose', count=True,
                                     help='Print additional information', rich_help_panel='Debugging'),
        debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                                   rich_help_panel='Debugging')
):
    '''
    Read one or more stratcal_gen.nml files and update the NAME_TEMPLATE string for each of the sweeps.

    '''
    cli = Console(verbose=verbose, debug=debug)
    updated_files = []
    for nml_file in nml_files:
        # Check if file exists
        if not nml_file.exists():
            cli.print(f'File not found: {nml_file}', style='red')
            raise typer.Abort()

        # Read NML file
        try:
            nml = f90nml.read(nml_file)
        except Exception as e:
            cli.print_traceback(e, '    ')
            cli.print(f'Failed to read file: {nml_file}', style='red')
            raise typer.Abort()

        no_sweeps = len(nml['simcal_sweep_list']) if 'simcal_sweep_list' in nml else 0
        cli.print(f'Read file: {nml_file}')
        cli.print(f'Found {no_sweeps} sweeps')

        # Skip NML file if it doesn't contain any sweeps
        if no_sweeps == 0:
            cli.print()
            continue

        # Iterate over the sweeps
        sweeps_updated = [False for i in range(no_sweeps)]
        for i, sweep in enumerate(nml['simcal_sweep_list']):
            t = '├─' if i + 1 < no_sweeps else '└─'
            # Skip if sweep doesn't contain name_template
            if 'name_template' not in sweep:
                cli.print(f'{t} [yellow]No name_template found for sweep {i + 1}[/]')
                continue

            # Skip if name_template doesn't contain search string
            name_template = str(sweep['name_template'])
            if search_str not in name_template:
                t1 = '│ ' if i + 1 < no_sweeps else '  '
                cli.print(f'{t} [yellow]{i + 1}: No match for search string: {search_str}[/]\n'
                          f'{t1}    [yellow]{name_template}[/]')
                continue

            new_name_template = name_template.replace(search_str, target_str)
            m = f'{i + 1}: {new_name_template}'

            # Check if the new name_template is valid
            if check:
                template = Path(new_name_template)
                directory = template.parent
                if not directory.exists():
                    cli.print(f'{t} [red]{i + 1}: Directory does not exist: {directory}[/]')
                    continue

                files = directory.glob(re.sub(r'\?+', '*', template.name))
                if not files:
                    cli.print(f'{t} [red]{m}[/]')
                else:
                    cli.print(f'{t} [green]{m}[/]')
                    sweeps_updated[i] = True
            else:
                cli.print(f'{t} {m}')
                sweeps_updated[i] = True

            # Update name_template
            sweep['name_template'] = new_name_template

        # Check if all sweeps where updated
        if not all(sweeps_updated):
            cli.print(f'Failed to update all sweeps for file: {nml_file}', style='red')
            raise typer.Abort()

        # Save updated NML to file
        nml.uppercase = True
        if out_dir.is_absolute():
            output_nml = out_dir / f'{nml_file.stem}_updated.nml'
        else:
            output_nml = nml_file.parent / out_dir / f'{nml_file.stem}_updated.nml'

        try:
            output_nml = output_nml.resolve()
            nml.write(output_nml, force=overwrite)
        except OSError as e:
            cli.print(f'[red]File {output_nml} already exists[/]')
            raise typer.Abort()

        # Replace newline characters
        output_nml.write_text(output_nml.read_text().replace('\r\n', '\n'), encoding='utf-8', newline='\n')

        # Update permissions
        if chmod:
            output_nml.chmod(mode=get_permissions_in_decimal(chmod_files))

        cli.print(f'Updated NML file saved in: {output_nml}\n')
        updated_files.append(output_nml)

    if updated_files:
        csv_out = Path(f'updated_nml_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv').resolve()
        if chmod:
            csv_out.touch(mode=get_permissions_in_decimal(chmod_files))
            csv_out.chmod(mode=get_permissions_in_decimal(chmod_files))
        with open(csv_out, 'w') as f:
            f.write('# nml_file\n')
            for file in updated_files:
                f.write(f'{file}\n')
            f.write(f'# Written by xtl.autoproc.fixnml at {datetime.now()}')
            cli.print(f'Wrote new .csv file: {csv_out}')

    return typer.Exit()