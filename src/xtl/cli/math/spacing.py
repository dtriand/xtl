import numpy as np
import typer

from xtl.cli.cliio import Console
from xtl.units.scattering.radial import RadialUnits, RadialValue
from xtl.math.constants import KEV_PER_A

app = typer.Typer()


def print_radial_ids(explain_ids: bool):
    if explain_ids:
        cli = Console()
        table = []
        for unit in RadialUnits:
            table.append([unit.pretty, ', '.join(unit.aliases)])
        cli.print_table(table, headers=['Radial units', 'Identifiers'])
        raise typer.Exit()


@app.command('spacing', help='Convert between 2\u03b8, d and q spacing')
def cli_math_spacing(
        value: float = typer.Argument(..., help='Value to convert',
                                      min=0.),
        quantity: str = typer.Argument(..., case_sensitive=False, help='Units for provided value'),
        to_quantity: str = typer.Option(None, '-t', '--to', help='Units to convert to'),
        # X-ray parameters
        wavelength: float = typer.Option(None, '-w', '--wavelength', help='Wavelength in \u212b',
                                         min=0.001, rich_help_panel='X-ray parameters'),
        energy: float = typer.Option(None, '-e', '--energy', help='Energy in keV',
                                     min=0.001, rich_help_panel='X-ray parameters'),
        # Additional help
        explain_ids: bool = typer.Option(False, '--ids', help='Print all valid radial units identifiers and exit',
                                         is_flag=True, is_eager=True, callback=print_radial_ids, rich_help_panel='Help'),
        # Debugging
        verbose: int = typer.Option(0, '-v', '--verbose', count=True, help='Print additional information',
                                    rich_help_panel='Debugging'),
        debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                                   rich_help_panel='Debugging')
):
    cli = Console(verbose=verbose, debug=debug, striped_table_rows=False)

    from_type = RadialUnits.by_alias(quantity)
    if from_type is None:
        cli.print(f'Unknown identifier {quantity!r} for radial units. '
                  f'Use --ids to see all valid identifiers.', style='red')
        raise typer.Abort()

    if to_quantity is None:
        to_type = None
    else:
        to_type = RadialUnits.by_alias(to_quantity)
        if to_type is None:
            cli.print(f'Unknown identifier --to={to_quantity!r} for radial units. '
                      f'Use --ids to see all valid identifiers.', style='red')
            raise typer.Abort()

    # Convert energy to wavelength
    if wavelength is not None and energy is not None:
        cli.print('Please specify either wavelength or energy, not both.', style='red')
        raise typer.Abort()
    if energy:
        wavelength = KEV_PER_A / energy
        if verbose:
            cli.print(f'Converted {energy:,.6f} keV to {wavelength:,.6f} \u212b', style='cyan')

    if to_type is None and wavelength is None:
        cli.print('Wavelength required to calculate all units', style='red')
        raise typer.Abort()

    from xtl.exceptions.utils import Catcher

    r = RadialValue(value=value, type=from_type)
    with Catcher(echo_func=cli.print, traceback_func=cli.print_traceback, silent=True) as catcher:
        if to_type:
            n = r.to(units=to_type, wavelength=wavelength)
            rs = '' if r.type is RadialUnits.TWOTHETA_DEG else ' '
            ns = '' if n.type is RadialUnits.TWOTHETA_DEG else ' '
            cli.print(f'{r.name.latex}={r.value:,.6f}{rs}{r.units.latex} is '
                      f'{n.name.latex}={n.value:,.6f}{ns}{n.units.latex}')
        else:
            tth_deg = r.to(RadialUnits.TWOTHETA_DEG, wavelength=wavelength)
            tth_rad = r.to(RadialUnits.TWOTHETA_RAD, wavelength=wavelength)
            d_A = r.to(RadialUnits.D_A, wavelength=wavelength)
            d_nm = r.to(RadialUnits.D_NM, wavelength=wavelength)
            q_A = r.to(RadialUnits.Q_A, wavelength=wavelength)
            q_nm = r.to(RadialUnits.Q_NM, wavelength=wavelength)

            table = []
            for line in [[tth_deg, d_A, q_A], [tth_rad, d_nm, q_nm]]:
                row = []
                for q in line:
                    if np.isnan(q.value):
                        text = '\u221e'
                    else:
                        qs = '' if q.type is RadialUnits.TWOTHETA_DEG else ' '
                        text = f'{q.value:,.6f}{qs}{q.units.latex}'
                    if q.type == r.type:
                        text = f'[i]{text}[/]'
                    row.append(text)
                table.append(row)

            cli.print_table(table, headers=[tth_deg.name.latex, d_A.name.latex, q_nm.name.latex],
                            table_kwargs={'caption': f'Assuming \u03bb={wavelength:,.6f} \u212b',
                                          'box': None})
    if catcher.raised:
        raise typer.Abort()

