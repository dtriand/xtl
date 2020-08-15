import click

from xtl import cfg
from xtl.cli.utils import GroupWithCommandOptions, OutputQueue, update_docstring, dict_to_table
import xtl.cli.utils_gsas2 as g2u
from xtl.GSAS2.projects import InformationProject

GI = g2u.GI
GI.G2sc.SetPrintLevel('error')
# GI.G2fil.G2printLevel = 'error'


@click.group(cls=GroupWithCommandOptions, context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
@click.pass_context
def info_group(ctx: click.core.Context, debug: bool, verbose: bool):
    if debug:
        click.secho(f'Debug mode is on.', fg='magenta')
    if debug and verbose:
        click.secho(f'Verbosity set to {verbose}.', fg='magenta')

    ctx.ensure_object(dict)
    ctx.obj = {
        'debug': debug,
        'verbose': verbose
    }


# Arguments: -phase, -hist, -constraint, -restraint, -rigidbody
@info_group.command(short_help='Print information about a project file.',
                    context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
@click.argument('file', nargs=1, type=click.Path(exists=True))
@click.option('-p', '--phase', show_default=True, metavar='<N: int>', help='Information about phase with ID=N.')
def info(ctx: click.core.Context, file: str, phase: int):
    """
    Prints information about a .gpx file, e.g. phases, histograms, restraints, constraints, etc.

    \f
    :param phase:
    :param ctx:
    :param file:
    :return:
    """
    gpx = InformationProject(filename=file)
    ctx.obj['gpx'] = gpx  # save gpx object to context

    # Invoke subcommands if info about a specific object (e.g. phase, histogram) is requested.
    # ctx is also passed along.
    if phase is not None:
        ctx.invoke(info_phase, file=file, phase=phase)
        return

    debug = ctx.obj['debug']
    verbose = ctx.obj['verbose']

    q = OutputQueue()

    controls_data = gpx.data['Controls']['data']
    no_phases, no_histograms, no_constraints, no_restraints, no_rigidbodies = gpx.get_no_of_items()

    info_general = [
        ['Author', controls_data['Author']],
        ['Last path', controls_data['LastSavedAs']],
        ['File size', gpx.get_filesize()],
        ['GSAS version', controls_data['LastSavedUsing']],
        ['Phases', no_phases],
        ['Histograms', no_histograms],
        ['Constraints', no_constraints],
        ['Restraints', no_restraints],
        ['Rigid bodies', no_rigidbodies]
    ]

    q.append_table(title='Project details', table=info_general, tablefmt='two_columns', colalign=('right', 'left'))

    info_phases = []
    for phase in gpx.phases():
        info_phases.append(g2u.get_phase_info(gpx, phase, verbose))
    q.append_table(title='Phases info', table=info_phases, headers='keys', disable_numparse=True, colalign=('right',),
                   tablefmt='simple')

    info_histograms = []
    for histogram in gpx.histograms():
        info_histograms.append(g2u.get_histogram_info(gpx, histogram, verbose))
    q.append_table(title='Histograms info', table=info_histograms, headers='keys', disable_numparse=True,
                   colalign=('right',), tablefmt='simple')

    q.print()


@click.command()
@click.argument('file', nargs=1)
@click.option('-p', '--phase')
@click.pass_context
def info_phase(ctx: click.core.Context, file, phase):
    """
    Subcommand for displaying info about a specific phase.

    Invoked by :func:`.info`.
    Not registered in the help menu.
    \f
    :param click.Context ctx:
    :param str file:
    :param str phase:
    :return:
    """
    debug = ctx.obj['debug']
    verbose = ctx.obj['verbose']
    gpx = ctx.obj['gpx']
    """
    :param xtl.GSAS2.projects.InformationProject gpx:
    """

    q = OutputQueue(debug)

    phase_id = phase
    phase = gpx.phase(phase_id)
    if not phase:
        click.secho(f'No phase with ID {phase}.', fg='red')
        return

    q.append_to_queue((f"Displaying information about phase '{phase.name}' (ID = {phase_id})\n", {}))

    info_cell = g2u.get_cell_info(phase, verbose)
    q.append_table(title='Unit-cell info', table=dict_to_table(info_cell), tablefmt='two_columns',
                   colalign=('left', 'left'))

    # Idea: Add space group information

    if gpx.get_phase_type(phase) == 'macromolecular':
        info_density = g2u.get_density_info(phase, verbose)
        q.append_table(title='Density info', table=dict_to_table(info_density), tablefmt='two_columns',
                       colalign=('right', 'left'))

    if gpx.has_map(phase):
        info_map = g2u.get_map_info(phase, verbose)
        q.append_table(title='Map info', table=dict_to_table(info_map), tablefmt='two_columns',
                       colalign=('right', 'left'))

    info_histograms = []
    for h in phase.histograms():
        histogram = gpx.histogram(h)
        info_histograms.append(g2u.get_histogram_info(gpx, histogram, verbose))
    q.append_table(title='Histograms info', table=info_histograms, headers='keys', disable_numparse=True,
                   colalign=('right',), tablefmt='simple')

    q.print()


@click.command(cls=click.CommandCollection, sources=[info_group], invoke_without_command=True,
               context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
@click.pass_context
@update_docstring(field='{version}', value=cfg['xtl']['version'].value)
def cli_gsas(ctx: click.core.Context, debug: bool, verbose: int):
    """
    \b
    Utilities for manipulating GSAS2 .gpx files.
    Installed by xtl (version {version})

    \f
    :param ctx:
    :param debug:
    :param verbose:
    :return:
    """
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


if __name__ == '__main__':
    cli_gsas(obj={})
