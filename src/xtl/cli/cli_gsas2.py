import click

from xtl import cfg
from xtl.cli.utils import GroupWithCommandOptions, OutputQueue, update_docstring, dict_to_table
import xtl.cli.utils_gsas2 as g2u
from xtl.GSAS2.parameters import InstrumentalParameters
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
@click.pass_context
@click.argument('file', nargs=1)
@click.option('-p', '--phase')
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


@click.group(cls=GroupWithCommandOptions, context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
def params_group(ctx: click.core.Context, debug: bool, verbose: bool):
    if debug:
        click.secho(f'Debug mode is on.', fg='magenta')
    if debug and verbose:
        click.secho(f'Verbosity set to {verbose}.', fg='magenta')

    ctx.ensure_object(dict)
    ctx.obj = {
        'debug': debug,
        'verbose': verbose
    }


@params_group.command(short_help='Create .instprm files.', context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
@click.argument('file', nargs=1, type=click.Path())
@click.option('-l', '--wavelength', type=str, help="Set wavelength. When editing a lab parameters file all three "
                                                   "'Lam1', 'Lam2' and 'I(L2)/I(L1)' must be provided as a semicolon-"
                                                   "separated string (e.g. -l l1;l2;r). Alternatively, use --source to "
                                                   "change all three at once.")
@click.option('--source', type=str, help='Set radiation source. Only for characteristic radiations (lab '
                                         'diffractometers). Can be an element symbol (e.g. Cu), name (e.g. copper) or '
                                         'atomic number (e.g. 29).')
@click.option('-zs', '--zero_shift', type=float, help='Set the angular offset of the instrumentation.')
@click.option('-p', '--polarization', type=float, help='Set the polarization factor for x-ray source.')
@click.option('-a', '--azimuth', type=float, help='Set the azimuthal angle (for texture analysis).')
@click.option('-U', '--gaussian_u', type=float, help='Set peak shape parameter U (Gaussian).')
@click.option('-V', '--gaussian_v', type=float, help='Set peak shape parameter V (Gaussian).')
@click.option('-W', '--gaussian_w', type=float, help='Set peak shape parameter W (Gaussian).')
@click.option('-X', '--lorentzian_x', type=float, help='Set peak shape parameter X (Lorentzian).')
@click.option('-Y', '--lorentzian_y', type=float, help='Set peak shape parameter Y (Lorentzian).')
@click.option('-Z', '--lorentzian_z', type=float, help='Set peak shape parameter Z (Lorentzian).')
@click.option('-shl', '--axial_divergence', type=float, help='Set peak shape parameter SH/L (asymmetry).')
@click.option('-t', '--template', type=str, help='Template .instprm file to copy parameters from.')
def params(ctx: click.core.Context, file, wavelength, source, zero_shift, polarization, azimuth,
           gaussian_u, gaussian_v, gaussian_w, lorentzian_x, lorentzian_y, lorentzian_z, axial_divergence, template):
    """
    Create instrumental parameters files.

    A template file can be used to generate the new file. Individual parameters can be overridden using
    the respective options. E.g.:

    \b
    gsas2 params file.instprm -t template.instprm
    gsas2 params file.instprm -t template.instprm -l 1.3 -zs 0.0004

    If no template file is provided, then the default GSAS2 synchrotron or laboratory .instprm file will be created. In
    that case, you need to specify the wavelength or the radiation source to create a default file.

    \b
    gsas2 params file.instprm -l 2.0
    gsas2 params file.instprm --source=Cu

    To add a directory as a source of template .instprm files use the following command:

    \b
    gsas2 params /edit_temp
    \f
    :param ctx:
    :param file:
    :param wavelength:
    :param source:
    :param zero_shift:
    :param polarization:
    :param azimuth:
    :param gaussian_u:
    :param gaussian_v:
    :param gaussian_w:
    :param lorentzian_x:
    :param lorentzian_y:
    :param lorentzian_z:
    :param axial_divergence:
    :param template:
    :return:
    """
    # Edit template directories
    if file == '/edit_temp':
        ctx.invoke(params_edit_template_dir)
        return

    # Wavelength processing
    if wavelength:
        wavelength_list = tuple(wavelength.split(';'))
        if len(wavelength_list) == 1:  # Synchrotron source, 'Lam'
            try:
                wavelength = float(wavelength_list[0])
            except ValueError:
                click.secho(f'Wavelength must be float. You entered: {wavelength_list[0]}', fg='red')
                return
        elif len(wavelength_list) == 3:  # Laboratory source, 'Lam1', 'Lam2', 'I(L2)/L(1)'
            wavelength = []
            for value in wavelength_list:
                try:
                    wavelength.append(float(value))
                except ValueError:
                    click.secho(f'Wavelength value must be float. You entered: {value}', fg='red')
                    return
        else:
            click.secho(f'Wavelength can be either a single value (synchrotron sources) or a semicolon-separated list '
                        f'of length 3 (lab sources). You entered: {wavelength}', fg='red')
            return

    # Source processing
    try:
        # Check if source is an atomic number
        source = int(source)
    except ValueError:
        pass

    # Check if file exists and prompt
    import os
    if os.path.exists(file):
        yes = click.confirm(f'{file} file already exists. Would you like to overwrite it?')
        if not yes:
            return

    # Invoke subcommand and pass the ctx along
    ctx.invoke(params_create, file=file, wavelength=wavelength, source=source, zero_shift=zero_shift,
               polarization=polarization, azimuth=azimuth, gaussian_u=gaussian_u, gaussian_v=gaussian_v,
               gaussian_w=gaussian_w, lorentzian_x=lorentzian_x, lorentzian_y=lorentzian_y,
               lorentzian_z=lorentzian_z, axial_divergence=axial_divergence, template=template)


@click.command()
@click.pass_context
@click.argument('file')
@click.option('--wavelength')
@click.option('--source')
@click.option('--zero_shift')
@click.option('--polarization')
@click.option('--azimuth')
@click.option('--gaussian_u')
@click.option('--gaussian_v')
@click.option('--gaussian_w')
@click.option('--lorentzian_x')
@click.option('--lorentzian_y')
@click.option('--lorentzian_z')
@click.option('--axial_divergence')
@click.option('--template')
def params_create(ctx: click.core.Context, file, wavelength, source, zero_shift, polarization, azimuth, gaussian_u,
                  gaussian_v, gaussian_w, lorentzian_x, lorentzian_y, lorentzian_z, axial_divergence, template):
    """
    Subcommand for creating .instprm files.

    Invoked by :func:`.params`.
    Not registered in the help menu.
    \f
    :param ctx:
    :param file:
    :param wavelength:
    :param source:
    :param zero_shift:
    :param polarization:
    :param azimuth:
    :param gaussian_u:
    :param gaussian_v:
    :param gaussian_w:
    :param lorentzian_x:
    :param lorentzian_y:
    :param lorentzian_z:
    :param axial_divergence:
    :param template:
    :return:
    """

    debug = ctx.obj['debug']
    verbose = ctx.obj['verbose']

    # Create dictionary with user-provided parameters minus wavelength/source
    # Parameters not specified by the user will be set to None
    user_iparams = {
        'Zero': zero_shift,
        'Polariz.': polarization,
        'Azimuth': azimuth,
        'U': gaussian_u,
        'V': gaussian_v,
        'W': gaussian_w,
        'X': lorentzian_x,
        'Y': lorentzian_y,
        'Z': lorentzian_z,
        'SH/L': axial_divergence
    }

    # Create an initial parameters object
    if template:  # A template .instprm is provided
        import os
        if not os.path.isabs(template):
            # Create a list of directories to search for the template file
            template_dirs = [
                os.getcwd()  # cwd takes priority
            ]
            cfg_dirs = cfg['cli']['gsas_instprm_template_dir'].value.split(';')
            for dir in cfg_dirs:
                if os.path.isdir(dir):
                    template_dirs.append(os.path.abspath(dir))

            # Search for the file in each of the provided directories
            if debug:
                click.secho('Searching in the following directories for template .instprm:', fg='magenta')
                click.secho('  ' + '\n  '.join(template_dirs), fg='magenta')
            for dir in template_dirs:
                if template in os.listdir(dir):
                    template = os.path.join(dir, template)
                    if debug:
                        click.secho(f'Found template file: {template}', fg='magenta')
                    break  # exit after founding one matching template file

        # Create InstrumentalParameters object. Template is now an absolute path.
        ip = InstrumentalParameters(file=template)

        # Edit wavelength/source if provided
        if wavelength:
            ip.wavelength = wavelength  # either single value or list of three items
        elif source:
            temp_ip = InstrumentalParameters.defaults_lab(tube=source)
            ip.wavelength = temp_ip.wavelength
            new_source = temp_ip.source

    else:  # If no template file is provided
        # Create a file with the GSAS default parameters, using the user-provided wavelength/source
        if wavelength:
            ip = InstrumentalParameters.defaults_synchrotron(wavelength=wavelength)
        elif source:
            ip = InstrumentalParameters.defaults_lab(tube=source)
        else:
            click.secho(f'Instrumental parameters file cannot be created without a wavelength or a radiation source.',
                        fg='red')
            return

    # Create a dict from the initial parameters object (only wavelengt/source is user-specified)
    gsas_dict = ip.dictionary  # gsas_dict = {'entry': [old_value, new_value, include_in_refinement]}
    iparams = {key: gsas_dict[key][1] for key in gsas_dict}

    # Set the rest of the user-specified values
    for iparam, value in user_iparams.items():
        if value is not None:  # If user provided a value
            iparams[iparam] = value
    if template and source:  # Fix for adding the source, which does not have its own setter
        if wavelength:  # If user overwrites the source wavelength
            iparams['Source'] = ''
        else:
            iparams['Source'] = new_source

    # Create a new parameters object with the new parameters
    ip = InstrumentalParameters.from_dictionary(iparams)

    # Save file
    q = OutputQueue()
    ip.save_to_file(name=file)
    q.append_to_queue((f'Created file {file}', {}))
    if verbose:
        q.append_table(f'{file}', table=dict_to_table(iparams), tablefmt='two_columns', colalign=('right', 'left'))
    q.print()


@click.command()
def params_edit_template_dir():
    """
    Subcommand for editing the locations to lookup for template .instprm files.

    Invoked by :func:`.params`.
    Not registered in the help menu.
    \f
    :return:
    """

    # Get template directories from config file
    click.secho('Editing the directories for template .instprm files', fg='cyan')
    temp_dirs = cfg['cli']['gsas_instprm_template_dir'].value.split(';')
    if temp_dirs:
        click.secho('The following directories are currently registered:')
        click.secho('\n'.join([f'{i+1}) {temp_dir}' for i, temp_dir in enumerate(temp_dirs)]))

    # Prompt user for operation
    while True:
        commands = {
            'a': 'Add new entry',
            'd': 'Delete entry',
            'l': 'List entries',
            'x': 'Exit menu',
            '?': 'Display help'
        }
        operation = click.prompt(f'What would you like to do? ({", ".join(tuple(commands.keys()))})', default='?')
        command = operation[0].lower()
        entry = ''
        if len(operation) > 1:  # E.g. 'd1' or 'd 1'
            entry = operation[1:].strip()

        if command == 'a':  # Append new entry
            if entry:  # If user provided an path upon invocation
                import os
                if not os.path.isdir(entry):
                    click.secho(f'{entry} is not a directory or does not exist.', fg='red')
                    entry = ''  # reset entry

            if not entry:
                entry = click.prompt(f'Which directory do you want to add?', type=click.Path(exists=True,
                                                                                             file_okay=False,
                                                                                             dir_okay=True))

            # Check if directory is already registered
            if entry in temp_dirs:
                click.secho(f'{entry} already registered!', fg='red')
                continue

            # Append entry to config
            temp_dirs.append(entry)
            cfg['cli']['gsas_instprm_template_dir'] = ';'.join(temp_dirs)
            cfg.save()

        elif command == 'd':  # Delete entry
            try:  # If user provided an entry upon invocation
                entry = int(entry)
            except ValueError:
                click.secho(f"Entry must be an integer: '{entry}'", fg='red')
                entry = ''  # reset entry

            if not entry:
                entry = click.prompt(f'Which entry do you want to delete?', type=int)

            # Check if entry is valid
            try:
                remove_dir = temp_dirs[entry-1]
            except IndexError:
                click.secho(f'Entry {entry} does not exist.', fg='red')
                continue

            # Remove entry from config
            temp_dirs.remove(remove_dir)
            cfg['cli']['gsas_instprm_template_dir'] = ';'.join(temp_dirs)
            cfg.save()

        elif command == 'l':  # List entries
            click.secho('The following directories are currently registered:')
            click.secho('\n'.join([f'{i + 1}) {temp_dir}' for i, temp_dir in enumerate(temp_dirs)]))

        elif command == 'x':  # Exit
            break

        elif command == '?':  # Display commands
            click.secho('\n'.join([f'{c}: {commands[c]}' for c in commands]))
            continue

        else:
            click.secho(f'Unknown command. Type ? to get a list of available commands.', fg='red')
            continue

        # Prompt for another operation
        if click.confirm(f'Do you want to perform another operation?'):
            continue
        else:
            break


@click.command(cls=click.CommandCollection, sources=[info_group, params_group], invoke_without_command=True,
               context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
@click.option('-d', '--debug', is_flag=True, default=False, help='Run on debug mode.')
@click.option('-v', '--verbose', count=True, type=click.IntRange(0, 3, clamp=True), help='Display additional info.')
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
