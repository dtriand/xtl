import asyncio
from datetime import datetime
from functools import partial, wraps
import json
import math
import os
from pathlib import Path
from pprint import pformat
import re
from time import sleep

import f90nml
import pandas as pd
import rich.box
from rich.markup import escape
import rich.table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn
import typer

from xtl.cli.cliio import Console
import xtl.cli.automate.autoproc_utils as apu
from xtl.cli.utils import typer_async
from xtl.common.os import get_permissions_in_decimal
from xtl.config import cfg
from xtl.diffraction.automate.autoproc import AutoPROCJob, AutoPROCWorkflowJob
from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.exceptions.utils import Catcher
from xtl.math import si_units


app = typer.Typer(name='xtl.autoproc', help='Execute multiple autoPROC jobs', rich_markup_mode='rich',
                  epilog='</> with ❤️ by [i magenta]_dtriand[/]')


@app.command('options', help='Show available autoPROC configuration options', epilog=app.info.epilog)
def cli_autoproc_options():
    cli = Console()

    table_kwargs = {
        'title_style': 'bold italic white on cornflower_blue',
        'box': rich.box.HORIZONTALS,
        'expand': True
    }

    cli.print('The following parameters can be passed as headers in the [b i u]datasets.csv[/b i u] file.')
    cli.print()
    cli.print_table(table=apu.get_attributes_dataset(),
                    headers=['XTL parameter', 'Type', 'Description'],
                    column_kwargs=[
                        {'style': 'cornflower_blue'},
                        {'style': 'italic'},
                        {'style': 'bright_black'}
                    ],
                    table_kwargs=table_kwargs | {'title': 'Dataset options',
                                                 # 'caption': 'An additional \'dataset_group\' parameter can be added to '
                                                 #            'the [u]datasets.csv[/u] file to process and merge multiple'
                                                 #            ' datasets together ([i]e.g.[/i] multi-sweep data)'
                                                 }
                    )
    cli.print()
    cli.print_table(table=apu.get_attributes_config(),
                    headers=['XTL parameter', 'autoPROC parameter', 'Type', 'Description'],
                    column_kwargs=[
                        {'style': 'cornflower_blue'},
                        {'style': 'medium_orchid'},
                        {'style': 'italic'},
                        {'style': 'bright_black'}
                    ],
                    table_kwargs=table_kwargs | {'title': 'autoPROC configuration options',
                                                 'caption': 'Parameters in purple are the equivalent autoPROC '
                                                            'parameters that will be passed to the process command. '
                                                            'Additional parameters can be passed as a dictionary in '
                                                            'the \'extra_params\' argument. A full list of the '
                                                            'available autoPROC parameters can be found '
                                                            '[link=https://www.globalphasing.com/autoproc/manual/appendix1.html]'
                                                            'here[/link].'}
                    )

    return typer.Exit(code=0)


@app.command('process', short_help='Run multiple autoPROC jobs', epilog=app.info.epilog)
@typer_async
async def cli_autoproc_process(
    input_files: list[Path] = typer.Argument(metavar='<DATASETS>',
                                         help='List of paths to the first image files of datasets or a datasets.csv '
                                              'file'),
    # Dataset parameters
    raw_dir: Path = typer.Option(None, '-i', '--raw-dir', help='Path to the raw data directory',
                                 rich_help_panel='Dataset parameters'),
    out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory',
                                 rich_help_panel='Dataset parameters'),
    out_subdir: str = typer.Option(None, '--out-subdir', help='Subdirectory within the output '
                                   'directory', rich_help_panel='Dataset parameters'),
    # autoPROC parameters
    unit_cell: str = typer.Option(None, '-u', '--unit-cell', help='Unit-cell parameters',
                                  rich_help_panel='autoPROC parameters'),
    space_group: str = typer.Option(None, '-s', '--space-group', help='Space group',
                                    rich_help_panel='autoPROC parameters'),
    mtz_rfree: Path = typer.Option(None, '-f', '--mtz-rfree',
                                   help='Path to a MTZ file with R-free flags', rich_help_panel='autoPROC parameters'),
    mtz_ref: Path = typer.Option(None, '-R', '--mtz-ref', help='Path to a reference MTZ file',
                                 rich_help_panel='autoPROC parameters'),
    resolution: str = typer.Option(None, '-r', '--resolution', help='Resolution range',
                                   rich_help_panel='autoPROC parameters'),
    cutoff: apu.ResolutionCriterion = typer.Option(apu.ResolutionCriterion.cc_half.value, '-c', '--cutoff',
                                                   help='Resolution cutoff criterion',
                                                   rich_help_panel='autoPROC parameters'),
    beamline: apu.Beamline = typer.Option(None, '-b', '--beamline', show_choices=False,
                                          help='Beamline name', rich_help_panel='autoPROC parameters'),
    exclude_ice_rings: bool = typer.Option(None, '-e', '--exclude-ice', is_flag=True,
                                           flag_value=True, help='Exclude ice rings',
                                           rich_help_panel='autoPROC parameters'),
    no_residues: int = typer.Option(None, '-N', '--no-residues',
                            help='Number of residues in the asymmetric unit', rich_help_panel='autoPROC parameters'),
    anomalous: bool = typer.Option(True, '--no-anomalous', is_flag=True, flag_value=False,
                                   show_default=False, help='Merge anomalous signal',
                                   rich_help_panel='autoPROC parameters'),
    extra_args: list[str] = typer.Option(None, '-x', '--extra',
                                         help='Extra arguments to pass to autoPROC',
                                         rich_help_panel='autoPROC parameters'),
    # Parallelization parameters
    no_concurrent_jobs: int = typer.Option(1, '-n', '--no-jobs',
                                           help='Number of datasets to process in parallel',
                                           rich_help_panel='Parallelization'),
    n_threads: int = typer.Option(os.cpu_count(), '-t', '--threads', help='Number of threads for all jobs',
                                  rich_help_panel='Parallelization', hidden=True),
    xds_njobs: int = typer.Option(None, '-j', '--xds-jobs', help='Number of XDS jobs',
                                  rich_help_panel='Parallelization'),
    xds_nproc: int = typer.Option(None, '-p', '--xds-proc', help='Number of XDS processors',
                                  rich_help_panel='Parallelization'),
    # Localization
    modules: list[str] = typer.Option(None, '-m', '--module',
                                      help='Module to load before running the jobs', rich_help_panel='Localization'),
    compute_site: apu.ComputeSite = typer.Option(cfg['automate']['compute_site'].value, '--compute-site',
                                                 help='Computation site for configuring the job execution',
                                                 rich_help_panel='Localization'),
    chmod: bool = typer.Option(cfg['automate']['change_permissions'].value, '--chmod',
                               help='Change permissions of the output directories', rich_help_panel='Localization'),
    chmod_files: int = typer.Option(cfg['automate']['permissions_files'].value, '--chmod-files',
                                    help='Permissions for files', rich_help_panel='Localization'),
    chmod_dirs: int = typer.Option(cfg['automate']['permissions_directories'].value, '--chmod-dirs',
                                   help='Permissions for directories', rich_help_panel='Localization'),
    # Debugging
    log_file: Path = typer.Option(None, '-l', '--log', help='Path to the log file',
                                  rich_help_panel='Debugging'),
    verbose: int = typer.Option(0, '-v', '--verbose', count=True,
                                help='Print additional information', rich_help_panel='Debugging'),
    debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                               rich_help_panel='Debugging'),
    dry_run: bool = typer.Option(False, '--dry', help='Dry run without running autoPROC',
                                 rich_help_panel='Debugging'),
    do_only: int = typer.Option(0, '--only', hidden=True, help='Do only X jobs',
                                rich_help_panel='Debugging'),
):
    '''
    Execute multiple autoPROC jobs in parallel.



    EXAMPLES
        Simplest possible usage:
        xtl.autoproc process datasets.csv

        Provide starting unit-cell parameters and space group:
        xtl.autoproc process datasets.csv -u "78 78 37 90 90 90" -s "P 43 21 2"

        Provide reference MTZ file:
        xtl.autoproc process datasets.csv -R reference.mtz

    DATASETS.CSV EXAMPLE
        # first_image
        /path/to/dataset1/dataset1_00001.cbf.gz
        /path/to/dataset2/dataset2_00001.cbf.gz
        ...
    '''
    if log_file is None and cfg['cli']['log_file'].value:
        log_file = Path(cfg['cli']['log_file'].value)
    log_filename = f'xtl.autoproc.process_{os.getlogin()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_permissions = chmod_files if chmod else None

    cli = Console(verbose=verbose, debug=debug, log_file=log_file, log_filename=log_filename,
                  log_permissions=log_permissions)

    # Check if dry_run
    if dry_run:
        cli.print(':two-hump_camel: Dry run enabled', style='magenta')

    # Sanitize user input
    sanitized_input = {}

    if raw_dir:
        raw_dir = raw_dir.resolve()
        if not raw_dir.exists():
            cli.print(f'Raw data directory {raw_dir} does not exist', style='red')
            raise typer.Abort()
        sanitized_input['Raw data directory'] = raw_dir

    directories_created = []
    if out_dir:
        out_dir = out_dir.resolve()
        if not out_dir.exists():
            cli.print(f'Creating output directory: {out_dir} ', end='')
            try:
                out_dir.mkdir(parents=True)
            except OSError:
                cli.print('Failed.', style='red')
                raise typer.Abort()
            cli.print('Done.', style='green')
            directories_created.append(out_dir)
        sanitized_input['Output directory'] = out_dir

    if out_subdir:
        sanitized_input['Output subdirectory'] = out_subdir

    if unit_cell:
        uc = apu.parse_unit_cell(unit_cell)
        sanitized_input['Unit-cell parameters'] = ", ".join(map(str, uc))
    else:
        uc = None

    if space_group:
        sanitized_input['Space group'] = space_group.replace(' ', '')

    if mtz_rfree:
        sanitized_input['MTZ file with R-free flags'] = mtz_rfree

    if mtz_ref:
        sanitized_input['Reference MTZ file'] = mtz_ref

    res_low, res_high = apu.parse_resolution_range(resolution)
    if res_low or res_high:
        if not res_low:
            res_low = 999.0
        if not res_high:
            res_high = 0.1
        sanitized_input['Resolution range'] = f'{res_low} - {res_high} Å'

    if cutoff != apu.ResolutionCriterion.none:
        if res_high:
            sanitized_input['Resolution cutoff criterion'] = (f'[strike]{cutoff.value}[/strike] [i](ignored because a '
                                                              f'resolution range was provided)[/i]')
            cutoff = apu.ResolutionCriterion.none
        else:
            sanitized_input['Resolution cutoff criterion'] = cutoff.value

    if beamline:
        beamline = beamline.value
        sanitized_input['Beamline'] = beamline

    if exclude_ice_rings:
        sanitized_input['Ice rings'] = 'excluded'

    if no_residues:
        if no_residues <= 0:
            no_residues = None
        else:
            sanitized_input['Number of residues'] = no_residues

    sanitized_input['Anomalous signal'] = 'kept' if anomalous else 'merged'

    extra = apu.parse_extra_params(extra_args)
    if extra:
        sanitized_input['Extra autoPROC arguments'] = '\n'.join([f'{k}={v}' for k, v in extra.items()])

    sanitized_input['Number of concurrent jobs'] = no_concurrent_jobs
    sanitized_input['Number of threads per job'] = math.floor(n_threads / no_concurrent_jobs)
    if xds_njobs:
        sanitized_input['Number of XDS jobs'] = xds_njobs
    if xds_nproc:
        sanitized_input['Number of XDS processors'] = xds_nproc

    cs = compute_site.get_site()
    sanitized_input['Computation site'] = f'{compute_site.value}' + (f' \[{cs.priority_system.system_type}]'
                                                                     if cs.priority_system.system_type else '')
    if modules:
        sanitized_input['Modules'] = '\n'.join(modules)

    if chmod != bool(cfg['automate']['change_permissions'].value):
        sanitized_input['Change permissions'] = 'enabled' if chmod else 'disabled'

    if chmod_files != int(cfg['automate']['permissions_files'].value):
        sanitized_input['Permissions for files'] = chmod_files

    if chmod_dirs != int(cfg['automate']['permissions_directories'].value):
        sanitized_input['Permissions for directories'] = chmod_dirs

    if do_only:
        sanitized_input['Total number of jobs'] = f'{do_only} [i](limited by --only)[/]'

    if verbose:
        cli.print('The following global parameters will be used unless overriden on the .csv file:')
        cli.print_table(table=[[key, str(value)] for key, value in sanitized_input.items()],
                        headers=['Parameter', 'Value'],
                        column_kwargs=[{'style': 'deep_pink1'}, {'style': 'orange3'}],
                        table_kwargs={'title': 'Global parameters', 'expand': True, 'box': rich.box.HORIZONTALS})
        cli.confirm('Would you like to proceed with the above parameters?', default=False)

    # Housekeeping
    csv_file = None
    datasets = []
    csv_dict = {}

    # Input for DiffractionDataset constructors
    #  raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir, output_subdir
    datasets_input = []

    # Check if a datasets.csv file was provided
    if len(input_files) == 1 and input_files[0].suffix == '.csv':
        if not input_files[0].exists():
            cli.print(f'File {input_files[0]} does not exist', style='red')
        csv_file = input_files[0]
        cli.print(f'📃 Parsing datasets from {csv_file}')

        cli.print('\n### CSV FILE CONTENTS \n' + csv_file.read_text() + '### END CSV FILE CONTENTS\n',
                  log_only=True)

        csv_dict = apu.parse_csv(csv_file, echo=cli.print)
        cli.print(f'📑 Found {len(csv_dict["headers"])} headers in the CSV file: ')
        cli.print('\n'.join(f' - {h} ' + escape(f'[{csv_dict["index"][h]}]') for h in csv_dict['headers']))

        # Check if dataset paths have been fully specified and collect the images
        datasets_input = apu.sanitize_csv_datasets(csv_dict=csv_dict, raw_dir=raw_dir, out_dir=out_dir,
                                                   out_subdir=out_subdir, echo=cli.print)
    else:
        for i, image in enumerate(input_files):
            # Prepend the raw_dir if an absolute path is not provided
            if raw_dir and raw_dir.exists():
                if not image.is_absolute():
                    image = raw_dir / image
            if not image.exists():
                cli.print(f'Image for dataset {i+1} does not exist: {image}', style='red')
                raise typer.Abort()
            datasets_input.append([None, None, None, image, out_dir, None])
    cli.print(f'🗃️ Found {len(datasets_input)} datasets from input')

    # Report the dataset attributes parsed from the CSV file
    if log_file or verbose:
        log_only = (verbose == 0)
        renderable_datasets = [list(map(apu.str_or_none, dataset_params)) for dataset_params in datasets_input]
        cli.print('The following parameters will be used for locating the images:', log_only=log_only)
        cli.print_table(table=renderable_datasets,
                        headers=['raw_data_dir', 'dataset_dir', 'dataset_name', 'first_image',
                                 'processed_data_dir', 'output_dir', 'output_subdir'],
                        column_kwargs=[{'overflow': 'fold', 'style': 'deep_pink1'},
                                       {'overflow': 'fold', 'style': 'medium_orchid1'},
                                       {'overflow': 'fold', 'style': 'plum1'},
                                       {'overflow': 'fold', 'style': 'orange3'},
                                       {'overflow': 'fold', 'style': 'dodger_blue1'},
                                       {'overflow': 'fold', 'style': 'steel_blue1'},
                                       {'overflow': 'fold', 'style': 'cornflower_blue'}],
                        table_kwargs={'title': 'Sanitized datasets input', 'expand': True,
                                      'box': rich.box.HORIZONTALS},
                        log_only=log_only)
        cli.print(log_only=log_only)

    # Create DiffractionDataset instances
    no_images = 0
    t0 = datetime.now()
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  transient=True) as progress:
        task = progress.add_task('🔎 Looking for images in directories...',
                                 total=len(datasets_input))
        with Catcher(silent=not debug) as catcher:  # debug will print the exceptions
            for i, (r_dir, d_dir, d_name, image, p_dir, o_dir, o_sdir) in enumerate(datasets_input):
                try:
                    if image:
                        reading_method = 'from_image'
                        dataset = DiffractionDataset.from_image(image=image, raw_dataset_dir=r_dir,
                                                                processed_data_dir=p_dir, output_dir=o_dir)
                    else:
                        reading_method = 'from_dirs'
                        dataset = DiffractionDataset(dataset_name=d_name, dataset_dir=d_dir, raw_data_dir=r_dir,
                                                     processed_data_dir=p_dir, output_dir=o_dir)
                except Exception as e:
                    catcher.log_exception(
                        {
                            'index': i + 1,
                            'method': reading_method,
                            'data': {
                                'raw_data_dir': r_dir, 'dataset_dir': d_dir, 'dataset_name': d_name,
                                'first_image': image, 'processed_data_dir': p_dir, 'output_dir': o_dir,
                                'output_subdir': o_sdir
                            },
                            'exception': e
                        }
                    )
                    continue
                if o_sdir:
                    setattr(dataset, 'output_subdir', o_sdir)
                    dataset._fstring_dict['processed_data_dir'] += f'/{o_sdir}'
                    dataset._check_dir_fstring('processed_data_dir')
                no_images += dataset.no_images
                dataset.reset_images_cache()
                datasets.append(dataset)
                progress.advance(task)
    t1 = datetime.now()
    cli.print(f'📷 Found {no_images:,} images in {len(datasets)} datasets in {t1 - t0}')

    # Exit if there were any errors while creating the datasets
    if catcher.errors:
        cli.print(f'The following {len(catcher.errors)} dataset(s) could not be processed:', style='red')
        for error in catcher.errors:
            cli.print(f':police_car_light: Dataset {error["index"]} read with method \'{error["method"]}\'',
                      style='red bold')
            for key, value in error['data'].items():
                cli.print(f'    - {key}: {value}', style='red dim')
            cli.print(f'\n    The following exception was raised:', style='red')
            cli.print_traceback(exc=error['exception'], indent='    ')
            cli.print()
        raise typer.Abort()

    # Report the actual attributes of the datasets
    renderable_params = []
    missing_dirs = 0
    for dataset in datasets:
        processed_data_dir = dataset.processed_data
        if not processed_data_dir.exists():
            processed_data_dir = f'[u red]{processed_data_dir}[/]'
            missing_dirs += 1
        params = [dataset.raw_data, dataset.dataset_dir, dataset.dataset_name, dataset.first_image,
                  processed_data_dir, dataset.output_dir]
        template, img_no_first, img_no_last = dataset.get_image_template(first_last=True)
        params += [template, dataset.file_extension, img_no_first, img_no_last, dataset.no_images]
        renderable_params.append(list(map(apu.str_or_none, params)))
    if verbose or missing_dirs or log_file:
        log_only = not(verbose or missing_dirs)
        headers = ['raw_data_dir', 'dataset_dir', 'dataset_name', 'first_image', 'processed_data_dir', 'output_dir',
                   'image_template', 'file_extension', 'img_no_first', 'img_no_last', 'no_images']
        cli.print('The following datasets were initialized:\n', log_only=log_only)
        cli.print_table(table=renderable_params,
                        headers=headers,
                        column_kwargs=[{'overflow': 'fold', 'style': 'deep_pink1'},
                                       {'overflow': 'fold', 'style': 'medium_orchid1'},
                                       {'overflow': 'fold', 'style': 'plum1'},
                                       {'overflow': 'fold', 'style': 'orange3'},
                                       {'overflow': 'fold', 'style': 'dodger_blue1'},
                                       {'overflow': 'fold', 'style': 'steel_blue1'},
                                       {'overflow': 'fold', 'style': 'dark_orange3'},
                                       {'overflow': 'fold', 'style': 'salmon1'},
                                       {'overflow': 'fold', 'style': 'cyan3'},
                                       {'overflow': 'fold', 'style': 'sea_green3'},
                                       {'overflow': 'fold', 'style': 'dark_cyan'}],
                        table_kwargs={'title': 'Datasets attributes', 'expand': True,
                                      'caption': 'Table also saved as [u]datasets_sanitized.csv[/u]',
                                      'box': rich.box.HORIZONTALS},
                        log_only=log_only)
        cli.print(log_only=log_only)
        if missing_dirs:
            cli.print(f'Any [u red]red and underlined paths[/] in table above indicate missing directories that will '
                      f'be generated.\n[yellow]Please ensure that these paths are correct [b]before launching the '
                      f'jobs![/][/]', log_only=log_only)
        cli.confirm('Do the above output directories look correct?', default=False)

        # Save datasets attributes to a CSV file
        output = Path('./datasets_sanitized.csv')
        with output.open('w') as f:
            f.write('# ' + ','.join(headers) + '\n')
            for params in renderable_params:
                f.write(','.join(params) + '\n')
            f.write(f'# Written by xtl.autoproc.process at {datetime.now()}')
        if chmod:
            output.chmod(mode=get_permissions_in_decimal(chmod_files))

    # Prepare the jobs
    jobs = []
    sanitized_configs = {}
    APJ = AutoPROCJob.update_concurrency_limit(no_concurrent_jobs)
    APJ._echo_success_kwargs = {'style': 'green'}
    APJ._echo_warning_kwargs = {'style': 'yellow'}
    APJ._echo_error_kwargs = {'style': 'red'}
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  transient=True, console=cli) as progress:
        APJ._echo = partial(progress.console.print, highlight=False, markup=False, overflow='fold', log_escape=True)
        task = progress.add_task('🛠️ Preparing jobs...', total=len(datasets))
        with Catcher(silent=not debug) as catcher:  # debug will print the exceptions
            progress.console.print('\n### JOB OPTIONS', log_only=True)
            for i, dataset in enumerate(datasets):
                if i >= do_only > 0:
                    progress.console.print(f'Skipping the rest of the datasets (--only={do_only})',
                                           style='magenta')
                    break
                config_input = apu.merge_configs(csv_dict=csv_dict, dataset_index=i, **{
                    'change_permissions': chmod, 'file_permissions': chmod_files, 'directory_permissions': chmod_dirs,
                    'unit_cell': uc, 'space_group': space_group, 'resolution_high': res_high, 'resolution_low': res_low,
                    'anomalous': anomalous, 'no_residues': no_residues, 'rfree_mtz': mtz_rfree,
                    'reference_mtz': mtz_ref, 'xds_njobs': xds_njobs, 'xds_nproc': xds_nproc,
                    'exclude_ice_rings': exclude_ice_rings, 'beamline': beamline,
                    'resolution_cutoff_criterion': cutoff.value, 'extra_params': extra
                })
                sanitized_config = {
                    'input': {
                        'datasets': [dataset],
                        'config': config_input
                    }
                }
                sanitized_configs[i] = sanitized_config
                try:
                    config = AutoPROCConfig(batch_mode=True, **config_input)
                    sanitized_configs[i]['config'] = config
                    job = APJ(datasets=dataset, config=config, compute_site=cs, modules=modules)
                    sanitized_configs[i]['job'] = job.__dict__
                except Exception as e:
                    catcher.log_exception({'index': i + 1, 'data': sanitized_config, 'exception': e})
                    continue
                finally:
                    if debug or log_file:
                        log_only = not debug
                        progress.console.print(f'Job options for dataset {i+1}:', log_only=log_only)
                        progress.console.pprint(sanitized_config, log_only=log_only)
                        progress.console.print('', log_only=log_only)

                jobs.append(job)
                progress.advance(task)
            progress.console.print('### END JOB OPTIONS', log_only=True)
    no_jobs = len(jobs)
    cli.print(f'🛠️ Prepared {no_jobs} job' + ('s' if no_jobs > 1 else ''))

    # Exit if there were any errors while creating the jobs
    if catcher.errors:
        cli.print(f'The following {len(catcher.errors)} job(s) could not be created:', style='red')
        for error in catcher.errors:
            cli.print(f':police_car_light: Job {error["index"]} was instantiated with the following data:',
                      style='red bold')
            cli.print(error['data'], style='red dim')
            cli.print(f'\n    The following exception was raised:', style='red')
            cli.print_traceback(exc=error['exception'], indent='    ')
            cli.print('')
        cli.print('All data passed to the jobs is saved in [u]jobs_input.txt[/]', style='magenta')
        with open('jobs_input.txt', 'w') as f:
            f.write(pformat(sanitized_configs))
        if chmod:
            Path('jobs_input.txt').chmod(mode=get_permissions_in_decimal(chmod_files))
        raise typer.Abort()

    message = f'🚀 Would you like to launch {no_jobs} job'
    if no_jobs > 1:
        message += 's'
    if no_jobs > no_concurrent_jobs > 1:
        message += f' in batches of {no_concurrent_jobs}'
    message += '?'
    cli.print(message, log_only=True)
    cli.confirm(message, default=False)

    # Run the jobs
    t0 = datetime.now()
    cli.print(f'\nLaunching jobs at {t0}...')
    output_csv = []
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  TextColumn('{task.fields[status]}'),
                  transient=True, console=cli) as progress:
        jobs_succeeded = 0
        jobs_tidyup_failed = 0
        jobs_failed = 0
        running = progress.add_task(':person_running: Running jobs... ', total=no_jobs,
                                    status='Status: Running...')

        # Attach progress bar's print to the jobs
        logger = partial(progress.console.print, highlight=False, overflow='fold', markup=False, log_escape=True)
        for job in jobs:
            job._echo = logger

        # Generate tasks list
        pending_tasks = [asyncio.create_task(job.run(execute_batch=not dry_run)) for job in jobs]
        with Catcher(silent=not debug) as catcher:
            while pending_tasks:
                try:
                    completed_tasks, pending_tasks = await asyncio.wait(pending_tasks,
                                                                        return_when=asyncio.FIRST_COMPLETED)
                    for completed_task in completed_tasks:
                        job = completed_task.result()
                        directories_created.append(job.job_dir)

                        for d in job.datasets:
                            c = job.config
                            o_sdir = d.output_subdir if hasattr(d, 'output_subdir') else None
                            output_csv.append([job.job_dir.resolve(), job.run_no, job._success, d.sweep_id,
                                               d.autoproc_id, d.dataset_name, d.dataset_dir, d.first_image,
                                               d.raw_data_dir, d.processed_data_dir, d.output_dir, o_sdir,
                                               c.mtz_project_name, c.mtz_crystal_name, c.mtz_dataset_name])


                        progress.advance(running)
                        if job._success:
                            jobs_succeeded += 1
                        else:
                            if job._results is None:
                                jobs_tidyup_failed += 1
                            else:
                                if not (job.job_dir / job._results._json_fname).exists():
                                    jobs_tidyup_failed += 1
                                else:
                                    jobs_failed += 1
                        progress.update(running, status=f'Status: :star-struck: [green]{jobs_succeeded}[/] '
                                                        f':thinking_face: [yellow]{jobs_tidyup_failed}[/] '
                                                        f':loudly_crying_face: [red]{jobs_failed}[/]')
                except Exception as e:
                    catcher.log_exception({'index': i + 1, 'job': job, 'exception': e})
                    continue
        csv_out = (out_dir / f'datasets_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv').resolve()
        if chmod:
            csv_out.touch(mode=get_permissions_in_decimal(chmod_files))
            csv_out.chmod(mode=get_permissions_in_decimal(chmod_files))
        with open(csv_out, 'w') as f:
            f.write('# ' + ','.join(['job_dir', 'run_no', 'success', 'sweep_id', 'autoproc_id',
                                     'dataset_name', 'dataset_dir', 'first_image', 'raw_data_dir',
                                     'processed_data_dir', 'output_dir', 'output_subdir', 'mtz_project_name',
                                     'mtz_crystal_name', 'mtz_dataset_name']) + '\n')
            for line in output_csv:
                values = [str(v) if v else '' for v in line]
                f.write(','.join(values) + '\n')
            f.write(f'# Written by xtl.autoproc.process at {datetime.now()}')
            cli.print(f'Wrote new .csv file: {csv_out}')
    cli.print('')
    t1 = datetime.now()

    file_size = 0
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  transient=True, console=cli) as progress:
        task = progress.add_task(':bookmark_tabs: Calculating disk space used...',
                                 total=len(directories_created))
        for created in directories_created:
            file_size += apu.get_directory_size(created)
            progress.advance(task)
        sleep(0.5)

    jobs_all = len(jobs) + 1
    if jobs_succeeded == jobs_all:
        cli.print(f'😎 All jobs completed at {t1}', style='green')
    else:
        cli.print(f'🫡 All jobs completed at {t1}', style='green')
        cli.print(f'Outlook: '
                  f'[green]:star-struck: {jobs_succeeded} succeeded[/], '
                  f'[yellow]:thinking_face: {jobs_tidyup_failed} tidy-up failed[/], '
                  f'[red]:loudly_crying_face: {jobs_failed} failed[/]',)
    cli.print(f':hourglass_done: Total elapsed time: {t1 - t0} (approx. {(t1 - t0) / len(jobs)} per job)')

    file_size_human_friendly = si_units(file_size, suffix='B', base=1024, digits=2)
    cli.print(f':bookmark_tabs: Total disk space used: {file_size_human_friendly}')

    # Write new csv file for downstream processing
    with open('jobs_output.txt', 'w') as f:
        f.write('\n'.join([str(created) for created in directories_created]))

    # TODO:
    #  [x] Change permissions
    #  [x] Fix output_dir bug
    #  [x] Write new csv file for downstream processing
    #  [x] Add option for logging stdout to file (--log)
    #  [x] Run GPhL workflow files
    #  [ ] Monitor resources in the progress bar [psutil.cpu_percent() and psutil.virtual_memory().percent]
    #  [ ] Rethink success criteria


@app.command('process_wf', short_help='Run multiple GPhL workflow jobs', epilog=app.info.epilog)
@typer_async
async def cli_autoproc_process_wf(
    input_files: list[Path] = typer.Argument(metavar='<DATASETS>',
                                         help='List of paths to the NML files of datasets or a datasets.csv '
                                              'file'),
    # Dataset parameters
    raw_dir: Path = typer.Option(None, '-i', '--raw-dir', help='Path to the raw data directory',
                                 rich_help_panel='Dataset parameters'),
    out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory',
                                 rich_help_panel='Dataset parameters'),
    out_subdir: str = typer.Option(None, '--out-subdir', help='Subdirectory within the output '
                                   'directory', rich_help_panel='Dataset parameters'),
    # autoPROC parameters
    unit_cell: str = typer.Option(None, '-u', '--unit-cell', help='Unit-cell parameters',
                                  rich_help_panel='autoPROC parameters'),
    space_group: str = typer.Option(None, '-s', '--space-group', help='Space group',
                                    rich_help_panel='autoPROC parameters'),
    mtz_rfree: Path = typer.Option(None, '-f', '--mtz-rfree',
                                   help='Path to a MTZ file with R-free flags', rich_help_panel='autoPROC parameters'),
    mtz_ref: Path = typer.Option(None, '-R', '--mtz-ref', help='Path to a reference MTZ file',
                                 rich_help_panel='autoPROC parameters'),
    resolution: str = typer.Option(None, '-r', '--resolution', help='Resolution range',
                                   rich_help_panel='autoPROC parameters'),
    cutoff: apu.ResolutionCriterion = typer.Option(apu.ResolutionCriterion.cc_half.value, '-c', '--cutoff',
                                                   help='Resolution cutoff criterion',
                                                   rich_help_panel='autoPROC parameters'),
    beamline: apu.Beamline = typer.Option(None, '-b', '--beamline', show_choices=False,
                                          help='Beamline name', rich_help_panel='autoPROC parameters'),
    exclude_ice_rings: bool = typer.Option(None, '-e', '--exclude-ice', is_flag=True,
                                           flag_value=True, help='Exclude ice rings',
                                           rich_help_panel='autoPROC parameters'),
    no_residues: int = typer.Option(None, '-N', '--no-residues',
                            help='Number of residues in the asymmetric unit', rich_help_panel='autoPROC parameters'),
    anomalous: bool = typer.Option(True, '--no-anomalous', is_flag=True, flag_value=False,
                                   show_default=False, help='Merge anomalous signal',
                                   rich_help_panel='autoPROC parameters'),
    extra_args: list[str] = typer.Option(None, '-x', '--extra',
                                         help='Extra arguments to pass to autoPROC',
                                         rich_help_panel='autoPROC parameters'),
    # Parallelization parameters
    no_concurrent_jobs: int = typer.Option(1, '-n', '--no-jobs',
                                           help='Number of datasets to process in parallel',
                                           rich_help_panel='Parallelization'),
    n_threads: int = typer.Option(os.cpu_count(), '-t', '--threads', help='Number of threads for all jobs',
                                  rich_help_panel='Parallelization', hidden=True),
    xds_njobs: int = typer.Option(None, '-j', '--xds-jobs', help='Number of XDS jobs',
                                  rich_help_panel='Parallelization'),
    xds_nproc: int = typer.Option(None, '-p', '--xds-proc', help='Number of XDS processors',
                                  rich_help_panel='Parallelization'),
    # Localization
    modules: list[str] = typer.Option(None, '-m', '--module',
                                      help='Module to load before running the jobs', rich_help_panel='Localization'),
    compute_site: apu.ComputeSite = typer.Option(cfg['automate']['compute_site'].value, '--compute-site',
                                                 help='Computation site for configuring the job execution',
                                                 rich_help_panel='Localization'),
    chmod: bool = typer.Option(cfg['automate']['change_permissions'].value, '--chmod',
                               help='Change permissions of the output directories', rich_help_panel='Localization'),
    chmod_files: int = typer.Option(cfg['automate']['permissions_files'].value, '--chmod-files',
                                    help='Permissions for files', rich_help_panel='Localization'),
    chmod_dirs: int = typer.Option(cfg['automate']['permissions_directories'].value, '--chmod-dirs',
                                   help='Permissions for directories', rich_help_panel='Localization'),
    # Debugging
    log_file: Path = typer.Option(None, '-l', '--log', help='Path to the log file',
                                  rich_help_panel='Debugging'),
    verbose: int = typer.Option(0, '-v', '--verbose', count=True,
                                help='Print additional information', rich_help_panel='Debugging'),
    debug: bool = typer.Option(False, '--debug', hidden=True, help='Print debug information',
                               rich_help_panel='Debugging'),
    dry_run: bool = typer.Option(False, '--dry', help='Dry run without running autoPROC',
                                 rich_help_panel='Debugging'),
    do_only: int = typer.Option(0, '--only', hidden=True, help='Do only X jobs',
                                rich_help_panel='Debugging'),
):
    '''
    Execute multiple aP_wf_process jobs in parallel.
    '''
    if log_file is None and cfg['cli']['log_file'].value:
        log_file = Path(cfg['cli']['log_file'].value)
    log_filename = f'xtl.autoproc.process_wf_{os.getlogin()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_permissions = chmod_files if chmod else None

    cli = Console(verbose=verbose, debug=debug, log_file=log_file, log_filename=log_filename,
                  log_permissions=log_permissions)

    # Check if dry_run
    if dry_run:
        cli.print(':two-hump_camel: Dry run enabled', style='magenta')

    # Sanitize user input
    sanitized_input = {}

    if raw_dir:
        raw_dir = raw_dir.resolve()
        if not raw_dir.exists():
            cli.print(f'Raw data directory {raw_dir} does not exist', style='red')
            raise typer.Abort()
        sanitized_input['Raw data directory'] = raw_dir

    directories_created = []
    if out_dir:
        out_dir = out_dir.resolve()
        if not out_dir.exists():
            cli.print(f'Creating output directory: {out_dir} ', end='')
            try:
                out_dir.mkdir(parents=True)
            except OSError:
                cli.print('Failed.', style='red')
                raise typer.Abort()
            cli.print('Done.', style='green')
            directories_created.append(out_dir)
        sanitized_input['Output directory'] = out_dir

    if out_subdir:
        sanitized_input['Output subdirectory'] = out_subdir

    if unit_cell:
        uc = apu.parse_unit_cell(unit_cell)
        sanitized_input['Unit-cell parameters'] = ", ".join(map(str, uc))
    else:
        uc = None

    if space_group:
        sanitized_input['Space group'] = space_group.replace(' ', '')

    if mtz_rfree:
        sanitized_input['MTZ file with R-free flags'] = mtz_rfree

    if mtz_ref:
        sanitized_input['Reference MTZ file'] = mtz_ref

    res_low, res_high = apu.parse_resolution_range(resolution)
    if res_low or res_high:
        if not res_low:
            res_low = 999.0
        if not res_high:
            res_high = 0.1
        sanitized_input['Resolution range'] = f'{res_low} - {res_high} Å'

    if cutoff != apu.ResolutionCriterion.none:
        if res_high:
            sanitized_input['Resolution cutoff criterion'] = (f'[strike]{cutoff.value}[/strike] [i](ignored because a '
                                                              f'resolution range was provided)[/i]')
            cutoff = apu.ResolutionCriterion.none
        else:
            sanitized_input['Resolution cutoff criterion'] = cutoff.value

    if beamline:
        beamline = beamline.value
        sanitized_input['Beamline'] = beamline

    if exclude_ice_rings:
        sanitized_input['Ice rings'] = 'excluded'

    if no_residues:
        if no_residues <= 0:
            no_residues = None
        else:
            sanitized_input['Number of residues'] = no_residues

    sanitized_input['Anomalous signal'] = 'kept' if anomalous else 'merged'

    extra = apu.parse_extra_params(extra_args)
    if extra:
        sanitized_input['Extra autoPROC arguments'] = '\n'.join([f'{k}={v}' for k, v in extra.items()])

    sanitized_input['Number of concurrent jobs'] = no_concurrent_jobs
    sanitized_input['Number of threads per job'] = math.floor(n_threads / no_concurrent_jobs)
    if xds_njobs:
        sanitized_input['Number of XDS jobs'] = xds_njobs
    if xds_nproc:
        sanitized_input['Number of XDS processors'] = xds_nproc

    cs = compute_site.get_site()
    sanitized_input['Computation site'] = f'{compute_site.value}' + (f' \[{cs.priority_system.system_type}]'
                                                                     if cs.priority_system.system_type else '')
    if modules:
        sanitized_input['Modules'] = '\n'.join(modules)

    if chmod != bool(cfg['automate']['change_permissions'].value):
        sanitized_input['Change permissions'] = 'enabled' if chmod else 'disabled'

    if chmod_files != int(cfg['automate']['permissions_files'].value):
        sanitized_input['Permissions for files'] = chmod_files

    if chmod_dirs != int(cfg['automate']['permissions_directories'].value):
        sanitized_input['Permissions for directories'] = chmod_dirs

    if do_only:
        sanitized_input['Total number of jobs'] = f'{do_only} [i](limited by --only)[/]'

    if verbose:
        cli.print('The following global parameters will be used unless overriden on the .csv file:')
        cli.print_table(table=[[key, str(value)] for key, value in sanitized_input.items()],
                        headers=['Parameter', 'Value'],
                        column_kwargs=[{'style': 'deep_pink1'}, {'style': 'orange3'}],
                        table_kwargs={'title': 'Global parameters', 'expand': True, 'box': rich.box.HORIZONTALS})
        cli.confirm('Would you like to proceed with the above parameters?', default=False)

    # Housekeeping
    csv_file = None
    csv_dict = {}

    # Input for AutoPROCWorkFlowJob
    #  nml_file, raw_data_dir, processed_data_dir, output_dir, output_subdir
    nml_input = []

    # Check if a datasets.csv file was provided
    if len(input_files) == 1 and input_files[0].suffix == '.csv':
        if not input_files[0].exists():
            cli.print(f'File {input_files[0]} does not exist', style='red')
        csv_file = input_files[0]
        cli.print(f'📃 Parsing datasets from {csv_file}')

        cli.print('\n### CSV FILE CONTENTS \n' + csv_file.read_text() + '### END CSV FILE CONTENTS\n',
                  log_only=True)

        csv_dict = apu.parse_csv(csv_file, extra_headers=['nml_file'], echo=cli.print)
        cli.print(f'📑 Found {len(csv_dict["headers"])} headers in the CSV file: ')
        cli.print('\n'.join(f' - {h} ' + escape(f'[{csv_dict["index"][h]}]') for h in csv_dict['headers']))

        # Check if CSV contains a 'nml_file' column
        if 'nml_file' not in csv_dict['headers']:
            cli.print('Column \'nml_file\' not found in the CSV file', style='red')
            raise typer.Abort()

        # Check if the
        nml_input = apu.sanitize_nml_datasets(csv_dict=csv_dict, raw_dir=raw_dir, out_dir=out_dir,
                                              out_subdir=out_subdir, echo=cli.print)

    else:
        for i, nml_file in enumerate(input_files):
            if nml_file.suffix != '.nml':
                continue
            nml_input.append([nml_file, raw_dir, out_dir, None, out_subdir])
    cli.print(f'🗃️ Found {len(nml_input)} NML files from input')

    # Report the dataset attributes parsed from the CSV file
    if log_file or verbose:
        log_only = (verbose == 0)
        renderable_datasets = [list(map(apu.str_or_none, dataset_params)) for dataset_params in nml_input]
        cli.print('The following parameters will be used for locating the images:', log_only=log_only)
        cli.print_table(table=renderable_datasets,
                        headers=['nml_file', 'raw_data_dir', 'processed_data_dir', 'output_dir', 'output_subdir'],
                        column_kwargs=[{'overflow': 'fold', 'style': 'orange3'},
                                       {'overflow': 'fold', 'style': 'deep_pink1'},
                                       {'overflow': 'fold', 'style': 'dodger_blue1'},
                                       {'overflow': 'fold', 'style': 'steel_blue1'},
                                       {'overflow': 'fold', 'style': 'cornflower_blue'}],
                        table_kwargs={'title': 'Sanitized job input', 'expand': True,
                                      'box': rich.box.HORIZONTALS},
                        log_only=log_only)
        cli.print(log_only=log_only)

        # Prepare the jobs
        jobs = []
        no_images = 0
        sanitized_configs = {}
        APJ = AutoPROCWorkflowJob.update_concurrency_limit(no_concurrent_jobs)
        APJ._echo_success_kwargs = {'style': 'green'}
        APJ._echo_warning_kwargs = {'style': 'yellow'}
        APJ._echo_error_kwargs = {'style': 'red'}
        t0 = datetime.now()
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                      transient=True, console=cli) as progress:
            APJ._echo = partial(progress.console.print, highlight=False, markup=False, overflow='fold', log_escape=True)
            task = progress.add_task('🛠️ Preparing jobs...', total=len(nml_input))
            with Catcher(silent=not debug) as catcher:  # debug will print the exceptions
                progress.console.print('\n### JOB OPTIONS', log_only=True)
                for i, (nml_file, raw_data_dir, processed_data_dir, output_dir, output_subdir) in enumerate(nml_input):
                    if i >= do_only > 0:
                        progress.console.print(f'Skipping the rest of the datasets (--only={do_only})',
                                               style='magenta')
                        break
                    config_input = apu.merge_configs(csv_dict=csv_dict, dataset_index=i, **{
                        'change_permissions': chmod, 'file_permissions': chmod_files,
                        'directory_permissions': chmod_dirs,
                        'unit_cell': uc, 'space_group': space_group, 'resolution_high': res_high,
                        'resolution_low': res_low,
                        'anomalous': anomalous, 'no_residues': no_residues, 'rfree_mtz': mtz_rfree,
                        'reference_mtz': mtz_ref, 'xds_njobs': xds_njobs, 'xds_nproc': xds_nproc,
                        'exclude_ice_rings': exclude_ice_rings, 'beamline': beamline,
                        'resolution_cutoff_criterion': cutoff.value, 'extra_params': extra
                    })
                    sanitized_config = {
                        'input': {
                            'nml_file': nml_file,
                            'raw_data_dir': raw_data_dir,
                            'processed_data_dir': processed_data_dir,
                            'output_dir': output_dir,
                            'output_subdir': output_subdir,
                            'config': config_input
                        }
                    }
                    sanitized_configs[i] = sanitized_config
                    try:
                        config = AutoPROCConfig(batch_mode=True, **config_input)
                        sanitized_configs[i]['config'] = config
                        job = APJ(nml_file=nml_file, config=config, compute_site=cs, modules=modules,
                                  raw_data_dir=raw_data_dir, processed_data_dir=processed_data_dir)

                        # Patch datasets
                        for dataset in job.datasets:
                            if output_dir:
                                setattr(dataset, 'output_dir', output_dir)
                            if output_subdir:
                                setattr(dataset, 'output_subdir', output_subdir)
                                dataset._fstring_dict['processed_data_dir'] += f'/{output_subdir}'
                                dataset._check_dir_fstring('processed_data_dir')
                            no_images += dataset.no_images
                            dataset.reset_images_cache()

                        # Update log files
                        if output_subdir:
                            job._stdout = job.job_dir / job._stdout.name
                            job._stderr = job.job_dir / job._stderr.name

                        sanitized_configs[i]['job'] = job.__dict__
                    except Exception as e:
                        catcher.log_exception({'index': i + 1, 'data': sanitized_config, 'exception': e})
                        continue
                    finally:
                        if debug or log_file:
                            log_only = not debug
                            progress.console.print(f'Job options for dataset {i + 1}:', log_only=log_only)
                            progress.console.pprint(sanitized_config, log_only=log_only)
                            progress.console.print('', log_only=log_only)

                    jobs.append(job)
                    progress.advance(task)
                progress.console.print('### END JOB OPTIONS', log_only=True)
        no_jobs = len(jobs)
        t1 = datetime.now()
        cli.print(f'📷 Found {no_images:,} images from {len(nml_input)} NML files in {t1 - t0}')
        cli.print(f'🛠️ Prepared {no_jobs} job' + ('s' if no_jobs > 1 else ''))

        # Exit if there were any errors while creating the jobs
        if catcher.errors:
            cli.print(f'The following {len(catcher.errors)} job(s) could not be created:', style='red')
            for error in catcher.errors:
                cli.print(f':police_car_light: Job {error["index"]} was instantiated with the following data:',
                          style='red bold')
                cli.print(error['data'], style='red dim')
                cli.print(f'\n    The following exception was raised:', style='red')
                cli.print_traceback(exc=error['exception'], indent='    ')
                cli.print('')
            cli.print('All data passed to the jobs is saved in [u]jobs_input.txt[/]', style='magenta')
            with open('jobs_input.txt', 'w') as f:
                f.write(pformat(sanitized_configs))
            if chmod:
                Path('jobs_input.txt').chmod(mode=get_permissions_in_decimal(chmod_files))
            raise typer.Abort()

        message = f'🚀 Would you like to launch {no_jobs} job'
        if no_jobs > 1:
            message += 's'
        if no_jobs > no_concurrent_jobs > 1:
            message += f' in batches of {no_concurrent_jobs}'
        message += '?'
        cli.print(message, log_only=True)
        cli.confirm(message, default=False)

        # Run the jobs
        t0 = datetime.now()
        cli.print(f'\nLaunching jobs at {t0}...')
        output_csv = []
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                      TextColumn('{task.fields[status]}'),
                      transient=True, console=cli) as progress:
            jobs_succeeded = 0
            jobs_tidyup_failed = 0
            jobs_failed = 0
            running = progress.add_task(':person_running: Running jobs... ', total=no_jobs,
                                        status='Status: Running...')

            # Attach progress bar's print to the jobs
            logger = partial(progress.console.print, highlight=False, overflow='fold', markup=False, log_escape=True)
            for job in jobs:
                job._echo = logger

            # Generate tasks list
            pending_tasks = [asyncio.create_task(job.run(execute_batch=not dry_run)) for job in jobs]
            with Catcher(silent=not debug) as catcher:
                while pending_tasks:
                    try:
                        completed_tasks, pending_tasks = await asyncio.wait(pending_tasks,
                                                                            return_when=asyncio.FIRST_COMPLETED)
                        for completed_task in completed_tasks:
                            job = completed_task.result()
                            directories_created.append(job.job_dir)

                            for d in job.datasets:
                                c = job.config
                                o_sdir = d.output_subdir if hasattr(d, 'output_subdir') else None
                                output_csv.append([job.job_dir.resolve(), job.run_no, job._success, d.sweep_id,
                                                   d.autoproc_id, d.dataset_name, d.dataset_dir, d.first_image,
                                                   d.raw_data_dir, d.processed_data_dir, d.output_dir, o_sdir,
                                                   c.mtz_project_name, c.mtz_crystal_name, c.mtz_dataset_name])

                            progress.advance(running)
                            if job._success:
                                jobs_succeeded += 1
                            else:
                                if job._results is None:
                                    jobs_tidyup_failed += 1
                                else:
                                    if not (job.job_dir / job._results._json_fname).exists():
                                        jobs_tidyup_failed += 1
                                    else:
                                        jobs_failed += 1
                            progress.update(running, status=f'Status: :star-struck: [green]{jobs_succeeded}[/] '
                                                            f':thinking_face: [yellow]{jobs_tidyup_failed}[/] '
                                                            f':loudly_crying_face: [red]{jobs_failed}[/]')
                    except Exception as e:
                        catcher.log_exception({'index': i + 1, 'job': job, 'exception': e})
                        continue
            csv_out = (out_dir / f'datasets_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv').resolve()
            if chmod:
                csv_out.touch(mode=get_permissions_in_decimal(chmod_files))
                csv_out.chmod(mode=get_permissions_in_decimal(chmod_files))
            with open(csv_out, 'w') as f:
                f.write('# ' + ','.join(['job_dir', 'run_no', 'success', 'sweep_id', 'autoproc_id',
                                         'dataset_name', 'dataset_dir', 'first_image', 'raw_data_dir',
                                         'processed_data_dir', 'output_dir', 'output_subdir', 'mtz_project_name',
                                         'mtz_crystal_name', 'mtz_dataset_name']) + '\n')
                for line in output_csv:
                    values = [str(v) if v else '' for v in line]
                    f.write(','.join(values) + '\n')
                f.write(f'# Written by xtl.autoproc.process_wf at {datetime.now()}')
                cli.print(f'Wrote new .csv file: {csv_out}')
        cli.print('')
        t1 = datetime.now()

        file_size = 0
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                      transient=True, console=cli) as progress:
            task = progress.add_task(':bookmark_tabs: Calculating disk space used...',
                                     total=len(directories_created))
            for created in directories_created:
                file_size += apu.get_directory_size(created)
                progress.advance(task)
            sleep(0.5)

        jobs_all = len(jobs) + 1
        if jobs_succeeded == jobs_all:
            cli.print(f'😎 All jobs completed at {t1}', style='green')
        else:
            cli.print(f'🫡 All jobs completed at {t1}', style='green')
            cli.print(f'Outlook: '
                      f'[green]:star-struck: {jobs_succeeded} succeeded[/], '
                      f'[yellow]:thinking_face: {jobs_tidyup_failed} tidy-up failed[/], '
                      f'[red]:loudly_crying_face: {jobs_failed} failed[/]', )
        cli.print(f':hourglass_done: Total elapsed time: {t1 - t0} (approx. {(t1 - t0) / len(jobs)} per job)')

        file_size_human_friendly = si_units(file_size, suffix='B', base=1024, digits=2)
        cli.print(f':bookmark_tabs: Total disk space used: {file_size_human_friendly}')

        # Write new csv file for downstream processing
        with open('jobs_output.txt', 'w') as f:
            f.write('\n'.join([str(created) for created in directories_created]))


@app.command('fixnml', short_help='Update paths in NML files', epilog=app.info.epilog)
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
        chmod: bool = typer.Option(cfg['automate']['change_permissions'].value, '--chmod',
                                   help='Change permissions of the output directories', rich_help_panel='Localization'),
        chmod_files: int = typer.Option(cfg['automate']['permissions_files'].value, '--chmod-files',
                                        help='Permissions for files', rich_help_panel='Localization'),
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


@app.command('json2csv', short_help='Create summary CSV from many JSON files', epilog=app.info.epilog)
def cli_autoproc_json_to_csv(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory'),
        debug: bool = typer.Option(False, '--debug', help='Print debug information')
):
    '''
    Reads a datasets_output.csv file and collects all the xtl_autoPROC.json files from the job directories to create a
    single summary CSV file.

    Note that the CSV file must contain the following columns: 'job_dir', 'sweep_id', 'autoproc_id'.
    '''
    cli = Console()
    # Check if csv file exists
    if not datasets_file.exists():
        cli.print(f'File {datasets_file} does not exist', style='red')
        raise typer.Abort()

    cli.print(f'Parsing dataset names from {datasets_file}... ')
    datasets = apu.parse_csv(datasets_file, extra_headers=['job_dir', 'sweep_id', 'autoproc_id'])
    cli.print(f'Found {len(datasets["extra"]["job_dir"])} datasets')

    data = []
    if debug:
        cli.print('# dataset_subdir, rename_dataset_subdir, autoproc_dir', style='magenta')
    for i, (j_dir, sweep_id, autoproc_id) in enumerate(zip(datasets["extra"]['job_dir'], datasets["extra"]['sweep_id'],
                                                           datasets["extra"]['autoproc_id'])):
        if j_dir:
            j = Path(j_dir) / 'xtl_autoPROC.json'
            if j.exists():
                d = {
                    'id': i,
                    'sweep_id': sweep_id,
                    'job_dir': j.parent.as_uri(),
                    'autoproc_id': autoproc_id
                }
                d.update(json.loads(j.read_text()))
                data.append(d)

    cli.print(f'Found {len(data)} JSON files')
    if not data:
        return typer.Exit(code=0)

    df = pd.json_normalize(data)
    df = apu.df_stringify(df)
    csv_file = out_dir / f'xtl_autoPROC_summary.csv'
    df.to_csv(csv_file, index=False)
    cli.print(f'Wrote summary to {csv_file}')


# @app.command('check_wavelength', help='Check wavelength with aP_fit_wvl_to_spots')
# def cli_autoproc_check_wavelength():
#     pass


if __name__ == '__main__':
    app()
