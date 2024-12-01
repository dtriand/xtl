import asyncio
import json
import os
from datetime import datetime
from enum import Enum
from functools import partial, wraps
import math
from pathlib import Path
from pprint import pformat
from time import sleep
import traceback

import pandas as pd
import rich.box
from rich.markup import escape
import rich.table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn
import typer
import tabulate

from xtl.automate.sites import LocalSite, BiotixHPC
from xtl.cli.cliio import CliIO, Console
from xtl.cli.automate.autoproc_utils import get_attributes_config, get_attributes_dataset, parse_csv2, \
    sanitize_csv_datasets, str_or_none, merge_configs, get_directory_size
from xtl.config import cfg
from xtl.diffraction.automate.autoproc import AutoPROCJobConfig, AutoPROCJob, AutoPROCJob2
from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.exceptions.utils import Catcher
from xtl.math import si_units


app = typer.Typer(name='xtl.autoproc', help='Execute multiple autoPROC jobs', rich_markup_mode='rich',
                  epilog='</> with ❤️ by [i magenta]_dtriand[/]')

def typer_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def parse_csv(csv_file: Path, extra_headers: list[str] = None):
    datasets_dict = {
        'dataset_subdir': [],
        'dataset_name': [],
        'first_image': [],
        'rename_dataset_subdir': [],
        'mtz_project_name': [],
        'mtz_crystal_name': [],
        'mtz_dataset_name': []
    }
    if extra_headers:
        for header in extra_headers:
            datasets_dict[header] = []
    indices = {key: None for key in datasets_dict.keys()}

    headers = csv_file.read_text().splitlines()[0].replace('#', '').replace(' ', '').split(',')
    for header in headers:
        if header in datasets_dict:
            indices[header] = headers.index(header)

    for line in csv_file.read_text().splitlines()[1:]:
        if line.startswith('#'):
            continue
        values = line.split(',')

        for key in datasets_dict.keys():
            if indices[key] is not None:
                v = values[indices[key]]
                if not v:
                    v = None
                datasets_dict[key].append(v)
            else:
                datasets_dict[key].append(None)

    used_headers = [key for key, value in indices.items() if value is not None]
    datasets_dict['headers'] = used_headers

    return datasets_dict

@app.command('check_files', help='Check a datasets.csv file')
def cli_autoproc_check_files(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names'),
        raw_dir: Path = typer.Option(None, '-i', '--raw-dir', help='Path to the raw data directory')
):
    cli = CliIO()
    if not datasets_file.exists():
        cli.echo(f'File {datasets_file} does not exist', level='error')
        raise typer.Abort()

    cli.echo(f'Parsing dataset names from {datasets_file}... ', nl=False)
    datasets = parse_csv(datasets_file)
    headers = datasets.pop('headers')
    cli.echo('Done.\n')

    cli.echo('The following metadata were parsed:')
    table = tabulate.tabulate(datasets, headers=headers, tablefmt='simple')
    cli.echo(table)

    if raw_dir is None:
        return typer.Exit(code=0)

    cli.echo(f'\nChecking raw data directory: {raw_dir}')
    missing_datasets = []
    for subdir, dataset, first_image, subdir_new, pname, xname, dname in zip(
            datasets['dataset_subdir'], datasets['dataset_name'], datasets['first_image'],
            datasets['rename_dataset_subdir'], datasets['mtz_project_name'], datasets['mtz_crystal_name'],
            datasets['mtz_dataset_name']
    ):
        image = raw_dir
        img_strs = [[f'{raw_dir}', typer.colors.GREEN]]
        if subdir:
            image /= subdir
            img_strs.append([f'{subdir}', typer.colors.CYAN])
        if first_image:
            image /= first_image
            img_strs.append([f'{first_image}', typer.colors.MAGENTA])
        if image.exists():
            cli.echo(f'✅ Found image: ', style={'fg': typer.colors.GREEN}, nl=False)
            no_segments = len(img_strs)
            for i, (segment, color) in enumerate(img_strs):
                cli.echo(segment, style={'fg': color}, nl=False)
                if i < no_segments - 1:
                    cli.echo('/', nl=False)
            cli.echo('')  # new line
            if subdir_new:
                cli.echo(f'   ✏️ Will be processed under: {subdir_new}')
            if pname or xname or dname:
                cli.echo(f'   ℹ️ MTZ: {pname if pname else ""} > {xname if xname else ""} > {dname if dname else ""}')
        else:
            cli.echo(f'❌ Did not find image: {image}', level='error')
            missing_datasets.append(image)
            parent = image.parent
            while True:
                if parent.exists():
                    cli.echo(f'   📁 Path valid up to: {parent}', level='warning')
                    break
                parent = parent.parent
                if parent == Path('/'):
                    break

    cli.echo(f'\nTotal number of datasets: {len(datasets["dataset_subdir"])}')
    cli.echo(f'Number of missing datasets: {len(missing_datasets)}')

    if missing_datasets:
        cli.echo('\nMissing datasets:', level='error')
        for missing in missing_datasets:
            cli.echo(f'    {missing}', level='error')


class ResolutionCriterion(Enum):
    none = 'None'
    cc_half = 'CC1/2'


class ComputeSite(Enum):
    local = 'local'
    biotix_hpc = 'biotix'

    def get_site(self):
        if self == ComputeSite.local:
            return LocalSite()
        elif self == ComputeSite.biotix_hpc:
            return BiotixHPC()
        else:
            raise ValueError(f'Unknown compute_site: {self}')


class Beamline(Enum):
    alba_bl13_xaloc = 'AlbaBL13Xaloc'
    als_1231 = 'Als1231'
    als_422 = 'Als422'
    als_831 = 'Als831'
    australian_sync_mx1 = 'AustralianSyncMX1'
    australian_sync_mx2 = 'AustralianSyncMX2'
    diamond_i04_mk = 'DiamondI04-MK'
    diamond_io4 = 'DiamondIO4'
    diamond_i23_day1 = 'DiamondI23-Day1'
    diamond_i23 = 'DiamondI23'
    esrf_id23_2 = 'EsrfId23-2'
    esrf_id29 = 'EsrfId29'
    esrf_id30_b = 'EsrfId30-B'
    ill_d19 = 'ILL_D19'
    petra_iii_p13 = 'PetraIIIP13'
    petra_iii_p14 = 'PetraIIIP14'
    sls_pxiii = 'SlsPXIII'
    soleil_proxima1 = 'SoleilProxima1'


def parse_resolution_range(resolution: str):
    resolution = resolution.replace(' ', '') if resolution else None
    if not resolution:
        return None, None
    elif '-' not in resolution:
        return None, float(resolution)
    elif resolution.startswith('-'):
        return None, float(resolution[1:])
    elif resolution.endswith('-'):
        return float(resolution[:-1]), None
    else:
        res_low, res_high = resolution.split('-')
        res_low, res_high = float(res_low), float(res_high)
        if res_high > res_low:
            return res_high, res_low
        return res_low, res_high


def parse_unit_cell(unit_cell: str):
    if not unit_cell:
        return []
    if ',' in unit_cell:
        uc = unit_cell.replace(' ', '').split(',')
        if len(uc) != 6:
            raise ValueError('Unit-cell parameters must be 6 comma-separated values')
        return [float(x) for x in uc]
    elif ' ' in unit_cell:
        uc = unit_cell.split()
        if len(uc) != 6:
            raise ValueError('Unit-cell parameters must be 6 space-separated values')
        return [float(x) for x in uc]
    else:
        raise ValueError('Unit-cell parameters must be 6 comma-separated or space-separated values')


def parse_extra_args(extra_args: list[str]):
    if not extra_args:
        return {}
    extra = {}
    for arg in extra_args:
        if '=' in arg:
            key, value = arg.split('=')
            extra[key] = value
    return extra



def create_config(unit_cell: list[float], space_group: str, resolution_high: float, resolution_low: float,
                  anomalous: bool, nresidues: int, free_mtz_file: Path, xds_njobs: int, xds_nproc: int,
                  exclude_ice_rings: bool, beamline: Beamline, cutoff: ResolutionCriterion, extra_args: dict = None):
    beamline = beamline.value if beamline else None
    cutoff = cutoff.value if cutoff != ResolutionCriterion.none else None
    config = partial(AutoPROCJobConfig, unit_cell=unit_cell, space_group=space_group, resolution_high=resolution_high,
                     resolution_low=resolution_low, anomalous=anomalous, nresidues=nresidues,
                     free_mtz_file=free_mtz_file, xds_njobs=xds_njobs, xds_nproc=xds_nproc,
                     exclude_ice_rings=exclude_ice_rings, beamline=beamline, resolution_cutoff_criterion=cutoff,
                     extra_kwargs=extra_args)
    return config


@app.command('run_many', help='Run multiple autoPROC jobs')
@typer_async
async def cli_autoproc_run_many(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names'),
        raw_dir: Path = typer.Option(None, '-i', '--raw-dir', help='Path to the raw data directory'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory'),
        no_concurrent_jobs: int = typer.Option(1, '-n', '--no-jobs', help='Number of datasets to process in parallel'),
        modules: list[str] = typer.Option(None, '-m', '--module', help='Module to load before running the jobs'),
        compute_site: ComputeSite = typer.Option(cfg['automate']['compute_site'].value, '--compute-site', help='Computation site for configuring the job execution'),
        xds_njobs: int = typer.Option(None, '-j', '--xds-jobs', help='Number of XDS jobs'),
        xds_nproc: int = typer.Option(None, '-p', '--xds-proc', help='Number of XDS processors'),
        beamline: Beamline = typer.Option(None, '-b', '--beamline', show_choices=False, help='Beamline name'),
        resolution: str = typer.Option(None, '-r', '--resolution', help='Resolution range'),
        cutoff: ResolutionCriterion = typer.Option(ResolutionCriterion.cc_half.value, '-c', '--cutoff', help='Resolution cutoff criterion'),
        unit_cell: str = typer.Option(None, '-u', '--unit-cell', help='Unit-cell parameters'),
        space_group: str = typer.Option(None, '-s', '--space-group', help='Space group'),
        no_residues: int = typer.Option(None, '-N', '--no-residues', help='Number of residues in the asymmetric unit'),
        anomalous: bool = typer.Option(True, '--no-anomalous', is_flag=True, flag_value=False, show_default=False, help='Merge anomalous signal'),
        exclude_ice_rings: bool = typer.Option(None, '-e', '--exclude-ice', is_flag=True, flag_value=True, help='Exclude ice rings'),
        mtz_rfree: Path = typer.Option(None, '-f', '--mtz-rfree', help='Path to a MTZ file with R-free flags'),
        dry_run: bool = typer.Option(False, '--dry', help='Dry run without running autoPROC'),
        do_only: int = typer.Option(None, '--only', hidden=True, help='Do only X jobs'),
        extra_args: list[str] = typer.Option(None, '-x', '--extra', help='Extra arguments to pass to autoPROC'),
):
    cli = CliIO()

    # Check if dry_run
    if dry_run:
        cli.echo('DRY_RUN enabled', style={'fg': typer.colors.MAGENTA})

    # Get compute_site
    cs = compute_site.get_site()
    cli.echo(f'Setting compute site to: {compute_site.value} ', nl=False)
    if not cs.priority_system._system_type:
        cli.echo('')
    elif cs.priority_system._system_type == 'nice':
        cli.echo(f'[nice={cs.priority_system.nice_level}]')
    else:
        cli.echo(f'[{cs.priority_system._system_type}]')

    # Set number of concurrent jobs
    if no_concurrent_jobs > 1:
        cli.echo(f'Setting no. of concurrent jobs to: {no_concurrent_jobs}')

    # Set XDS keywords
    if xds_njobs:
        cli.echo(f'Setting XDS keyword MAXIMUM_NUMBER_OF_JOBS to: {xds_njobs}')
    if xds_nproc:
        cli.echo(f'Setting XDS keyword MAXIMUM_NUMBER_OF_PROCESSORS to: {xds_nproc}')

    # Set beamline
    if beamline:
        cli.echo(f'Setting beamline to: {beamline.value}')

    # Set resolution range
    res_low, res_high = parse_resolution_range(resolution)
    if res_low and res_high:
        cli.echo(f'Setting resolution range to: {res_low} - {res_high} Å')
    elif res_low:
        cli.echo(f'Setting low resolution cutoff to: {res_low} Å')
    elif res_high:
        cli.echo(f'Setting high resolution cutoff to: {res_high} Å')

    # Set resolution cutoff criterion
    if cutoff != ResolutionCriterion.none:
        if res_high:
            cli.echo(f'Ignoring resolution cutoff criterion {cutoff.value}, because high resolution cutoff was '
                     f'explicitly specified', level='warning')
            cutoff = ResolutionCriterion.none
        else:
            cli.echo(f'Setting high resolution cutoff criterion to: {cutoff.value}')

    # Set unit-cell
    if unit_cell:
        uc = parse_unit_cell(unit_cell)
        uc = " ".join(map(str, uc))
        cli.echo(f'Setting unit-cell to: {uc}')
    else:
        uc = None

    # Set space group
    if space_group:
        space_group = space_group.replace(' ', '')
        cli.echo(f'Setting space group to: {space_group}')

    # Set number of residues
    if no_residues:
        if no_residues < 0:
            cli.echo('Number of residues in the asymmetric unit cannot be negative', level='error')
            raise typer.Abort()
        elif no_residues == 0:
            no_residues = None
        else:
            cli.echo(f'Setting number of residues in the asymmetric unit to: {no_residues}')

    # Set anomalous signal
    cli.echo(f'Anomalous signal will {"" if anomalous else "not "}be kept')

    # Set ice rings exclusion
    if exclude_ice_rings:
        cli.echo('Ice rings will be excluded')

    # R-free MTZ file
    if mtz_rfree:
        if not mtz_rfree.exists():
            cli.echo(f'MTZ file with R-free flags does not exist: {mtz_rfree}', level='error')
            raise typer.Abort()
        cli.echo(f'Using R-free flags from: {mtz_rfree}')

    # Extra arguments
    extra = parse_extra_args(extra_args)
    if extra:
        cli.echo('Extra arguments:')
        for key, value in extra.items():
            cli.echo(f'    {key}={value}')

    # Modules to be loaded
    if modules:
        cli.echo(f'Modules to load: {", ".join(modules)}')

    # Check if raw data directory exists
    if raw_dir is None:
        cli.echo('Please provide the path to the raw data directory', level='error')
        raise typer.Abort()
    if not raw_dir.exists():
        cli.echo(f'Raw data directory {raw_dir} does not exist', level='error')
        raise typer.Abort()
    raw_dir = raw_dir.resolve()
    cli.echo(f'Reading raw data from: {raw_dir}')

    # Check output directory
    out_dir = out_dir.resolve()
    if out_dir.exists():
        cli.echo(f'Output directory: {out_dir}')
    else:
        cli.echo(f'Creating output directory: {out_dir} ', nl=False)
        try:
            out_dir.mkdir(parents=True)
        except OSError:
            cli.echo(f'\nFailed to create output directory: {out_dir}', level='error')
            raise typer.Abort()
        if out_dir.exists():
            cli.echo('Done.')

    # Check if csv file exists
    if not datasets_file.exists():
        cli.echo(f'File {datasets_file} does not exist', level='error')
        raise typer.Abort()

    cli.echo(f'Parsing dataset names from {datasets_file}... ', nl=False)
    datasets = parse_csv(datasets_file)
    cli.echo('Done.')
    cli.echo(f'Found {len(datasets["dataset_subdir"])} datasets to process')

    # Create partial AutoPROCJobConfig
    config = create_config(unit_cell=uc, space_group=space_group, resolution_high=res_high, resolution_low=res_low,
                           anomalous=anomalous, nresidues=no_residues, free_mtz_file=mtz_rfree, xds_njobs=xds_njobs,
                           xds_nproc=xds_nproc, exclude_ice_rings=exclude_ice_rings, beamline=beamline,
                           cutoff=cutoff, extra_args=extra)

    # Create a new datasets.csv file for the processed datasets
    csv_new = out_dir / f'datasets_new_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    csv_dict = {key: [] for key in datasets.keys()}
    csv_dict['job_dir'] = []
    csv_dict['autoproc_id'] = []
    csv_dict.pop('headers')

    # Create AutoPROCJob instances
    cli.echo('Preparing jobs... ', nl=False)
    apj = AutoPROCJob.update_concurrency_limit(no_concurrent_jobs)
    jobs = []
    for i, (subdir, dataset, first_image, subdir_rename, pname, xname, dname) in enumerate(zip(
            datasets['dataset_subdir'], datasets['dataset_name'], datasets['first_image'],
            datasets['rename_dataset_subdir'], datasets['mtz_project_name'], datasets['mtz_crystal_name'],
            datasets['mtz_dataset_name']
    )):
        if not subdir or not dataset or not first_image:
            cli.echo('Skipping dataset with missing information', level='warning')
            continue
        job_config = config(
            raw_data_dir=raw_dir,
            processed_data_dir=out_dir,
            dataset_subdir=subdir,
            dataset_subdir_rename=subdir_rename,
            dataset_name=dataset,
            first_image=first_image,
            mtz_project_name=pname,
            mtz_crystal_name=xname,
            mtz_dataset_name=dname
        )
        job = apj(job_config, compute_site=cs, module=' '.join(modules))
        jobs.append(job)
    cli.echo('Done.')

    if do_only:
        cli.echo(f'Running only {do_only} jobs', style={'fg': typer.colors.MAGENTA})
        jobs = jobs[:do_only]

    # Ask for user confirmation
    cli.echo(f'\nReady to run {len(jobs)} jobs '
             f'{"in batches of " + str(no_concurrent_jobs) if no_concurrent_jobs > 1 else ""}')
    if not typer.confirm('Do you want to proceed?'):
        cli.echo('Aborted by user', level='warning')
        raise typer.Abort()

    # Run the jobs
    t0 = datetime.now()
    cli.echo(f'\nLaunching jobs at {t0}...')
    do_run = not dry_run
    tasks = [asyncio.create_task(job.run(do_run=do_run)) for job in jobs]
    # TODO: Add a KeyboardInterrupt handler to cancel the jobs and write partial results to the csv file
    #  https://www.roguelynn.com/words/asyncio-graceful-shutdowns/
    # TODO: Add a pause handler that will not submit new jobs until the user confirms to continue
    #  https://stackoverflow.com/questions/61148269/how-to-pause-an-asyncio-created-task-using-a-lock-or-some-other-method
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        cli.echo('Uh oh... An error occurred while running the jobs', level='error')
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        # Update the csv_dict with the completed jobs
        for job, task in zip(jobs, tasks):
            if task.done():
                csv_dict['dataset_subdir'].append(job.config.dataset_subdir)
                csv_dict['dataset_name'].append(job.config.dataset_name)
                csv_dict['first_image'].append(job.config.first_image)
                csv_dict['rename_dataset_subdir'].append(job.config.dataset_subdir_rename)
                csv_dict['mtz_project_name'].append(job.config.mtz_project_name)
                csv_dict['mtz_crystal_name'].append(job.config.mtz_crystal_name)
                csv_dict['mtz_dataset_name'].append(job.config.mtz_dataset_name)
                csv_dict['job_dir'].append(job.config._job_subdir)
                csv_dict['autoproc_id'].append(job.config._idn)

    t1 = datetime.now()
    cli.echo(f'All jobs completed at {t1}')
    cli.echo(f'Total elapsed time: {t1 - t0} [approx. {(t1 - t0) / len(jobs)} per job]')

    # Write new csv file for downstream processing
    cli.echo(f'Writing new .csv file: {csv_new}')
    with csv_new.open('w') as f:
        f.write('# ' + ','.join(csv_dict.keys()) + '\n')
        for i in range(len(list(csv_dict.values())[0])):
            values = []
            for key in csv_dict.keys():
                value = csv_dict[key][i]
                if value is None:
                    value = ''
                values.append(value)
            f.write(','.join(values) + '\n')
        f.write(f'# Written by xtl.autoproc.run_many at {datetime.now()}')

    cli.echo('xtl.autoproc finished graciously <3\n')


def df_stringify(df: pd.DataFrame):
    df2 = df.copy(deep=True)
    columns = df.columns
    for col in columns:
        if df[col].dtype == 'object':
            first_object = df[col].dropna().iloc[0]
            if isinstance(first_object, list | tuple):
                df2[col] = df[col].apply(lambda x: ' '.join(map(str, x)))
    return df2


@app.command('json2csv', help='Create summary CSV from many JSON files')
@typer_async
async def cli_autoproc_json_to_csv(
        datasets_file: Path = typer.Argument(metavar='<DATASETS.CSV>', help='Path to a CSV file containing dataset names'),
        out_dir: Path = typer.Option(Path('./'), '-o', '--out-dir', help='Path to the output directory'),
        debug: bool = typer.Option(False, '--debug', help='Print debug information')
):
    cli = CliIO()
    # Check if csv file exists
    if not datasets_file.exists():
        cli.echo(f'File {datasets_file} does not exist', level='error')
        raise typer.Abort()

    cli.echo(f'Parsing dataset names from {datasets_file}... ', nl=False)
    datasets = parse_csv(datasets_file, extra_headers=['autoproc_dir', 'autoproc_id', 'mtz_dataset_name'])
    cli.echo('Done.')
    cli.echo(f'Found {len(datasets["dataset_subdir"])} datasets')

    data = []
    if debug:
        cli.echo('# dataset_subdir, rename_dataset_subdir, autoproc_dir', style={'fg': typer.colors.BRIGHT_MAGENTA})
    for i, (dataset_dir, output_subdir, job_dir, autoproc_id, mtz) in enumerate(zip(datasets['dataset_subdir'],
                                                                                    datasets['rename_dataset_subdir'],
                                                                                    datasets['autoproc_dir'],
                                                                                    datasets['autoproc_id'],
                                                                                    datasets['mtz_dataset_name'])):
        if debug:
            cli.echo(f'{dataset_dir}, {output_subdir}, {job_dir}', style={'fg': typer.colors.BRIGHT_MAGENTA})
        if not output_subdir:
            output_subdir = dataset_dir
        if job_dir:
            j = out_dir / output_subdir / job_dir / 'xtl_autoPROC.json'
            if j.exists():
                d = {
                    'id': i,
                    'dataset_name': mtz,
                    'job_dir': j.parent.as_uri(),
                    'autoproc_id': autoproc_id
                }
                d.update(json.loads(j.read_text()))
                data.append(d)

    cli.echo(f'Found {len(data)} JSON files')
    if not data:
        return typer.Exit(code=0)

    df = pd.json_normalize(data)
    df = df_stringify(df)
    csv_file = Path('.') / f'xtl_autoPROC_summary.csv'
    df.to_csv(csv_file, index=False)
    cli.echo(f'Wrote summary to {csv_file}')


@app.command('options', help='Show available autoPROC configuration options', epilog=app.info.epilog)
def cli_autoproc_options():
    cli = Console()

    table_kwargs = {
        'title_style': 'bold italic white on cornflower_blue',
        'box': rich.box.HORIZONTALS,
        'expand': True
    }

    cli.print('The following parameters can be passed as arguments to [b i u]xtl.autoproc process[/b i u], '
              'or as headers in the [b i u]datasets.csv[/b i u] file.')
    cli.print()
    cli.print_table(table=get_attributes_dataset(),
                    headers=['XTL parameter', 'Type', 'Description'],
                    column_kwargs=[
                        {'style': 'cornflower_blue'},
                        {'style': 'italic'},
                        {'style': 'bright_black'}
                    ],
                    table_kwargs=table_kwargs | {'title': 'Dataset options',
                                                 'caption': 'An additional \'dataset_group\' parameter can be added to '
                                                            'the [u]datasets.csv[/u] file to process and merge multiple'
                                                            ' datasets together ([i]e.g.[/i] multi-sweep data)'}
                    )
    cli.print()
    cli.print_table(table=get_attributes_config(),
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
    cutoff: ResolutionCriterion = typer.Option(ResolutionCriterion.cc_half.value, '-c', '--cutoff',
                                               help='Resolution cutoff criterion',
                                               rich_help_panel='autoPROC parameters'),
    beamline: Beamline = typer.Option(None, '-b', '--beamline', show_choices=False,
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
                                  rich_help_panel='Parallelization'),
    xds_njobs: int = typer.Option(None, '-j', '--xds-jobs', help='Number of XDS jobs',
                                  rich_help_panel='Parallelization'),
    xds_nproc: int = typer.Option(None, '-p', '--xds-proc', help='Number of XDS processors',
                                  rich_help_panel='Parallelization'),
    # Localization
    modules: list[str] = typer.Option(None, '-m', '--module',
                                      help='Module to load before running the jobs', rich_help_panel='Localization'),
    compute_site: ComputeSite = typer.Option(cfg['automate']['compute_site'].value, '--compute-site',
                                             help='Computation site for configuring the job execution',
                                             rich_help_panel='Localization'),
    chmod: bool = typer.Option(cfg['automate']['change_permissions'].value, '--chmod',
                               help='Change permissions of the output directories', rich_help_panel='Localization'),
    chmod_files: int = typer.Option(cfg['automate']['permissions_files'].value, '--chmod-files',
                                    help='Permissions for files', rich_help_panel='Localization'),
    chmod_dirs: int = typer.Option(cfg['automate']['permissions_directories'].value, '--chmod-dirs',
                                   help='Permissions for directories', rich_help_panel='Localization'),
    # Debugging
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
    Examples:
        asdfasdf
        asdfasdf
    And a simple CSV file
    '''
    cli = Console(verbose=verbose, debug=debug)

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
        uc = parse_unit_cell(unit_cell)
        sanitized_input['Unit-cell parameters'] = ", ".join(map(str, uc))
    else:
        uc = None

    if space_group:
        sanitized_input['Space group'] = space_group.replace(' ', '')

    if mtz_rfree:
        sanitized_input['MTZ file with R-free flags'] = mtz_rfree

    if mtz_ref:
        sanitized_input['Reference MTZ file'] = mtz_ref

    res_low, res_high = parse_resolution_range(resolution)
    if res_low or res_high:
        if not res_low:
            res_low = 999.0
        if not res_high:
            res_high = 0.1
        sanitized_input['Resolution range'] = f'{res_low} - {res_high} Å'

    if cutoff != ResolutionCriterion.none:
        if res_high:
            sanitized_input['Resolution cutoff criterion'] = (f'[strike]{cutoff.value}[/strike] [i](ignored because a '
                                                              f'resolution range was provided)[/i]')
            cutoff = ResolutionCriterion.none
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

    extra = parse_extra_args(extra_args)
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
    #  raw_data_dir, dataset_dir, dataset_name, first_image, processed_data_dir, output_dir
    datasets_input = []

    # Check if a datasets.csv file was provided
    if len(input_files) == 1 and input_files[0].suffix == '.csv':
        if not input_files[0].exists():
            cli.print(f'File {input_files[0]} does not exist', style='red')
            raise typer.Abort()
        csv_file = input_files[0]
        cli.print(f'📃 Parsing datasets from {csv_file}')
        csv_dict = parse_csv2(csv_file)
        cli.print(f'📑 Found {len(csv_dict["headers"])} headers in the CSV file: ')
        cli.print('\n'.join(f' - {h} ' + escape(f'[{csv_dict["index"][h]}]') for h in csv_dict['headers']))

        # Check if dataset paths have been fully specified and collect the images
        datasets_input = sanitize_csv_datasets(csv_dict=csv_dict, raw_dir=raw_dir, out_dir=out_dir,
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
    if verbose:
        renderable_datasets = [list(map(str_or_none, dataset_params)) for dataset_params in datasets_input]
        cli.print('The following parameters will be used for locating the images:')
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
                                      'box': rich.box.HORIZONTALS})
        cli.print()

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
                                                                processed_data_dir=p_dir)
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
        renderable_params.append(list(map(str_or_none, params)))
    if verbose or missing_dirs:
        headers = ['raw_data_dir', 'dataset_dir', 'dataset_name', 'first_image', 'processed_data_dir', 'output_dir',
                   'image_template', 'file_extension', 'img_no_first', 'img_no_last', 'no_images']
        cli.print('The following datasets were initialized:\n')
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
                                      'box': rich.box.HORIZONTALS})
        cli.print()
        if missing_dirs:
            cli.print(f'Any [u red]red and underlined paths[/] in table above indicate missing directories that will '
                      f'be generated.\n[yellow]Please ensure that these paths are correct [b]before launching the '
                      f'jobs![/][/]')
        cli.confirm('Do the above output directories look correct?', default=False)

        # Save datasets attributes to a CSV file
        output = Path('./datasets_sanitized.csv')
        with output.open('w') as f:
            f.write('# ' + ','.join(headers) + '\n')
            for params in renderable_params:
                f.write(','.join(params) + '\n')
            f.write(f'# Written by xtl.autoproc.process at {datetime.now()}')

    # Prepare the jobs
    jobs = []
    sanitized_configs = {}
    APJ = AutoPROCJob2.update_concurrency_limit(no_concurrent_jobs)
    APJ._echo_success_kwargs = {'style': 'green'}
    APJ._echo_warning_kwargs = {'style': 'yellow'}
    APJ._echo_error_kwargs = {'style': 'red'}
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  transient=True, console=cli) as progress:
        APJ._echo = partial(progress.console.print, highlight=False, markup=False, overflow='fold')
        task = progress.add_task('🛠️ Preparing jobs...', total=len(datasets))
        with Catcher(silent=not debug) as catcher:  # debug will print the exceptions
            for i, dataset in enumerate(datasets):
                if i >= do_only > 0:
                    cli.print(f'Skipping the rest of the datasets (--only={do_only})', style='magenta')
                    break
                config_input = merge_configs(csv_dict=csv_dict, dataset_index=i, **{
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
                    config = AutoPROCConfig(**config_input)
                    sanitized_configs[i]['config'] = config
                    job = APJ(datasets=dataset, config=config, compute_site=cs, modules=modules)
                    sanitized_configs[i]['job'] = job.__dict__
                except Exception as e:
                    catcher.log_exception({'index': i + 1, 'data': sanitized_config, 'exception': e})
                    continue
                finally:
                    if debug:
                        progress.console.print(f'Job options for dataset {i+1}:')
                        progress.console.print(sanitized_config)
                        progress.console.print()

                jobs.append(job)
                progress.advance(task)
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
            cli.print()
        cli.print('All data passed to the jobs is saved in [u]jobs_input.txt[/]', style='magenta')
        with open('jobs_input.txt', 'w') as f:
            f.write(pformat(sanitized_configs))
        raise typer.Abort()

    message = f'🚀 Would you like to launch {no_jobs} job'
    if no_jobs > 1:
        message += 's'
    if no_jobs > no_concurrent_jobs > 1:
        message += f' in batches of {no_concurrent_jobs}'
    message += '?'
    cli.confirm(message, default=False)

    # Run the jobs
    t0 = datetime.now()
    cli.print(f'\nLaunching jobs at {t0}...')
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  TextColumn('Status {task.fields[status]}'),
                  transient=True, console=cli) as progress:
        jobs_succeeded = 0
        jobs_tidyup_failed = 0
        jobs_failed = 0
        running = progress.add_task(':person_running: Running jobs... ', total=no_jobs,
                                    status=': Running...')

        # Attach progress bar's print to the jobs
        logger = partial(progress.console.print, highlight=False, overflow='fold', markup=False)
        for job in jobs:
            job._echo = logger

        # Generate tasks list
        pending_tasks = [asyncio.create_task(job.run(execute_batch=not dry_run)) for job in jobs]
        # with Catcher(silent=True) as catcher:
        while pending_tasks:
            # try:
            completed_tasks, pending_tasks = await asyncio.wait(pending_tasks,
                                                                return_when=asyncio.FIRST_COMPLETED)
            # except Exception as e:
            #     catcher.log_exception({'index': i + 1, 'job': job, 'exception': e})
            for completed_task in completed_tasks:
                job = completed_task.result()
                directories_created.append(job.job_dir)
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
                progress.update(running, status=f':star-struck: [green]{jobs_succeeded}[/] '
                                                f':thinking_face: [yellow]{jobs_tidyup_failed}[/] '
                                                f':loudly_crying_face: [red]{jobs_failed}[/]')
    cli.print()
    t1 = datetime.now()

    file_size = 0
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), MofNCompleteColumn(),
                  transient=True, console=cli) as progress:
        task = progress.add_task(':bookmark_tabs: Calculating disk space used...',
                                 total=len(directories_created))
        for created in directories_created:
            file_size += get_directory_size(created)
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
    #  [ ] Write new csv file for downstream processing
    #  [ ] Add option for logging stdout to file (--log)
    #  [ ] Run GPhL workflow files
    #  [ ] Monitor resources in the progress bar [psutil.cpu_percent() and psutil.virtual_memory().percent]
    #  [ ] Rethink success criteria


# @app.command('check_wavelength', help='Check wavelength with aP_fit_wvl_to_spots')
# def cli_autoproc_check_wavelength():
#     pass


if __name__ == '__main__':
    app()
