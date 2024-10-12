import asyncio
from datetime import datetime
from enum import Enum
from functools import partial, wraps
from pathlib import Path

import typer
import tabulate

from xtl.diffraction.automate.autoproc import AutoPROCJobConfig, AutoPROCJob
from xtl.automate.sites import LocalSite, BiotixHPC
from xtl.cli.cliio import CliIO
from xtl.config import cfg


app = typer.Typer(name='xtl.autoproc', help='Execute multiple autoPROC jobs')


def typer_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def parse_csv(csv_file: Path):
    datasets_dict = {
        'dataset_subdir': [],
        'dataset_name': [],
        'first_image': [],
        'rename_dataset_subdir': [],
        'mtz_project_name': [],
        'mtz_crystal_name': [],
        'mtz_dataset_name': []
    }
    indices = {key: None for key in datasets_dict.keys()}

    headers = csv_file.read_text().splitlines()[0].replace('#', '').replace(' ', '').split(',')
    for header in headers:
        if header in datasets_dict:
            indices[header] = headers.index(header)

    for line in csv_file.read_text().splitlines():
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
    none = 'none'
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

    # Create AutoPROCJob instances
    cli.echo('Preparing jobs... ', nl=False)
    apj = AutoPROCJob.update_concurrency_limit(no_concurrent_jobs)
    jobs = []
    for subdir, dataset, first_image, subdir_rename, pname, xname, dname in zip(
            datasets['dataset_subdir'], datasets['dataset_name'], datasets['first_image'],
            datasets['rename_dataset_subdir'], datasets['mtz_project_name'], datasets['mtz_crystal_name'],
            datasets['mtz_dataset_name']
    ):
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
    cli.echo(f'\nReady to run {len(jobs)} jobs')
    if not typer.confirm('Do you want to proceed?'):
        cli.echo('Aborted by user', level='warning')
        raise typer.Abort()

    # Run the jobs
    do_run = not dry_run
    t0 = datetime.now()
    cli.echo(f'\nLaunching jobs at {t0}...')
    await asyncio.gather(*[job.run(do_run=do_run) for job in jobs])
    t1 = datetime.now()
    cli.echo(f'All jobs completed at {t1}')
    cli.echo('xtl.automate finished graciously <3\n')
    cli.echo(f'Total elapsed time: {t1 - t0} [approx. {(t1 - t0) / len(jobs)} per job]')


if __name__ == '__main__':
    app()
