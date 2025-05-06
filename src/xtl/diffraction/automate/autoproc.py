import asyncio
import copy
from datetime import datetime
from functools import partial
import f90nml
import os
from pathlib import Path
import platform
from random import randint
import re
import traceback
from typing import Sequence, Optional

from xtl import __version__
from xtl.automate.batchfile import BatchFile
from xtl.automate.shells import Shell, BashShell
from xtl.automate.sites import ComputeSite
from xtl.automate.jobs import Job, limited_concurrency
from xtl.common.os import get_os_name_and_version, chmod_recursively
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig, AutoPROCJobResults
from xtl.exceptions.utils import Catcher


class AutoPROCJob(Job):
    _no_parallel_jobs = 1
    _default_shell = BashShell
    _supported_shells = [BashShell]
    _job_prefix = 'autoproc'

    _echo_success_kwargs = {}
    _echo_warning_kwargs = {}
    _echo_error_kwargs = {}

    def __init__(self, datasets: DiffractionDataset | Sequence[DiffractionDataset],
                 config: AutoPROCConfig | Sequence[AutoPROCConfig],
                 compute_site: Optional[ComputeSite] = None, shell: Optional[Shell] = None,
                 modules: Optional[Sequence[str]] = None, stdout_log: Optional[str | Path] = None,
                 stderr_log: Optional[str | Path] = None, output_exists: bool = False):

        # Initialize the Job class
        super().__init__(
            name=f'xtl_autoPROC_{randint(0, 9999):04d}',
            shell=shell,
            compute_site=compute_site,
            stdout_log=stdout_log,
            stderr_log=stderr_log
        )
        self._job_type = 'xtl.autoproc.process'
        self._executable = 'process'
        self._executable_location: Optional[str] = None
        self._executable_version: Optional[str] = None

        # Datasets and config
        self._datasets: Sequence[DiffractionDataset]
        self._config: AutoPROCConfig | Sequence[AutoPROCConfig]

        # Initialization modes
        self._single_sweep: bool  # True if only one dataset is provided
        self._common_config: bool  # True if only one config is provided
        self._validate_datasets_configs(datasets=datasets, configs=config)

        # Determine if the datasets are in HDF5 format
        self._is_h5 = self.datasets[0].is_h5

        # Determine the run number for the job
        self._reading_mode = output_exists
        self._run_no: int = None
        self._determine_run_no()

        # Move the log files to the job directory
        self._stdout = self.job_dir / 'xtl_autoPROC.stdout.log'
        self._stderr = self.job_dir / 'xtl_autoPROC.stderr.log'

        # Set the job identifier
        self._idn = f'{self.config.idn_prefix}{randint(0, 9999):04d}' if not self.config.idn else self.config.idn

        # Attach additional attributes to the datasets (sweep_id, autoproc_id, autoproc_idn (not for h5), job_dir)
        self._patch_datasets()

        self._modules = modules if modules else []

        # Results
        self._success: bool = None
        self._success_file: str = AutoPROCJobResults._success_fname
        self._results: AutoPROCJobResults = None

        # Batch and macro file
        self._batch_file: Path
        self._macro_file: Path

        # Set exception and warnings catcher
        self._exception_catcher = partial(Catcher, echo_func=self.echo, error_kwargs=self._echo_error_kwargs,
                                          warning_kwargs=self._echo_warning_kwargs)


    def _validate_datasets_configs(self, datasets: DiffractionDataset | Sequence[DiffractionDataset],
                                   configs: AutoPROCConfig | Sequence[AutoPROCConfig]):
        # Check that the datasets are valid
        if isinstance(datasets, DiffractionDataset):
            self._datasets = [datasets]
            self._single_sweep = True
        elif isinstance(datasets, Sequence):
            for i, ds in enumerate(datasets):
                if not isinstance(ds, DiffractionDataset):
                    raise ValueError(f'Invalid type for datasets\[{i}]: {type(ds)}')
            self._datasets = datasets
            self._single_sweep = len(datasets) == 1  # True if only one dataset is provided
        else:
            raise ValueError(f'\'datasets\' must be of type {DiffractionDataset.__name__} or a sequence of them, '
                             f'not {type(datasets)}')

        # Check that the config is valid
        if isinstance(configs, AutoPROCConfig):
            self._config = configs
            self._common_config = True
        elif isinstance(configs, Sequence):
            if len(configs) != len(self._datasets):
                raise ValueError(f'Length mismatch: datasets={len(self._datasets)} != config={len(configs)}')
            for i, c in enumerate(configs):
                if not isinstance(c, AutoPROCConfig):
                    raise ValueError(f'Invalid type for config\[{i}]: {type(c)}')
            self._config = configs
            self._common_config = len(configs) == 1  # True if only one config is provided
        else:
            raise ValueError(f'\'config\' must be of type {AutoPROCConfig.__name__} or a sequence of them, '
                             f'not {type(configs)}')

    @property
    def executable(self) -> str:
        return self._executable

    @property
    def config(self) -> AutoPROCConfig:
        """
        Return the main config.
        """
        return self._config if self._common_config else self._config[0]

    @property
    def datasets(self) -> list[DiffractionDataset]:
        """
        Return a list of all datasets.
        """
        return self._datasets

    @property
    def configs(self) -> list[AutoPROCConfig]:
        """
        Return a list of all configs.
        """
        return [self._config] if self._common_config else self._config

    @property
    def run_no(self) -> int:
        return self._run_no

    @property
    def job_dir(self) -> Path:
        processed_data = self._datasets[0].processed_data
        return processed_data / f'{self._job_prefix}_run{self.run_no:02d}'

    @property
    def autoproc_dir(self) -> Path:
        return self.job_dir / self.config.autoproc_output_subdir

    def _determine_run_no(self) -> int:
        """
        Determine the job run number without creating the job_dir.
        """
        self._run_no = self.config.run_number
        processed_data = self._datasets[0].processed_data
        if not processed_data.exists():
            return self._run_no
        while not self._reading_mode:  # Run number determination is skipped when in reading mode
            if not self.job_dir.exists():
                break
            if self._run_no > 99:
                raise FileExistsError(f'\'job_dir\' already exists: {self.job_dir}\n'
                                      f'All run numbers from 01 to 99 are already taken!')
            self._run_no += 1
        if self._run_no != self.config.run_number:  # Check if the run number was changed
            self.echo(f'Run number incremented to {self._run_no:02d} to avoid overwriting existing directories',
                      **self._echo_warning_kwargs)
        return self._run_no

    def _patch_datasets(self) -> None:
        """
        Attach additional attributes to the datasets (sweep_id, autoproc_id, autoproc_idn, output_dir).
        """
        for i, ds in enumerate(self._datasets):
            # Set the sweep_id
            setattr(ds, 'sweep_id', i + 1)

            # Set the autoproc_id
            if self._single_sweep:
                setattr(ds, 'autoproc_id', f'{self._idn}')
            else:
                setattr(ds, 'autoproc_id', f'{self._idn}s{ds.sweep_id:02d}')
                # Set the output directory to be the same for all datasets
                setattr(ds, 'output_dir', self._datasets[0].output_dir)

            if ds.is_h5:
                # Set the autoproc_idn to be passed on the -Id/-h5 flag
                # NOTE: Check if HDF5 images can also be parsed with -Id flag
                #  According to the documentation, the image template should be <dataset_name>_master.h5
                #  but how would we determine the first and last images? Are they required?
                image_template, first_image, last_image = ds.get_image_template(as_path=True, first_last=True)
                setattr(ds, 'autoproc_idn', f'{image_template}')
            else:
                image_template, first_image, last_image = ds.get_image_template(as_path=False, first_last=True)
                if first_image is None or last_image is None:
                    raise ValueError(f'Failed to determine first and last images for dataset[{i}]: {ds}\n'
                                     f'template: {image_template}, first: {first_image}, last: {last_image}')
                setattr(ds, 'autoproc_idn',
                        f'{ds.autoproc_id},{ds.raw_data},{image_template},{first_image},{last_image}')

    def _get_modules_commands(self) -> list[str]:
        commands = []
        if not self._modules:
            return commands
        purge_cmd = self._compute_site.purge_modules()
        commands.append(purge_cmd) if purge_cmd else None
        load_cmd = self._compute_site.load_modules(self._modules)
        commands.append(load_cmd) if load_cmd else None
        return commands

    def _get_batch_commands(self):
        """
        Creates a list of commands to be executed in the batch script. This includes the loading of modules if
        necessary.

        The command to be executed is: `process -M <MACRO>.dat -d <OUTPUT_DIR>`

        The rest of the configuration, including the dataset sweeps definition, is provided in the macro file.
        """

        # If a module was provided, then purge all modules and load the specified one
        commands = self._get_modules_commands()

        # Add the autoPROC command by applying the appropriate priority system
        process_cmd = f'{self.executable} -M {self.job_dir / self.config.macro_filename} -d {self.autoproc_dir}'
        commands.append(self._compute_site.prepare_command(process_cmd))
        return commands

    def _create_batch_file(self) -> BatchFile:
        commands = self._get_batch_commands()
        if not self.job_dir.exists():
            self.job_dir.mkdir(parents=True, exist_ok=True)
        batch = BatchFile(filename=self.job_dir / self.config.batch_filename, compute_site=self._compute_site,
                          shell=self._shell)
        batch.add_commands(commands)
        batch.save(change_permissions=True)
        self._batch_file = batch.file
        return batch

    def _get_macro_content(self) -> str:
        # Header
        content = [
            f'# autoPROC macro file',
            f'# Generated by xtl v.{__version__} on {datetime.now().isoformat()}',
            f'#  {os.getlogin()}@{platform.node()} [{get_os_name_and_version()}]',
            f''
        ]

        # Dataset definitions
        content += [
            f'### Dataset definitions',
            f'# autoproc_id = {self._idn}',
            f'# no_sweeps = {len(self.datasets)}'
        ]
        idns = []
        for dataset in self.datasets:
            content += [
                f'## Sweep {dataset.sweep_id} [{dataset.autoproc_id}]: {dataset.dataset_name}',
                f'#   raw_data = {dataset.raw_data}',
                f'#   first_image = {dataset.first_image.name}',
            ]
            if dataset.is_h5:
                idn = dataset.autoproc_idn  # equivalent to dataset.first_image
                idns.append(idn)
                content += [
                    f'#   idn = {idn}'
                ]
            else:
                idn = dataset.autoproc_idn
                idns.append(idn)
                _, _, image_template, img_no_first, img_no_last = idn.split(',')
                content += [
                    f'#   image_template = {image_template}',
                    f'#   img_no_first = {img_no_first}',
                    f'#   img_no_last = {img_no_last}',
                    f'#   idn = {idn}'
                ]
        content.append('')

        # __args parameter
        __args = ''
        for idn, dataset in zip(idns, self.datasets):
            prefix = '-h5' if dataset.is_h5 else '-Id'
            __args += f'{prefix} "{idn}" '
        __args += self.config.get_param_value('_args')['__args']

        content += [
            f'### CLI arguments (including dataset definitions and macros)',
            f'__args=\'{__args}\'',
            f''
        ]

        # Parameters from AutoPROCConfig
        all_params = self.config.get_all_params(modified_only=True, grouped=True)
        for group in all_params.values():
            content.append(f'### {group["comment"]}')
            for key, value in group['params'].items():
                content.append(f'{key}={value}')
            content.append('')

        # Extra parameters not included in the config definition
        extra_params = self.config.get_group('extra_params')['_extra_params']
        if extra_params:
            content.append('### Extra parameters')
            for key, value in extra_params.items():
                content.append(f'{key}={value}')
            content.append('')

        # Environment information
        content += [
            f'### XTL environment',
            f'# job_type = {self._job_type}',
            f'# run_number = {self.run_no}',
            f'# job_dir = {self.job_dir}',
            f'# autoproc_output_dir = {self.job_dir / self.config.autoproc_output_subdir}',
            f'## Initialization mode',
            f'# single_sweep = {self._single_sweep}',
            f'## Localization',
            f'# shell = {self._shell.name} [{self._shell.executable}]',
            f'# compute_site = {self._compute_site.__class__.__name__} '
            f'[{self._compute_site.priority_system.system_type}]',
            f'# files_permissions = {self.config.file_permissions}',
            f'# directories_permissions = {self.config.directory_permissions}',
            f'# change_permissions = {self.config.change_permissions}',
            f'# modules = {self._modules}',
            f'# executable = {self._executable_location}',
            f'# version = {self._executable_version}',
        ]
        return '\n'.join(content)

    def _create_macro_file(self) -> Path:
        self._macro_file = self.job_dir / self.config.macro_filename
        content = self._get_macro_content()
        self._macro_file.write_text(content, encoding='utf-8')
        return self._macro_file

    @limited_concurrency(_no_parallel_jobs)
    async def run(self, execute_batch: bool = True):
        # Check if the executable exists
        await self._determine_executable_location()
        await self._determine_executable_version()
        if not self._executable_location:
            self.echo(f'Executable \'{self.executable}\' not found in PATH')
            self.echo('Skipping job execution')
            return self

        # Create the job directory
        self.echo('Creating job directory...')
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.echo('Directory created')

        # Create a macro file with the user parameters
        self.echo('Creating autoPROC macro file...')
        m = self._create_macro_file()
        self.echo(f'Macro file created: {m}')

        # Create the batch script file
        self.echo('Creating batch script...')
        s = self._create_batch_file()
        self.echo(f'Batch script created: {s.file}')

        # Set up the log files
        self.echo('Initializing log files...')
        self._stdout.touch(exist_ok=True)
        self._stderr.touch(exist_ok=True)
        self.echo('Log files initialized')

        # Run the batch script
        if execute_batch:
            self.echo('Running batch script...')
            await self.run_batch(batchfile=s, stdout_log=self._stdout, stderr_log=self._stderr)
            self.echo('Batch script completed')
            self.tidy_up()
        else:
            self.echo('Skipping batch script execution and sleeping for 5 seconds...', **self._echo_warning_kwargs)
            await asyncio.sleep(5)
            self._success = True
            self.echo('Done sleeping!', **self._echo_success_kwargs)
        return self

    def tidy_up(self):
        self.echo('Tidying up results...')

        # Instantiate the results object
        with self._exception_catcher() as catcher:
            self._results = AutoPROCJobResults(job_dir=self.autoproc_dir, datasets=self.datasets)
        if catcher.raised:
            self.echo(f'Failed to create {AutoPROCJobResults.__class__.__name__} instance')
            return
        self._success = self._results.success

        # Determine prefix for copied files
        prefix = [self.config.mtz_dataset_name] if self.config.mtz_dataset_name else None

        # Copy files to the processed data directory
        dest_dir = self.job_dir
        self.echo(f'Copying files to {dest_dir}... ')
        with self._exception_catcher() as catcher:
            self._results.copy_files(dest_dir=dest_dir, prefixes=prefix)
        if catcher.raised:
            self.echo('Failed to copy files', **self._echo_error_kwargs)
            return
        self.echo('Files copied')

        if not self._success:
            self.echo('autoPROC did not complete successfully, look at summary.html', **self._echo_warning_kwargs)
        else:
            self.echo('autoPROC completed successfully, now parsing the log files...')
            with self._exception_catcher() as catcher:
                    self._results.parse_logs()
            if catcher.raised:
                self.echo('Failed to parse log files', **self._echo_error_kwargs)
                return
            with self._exception_catcher() as catcher:
                j = self._results.save_json(dest_dir)
            if catcher.raised:
                self.echo('Failed to save results to JSON', **self._echo_error_kwargs)
                return
            self.echo(f'Log files parsed and results saved to {j}')

        # Update permissions
        if self.config.change_permissions:
            self.echo('Updating permissions...')
            with self._exception_catcher() as catcher:
                chmod_recursively(self.job_dir, files_permissions=self.config.file_permissions,
                                  directories_permissions=self.config.directory_permissions)
                self.echo(f'File permissions updated to {self.config.file_permissions} and directory permissions '
                          f'updated to {self.config.directory_permissions}')
                if catcher.raised:
                    self.echo(f'Failed to update permissions to F {self.config.file_permissions} and '
                              f'D {self.config.directory_permissions}', **self._echo_error_kwargs)
                    return
        self.echo('Tidying up complete!')

    async def _run_command(self, command: str, prefix: str = 'custom_command', remove_logs: bool = True) \
            -> tuple[Optional[str], Optional[str]]:
        if not isinstance(command, str):
            raise TypeError(f'Invalid type for command: {type(command)}')

        # Create job directory
        dir_exists = self.job_dir.exists()
        if not dir_exists:
            self.job_dir.mkdir(parents=True, exist_ok=True)

        # Gather commands
        commands = self._get_modules_commands()
        commands.append(self._compute_site.prepare_command(command))

        # Create batch file
        batch = BatchFile(filename=self.job_dir / prefix, compute_site=self._compute_site,
                          shell=self._shell)
        batch.add_commands(commands)
        batch.save(change_permissions=True)

        # Run batch file
        try:
            stdout = self.job_dir / f'{prefix}.stdout.log'
            stderr = self.job_dir / f'{prefix}.stderr.log'
            await self.run_batch(batchfile=batch, stdout_log=stdout, stderr_log=stderr)
            result = stdout.read_text(), stderr.read_text()

            # Remove files
            if remove_logs:
                stdout.unlink()
                stderr.unlink()
                batch.file.unlink()
                if not dir_exists:  # directory was only created for command execution
                    self.job_dir.rmdir()

            return result
        except Exception as e:
            self.echo(f'Error running command: \'{command}\'')
            for line in traceback.format_exception(type(e), e, e.__traceback__):
                self.echo(f'    {line}')
            return None, None

    async def _determine_executable_location(self) -> Optional[str]:
        stdout, stderr = await self._run_command(f'which {self.executable}', prefix='which_process',
                                                 remove_logs=True)
        if not stdout:
            return None
        result = stdout.splitlines()[-1].strip()
        if result.endswith(self.executable):
            self._executable_location = result
        return self._executable_location

    async def _determine_executable_version(self) -> Optional[str]:
        stdout, stderr = await self._run_command(f'{self.executable} -h', prefix='process_version',
                                                 remove_logs=True)
        if not stdout:
            return None
        for line in stdout.splitlines():
            if 'Version:' in line:
                self._executable_version = line.replace('Version:', '').strip()
                return self._executable_version
        return None


class AutoPROCWorkflowJob(AutoPROCJob):
    _job_prefix = 'autoproc_wf'

    def __init__(self, nml_file: Path, config: AutoPROCConfig | Sequence[AutoPROCConfig],
                 compute_site: Optional[ComputeSite] = None,  shell: Optional[Shell] = None,
                 modules: Optional[Sequence[str]] = None, stdout_log: Optional[str | Path] = None,
                 stderr_log: Optional[str | Path] = None,
                 raw_data_dir: Optional[str | Path | Sequence[str | Path]] = None,
                 processed_data_dir: Optional[str | Path | Sequence[str | Path]] = None):

        # Extract datasets from NML file
        self._nml_file = Path(nml_file)
        self._nml = self._parse_nml(nml_file=self._nml_file)
        datasets = self._get_datasets_from_nml(raw_data_dir=raw_data_dir, processed_data_dir=processed_data_dir)

        # Copy the config for each of the datasets
        if isinstance(config, AutoPROCConfig):
            config = [copy.deepcopy(config) for i in range(len(datasets))]

        # Parent constructor
        super().__init__(
            datasets=datasets, config=config, compute_site=compute_site, shell=shell, modules=modules,
            stdout_log=stdout_log, stderr_log=stderr_log
        )

        # Modify attributes
        self._job_type = 'xtl.autoproc.aP_wf_process'
        self._executable = 'aP_wf_process'

    def _parse_nml(self, nml_file: Path) -> dict:
        nml_dict = {'datasets': [], 'crystal': {}}
        try:
            nml = f90nml.read(nml_file)
        except Exception as e:
            raise ValueError(f'Failed to parse namelist file: {nml_file}') from e

        for key in ['goniostat_setting_list', 'centred_goniostat_setting_list', 'detector_setting_list',
                    'simcal_beam_setting_list', 'simcal_sweep_list', 'process_crystal_list']:
            if key not in nml:
                raise ValueError(f'Namelist file does not contain \'{key}\' group')

        # Get information about the crystal symmetry
        nml_dict['crystal'] = {
            'input': {
                'unit_cell': nml['process_crystal_list']['prior_cell_dim'] +
                             nml['process_crystal_list']['prior_cell_ang_deg'],
                'space_group': nml['process_crystal_list']['prior_sg_name']
            },
            'indexed': {
                'unit_cell': nml['process_crystal_list']['cell_dim'] +
                             nml['process_crystal_list']['cell_ang_deg'],
                'space_group': nml['process_crystal_list']['sg_name']
            }
        }

        # Get sweep information
        for i, sweep in enumerate(nml['simcal_sweep_list']):
            dataset = {
                'sweep_id': i + 1,
                'name_template': sweep['name_template'],
                'img_first': sweep['image_no'],
                'img_last': sweep['n_frames'],
                'exposure': sweep['exposure'],
                'ap_params': {  # these will be passed as extra_params to config
                    'goniostat_Omega_angle': sweep['start_deg'],
                    'osc': sweep['step_deg']
                }
            }

            # Get goniostat information
            goniostat_id = None
            centred_goniostat_id = sweep['centred_goniostat_setting_id']
            for centred_goniostat in nml['centred_goniostat_setting_list']:
                if centred_goniostat['id'] == centred_goniostat_id:
                    dataset['ap_params']['Xparm2Simin_TransX'] = centred_goniostat['trans_1']
                    dataset['ap_params']['Xparm2Simin_TransY'] = centred_goniostat['trans_2']
                    dataset['ap_params']['Xparm2Simin_TransZ'] = centred_goniostat['trans_3']
                    goniostat_id = centred_goniostat['goniostat_setting_id']
                    break

            for goniostat in nml['goniostat_setting_list']:
                if goniostat['id'] == goniostat_id:
                    dataset['ap_params']['goniostat_Kappa_angle'] = goniostat['kappa_deg']
                    dataset['ap_params']['goniostat_Phi_angle'] = goniostat['phi_deg']
                    break

            # Get beam information
            beam_id = sweep['beam_setting_id']
            beam = nml['simcal_beam_setting_list']
            if beam['id'] == beam_id:
                dataset['ap_params']['wave'] = beam['lambda']

            # Get detector information
            detector_id = sweep['detector_setting_id']
            detector = nml['detector_setting_list']
            if detector['id'] == detector_id:
                dataset['ap_params']['dist'] = detector['det_coord']

            nml_dict['datasets'].append(dataset)
        return nml_dict


    def _get_datasets_from_nml(self, raw_data_dir: Optional[str | Path | Sequence[str | Path]] = None,
                               processed_data_dir: Optional[str | Path | Sequence[str | Path]] = None) -> list[DiffractionDataset]:
        datasets = []
        no_datasets = len(self._nml['datasets'])

        dirs = {'raw_data_dirs': [], 'processed_data_dirs': []}
        for dtype, directories in zip(dirs.keys(), [raw_data_dir, processed_data_dir]):
            if directories is None:
                dirs[dtype] = [None for _ in range(no_datasets)]
            elif isinstance(directories, (str, Path)):
                dirs[dtype] = [Path(directories) for _ in range(no_datasets)]
            elif isinstance(directories, Sequence):
                if len(dirs) != no_datasets:
                    raise ValueError(f'Length mismatch for \'{dtype}\': datasets={no_datasets} != '
                                     f'directories={len(directories)}')
                dirs[dtype] = [Path(d) for d in directories]

        for i, dataset in enumerate(self._nml['datasets']):
            # Get path to first image from the template string
            no_digits = dataset['name_template'].count('?')
            first_image = re.sub(r'\?.*', str(dataset['img_first']).zfill(no_digits), dataset['name_template'],
                                 count=1)
            first_image = Path(first_image)

            # Generate dataset from the first image
            ds = DiffractionDataset.from_image(image=first_image, raw_dataset_dir=dirs['raw_data_dirs'][i],
                                               processed_data_dir=dirs['processed_data_dirs'][i])
            datasets.append(ds)
        return datasets


    def _patch_datasets(self) -> None:
        # iterate over _datasets and _nml and patch both datasets and configs
        for ds, cfg, nml in zip(self.datasets, self.configs, self._nml['datasets']):
            # Set the sweep_id
            setattr(ds, 'sweep_id', nml['sweep_id'])

            # Set the autoproc_id
            setattr(ds, 'autoproc_id', f'{ds.sweep_id:03d}')

            # Set the extra_params for the config
            cfg.extra_params['nml_params'] = nml['ap_params']


    def _get_macro_content(self) -> str:
        # Header
        content = [
            f'# aP_wf_process macro file',
            f'# Generated by xtl v.{__version__} on {datetime.now().isoformat()}',
            f'#  {os.getlogin()}@{platform.node()} [{get_os_name_and_version()}]',
            f''
        ]

        # Dataset definitions
        content += [
            f'### Dataset definitions',
            f'# autoproc_id = {self._idn}',
            f'# no_sweeps = {len(self.datasets)}'
        ]
        for dataset, config in zip(self.datasets, self.configs):
            content += [
                f'## Sweep {dataset.sweep_id} [{dataset.autoproc_id}]: {dataset.dataset_name}',
                f'#   raw_data = {dataset.raw_data}',
                f'#   first_image = {dataset.first_image.name}',
            ]
            if not self._is_h5:
                image_template, img_no_first, img_no_last = dataset.get_image_template(as_path=False, first_last=True)
                content += [
                    f'#   image_template = {image_template}',
                    f'#   img_no_first = {img_no_first}',
                    f'#   img_no_last = {img_no_last}',
                ]
            for key, value in config.extra_params.pop('nml_params', {}).items():
                content.append(f'#   {key}_{dataset.autoproc_id} = {value}')
        content.append('')

        # Parameters from AutoPROCConfig
        all_params = self.config.get_all_params(modified_only=True, grouped=True)
        for group in all_params.values():
            content.append(f'### {group["comment"]}')
            for key, value in group['params'].items():
                content.append(f'{key}={value}')
            content.append('')

        # Extra parameters not included in the config definition
        extra_params = self.config.get_group('extra_params')['_extra_params']
        if extra_params:
            content.append('### Extra parameters')
            for key, value in extra_params.items():
                content.append(f'{key}={value}')
            content.append('')

        # Environment information
        content += [
            f'### XTL environment',
            f'# job_type = {self._job_type}',
            f'# run_number = {self.run_no}',
            f'# job_dir = {self.job_dir}',
            f'# autoproc_output_dir = {self.job_dir / self.config.autoproc_output_subdir}',
            f'## Initialization mode',
            f'# single_sweep = {self._single_sweep}',
            f'## Localization',
            f'# shell = {self._shell.name} [{self._shell.executable}]',
            f'# compute_site = {self._compute_site.__class__.__name__} '
            f'[{self._compute_site.priority_system.system_type}]',
            f'# files_permissions = {self.config.file_permissions}',
            f'# directories_permissions = {self.config.directory_permissions}',
            f'# change_permissions = {self.config.change_permissions}',
            f'# modules = {self._modules}',
            f'# executable = {self._executable_location}',
            f'# version = {self._executable_version}',
        ]
        return '\n'.join(content)

    def _get_batch_commands(self):
        # If a module was provided, then purge all modules and load the specified one
        commands = self._get_modules_commands()

        # Add the autoPROC command by applying the appropriate priority system
        process_cmd = f'{self.executable} -i {self._nml_file} -M {self.job_dir / self.config.macro_filename} -o {self.autoproc_dir}'
        commands.append(self._compute_site.prepare_command(process_cmd))
        return commands


# class CheckWavelengthJob(Job):
#
#     def __init__(self):
#         super().__init__(
#             name=f'xtl_check_wavelength_{randint(0, 9999):04d}',
#             compute_site=compute_site,
#             stdout_log=stdout_log,
#             stderr_log=stderr_log
#         )
#         self._job_type = 'check_wavelength'
#         self._module = module
