import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from random import randint

from xtl.automate.jobs import Job, limited_concurrency
from xtl.automate.sites import ComputeSite
from xtl import __version__


def default_xds_idxref_refine_params():
    return ['BEAM', 'AXIS', 'ORIENTATION', 'CELL']


@dataclass
class AutoPROCJobConfig:
    # Data directories
    raw_data_dir: Path
    processed_data_dir: Path
    dataset_subdir: str
    dataset_name: str
    processing_subdir: str = None
    job_prefix: str = 'autoproc'
    run_number: int = 1
    first_image: str = None
    raw_data_dir_fstring: str = '{raw_data_dir}/{dataset_subdir}'
    processed_data_dir_fstring: str = '{processed_data_dir}/{dataset_subdir}{processing_subdir}/{job_subdir}'

    # Macros
    macros: list[str] = field(default_factory=list)

    # User-defined parameters
    unit_cell: str = None
    space_group: str = None
    wavelength: float = None
    resolution_high: float = None
    resolution_low: float = None
    anomalous: bool = False
    nresidues: int = None
    mosaicity: float = None
    free_mtz_file: Path = None
    reference_mtz_file: Path = None

    # MTZ data hierarchy
    mtz_project_name: str = None
    mtz_crystal_name: str = None
    mtz_dataset_name: str = None

    # Automatic image finding
    nimages: int = 500
    compressed_img: bool = True

    # XDS.INP parameters
    xds_njobs: int = 64
    xds_nproc: int = 8
    xds_pol_fraction: float = 0.98
    xds_idxref_refine_params: list[str] = field(default_factory=default_xds_idxref_refine_params)
    xds_idxref_optimize: bool = True
    xds_n_bckg_images: int = 100
    xds_defpix_start: int = 1000
    xds_lib: str = None

    # Miscellaneous parameters
    exclude_ice_rings: bool = None
    beamline: str = None
    resolution_cutoff_criterion: str = None
    tricky_data: bool = None

    # Extra kwargs
    extra_kwargs: dict = field(default_factory=dict)

    # Internal attributes
    _autoproc_output_subdir: str = 'autoproc'
    _file_ext: str = ''
    _is_h5_image: bool = False
    _is_compressed_image: bool = False
    _img_template: str = ''
    _starting_image: int = 1
    _no_images: int = 0
    _idn: str = None
    _idn_prefix: str = 'xtl'
    _known_beamlines = ['AlbaBL13Xaloc', 'Als1231', 'Als422', 'Als831', 'AustralianSyncMX1', 'AustralianSyncMX2',
                        'DiamondI04-MK', 'DiamondIO4', 'DiamondI23-Day1', 'DiamondI23', 'EsrfId23-2', 'EsrfId29',
                        'EsrfId30-B', 'ILL_D19', 'PetraIIIP13', 'PetraIIIP14', 'SlsPXIII', 'SoleilProxima1']
    _known_resolution_cutoff_criterions = {'CC1/2': 'HighResCutOnCChalf', 'none': 'NoHighResCut'}
    _echo = print

    def __post_init__(self):
        self._idn = f'{self._idn_prefix}_{randint(0, 9999):04d}'

    @property
    def _job_subdir(self):
        return f'{self.job_prefix}_run{self.run_number:02d}'

    def _check_path_fstring(self, fstring_type: str):
        subkeys = {
            'raw_data': ['raw_data_dir', 'dataset_subdir'],
            'processed_data': ['processed_data_dir', 'dataset_subdir', 'processing_subdir', 'job_subdir']
        }
        if fstring_type not in subkeys.keys():
            raise ValueError(f"Invalid fstring type: {fstring_type}")
        for subkey in subkeys[fstring_type]:
            if f'{{{subkey}}}' not in getattr(self, f'{fstring_type}_dir_fstring'):
                raise ValueError(f"Invalid f-string! Missing key '{subkey}' in {fstring_type} directory f-string")

    def get_raw_data_path(self):
        self._check_path_fstring('raw_data')
        return Path(self.raw_data_dir_fstring.format(raw_data_dir=self.raw_data_dir,
                                                     dataset_subdir=self.dataset_subdir))

    def get_processed_data_path(self):
        self._check_path_fstring('processed_data')
        processing_subdir = f'/{self.processing_subdir}' if self.processing_subdir is not None else ''
        return Path(self.processed_data_dir_fstring.format(processed_data_dir=self.processed_data_dir,
                                                           dataset_subdir=self.dataset_subdir,
                                                           processing_subdir=processing_subdir,
                                                           job_subdir=self._job_subdir))

    def get_autoproc_output_path(self):
        return self.get_processed_data_path() / self._autoproc_output_subdir

    def check_paths(self):
        # Check if raw data directory exists
        raw = self.get_raw_data_path()
        if not raw:
            raise FileNotFoundError(f"Raw data directory does not exist: {raw}")

        # Check if files exist in the raw data directory
        raw_images = [img for img in raw.iterdir() if img.is_file()]
        if not raw_images:
            raise FileNotFoundError(f"No files found in the raw_data directory: {raw}")

        # Check if images exist for the specified dataset
        dataset_images = [img for img in raw_images if img.name.startswith(self.dataset_name)]
        if not dataset_images:
            raise FileNotFoundError(f"No images for dataset {self.dataset_name} found in the raw_data directory: {raw}")

        # Determine file extension
        first_image = Path(self.first_image) if self.first_image is not None else dataset_images[0]
        if first_image.suffix == '.h5':
            self._file_ext = '.h5'
            self._is_h5_image = True
            self._is_compressed_image = False
        elif first_image.suffix == '.gz':
            self._is_compressed_image = True
            uncompressed_image = Path(first_image.stem)
            self._file_ext = uncompressed_image.suffix + '.gz'
        else:
            self._file_ext = first_image.suffix
            self._is_compressed_image = False

        # Determine image template, e.g. dataset_name_#####.cbf.gz
        dataset_images.sort()  # sort alphabetically
        # ToDo: Handling of .h5 images with master and data images
        self._img_template = self._get_image_template(dataset_images[0], dataset_images[-1])

        processed = self.get_processed_data_path()
        if processed.exists():
            while True:
                self.run_number += 1
                processed = self.get_processed_data_path()
                if not processed.exists():
                    break
                if self.run_number > 99:
                    raise FileExistsError(f'Directory for processed_data already exists: {processed}\n'
                                          f'All run numbers from 01 to 99 are already taken!')
            self._echo(f'Run number incremented to {self.run_number:02d} to avoid overwriting processed_data existing '
                       f'directory')

        processed.mkdir(parents=True, exist_ok=True)
        if not processed.exists():
            raise FileNotFoundError(f"Could not create directory for processed_data: {processed}")


    def _get_image_template(self, img1: Path, img2: Path):
        # Extract filenames without extensions, this accounts for compressed images
        fname0 = img1.name.split(self._file_ext)[0]
        fname1 = img2.name.split(self._file_ext)[0]

        # Find the longest match between the two filenames
        match = SequenceMatcher(None, fname0, fname1).find_longest_match()
        if match.a == match.b == 0:
            template = fname0[match.a:match.a+match.size]
            frame_digits = len(fname0[match.a+match.size:])
            self._no_images = int(fname1[match.b+match.size:])
            return f'{template}{"#" * frame_digits}{self._file_ext}'

    def get_image_identifier(self):
        return f'{self._idn},{self.get_raw_data_path()},{self._img_template},{self._starting_image},{self._no_images}'

    def get_user_params(self):
        user_params = dict()
        if self.unit_cell is not None:
            user_params['cell'] = self.unit_cell
        if self.space_group is not None:
            user_params['symm'] = self.space_group.replace(' ', '')
        if self.wavelength is not None:
            user_params['wave'] = self.wavelength
        if self.resolution_high is not None and self.resolution_low is not None:
            user_params['init_reso'] = f'{self.resolution_high:.2f} {self.resolution_low:.2f}'
        if self.anomalous:
            user_params['anom'] = 'yes'
        if self.nresidues is not None:
            user_params['nres'] = self.nresidues
        if self.mosaicity is not None:
            user_params['mosaic'] = self.mosaicity
        if self.free_mtz_file is not None:
            user_params['free_mtz'] = str(self.free_mtz_file)
        if self.reference_mtz_file is not None:
            user_params['ref_mtz'] = str(self.reference_mtz_file)

        if self.mtz_project_name is not None:
            user_params['pname'] = self.mtz_project_name
        if self.mtz_crystal_name is not None:
            user_params['xname'] = self.mtz_crystal_name
        if self.mtz_dataset_name is not None:
            user_params['dname'] = self.mtz_dataset_name
        return user_params

    def get_xds_params(self):
        xds_params = {
            'autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS': self.xds_njobs,
            'autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_PROCESSORS': self.xds_nproc,
            'autoPROC_XdsKeyword_FRACTION_OF_POLARIZATION': self.xds_pol_fraction,
            'autoPROC_XdsKeyword_REFINEIDXREF': ' '.join(self.xds_idxref_refine_params),
            'XdsOptimizeIdxref': "yes" if self.xds_idxref_optimize else "no",
            'XdsNumImagesBackgroundRange': self.xds_n_bckg_images,
            'XdsOptimizeDefpixStart': self.xds_defpix_start
        }
        if self.xds_lib is not None:
            xds_params['autoPROC_XdsKeyword_LIB'] = self.xds_lib
        return xds_params

    def get_misc_params(self):
        misc_params = dict()
        if self.exclude_ice_rings is not None:
            misc_params['_comment'] = 'Ice rings exclusion'
            misc_params['XdsExcludeIceRingsAutomatically'] = 'yes' if self.exclude_ice_rings else 'no'
            misc_params['RunIdxrefExcludeIceRingShells'] = 'yes' if self.exclude_ice_rings else 'no'
        return misc_params

    def get_extra_kwargs(self):
        return self.extra_kwargs

    def _format_key_value_pair(self, key, value):
        if isinstance(value, float) or isinstance(value, int):
            return f'{key}={value}\n'
        elif key == '_comment':
            return f'# {value}\n'
        else:
            return f'{key}="{value}"\n'

    def get_params_macro(self):
        macro = f'# autoPROC macro file for dataset {self.dataset_name}\n'
        macro += f'# Generated by xtl v.{__version__} on {datetime.now().isoformat()}\n\n'

        macro += f'### XTL input\n'
        macro += f'## Directories\n'
        macro += f'# raw_data_dir = "{self.get_raw_data_path()}"\n'
        macro += f'# image_template = "{self._img_template}"\n'
        macro += f'# starting_image = {self._starting_image}\n'
        macro += f'# final_image = {self._no_images}\n'
        macro += f'# dataset_idn = "{self._idn}"\n'
        macro += f'# processed_data_dir = "{self.get_processed_data_path()}"\n'
        macro += f'# autoproc_output_dir = "{self.get_autoproc_output_path()}"\n\n'

        if self.get_user_params():
            macro += '### User parameters\n'
            for key, value in self.get_user_params().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        macro += '### XDS parameters\n'
        for key, value in self.get_xds_params().items():
            macro += self._format_key_value_pair(key, value)
        macro += '\n'

        if self.get_misc_params():
            macro += '### Miscellaneous parameters\n'
            for key, value in self.get_misc_params().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        if self.get_extra_kwargs():
            macro += '### Extra parameters\n'
            for key, value in self.get_extra_kwargs().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        return macro

    def get_beamline(self):
        for bl in self._known_beamlines:
            if bl.lower() == self.beamline.lower():
                self.beamline = bl
                return [bl]
        self.beamline = None
        return []

    def get_resolution_cutoff_criterion(self):
        for rc in self._known_resolution_cutoff_criterions.keys():
            if rc.lower() == self.resolution_cutoff_criterion.lower():
                self.resolution_cutoff_criterion = rc
                return [self._known_resolution_cutoff_criterions[rc]]
        self.resolution_cutoff_criterion = None
        return []

    def get_tricky_data(self):
        if self.tricky_data is not None:
            return ['LowResOrTricky'] if self.tricky_data else []
        return []

    def get_command(self):
        command = ['process']

        # Run in batch mode
        command.append('-B')

        # Provide all other options in a macro file
        macro = self.get_processed_data_path() / 'xtl_autoPROC.dat'
        all_macros = ([macro] + self.macros + self.get_beamline() + self.get_resolution_cutoff_criterion()
                      + self.get_tricky_data())
        for m in set(all_macros):  # remove duplicates
            command.append(f'-M "{m}"')

        # Provide an image identifier (arbitrary id, raw data dir, image template, starting image, final image)
        command.append(f'-Id "{self.get_image_identifier()}"')

        # Specify output for processed data
        command.append(f'-d "{self.get_autoproc_output_path()}"')

        return ' '.join(command)


class AutoPROCJob(Job):
    _no_parallel_jobs = 5

    def __init__(self, config: AutoPROCJobConfig, compute_site: ComputeSite = None, module: str = None,
                 stdout_log: str | Path = None, stderr_log: str | Path = None, macro_filename: str = 'xtl_autoPROC.dat',
                 batch_filename: str = 'xtl_autoPROC.sh'):
        super().__init__(
            name=f'xtl_autoPROC_{randint(0, 9999):04d}',
            compute_site=compute_site,
            stdout_log=stdout_log,
            stderr_log=stderr_log
        )
        self._job_type = 'autoPROC'
        self._module = module

        # Load config parameters
        if not isinstance(config, AutoPROCJobConfig):
            raise ValueError(f'config must be an instance of {AutoPROCJobConfig.__name__}, not {type(config)}')
        self._config: AutoPROCJobConfig = config
        self._config._echo = self.echo

        # Filenames for the macro and batch file
        self._macro_filename = f'{macro_filename}{".dat" if not macro_filename.endswith(".dat") else ""}'
        self._batch_filename = f'{batch_filename}{".sh" if not batch_filename.endswith(".sh") else ""}'

        # Generate the commands for the batch scripts
        self._batch_commands = []

    @property
    def _output_dir(self):
        return self._config.get_processed_data_path()

    def _build_batch_commands(self):
        """
        Creates a list of commands to be executed in the batch script. This includes the loading of modules if
        necessary.
        """
        # If a module was provided, then purge all modules and load the specified one
        if self._module:
            purge_cmd = self._compute_site.purge_modules()
            if purge_cmd:
                self._batch_commands.append(purge_cmd)
            load_cmd = self._compute_site.load_modules(self._module)
            if load_cmd:
                self._batch_commands.append(load_cmd)
        # Add the autoPROC command by applying the appropriate priority system
        process_cmd = self._config.get_command()
        self._batch_commands.append(self._compute_site.prepare_command(process_cmd))
        return self._batch_commands

    @property
    def config(self):
        return self._config

    @limited_concurrency(_no_parallel_jobs)
    async def run(self, do_run: bool = True):
        # Check that the provided raw/processed data directories are valid and that images exist
        self.echo('Checking paths...')
        self.config.check_paths()
        self.echo('Paths are valid')

        # Create a macro file with the user parameters
        self.echo('Creating autoPROC macro file...')
        m = self.create_macro()
        self.echo(f'Macro file created: {m}')

        # Create the batch script file
        self.echo('Creating batch script...')
        self._build_batch_commands()
        s = self.create_batch(str(self._output_dir / self._batch_filename), self._batch_commands)
        self.echo(f'Batch script created: {s}')

        # Set the log files
        self.echo('Initializing log files...')
        log_stdout = self._output_dir / f'xtl_{self._job_type}.stdout.log'
        log_stderr = self._output_dir / f'xtl_{self._job_type}.stderr.log'
        log_stdout.touch(exist_ok=True)
        log_stderr.touch(exist_ok=True)
        self.echo('Log files initialized')

        # Run the batch script
        if do_run:
            self.echo('Running batch script...')
            await self.run_batch(batchfile=s, stdout_log=log_stdout, stderr_log=log_stderr)
            self.echo('Batch script completed')
        else:
            self.echo('Skipping batch script execution and sleeping for 5 seconds...')
            await asyncio.sleep(5)
            self.echo('Done sleeping!')

    def create_macro(self):
        macro = self._config.get_params_macro()
        macro_file = self.save_to_file(str(self._output_dir / self._macro_filename), macro)
        return macro_file