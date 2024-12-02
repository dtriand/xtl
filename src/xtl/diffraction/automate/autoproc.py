import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from difflib import SequenceMatcher
from functools import partial
import json
import os
from pathlib import Path
import platform
from random import randint
import shutil
import traceback
from typing import Any, Sequence, Optional
import warnings

from xtl import __version__
from xtl.automate.batchfile import BatchFile
from xtl.automate.shells import Shell, BashShell
from xtl.automate.sites import ComputeSite
from xtl.automate.jobs import Job, limited_concurrency
from xtl.common.os import get_os_name_and_version, chmod_recursively
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.diffraction.automate.autoproc_utils import AutoPROCConfig, AutoPROCJobResults2, ImgInfo, TruncateUnique, StaranisoUnique
from xtl.diffraction.automate.xds_utils import CorrectLp
from xtl.exceptions.utils import Catcher


def value_to_str(value):
    if value is None:
        return ''
    return str(value)


def default_xds_idxref_refine_params():
    return ['BEAM', 'AXIS', 'ORIENTATION', 'CELL']


@dataclass
class AutoPROCJobConfig:
    # Data directories
    raw_data_dir: Path
    processed_data_dir: Path
    dataset_subdir: str
    dataset_name: str
    dataset_subdir_rename: str = None
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
    xds_njobs: int = None
    xds_nproc: int = None
    xds_pol_fraction: float = None
    xds_idxref_refine_params: list[str] = None
    xds_idxref_optimize: bool = None
    xds_n_bckg_images: int = None
    xds_defpix_start: int = None
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
        dataset_subdir = self.dataset_subdir_rename if self.dataset_subdir_rename is not None else self.dataset_subdir
        processing_subdir = f'/{self.processing_subdir}' if self.processing_subdir is not None else ''
        return Path(self.processed_data_dir_fstring.format(processed_data_dir=self.processed_data_dir,
                                                           dataset_subdir=dataset_subdir,
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
        xds_params = dict()
        if self.xds_njobs is not None:
            xds_params['autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS'] = self.xds_njobs
        if self.xds_nproc is not None:
            xds_params['autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_PROCESSORS'] = self.xds_nproc
        if self.xds_pol_fraction is not None:
            xds_params['autoPROC_XdsKeyword_FRACTION_OF_POLARIZATION'] = self.xds_pol_fraction
        if self.xds_idxref_refine_params is not None:
            xds_params['autoPROC_XdsKeyword_REFINEIDXREF'] = ' '.join(self.xds_idxref_refine_params)
        if self.xds_idxref_optimize is not None:
            xds_params['XdsOptimizeIdxref'] = 'yes' if self.xds_idxref_optimize else 'no'
        if self.xds_n_bckg_images is not None:
            xds_params['XdsNumImagesBackgroundRange'] = self.xds_n_bckg_images
        if self.xds_defpix_start is not None:
            xds_params['XdsOptimizeDefpixStart'] = self.xds_defpix_start
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
        has_commands = False
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
            has_commands = True
            macro += '### User parameters\n'
            for key, value in self.get_user_params().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        if self.get_xds_params():
            has_commands = True
            macro += '### XDS parameters\n'
            for key, value in self.get_xds_params().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        if self.get_misc_params():
            has_commands = True
            macro += '### Miscellaneous parameters\n'
            for key, value in self.get_misc_params().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        if self.get_extra_kwargs():
            has_commands = True
            macro += '### Extra parameters\n'
            for key, value in self.get_extra_kwargs().items():
                macro += self._format_key_value_pair(key, value)
            macro += '\n'

        if not has_commands:
            return ''
        return macro

    def get_beamline(self):
        if self.beamline:
            for bl in self._known_beamlines:
                if bl.lower() == self.beamline.lower():
                    self.beamline = bl
                    return [bl]
        self.beamline = None
        return []

    def get_resolution_cutoff_criterion(self):
        if self.resolution_cutoff_criterion:
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

        # Provide all other user-specified parameters in a macro file
        if self.get_params_macro():
            macro = [self.get_processed_data_path() / 'xtl_autoPROC.dat']
        else:
            macro = []

        # Add all macros to the command
        all_macros = set(macro + self.macros + self.get_beamline() + self.get_resolution_cutoff_criterion()
                      + self.get_tricky_data())  # remove duplicates
        if all_macros:
            command.extend([f'-M "{m}"' for m in all_macros])

        # Provide an image identifier (arbitrary id, raw data dir, image template, starting image, final image)
        command.append(f'-Id "{self.get_image_identifier()}"')

        # Specify output for processed data
        command.append(f'-d "{self.get_autoproc_output_path()}"')

        return ' '.join(command)


@dataclass
class AutoPROCJobResults:
    job_dir: Path
    job_id: str

    _json_fname = 'xtl_autoPROC.json'

    # Files to process
    _summary_fname = 'summary.html'  # fix links to relative paths
    _imginfo_fname = 'imginfo.xml'

    _report_iso_fname = 'report.pdf'
    _mtz_iso_fname = 'truncate-unique.mtz'
    _stats_iso_fname = 'truncate-unique.xml'  # or autoPROC.xml

    _report_aniso_fname = 'report_staraniso.pdf'
    _mtz_aniso_fname = 'staraniso_alldata-unique.mtz'
    _stats_aniso_fname = 'staraniso_alldata-unique.xml'  # or autoPROC_staraniso.xml

    _correct_lp_fname = 'CORRECT.LP'

    _success_fname = _mtz_aniso_fname

    def __post_init__(self):
        # Create paths to the log files
        self._summary_file = self.job_dir / self._summary_fname
        self._imginfo_file = self.job_dir / self._imginfo_fname
        self._dat_file = self.job_dir / f'{self.job_id}.dat'

        self._report_iso_file = self.job_dir / self._report_iso_fname
        self._mtz_iso_file = self.job_dir / self._mtz_iso_fname
        self._stats_iso_file = self.job_dir / self._stats_iso_fname

        self._report_aniso_file = self.job_dir / self._report_aniso_fname
        self._mtz_aniso_file = self.job_dir / self._mtz_aniso_fname
        self._stats_aniso_file = self.job_dir / self._stats_aniso_fname

        self._correct_lp_file = self.job_dir / self._correct_lp_fname

        # Determine the success of the job
        self._success_file = self.job_dir / self._success_fname
        self._success = self._success_file.exists()

        # Keep track of the parsed log files
        self._logs: list[Path] = []
        self._logs_exists: list[bool] = []
        self._logs_is_parsed: list[bool] = []
        self._logs_is_processed: list[bool] = []
        self._all_logs_processed: bool = False

        # Parsed log files
        self._imginfo: ImgInfo = None
        self._truncate: TruncateUnique = None
        self._staraniso: StaranisoUnique = None
        self._correct: CorrectLp = None

        # Results dictionary
        self._data = {
            'autoproc.imginfo': {
                '_file': None,
                '_file_exists': False,
                '_is_parsed': False,
                '_is_processed': False
            },
            'autoproc.truncate': {
                '_file': None,
                '_file_exists': False,
                '_is_parsed': False,
                '_is_processed': False
            },
            'autoproc.staraniso': {
                '_file': None,
                '_file_exists': False,
                '_is_parsed': False,
                '_is_processed': False
            },
            'xds.correct': {
                '_file': None,
                '_file_exists': False,
                '_is_parsed': False,
                '_is_processed': False
            }
        }

    @property
    def success(self):
        return self._success

    def copy_files(self, dest_dir: Path = None, prefixes: list[str] = None):
        if dest_dir is None:
            dest_dir = self.job_dir.parent
        self._copy_summary_html(dest_dir)

        to_keep = [self._dat_file, self._report_iso_file, self._report_aniso_file]
        to_rename = [self._mtz_iso_file, self._mtz_aniso_file]
        if prefixes is None:
            for file in to_keep + to_rename:
                self._copy_rename(file, dest_dir)
        else:
            if not (isinstance(prefixes, list) or isinstance(prefixes, tuple)):
                raise ValueError(f'prefixes must be a list or tuple, not {type(prefixes)}')
            for prefix in prefixes:
                for file in to_rename:
                    self._copy_rename(file, dest_dir, prefix)
            for file in to_keep:
                self._copy_rename(file, dest_dir)

    def _copy_summary_html(self, dest_dir):
        summary_old = self._summary_file
        summary_new = dest_dir / self._summary_fname
        if self._summary_file.exists():
            shutil.copy(summary_old, summary_new)

        # Fix links to relative paths
        if summary_new.exists():
            content_updated = False
            new_dir = dest_dir
            old_dir = self.job_dir
            relative_path = Path(os.path.relpath(path=old_dir, start=new_dir))

            # Update all links to the plots
            # BUG: The links are sometimes relative and sometimes absolute in summary.html - why?
            link_text_old = f'<a href="{old_dir}'
            link_text_new = f'<a href="{relative_path}'
            content_old = summary_old.read_text()
            content_new = content_old.replace(link_text_old, link_text_new)
            content_updated = (content_old != content_new)
            if not content_updated:
                warnings.warn(f'Failed to replace the links in {summary_new.name}\n'
                              f'link_text_old: {link_text_old}\n'
                              f'link_text_new: {link_text_new}')

            # Update link to GPhL logo
            gphl_logo_old = '<img src="gphl_logo.png"'
            gphl_logo_new = f'<img src="{relative_path}/gphl_logo.png"'
            content_old = content_new
            content_new = content_old.replace(gphl_logo_old, gphl_logo_new)
            if content_old == content_new:
                warnings.warn(f'Failed to replace the GPhL logo link in {summary_new.name}\n'
                              f'gphl_logo_old: {gphl_logo_old}\n'
                              f'gphl_logo_new: {gphl_logo_new}')

            # Create new summary.html file with the updated content
            content_updated = any([content_updated, content_old != content_new])
            if content_updated:
                summary_updated = dest_dir / f'{summary_new.stem}_updated.html'
                summary_updated.write_text(content_new)

    def _copy_rename(self, src_file: Path, dest_dir: Path, prefix: str = None):
        if src_file.exists():
            if prefix:
                dest_file = dest_dir / f'{prefix}_{src_file.name}'
            else:
                dest_file = dest_dir / src_file.name
            shutil.copy(src_file, dest_file)
        else:
            warnings.warn(f'File not found: {src_file}')

    def parse_logs(self):
        self.parse_imginfo_xml()
        self.parse_truncate_xml()
        self.parse_staraniso_xml()
        self.parse_correct_lp()
        if len(self._logs_is_processed) == 0:
            self._all_logs_processed = False
        else:
            self._all_logs_processed = all(self._logs_is_processed)

    def _update_parsing_status(self, key: str, parser):
        self._logs.append(parser.file)
        self._logs_exists.append(parser._file_exists)
        self._logs_is_parsed.append(parser._is_parsed)
        self._logs_is_processed.append(parser._is_processed)

        self._data[key]['_file'] = parser.file
        self._data[key]['_file_exists'] = parser._file_exists
        self._data[key]['_is_parsed'] = parser._is_parsed
        self._data[key]['_is_processed'] = parser._is_processed

    def parse_imginfo_xml(self):
        self._imginfo = ImgInfo(filename=self._imginfo_file, safe_parse=True)
        self._update_parsing_status('autoproc.imginfo', self._imginfo)
        for key, value in self._imginfo.data.items():
            self._data['autoproc.imginfo'][key] = value

    @property
    def imginfo(self):
        if self._imginfo is None:
            self.parse_imginfo_xml()
        return self._imginfo

    def parse_truncate_xml(self):
        self._truncate = TruncateUnique(filename=self._stats_iso_file, safe_parse=True)
        self._update_parsing_status('autoproc.truncate', self._truncate)
        for key, value in self._truncate.data.items():
            self._data['autoproc.truncate'][key] = value

    @property
    def truncate(self):
        if self._truncate is None:
            self.parse_truncate_xml()
        return self._truncate

    def parse_staraniso_xml(self):
        self._staraniso = StaranisoUnique(filename=self._stats_aniso_file, safe_parse=True)
        self._update_parsing_status('autoproc.staraniso', self._staraniso)
        for key, value in self._staraniso.data.items():
            self._data['autoproc.staraniso'][key] = value

    @property
    def staraniso(self):
        if self._staraniso is None:
            self.parse_staraniso_xml()
        return self._staraniso

    def parse_correct_lp(self):
        self._correct = CorrectLp(filename=self._correct_lp_file, safe_parse=True)
        self._update_parsing_status('xds.correct', self._correct)
        for key, value in self._correct.data.items():
            self._data['xds.correct'][key] = value

    @property
    def correct_lp(self):
        if self._correct is None:
            self.parse_correct_lp()
        return self._correct

    @property
    def data(self):
        data = {
            'success': self.success,
            'all_logs_processed': self._all_logs_processed
        }
        data.update(self._data)
        return data

    @staticmethod
    def _json_serializer(obj):
        try:
            return obj.toJSON()
        except AttributeError:
            if isinstance(obj, datetime) or isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return obj.as_uri()
            else:
                print('Unknown object type:', type(obj))
                return obj.__dict__

    def to_json(self):
        return json.dumps(self.data, indent=4, default=self._json_serializer)

    def save_json(self, dest_dir: Path):
        dest_dir = Path(dest_dir)
        json_file = dest_dir / self._json_fname
        json_file.write_text(self.to_json())
        return json_file

    def get_csv_header(self):
        header = []

        # imginfo.xml header
        header += ['_file', '_file_exists', '_is_parsed', '_is_processed']
        for key in self.imginfo.data.keys():
            header.append(key)

        return '# ' + ','.join(map(value_to_str, header)) + '\n'

    def get_csv_line(self):
        line = []

        # imginfo.xml line
        imginfo = self.imginfo
        line += [imginfo._file, imginfo._file_exists, imginfo._is_parsed, imginfo._is_processed]
        for key, value in imginfo.data.items():
            line.append(value)

        return ','.join(map(value_to_str, line)) + '\n'


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
        self._success_file = 'staraniso_alldata-unique.mtz'
        self._success: bool = None
        self._results: AutoPROCJobResults = None

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

    def create_macro(self):
        macro = self._config.get_params_macro()
        # Don't create a macro file if there are no parameters provided
        if not macro:
            return None
        macro_file = self.save_to_file(str(self._output_dir / self._macro_filename), macro)
        return macro_file

    @limited_concurrency(_no_parallel_jobs)
    async def run(self, do_run: bool = True):
        # Check that the provided raw/processed data directories are valid and that images exist
        self.echo('Checking paths...')
        self.config.check_paths()
        self.echo('Paths are valid')

        # Create a macro file with the user parameters
        self.echo('Creating autoPROC macro file...')
        m = self.create_macro()
        if not m:
            self.echo('No parameters provided, skipping macro file creation')
        else:
            self.echo(f'Macro file created: {m}')

        # Create the batch script file
        self.echo('Creating batch script...')
        self._build_batch_commands()
        s = self.create_batch(str(self._output_dir / self._batch_filename), self._batch_commands)
        self.echo(f'Batch script created: {s}')

        # Set up the log files
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
            self.tidy_up()
        else:
            self.echo('Skipping batch script execution and sleeping for 5 seconds...')
            await asyncio.sleep(5)
            self.echo('Done sleeping!')


    def tidy_up(self):
        self.echo('Tidying up results...')
        self._results = AutoPROCJobResults(job_dir=self.config.get_autoproc_output_path(), job_id=self.config._idn)
        self._success = self._results.success

        # Destination directory for copied files
        dest_dir = self.config.get_processed_data_path()

        # Determine prefix for copied files
        if self.config.mtz_dataset_name:
            prefix = self.config.mtz_dataset_name
        else:
            prefix = self.config.dataset_name

        # Copy files to the processed data directory
        self.echo(f'Copying files to {dest_dir}... ')
        self._results.copy_files(dest_dir=dest_dir, prefixes=[prefix])
        self.echo('Files copied')

        if not self._success:
            self.echo('autoPROC did not complete successfully, look at summary.html')
        else:
            self.echo('autoPROC completed successfully, now parsing the log files...')
            self._results.parse_logs()
            j = self._results.save_json(dest_dir)
            self.echo(f'Log files parsed and results saved to {j}')
        self.echo('Tidying up complete!')


class AutoPROCJob2(Job):
    _no_parallel_jobs = 5
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
        self._success_file: str = AutoPROCJobResults2._success_fname
        self._results: AutoPROCJobResults2 = None

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

            if not self._is_h5:
                # Set the autoproc_idn to be passed on the -Id flag
                # NOTE: Check if HDF5 images can also be parsed with -Id flag
                #  According to the documentation, the image template should be <dataset_name>_master.h5
                #  but how would we determine the first and last images? Are they required?
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
            if self._is_h5:
                # NOTE: dataset.autoproc_idn is not set for HDF5 images because it's currently not fully supported
                idns.append(dataset.first_image)
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
        prefix = '-h5' if self._is_h5 else '-Id'
        __args = ' '.join([f'{prefix} "{idn}"' for idn in idns]) + ' '
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
            f'# is_h5 = {self._is_h5}',
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
            self._results = AutoPROCJobResults2(job_dir=self.autoproc_dir, datasets=self.datasets)
        if catcher.raised:
            self.echo(f'Failed to create {AutoPROCJobResults2.__class__.__name__} instance')
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
