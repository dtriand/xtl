import copy
from dataclasses import dataclass, field
from datetime import datetime, date
import dateutil.parser
import json
from pathlib import Path
import os
import re
import shutil
from typing import Optional, Sequence
import warnings

from defusedxml import ElementTree as DET

from xtl.common import afield, pfield, cfield
from xtl.common.annotated_dataclass import _ifield
from xtl.common.os import get_permissions_in_decimal
from xtl.diffraction.automate.gphl_utils import GPhLConfig
from xtl.diffraction.automate.xds_utils import CorrectLp
from xtl.diffraction.images.datasets import DiffractionDataset
from xtl.exceptions.warnings import FileNotFoundWarning, HTMLUpdateWarning


@dataclass
class AutoPROCConfig(GPhLConfig):
    # Housekeeping
    run_number: int = pfield(desc='Run number', group='housekeeping',
                             default=1, validator={'ge': 1})
    macro_filename: str = pfield(desc='Macro filename', group='housekeeping',
                                 default='xtl_autoPROC.dat',
                                 formatter=lambda x: f'{x}.dat' if not x.endswith('.dat') else x)
    batch_filename: str = pfield(desc='Batch filename', group='housekeeping',
                                 default='xtl_autoPROC.sh',
                                 formatter=lambda x: f'{x}.sh' if not x.endswith('.sh') else x)
    idn: str = pfield(desc='Dataset identifier', group='housekeeping',
                      default='')
    idn_prefix: str = pfield(desc='Prefix for the dataset identifier passed on to autoPROC',
                             default='xtl', group='housekeeping',)
    autoproc_output_subdir: str = pfield(desc='Subdirectory for autoPROC output',
                                         default='autoproc', group='housekeeping')
    file_permissions: int = pfield(desc='File permissions for all the output files',
                                   default=640, group='housekeeping',
                                   validator={'func': get_permissions_in_decimal})
    directory_permissions: int = pfield(desc='Directory permissions for all the output directories',
                                        default=750, group='housekeeping',
                                        validator={'func': get_permissions_in_decimal})
    change_permissions: bool = pfield(desc='Change permissions of the output files and directories',
                                      default=False, group='housekeeping')

    # User parameters
    unit_cell: list[float] = afield(desc='Target unit-cell for the dataset',
                                    default=None,
                                    alias='cell', group='user_params',
                                    validator={'len': 6},
                                    formatter=lambda x: ' '.join([str(y) for y in x]) if x else None)
    space_group: str = afield(desc='Target space group for the dataset',
                              default=None,
                              alias='symm', group='user_params',
                              formatter=lambda x: x.replace(' ', '') if x else None)
    wavelength: float = afield(desc='Wavelength of the X-ray beam in Angstroms',
                               default=None,
                               alias='wave', group='user_params',
                               validator={'gt': 0.0})
    resolution_low: float = afield(desc='Low resolution limit for the dataset',
                                   default=None,
                                   alias='init_reso', group='user_params',
                                   alias_fstring='{resolution_low:.2f} {resolution_high:.2f}',
                                   alias_fstring_keys=['resolution_low', 'resolution_high'],
                                   validator={'gt': 0.0})
    resolution_high: float = afield(desc='High resolution limit for the dataset',
                                    default=None,
                                    alias='init_reso', group='user_params',
                                    alias_fstring='{resolution_low:.2f} {resolution_high:.2f}',
                                    alias_fstring_keys=['resolution_low', 'resolution_high'],
                                    validator={'gt': 0.0})
    anomalous: bool = afield(desc='Keep anomalous signal',
                             default=None,
                             alias='anom', group='user_params')
    no_residues: int = afield(desc='Number of residues in the asymmetric unit',
                              default=None,
                              alias='nres', group='user_params',
                              validator={'gt': 0})
    mosaicity: float = afield(desc='Starting mosaicity value in degrees',
                              default=None,
                              alias='mosaic', group='user_params',
                              validator={'gt': 0.0})
    rfree_mtz: Path = afield(desc='Path to the MTZ file with R-free flags',
                             default=None,
                             alias='free_mtz', group='user_params')
    reference_mtz: Path = afield(desc='Path to the reference MTZ file for unit-cell, space group, indexing and R-free flags',
                                 default=None,
                                 alias='ref_mtz', group='user_params')
    mtz_project_name: str = afield(desc='Project name for the MTZ file',
                                   default=None,
                                   alias='pname', group='user_params')
    mtz_crystal_name: str = afield(desc='Crystal name for the MTZ file',
                                   default=None,
                                   alias='xname', group='user_params')
    mtz_dataset_name: str = afield(desc='Dataset name for the MTZ file',
                                   default=None,
                                   alias='dname', group='user_params')

    # XDS parameters
    xds_njobs: int = afield(desc='Maximum number of jobs for XDS',
                            default=None,
                            alias='autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS', group='xds_params',
                            validator={'ge': 1})
    xds_nproc: int = afield(desc='Maximum number of processors for XDS',
                            default=None,
                            alias='autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_PROCESSORS', group='xds_params',
                            validator={'ge': 1})
    xds_lib: Path = afield(desc='Path to external libraries',
                           default=None,
                           alias='autoPROC_XdsKeyword_LIB', group='xds_params')
    xds_polarization_fraction: float = afield(desc='Polarization fraction',
                                              default=None,
                                              alias='autoPROC_XdsKeyword_FRACTION_OF_POLARIZATION', group='xds_params',
                                              validator={'ge': 0.0, 'le': 1.0})
    xds_idxref_refine_params: list[str] = afield(desc='Parameters to refine in IDXREF',
                                                 alias='autoPROC_XdsKeyword_REFINEIDXREF', group='xds_params',
                                                 validator={
                                                    'choices': ['POSITION', 'BEAM', 'AXIS', 'ORIENTATION', 'CELL',
                                                                'SEGMENT']
                                                 },
                                                 formatter=lambda x: ' '.join(x) if x else [],
                                                 default_factory=lambda: [])
    xds_integrate_refine_params: list[str] = afield(desc='Parameters to refine in INTEGRATE',
                                                    alias='autoPROC_XdsKeyword_REFINEINTEGRATE', group='xds_params',
                                                    validator={
                                                       'choices': ['POSITION', 'BEAM', 'AXIS', 'ORIENTATION', 'CELL']
                                                    },
                                                    formatter=lambda x: ' '.join(x) if x else [],
                                                    default_factory=lambda: [])
    xds_correct_refine_params: list[str] = afield(desc='Parameters to refine in CORRECT',
                                                  alias='autoPROC_XdsKeyword_REFINECORRECT', group='xds_params',
                                                  validator={
                                                     'choices': ['POSITION', 'BEAM', 'AXIS', 'ORIENTATION', 'CELL',
                                                                 'SEGMENT']},
                                                  formatter=lambda x: ' '.join(x) if x else [],
                                                  default_factory=lambda: [])
    xds_defpix_optimize: bool = afield(desc='Optimize parameters for DEFPIX',
                                       default=None,
                                       alias='XdsOptimizeDefpix', group='xds_params')
    xds_idxref_optimize: bool = afield(desc='Optimize parameters for IDXREF',
                                       default=None,
                                       alias='XdsOptimizeIdxref', group='xds_params')
    xds_n_background_images: int = afield(desc='Number of images for background estimation',
                                          default=None,
                                          alias='XdsNumImagesBackgroundRange', group='xds_params')

    # Compound parameters
    _XdsExcludeIceRingsAutomatically: bool = afield(alias='XdsExcludeIceRingsAutomatically', default=None)
    _RunIdxrefExcludeIceRingShells: bool = afield(alias='RunIdxrefExcludeIceRingShells', default=None,)
    exclude_ice_rings: bool = cfield(desc='Exclude ice rings from the data',
                                     default=None,
                                     group='ice_rings_params',
                                     members=['_XdsExcludeIceRingsAutomatically', '_RunIdxrefExcludeIceRingShells'],
                                     in_sync=True)

    # Extra parameters
    extra_params: dict = afield(desc='Extra parameters for autoPROC',
                                alias='_extra_params',
                                group='extra_params',
                                formatter=lambda x: x if x else {},
                                default_factory=dict)

    # Macros
    beamline: str = afield(desc='Beamline name',
                           default=None,
                           alias='_beamline_macro',
                           validator={
                               'choice': [
                                   'AlbaBL13Xaloc', 'Als1231', 'Als422', 'Als831', 'AustralianSyncMX1',
                                   'AustralianSyncMX2', 'DiamondI04-MK', 'DiamondIO4', 'DiamondI23-Day1', 'DiamondI23',
                                   'EsrfId23-2', 'EsrfId29', 'EsrfId30-B', 'ILL_D19', 'PetraIIIP13', 'PetraIIIP14',
                                   'SlsPXIII', 'SoleilProxima1'
                               ]
                           })
    resolution_cutoff_criterion: str = afield(desc='Resolution cutoff criterion',
                                              default=None,
                                              alias='_resolution_cutoff_macro',
                                              validator={'choice': ['CC1/2', 'None']},
                                              formatter=lambda x: {'cc1/2': 'HighResCutOnCChalf' ,
                                                                   'none': 'NoHighResCut'}.get(x.lower(), None) if x else None)

    _macros: list[str] = cfield(desc='autoPROC macros',
                                default_factory=list,
                                group='macros',
                                members=['beamline', 'resolution_cutoff_criterion'],
                                formatter=lambda x: {'_macros':
                                                         ' '.join(f'-M {v}' for v in x.values() if v is not None)
                                                         if any(x.values()) else []}
                                )

    # Batch mode flag
    batch_mode: bool = pfield(desc='autoPROC in batch mode',
                              default=None,
                              formatter=lambda x: '-B' if x else None)

    # List of CLI arguments
    _args: list[str] = cfield(desc='autoPROC arguments',
                              default_factory=list,
                              alias='__args',
                              members=['batch_mode', '_macros'],
                              formatter=lambda x: {'__args': ' '.join(v for v in x.values() if v is not None)
                                                   if any(x.values()) else ''})

    _groups: dict = _ifield(default_factory=lambda: {
        'user_params': 'User parameters',
        'xds_params': 'XDS parameters',
        'ice_rings_params': 'Ice rings parameters'
    })

    def __post_init__(self):
        super().__post_init__()

        # Format the extra_params
        if self.extra_params:
            self.extra_params = {k: self._format_value(v) for k, v in self.extra_params.items()}
            self._validate_param(self._get_param('extra_params'))


@dataclass
class AutoProcXmlParser:
    filename: str | Path

    _xml_text: str = None
    _file_exists: bool = False
    _is_parsed: bool = False
    _is_processed: bool = False

    def __post_init__(self):
        self._file = Path(self.filename)
        # try:
        if not self._file.exists():
            raise FileNotFoundError(self._file)
        self._xml_text = self._file.read_text()
        self._file_exists = True
        self._parse_xml()
        self._process_xml()

    def _parse_xml(self):
        self._tree = DET.fromstring(self._xml_text)
        self._is_parsed = True

    def _process_xml(self):
        raise NotImplementedError

    @property
    def file(self):
        return self._file

    @property
    def data(self):
        raise NotImplementedError


@dataclass
class ImgInfo(AutoProcXmlParser):
    _collection_time: datetime = None
    _exposure_time: float = None
    _detector_distance: float = None
    _wavelength: float = None
    _phi_angle: float = None
    _omega_angle_start: float = None
    _omega_angle_end: float = None
    _omega_angle_step: float = None
    _kappa_angle: float = None
    _two_theta_angle: float = None
    _beam_center_x: float = None
    _beam_center_y: float = None
    _no_images: int = None
    _image_first: int = None
    _image_last: int = None

    def _parse_xml(self):
        self._xml_text = re.sub(r'(<\?xml[^>]+\?>)', r'\1<xtl_root>', self._file.read_text()) + '</xtl_root>'
        super()._parse_xml()

    def _process_xml(self):
        items = self._tree.findall('item')
        for item in items:
            id_, unit, value = item.find('id'), item.find('unit'), item.find('value')
            if id_ is None:
                continue
            id_text = id_.text
            if id_text == 'date':
                self._process_collection_time(value.text)
            elif id_text == 'exposure time':
                self._process_exposure_time(value.text, unit.text)
            elif id_text == 'distance':
                self._process_detector_distance(value.text, unit.text)
            elif id_text == 'wavelength':
                self._process_wavelength(value.text, unit.text)
            elif id_text == 'Phi-angle':
                self._process_phi_angle(value.text, unit.text)
            elif id_text == 'Omega-angle (start, end)':
                self._process_omega_angle_range(value.text, unit.text)
            elif id_text == 'Oscillation-angle in Omega':
                self._process_omega_angle_step(value.text, unit.text)
            elif id_text == 'Kappa-angle':
                self._process_kappa_angle(value.text, unit.text)
            elif id_text == '2-Theta angle':
                self._process_two_theta_angle(value.text, unit.text)
            elif id_text == 'Beam centre in X':
                self._process_beam_center_x(value.text, unit.text)
            elif id_text == 'Beam centre in Y':
                self._process_beam_center_y(value.text, unit.text)
            elif id_text == 'Number of images in sweep':
                self._process_no_images(value.text)
        self._is_processed = True

    def _process_collection_time(self, value):
        self._collection_time = dateutil.parser.parse(value)

    def _process_exposure_time(self, value, unit):
        v = float(value)
        if unit in ['s', 'seconds']:
            self._exposure_time = v
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_detector_distance(self, value, unit):
        v = float(value)
        if unit == 'm':
            self._detector_distance = v
        elif unit == 'cm':
            self._detector_distance = v / 100
        elif unit == 'mm':
            self._detector_distance = v / 1000
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_wavelength(self, value, unit):
        v = float(value)
        if unit == 'A':
            self._wavelength = v
        elif unit == 'nm':
            self._wavelength = v / 10
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_angle(self, value, unit):
        v = float(value)
        if unit == 'deg' or unit == 'degree':
            return v
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_phi_angle(self, value, unit):
        self._phi_angle = self._process_angle(value, unit)

    def _process_omega_angle_range(self, value, unit):
        angles = value.split(' ')
        if len(angles) != 2:
            raise ValueError(f"Expected 2 angles, got {len(angles)}: value='{value}'")
        omega_start, omega_end = angles
        self._omega_angle_start = self._process_angle(omega_start, unit)
        self._omega_angle_end = self._process_angle(omega_end, unit)

    def _process_omega_angle_step(self, value, unit):
        self._omega_angle_step = self._process_angle(value, unit)

    def _process_kappa_angle(self, value, unit):
        self._kappa_angle = self._process_angle(value, unit)

    def _process_two_theta_angle(self, value, unit):
        self._two_theta_angle = self._process_angle(value, unit)

    def _process_beam_center_x(self, value, unit):
        v = float(value)
        if unit == 'px' or unit == 'pixel':
            self._beam_center_x = v
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_beam_center_y(self, value, unit):
        v = float(value)
        if unit == 'px' or unit == 'pixel':
            self._beam_center_y = v
        else:
            raise ValueError(f"Unknown unit: {unit}")

    def _process_no_images(self, value):
        pattern = r'(\d+)\s\((\d+)\s-\s(\d+)\)'  # match a string like: '3600 (1 - 3600)'
        matches = re.findall(pattern, value)
        if len(matches) != 1:
            raise ValueError(f"Expected 1 match, got {len(matches)}: value='{value}', matches={matches}")

        numbers = [int(m) for m in matches[0]]
        if len(numbers) != 3:
            raise ValueError(f"Expected 3 numbers, got {len(numbers)}: value='{value}', numbers={numbers}")

        self._no_images = numbers[0]
        self._image_first = numbers[1]
        self._image_last = numbers[2]

    @property
    def file(self):
        return self._file

    @property
    def collection_time(self):
        return self._collection_time

    @property
    def exposure_time(self):
        return self._exposure_time

    @property
    def detector_distance(self):
        return self._detector_distance

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def phi_angle(self):
        return self._phi_angle

    @property
    def omega_angle_start(self):
        return self._omega_angle_start

    @property
    def omega_angle_end(self):
        return self._omega_angle_end

    @property
    def omega_angle_step(self):
        return self._omega_angle_step

    @property
    def kappa_angle(self):
        return self._kappa_angle

    @property
    def two_theta_angle(self):
        return self._two_theta_angle

    @property
    def beam_center_x(self):
        return self._beam_center_x

    @property
    def beam_center_y(self):
        return self._beam_center_y

    @property
    def no_images(self):
        return self._no_images

    @property
    def image_first(self):
        return self._image_first

    @property
    def image_last(self):
        return self._image_last

    @property
    def data(self):
        return {
            'collection_time': self.collection_time,
            'exposure_time': self.exposure_time,
            'detector_distance': self.detector_distance,
            'wavelength': self.wavelength,
            'phi_angle': self.phi_angle,
            'omega_angle_start': self.omega_angle_start,
            'omega_angle_end': self.omega_angle_end,
            'omega_angle_step': self.omega_angle_step,
            'kappa_angle': self.kappa_angle,
            'two_theta_angle': self.two_theta_angle,
            'beam_center_x': self.beam_center_x,
            'beam_center_y': self.beam_center_y,
            'no_images': self.no_images,
            'image_first': self.image_first,
            'image_last': self.image_last,
        }


@dataclass
class ReflectionsXml(AutoProcXmlParser):
    _processing_time: datetime = None

    _space_group: str = None
    _cell_a: float = None
    _cell_b: float = None
    _cell_c: float = None
    _cell_alpha: float = None
    _cell_beta: float = None
    _cell_gamma: float = None
    _wavelength: float = None

    _resolution_shell: list[str] = field(default_factory=list)
    _resolution_low: list[float] = field(default_factory=list)
    _resolution_high: list[float] = field(default_factory=list)
    _r_merge: list[float] = field(default_factory=list)
    _r_meas_within_i_plus_minus: list[float] = field(default_factory=list)
    _r_meas_all_i_plus_i_minus: list[float] = field(default_factory=list)
    _r_pim_within_i_plus_minus: list[float] = field(default_factory=list)
    _r_pim_all_i_plus_i_minus: list[float] = field(default_factory=list)
    _no_observations: list[int] = field(default_factory=list)
    _no_observations_unique: list[int] = field(default_factory=list)
    _i_over_sigma_mean: list[float] = field(default_factory=list)
    _completeness: list[float] = field(default_factory=list)
    _multiplicity: list[float] = field(default_factory=list)
    _cc_half: list[float] = field(default_factory=list)
    _anomalous_completeness: list[float] = field(default_factory=list)
    _anomalous_multiplicity: list[float] = field(default_factory=list)
    _anomalous_cc: list[float] = field(default_factory=list)
    _dano_over_sigma_dano: list[float] = field(default_factory=list)

    def _process_xml(self):
        autoproc_element = self._tree.find('AutoProc')
        if autoproc_element is not None:
            self._process_autoproc_element(autoproc_element)
        autoproc_scaling_container = self._tree.find('AutoProcScalingContainer')
        if autoproc_scaling_container is not None:
            autoproc_scaling_element = autoproc_scaling_container.find('AutoProcScaling')
            if autoproc_scaling_element is not None:
                self._process_autoproc_scaling_element(autoproc_scaling_element)
            autoproc_scaling_stats = autoproc_scaling_container.findall('AutoProcScalingStatistics')
            for stats in autoproc_scaling_stats:
                if stats is not None:
                    self._process_autoproc_scaling_stats(stats)
        self._is_processed = True

    def _process_autoproc_element(self, element: 'xml.etree.ElementTree.Element'):
        space_group = element.find('spaceGroup')
        if space_group is not None:
            self._space_group = space_group.text.replace(' ', '')
        wavelength = element.find('wavelength')
        if wavelength is not None:
            self._wavelength = float(wavelength.text)
        cell_a = element.find('refinedCell_a')
        if cell_a is not None:
            self._cell_a = float(cell_a.text)
        cell_b = element.find('refinedCell_b')
        if cell_b is not None:
            self._cell_b = float(cell_b.text)
        cell_c = element.find('refinedCell_c')
        if cell_c is not None:
            self._cell_c = float(cell_c.text)
        cell_alpha = element.find('refinedCell_alpha')
        if cell_alpha is not None:
            self._cell_alpha = float(cell_alpha.text)
        cell_beta = element.find('refinedCell_beta')
        if cell_beta is not None:
            self._cell_beta = float(cell_beta.text)
        cell_gamma = element.find('refinedCell_gamma')
        if cell_gamma is not None:
            self._cell_gamma = float(cell_gamma.text)

    def _process_autoproc_scaling_element(self, element: 'xml.etree.ElementTree.Element'):
        record_timestamp = element.find('recordTimeStamp')
        if record_timestamp is not None:
            self._processing_time = dateutil.parser.parse(record_timestamp.text)

    def _process_autoproc_scaling_stats(self, element: 'xml.etree.ElementTree.Element'):
        for e in element.iter():
            if e.tag == 'scalingStatisticsType':
                self._resolution_shell.append(e.text.replace('Shell', ''))
            elif e.tag == 'resolutionLimitLow':
                self._resolution_low.append(float(e.text))
            elif e.tag == 'resolutionLimitHigh':
                self._resolution_high.append(float(e.text))
            elif e.tag == 'rMerge':
                self._r_merge.append(float(e.text))
            elif e.tag == 'rMeasWithinIPlusIMinus':
                self._r_meas_within_i_plus_minus.append(float(e.text))
            elif e.tag == 'rMeasAllIPlusIMinus':
                self._r_meas_all_i_plus_i_minus.append(float(e.text))
            elif e.tag == 'rPimWithinIPlusIMinus':
                self._r_pim_within_i_plus_minus.append(float(e.text))
            elif e.tag == 'rPimAllIPlusIMinus':
                self._r_pim_all_i_plus_i_minus.append(float(e.text))
            elif e.tag == 'nTotalObservations':
                self._no_observations.append(int(e.text))
            elif e.tag == 'nTotalUniqueObservations':
                self._no_observations_unique.append(int(e.text))
            elif e.tag == 'meanIOverSigI':
                self._i_over_sigma_mean.append(float(e.text))
            elif e.tag == 'completeness':
                self._completeness.append(float(e.text))
            elif e.tag == 'multiplicity':
                self._multiplicity.append(float(e.text))
            elif e.tag == 'ccHalf':
                self._cc_half.append(float(e.text))
            elif e.tag == 'anomalousCompleteness':
                self._anomalous_completeness.append(float(e.text))
            elif e.tag == 'anomalousMultiplicity':
                self._anomalous_multiplicity.append(float(e.text))
            elif e.tag == 'ccAnomalous':
                self._anomalous_cc.append(float(e.text))
            elif e.tag == 'DanoOverSigDano':
                self._dano_over_sigma_dano.append(float(e.text))

    @property
    def processing_time(self):
        return self._processing_time

    @property
    def space_group(self):
        return self._space_group

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def cell_a(self):
        return self._cell_a

    @property
    def cell_b(self):
        return self._cell_b

    @property
    def cell_c(self):
        return self._cell_c

    @property
    def cell_alpha(self):
        return self._cell_alpha

    @property
    def cell_beta(self):
        return self._cell_beta

    @property
    def cell_gamma(self):
        return self._cell_gamma

    @property
    def resolution_shell(self):
        return self._resolution_shell

    @property
    def resolution_low(self):
        return self._resolution_low

    @property
    def resolution_high(self):
        return self._resolution_high

    @property
    def r_merge(self):
        return self._r_merge

    @property
    def r_meas_within_i_plus_minus(self):
        return self._r_meas_within_i_plus_minus

    @property
    def r_meas_all_i_plus_i_minus(self):
        return self._r_meas_all_i_plus_i_minus

    @property
    def r_pim_within_i_plus_minus(self):
        return self._r_pim_within_i_plus_minus

    @property
    def r_pim_all_i_plus_i_minus(self):
        return self._r_pim_all_i_plus_i_minus

    @property
    def no_observations(self):
        return self._no_observations

    @property
    def no_observations_unique(self):
        return self._no_observations_unique

    @property
    def i_over_sigma_mean(self):
        return self._i_over_sigma_mean

    @property
    def completeness(self):
        return self._completeness

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def cc_half(self):
        return self._cc_half

    @property
    def anomalous_completeness(self):
        return self._anomalous_completeness

    @property
    def anomalous_multiplicity(self):
        return self._anomalous_multiplicity

    @property
    def anomalous_cc(self):
        return self._anomalous_cc

    @property
    def dano_over_sigma_dano(self):
        return self._dano_over_sigma_dano

    @property
    def unit_cell(self):
        return [self.cell_a, self.cell_b, self.cell_c, self.cell_alpha, self.cell_beta, self.cell_gamma]

    @property
    def statistics(self):
        statistics = {}
        for i, shell in enumerate(self.resolution_shell):
            statistics[shell] = {
                'resolution_low': self.resolution_low[i],
                'resolution_high': self.resolution_high[i],
                'r_merge': self.r_merge[i],
                'r_meas_within_i_plus_minus': self.r_meas_within_i_plus_minus[i],
                'r_meas_all_i_plus_i_minus': self.r_meas_all_i_plus_i_minus[i],
                'r_pim_within_i_plus_minus': self.r_pim_within_i_plus_minus[i],
                'r_pim_all_i_plus_i_minus': self.r_pim_all_i_plus_i_minus[i],
                'no_observations': self.no_observations[i],
                'no_observations_unique': self.no_observations_unique[i],
                'i_over_sigma_mean': self.i_over_sigma_mean[i],
                'completeness': self.completeness[i],
                'multiplicity': self.multiplicity[i],
                'cc_half': self.cc_half[i],
                'anomalous_completeness': self.anomalous_completeness[i],
                'anomalous_multiplicity': self.anomalous_multiplicity[i],
                'anomalous_cc': self.anomalous_cc[i],
                'dano_over_sigma_dano': self.dano_over_sigma_dano[i],
            }
        return statistics

    @property
    def data(self):
        return {
            'processing_time': self.processing_time,
            'space_group': self.space_group,
            'unit_cell': self.unit_cell,
            'wavelength': self.wavelength,
            'statistics': self.statistics,
        }


@dataclass
class TruncateUnique(ReflectionsXml):
    ...


@dataclass
class StaranisoUnique(ReflectionsXml):
    _resolution_ellipsoid_axis_11: float = None
    _resolution_ellipsoid_axis_12: float = None
    _resolution_ellipsoid_axis_13: float = None
    _resolution_ellipsoid_axis_21: float = None
    _resolution_ellipsoid_axis_22: float = None
    _resolution_ellipsoid_axis_23: float = None
    _resolution_ellipsoid_axis_31: float = None
    _resolution_ellipsoid_axis_32: float = None
    _resolution_ellipsoid_axis_33: float = None
    _resolution_ellipsoid_value_1: float = None
    _resolution_ellipsoid_value_2: float = None
    _resolution_ellipsoid_value_3: float = None

    def _process_autoproc_scaling_element(self, element: 'xml.etree.ElementTree.Element'):
        super()._process_autoproc_scaling_element(element)
        for e in element.iter():
            if e.tag == 'recordTimeStamp':
                continue
            elif e.tag == 'resolutionEllipsoidAxis11':
                self._resolution_ellipsoid_axis_11 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis12':
                self._resolution_ellipsoid_axis_12 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis13':
                self._resolution_ellipsoid_axis_13 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis21':
                self._resolution_ellipsoid_axis_21 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis22':
                self._resolution_ellipsoid_axis_22 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis23':
                self._resolution_ellipsoid_axis_23 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis31':
                self._resolution_ellipsoid_axis_31 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis32':
                self._resolution_ellipsoid_axis_32 = float(e.text)
            elif e.tag == 'resolutionEllipsoidAxis33':
                self._resolution_ellipsoid_axis_33 = float(e.text)
            elif e.tag == 'resolutionEllipsoidValue1':
                self._resolution_ellipsoid_value_1 = float(e.text)
            elif e.tag == 'resolutionEllipsoidValue2':
                self._resolution_ellipsoid_value_2 = float(e.text)
            elif e.tag == 'resolutionEllipsoidValue3':
                self._resolution_ellipsoid_value_3 = float(e.text)

    @property
    def resolution_ellipsoid_axis_11(self):
        return self._resolution_ellipsoid_axis_11

    @property
    def resolution_ellipsoid_axis_12(self):
        return self._resolution_ellipsoid_axis_12

    @property
    def resolution_ellipsoid_axis_13(self):
        return self._resolution_ellipsoid_axis_13

    @property
    def resolution_ellipsoid_axis_21(self):
        return self._resolution_ellipsoid_axis_21

    @property
    def resolution_ellipsoid_axis_22(self):
        return self._resolution_ellipsoid_axis_22

    @property
    def resolution_ellipsoid_axis_23(self):
        return self._resolution_ellipsoid_axis_23

    @property
    def resolution_ellipsoid_axis_31(self):
        return self._resolution_ellipsoid_axis_31

    @property
    def resolution_ellipsoid_axis_32(self):
        return self._resolution_ellipsoid_axis_32

    @property
    def resolution_ellipsoid_axis_33(self):
        return self._resolution_ellipsoid_axis_33

    @property
    def resolution_ellipsoid_value_1(self):
        return self._resolution_ellipsoid_value_1

    @property
    def resolution_ellipsoid_value_2(self):
        return self._resolution_ellipsoid_value_2

    @property
    def resolution_ellipsoid_value_3(self):
        return self._resolution_ellipsoid_value_3

    @property
    def resolution_ellipsoid_axes(self):
        return [
            [self.resolution_ellipsoid_axis_11, self.resolution_ellipsoid_axis_12, self.resolution_ellipsoid_axis_13],
            [self.resolution_ellipsoid_axis_21, self.resolution_ellipsoid_axis_22, self.resolution_ellipsoid_axis_23],
            [self.resolution_ellipsoid_axis_31, self.resolution_ellipsoid_axis_32, self.resolution_ellipsoid_axis_33]
        ]

    @property
    def resolution_limits(self):
        return [self.resolution_ellipsoid_value_1, self.resolution_ellipsoid_value_2, self.resolution_ellipsoid_value_3]

    @property
    def data(self):
        data = super().data
        data.update({
            'resolution_ellipsoid_axes': self.resolution_ellipsoid_axes,
            'resolution_limits': self.resolution_limits,
        })
        return data


@dataclass
class AutoPROCJobResults2:
    job_dir: Path
    datasets: list[DiffractionDataset]

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
        # Check that all datasets are instances of DiffractionDataset
        for i, dataset in enumerate(self.datasets):
            if not isinstance(dataset, DiffractionDataset):
                raise TypeError(f'datasets[{i}] is not an instance of {DiffractionDataset.__class__.__name__}')

        # Dataset directories
        self._single_sweep = len(self.datasets) == 1
        self._dataset_dirs = [self.job_dir / dataset.autoproc_id for dataset in self.datasets]

        # Create paths to the log files
        # Files under job_dir regardless if multi-sweep mode
        self._summary_file = self.job_dir / self._summary_fname
        self._dat_files = [self.job_dir / f'{dataset.autoproc_id}.dat' for dataset in self.datasets]
        self._report_iso_file = self.job_dir / self._report_iso_fname
        self._mtz_iso_file = self.job_dir / self._mtz_iso_fname
        self._stats_iso_file = self.job_dir / self._stats_iso_fname

        self._report_aniso_file = self.job_dir / self._report_aniso_fname
        self._mtz_aniso_file = self.job_dir / self._mtz_aniso_fname
        self._stats_aniso_file = self.job_dir / self._stats_aniso_fname

        # Files that can be under job_dir or job_dir/autoproc_id
        if self._single_sweep:
            self._imginfo_files = [self.job_dir / self._imginfo_fname]
            self._correct_lp_files = [self.job_dir / self._correct_lp_fname]
        else:
            self._imginfo_files = [dataset_dir / self._imginfo_fname for dataset_dir in self._dataset_dirs]
            self._correct_lp_files = [dataset_dir / self._correct_lp_fname for dataset_dir in self._dataset_dirs]

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
        self._imginfo: Optional[Sequence[ImgInfo]] = None
        self._truncate: Optional[TruncateUnique] = None
        self._staraniso: Optional[StaranisoUnique] = None
        self._correct: Optional[Sequence[CorrectLp]] = None

        # Set the results data dictionary
        template_dict = {
            '_file': None,
            '_file_exists': False,
            '_is_parsed': False,
            '_is_processed': False
        }
        self._data = {
            'datasets': [
                {
                    'dataset_name': dataset.dataset_name,
                    'raw_data_dir': dataset.raw_data_dir,
                    'dataset_dir': dataset.dataset_dir,
                    'first_image': dataset.first_image,
                    'processed_data_dir': dataset.processed_data_dir,
                    'output_dir': dataset.output_dir,
                    'autoproc_id': dataset.autoproc_id,
                } for dataset in self.datasets
            ],
            'autoproc.imginfo': None,
            'autoproc.truncate': copy.deepcopy(template_dict),
            'autoproc.staraniso': copy.deepcopy(template_dict),
            'xds.correct': None,
        }

        if self._single_sweep:
            self._data['autoproc.imginfo'] = copy.deepcopy(template_dict)
            self._data['xds.correct'] = copy.deepcopy(template_dict)
        else:
            self._data['autoproc.imginfo'] = \
                {dataset.autoproc_id: copy.deepcopy(template_dict) for dataset in self.datasets}
            self._data['xds.correct'] = \
                {dataset.autoproc_id: copy.deepcopy(template_dict) for dataset in self.datasets}


    @property
    def success(self):
        return self._success

    def copy_files(self, dest_dir: Path = None, prefixes: list[str] = None):
        if dest_dir is None:
            dest_dir = self.job_dir.parent
        self._copy_summary_html(dest_dir)

        to_keep = [*self._dat_files, self._report_iso_file, self._report_aniso_file]
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
                              f'link_text_new: {link_text_new}', category=HTMLUpdateWarning)

            # Update link to GPhL logo
            gphl_logo_old = '<img src="gphl_logo.png"'
            gphl_logo_new = f'<img src="{relative_path}/gphl_logo.png"'
            content_old = content_new
            content_new = content_old.replace(gphl_logo_old, gphl_logo_new)
            if content_old == content_new:
                warnings.warn(f'Failed to replace the GPhL logo link in {summary_new.name}\n'
                              f'gphl_logo_old: {gphl_logo_old}\n'
                              f'gphl_logo_new: {gphl_logo_new}', category=HTMLUpdateWarning)

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
            warnings.warn(str(src_file), category=FileNotFoundWarning)

    def parse_logs(self):
        self.parse_imginfo_xml()
        self.parse_truncate_xml()
        self.parse_staraniso_xml()
        self.parse_correct_lp()
        if len(self._logs_is_processed) == 0:
            self._all_logs_processed = False
        else:
            self._all_logs_processed = all(self._logs_is_processed)

    def _update_parsing_status(self, key: str, parser, dataset_id: str = None):
        self._logs.append(parser.file)
        self._logs_exists.append(parser._file_exists)
        self._logs_is_parsed.append(parser._is_parsed)
        self._logs_is_processed.append(parser._is_processed)

        if dataset_id is None:
            self._data[key]['_file'] = parser.file
            self._data[key]['_file_exists'] = parser._file_exists
            self._data[key]['_is_parsed'] = parser._is_parsed
            self._data[key]['_is_processed'] = parser._is_processed
        else:  # for parsers that are dataset-specific, e.g. imginfo, correct_lp
            self._data[key][dataset_id]['_file'] = parser.file
            self._data[key][dataset_id]['_file_exists'] = parser._file_exists
            self._data[key][dataset_id]['_is_parsed'] = parser._is_parsed
            self._data[key][dataset_id]['_is_processed'] = parser._is_processed

    def parse_imginfo_xml(self):
        parsers = []
        for imginfo_file, dataset in zip(self._imginfo_files, self.datasets):
            parser = ImgInfo(filename=imginfo_file)
            if self._single_sweep:
                self._update_parsing_status(key='autoproc.imginfo', parser=parser)
                for key, value in parser.data.items():
                    self._data['autoproc.imginfo'][key] = value
            else:
                self._update_parsing_status(key='autoproc.imginfo', parser=parser, dataset_id=dataset.autoproc_id)
                for key, value in parser.data.items():
                    self._data['autoproc.imginfo'][dataset.autoproc_id][key] = value
            parsers.append(parser)
        if self._single_sweep:
            self._imginfo = parsers[0]
        else:
            self._imginfo = parsers

    @property
    def imginfo(self) -> ImgInfo | Sequence[ImgInfo]:
        if self._imginfo is None:
            self.parse_imginfo_xml()
        return self._imginfo

    def parse_truncate_xml(self):
        self._truncate = TruncateUnique(filename=self._stats_iso_file)
        self._update_parsing_status('autoproc.truncate', self._truncate)
        for key, value in self._truncate.data.items():
            self._data['autoproc.truncate'][key] = value

    @property
    def truncate(self) -> TruncateUnique:
        if self._truncate is None:
            self.parse_truncate_xml()
        return self._truncate

    def parse_staraniso_xml(self):
        self._staraniso = StaranisoUnique(filename=self._stats_aniso_file)
        self._update_parsing_status('autoproc.staraniso', self._staraniso)
        for key, value in self._staraniso.data.items():
            self._data['autoproc.staraniso'][key] = value

    @property
    def staraniso(self) -> StaranisoUnique:
        if self._staraniso is None:
            self.parse_staraniso_xml()
        return self._staraniso

    def parse_correct_lp(self):
        parsers = []
        for correctlp_file, dataset in zip(self._correct_lp_files, self.datasets):
            parser = CorrectLp(filename=correctlp_file)
            if self._single_sweep:
                self._update_parsing_status(key='xds.correct', parser=parser)
                for key, value in parser.data.items():
                    self._data['xds.correct'][key] = value
            else:
                self._update_parsing_status(key='xds.correct', parser=parser, dataset_id=dataset.autoproc_id)
                for key, value in parser.data.items():
                    self._data['xds.correct'][dataset.autoproc_id][key] = value
            parsers.append(parser)
        if self._single_sweep:
            self._correct = parsers[0]
        else:
            self._correct = parsers

    @property
    def correct_lp(self) -> CorrectLp | Sequence[CorrectLp]:
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
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    print(f'Object does not have a __dict__ method: {obj}')
                    return 'object_with_no_dict'

    def to_json(self):
        return json.dumps(self.data, indent=4, default=self._json_serializer)

    def save_json(self, dest_dir: Path):
        dest_dir = Path(dest_dir)
        json_file = dest_dir / self._json_fname
        json_file.write_text(self.to_json())
        return json_file

