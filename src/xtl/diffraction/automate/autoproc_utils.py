from dataclasses import dataclass, field
from datetime import datetime
import dateutil.parser
from pathlib import Path
import re
import traceback
import warnings

from defusedxml import ElementTree as DET


@dataclass
class AutoProcXmlParser:
    filename: str | Path
    safe_parse: bool = False

    _xml_text: str = None
    _file_exists: bool = False
    _is_parsed: bool = False
    _is_processed: bool = False

    def __post_init__(self):
        self._file = Path(self.filename)
        try:
            if not self._file.exists():
                raise FileNotFoundError(f'File not found: {self._file}')
            self._xml_text = self._file.read_text()
            self._file_exists = True
            self._parse_xml()
            self._process_xml()
        except Exception as e:
            if self.safe_parse:
                warnings.warn(f'Error parsing file: {self._file}: {e}\n' +
                              traceback.format_exception(type(e), e, e.__traceback__))
            else:
                raise e

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
    def data(self):
        return {
            'processing_time': self.processing_time,
            'space_group': self.space_group,
            'wavelength': self.wavelength,
            'cell_a': self.cell_a,
            'cell_b': self.cell_b,
            'cell_c': self.cell_c,
            'cell_alpha': self.cell_alpha,
            'cell_beta': self.cell_beta,
            'cell_gamma': self.cell_gamma,
            'resolution_shell': self.resolution_shell,
            'resolution_low': self.resolution_low,
            'resolution_high': self.resolution_high,
            'r_merge': self.r_merge,
            'r_meas_within_i_plus_minus': self.r_meas_within_i_plus_minus,
            'r_meas_all_i_plus_i_minus': self.r_meas_all_i_plus_i_minus,
            'r_pim_within_i_plus_minus': self.r_pim_within_i_plus_minus,
            'r_pim_all_i_plus_i_minus': self.r_pim_all_i_plus_i_minus,
            'no_observations': self.no_observations,
            'no_observations_unique': self.no_observations_unique,
            'i_over_sigma_mean': self.i_over_sigma_mean,
            'completeness': self.completeness,
            'multiplicity': self.multiplicity,
            'cc_half': self.cc_half,
            'anomalous_completeness': self.anomalous_completeness,
            'anomalous_multiplicity': self.anomalous_multiplicity,
            'anomalous_cc': self.anomalous_cc,
            'dano_over_sigma_dano': self.dano_over_sigma_dano,
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
    def data(self):
        data = super().data
        data.update({
            'resolution_ellipsoid_axis_11': self.resolution_ellipsoid_axis_11,
            'resolution_ellipsoid_axis_12': self.resolution_ellipsoid_axis_12,
            'resolution_ellipsoid_axis_13': self.resolution_ellipsoid_axis_13,
            'resolution_ellipsoid_axis_21': self.resolution_ellipsoid_axis_21,
            'resolution_ellipsoid_axis_22': self.resolution_ellipsoid_axis_22,
            'resolution_ellipsoid_axis_23': self.resolution_ellipsoid_axis_23,
            'resolution_ellipsoid_axis_31': self.resolution_ellipsoid_axis_31,
            'resolution_ellipsoid_axis_32': self.resolution_ellipsoid_axis_32,
            'resolution_ellipsoid_axis_33': self.resolution_ellipsoid_axis_33,
            'resolution_ellipsoid_value_1': self.resolution_ellipsoid_value_1,
            'resolution_ellipsoid_value_2': self.resolution_ellipsoid_value_2,
            'resolution_ellipsoid_value_3': self.resolution_ellipsoid_value_3,
        })
        return data