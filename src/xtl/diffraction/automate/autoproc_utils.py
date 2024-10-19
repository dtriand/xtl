from dataclasses import dataclass
from datetime import datetime
import dateutil.parser
from pathlib import Path
import re
import traceback
import warnings

from defusedxml import ElementTree as DET


@dataclass
class ImgInfo:
    filename: str | Path
    safe_parse: bool = False

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

    _file_exists: bool = False
    _is_parsed: bool = False
    _is_processed: bool = False

    def __post_init__(self):
        self._file = Path(self.filename)
        try:
            if not self._file.exists():
                raise FileNotFoundError(f'File not found: {self._file}')
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
        xml_fixed = re.sub(r'(<\?xml[^>]+\?>)', r'\1<xtl_root>', self._file.read_text()) + '</xtl_root>'
        self._tree = DET.fromstring(xml_fixed)
        self._is_parsed = True

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
            'exposure_time (s)': self.exposure_time,
            'detector_distance (m)': self.detector_distance,
            'wavelength (A)': self.wavelength,
            'phi_angle (deg)': self.phi_angle,
            'omega_angle_start (deg)': self.omega_angle_start,
            'omega_angle_end (deg)': self.omega_angle_end,
            'omega_angle_step (deg)': self.omega_angle_step,
            'kappa_angle (deg)': self.kappa_angle,
            'two_theta_angle (deg)': self.two_theta_angle,
            'beam_center_x (px)': self.beam_center_x,
            'beam_center_y (px)': self.beam_center_y,
            'no_images': self.no_images,
            'image_first': self.image_first,
            'image_last': self.image_last,
        }