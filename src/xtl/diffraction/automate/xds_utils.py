from dataclasses import dataclass, field
from datetime import datetime
from dateutil import parser as date_parser
from pathlib import Path
import re
import traceback
import warnings

from xtl.exceptions.warnings import RegexWarning


@dataclass
class XdsLpParser:
    filename: str
    _lp_type: str

    _xds_version: datetime = None
    _xds_build: str = None
    _xds_release_date: datetime = None

    _file_exists: bool = False
    _is_parsed: bool = False
    _is_processed: bool = False

    def __post_init__(self):
        self._file = Path(self.filename)
        if not self._file.exists():
            raise FileNotFoundError(self._file)
        self._lp_text = self._file.read_text()
        self._file_exists = True
        self._parse_lp()

    @property
    def file(self):
        return self._file

    def _parse_lp(self):
        version_parsed = False
        for line in self._lp_text.splitlines():
            if version_parsed:
                break
            if line.startswith(f' ***** {self._lp_type} *****'):
                self._parse_version(line)

    def _perform_regex(self, pattern: str, line: str, block: str = None):
        match = re.search(pattern, line)
        if not match:
            warnings.warn(f'Regex failed while parsing {"block " + block + " " if isinstance(block, str) else ""}: '
                          f'{line}\nPattern: {pattern}', category=RegexWarning)
            return None
        return match

    def _raise_processing_warning(self, line: str, exception: Exception, block: str = None):
        warnings.warn(f'Processing failed while parsing {"block " + block + " " if isinstance(block, str) else ""}: '
                      f'{line}: {exception}\n' +
                      '\n'.join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
                      category=RegexWarning)

    def _parse_line_key_value(self, line: str, key: str, value_type: str, sep: str = '', block: str = None):
        if value_type not in ['int', 'float', 'sci-float', 'string', 'boolean']:
            raise ValueError(f'Invalid value type: {value_type}')

        # Escape special characters on the key
        key = key.replace('(', '\(').replace(')', '\)').replace('[', '\[').replace(']', '\]')

        if value_type == 'int':
            # optional space, one or more digits
            pattern = rf'{key}{sep}\s*(-?\d+)'
        elif value_type == 'float':
            # optional space, zero or more digits, decimal point, zero or more digits
            pattern = rf'{key}{sep}\s*(-?\d*\.\d*)'
        elif value_type == 'sci-float':
            # optional space, zero or more digits, decimal point, zero or more digits, e, zero or more digits
            pattern = rf'{key}{sep}\s*(-?\d*\.\d*[e|E][-|+]?\d*)'
        elif value_type == 'string':
            # optional space, one or more non-space characters
            pattern = rf'{key}{sep}\s*(\S+)'
        elif value_type == 'boolean':
            # optional space, TRUE or FALSE
            pattern = rf'{key}{sep}\s*(TRUE|FALSE)'

        match = self._perform_regex(pattern=pattern, line=line, block=f'{block}/{key}')
        if not match:
            return None

        if value_type == 'int':
            return int(match.group(1))
        elif value_type in ['float', 'sci-float']:
            return float(match.group(1))
        elif value_type == 'string':
            return match.group(1)
        elif value_type == 'boolean':
            return bool(match.group(1))

    def _parse_line_key_values(self, line: str, key: str, values_type: str, sep: str = '', block: str = None):
        if values_type not in ['int', 'float', 'sci-float', 'string', 'boolean']:
            raise ValueError(f'Invalid value type: {values_type}')

        # Escape special characters on the key
        key = key.replace('(', '\(').replace(')', '\)').replace('[', '\[').replace(']', '\]')

        if values_type == 'int':
            # space-separated list of [one or more digits]
            pattern = rf'{key}{sep}\s*((?:-?\d+\s*)+)'
        elif values_type == 'float':
            # space-separated list of [zero or more digits, decimal point, zero or more digits]
            pattern = rf'{key}{sep}\s*((?:-?\d*\.\d*\s*)+)'
        elif values_type == 'sci-float':
            # space-separated list of [zero or more digits, decimal point, zero or more digits, e, zero or more digits]
            pattern = rf'{key}{sep}\s*((?:-?\d*\.\d*[e|E][-|+]?\d*\s*)+)'
        elif values_type == 'string':
            # space-separated list of [one or more non-space characters]
            pattern = rf'{key}{sep}\s*((?:\S+\s*)+)'
        elif values_type == 'boolean':
            # space-separated list of [TRUE or FALSE]
            pattern = rf'{key}{sep}\s*((?:TRUE|FALSE\s*)+)'

        match = self._perform_regex(pattern=pattern, line=line, block=f'{block}/{key}')
        if not match:
            return None

        if values_type == 'int':
            return [int(value.strip()) for value in match.group(1).split()]
        elif values_type in ['float', 'sci-float']:
            return [float(value.strip()) for value in match.group(1).split()]
        elif values_type == 'string':
            return [value.strip() for value in match.group(1).split()]
        elif values_type == 'boolean':
            return [bool(value.strip()) for value in match.group(1).split()]

    def _parse_line_keys_values(self, line: str, keys: list[str], value_types: list[str], sep: str = '',
                                block: str = None):
        if len(keys) != len(value_types):
            raise ValueError(f'keys and value_types lists must have the same length')

        values = []
        for key, value_type in zip(keys, value_types):
            values.append(self._parse_line_key_value(line=line, key=key, value_type=value_type, sep=sep,
                                                     block=block))

        return values

    def _parse_version(self, line: str):
        block = 'xds/version'
        pattern = r"\(VERSION ([A-Za-z]{3} \s*\d{1,2}, \d{4})  BUILT=(\d{8})\)"
        match = self._perform_regex(pattern, line, block=block)
        if not match:
            return
        try:
            self._xds_version = date_parser.parse(match.group(1))
            self._xds_build = match.group(2)
        except Exception as e:
            self._raise_processing_warning(line, e, block=block)

    @property
    def xds_version(self):
        if not self._xds_version:
            return None
        return datetime.date(self._xds_version)

    @property
    def xds_build(self):
        return self._xds_build

    @property
    def data(self):
        return {
            'xds': {
                'version': self.xds_version,
                'build': self.xds_build
            },
        }


class CorrectLp(XdsLpParser):

    _input_space_group_number: int = None
    _input_unit_cell: list[float] = field(default_factory=list)
    _input_friedels_law: bool = None
    _input_image_template: str = None
    _input_data_range: list[int] = field(default_factory=list)
    _input_rotation_axis: list[float] = field(default_factory=list)
    _input_oscillation_angle: float = None
    _input_wavelength: float = None
    _input_polarization_fraction: float = None
    _input_detector: str = None
    _input_no_pixels_x: int = None
    _input_no_pixels_y: int = None
    _input_pixel_size_x: float = None
    _input_pixel_size_y: float = None
    _input_beam_center_x: float = None
    _input_beam_center_y: float = None
    _input_detector_distance: float = None
    _input_beam_divergence_esd: float = None
    _input_reflecting_range_esd: float = None
    _input_maximum_error_of_spot_position: float = None
    _input_maximum_error_of_spindle_position: float = None

    _refined_indexed_spots: int = None
    _refined_deviation_spot_position: float = None
    _refined_deviation_spindle_position: float = None
    _refined_space_group_number: int = None
    _refined_unit_cell: list[float] = field(default_factory=list)
    _refined_unit_cell_esd: list[float] = field(default_factory=list)
    _refined_mosaicity: float = None
    _refined_beam_center_x: float = None
    _refined_beam_center_y: float = None
    _refined_detector_distance: float = None

    _isa: float = None
    _isa_params: list[float] = field(default_factory=list)

    _keywords_input_params = [
        ' SPACE_GROUP_NUMBER=',
        ' UNIT_CELL_CONSTANTS=',
        ' FRIEDEL\'S_LAW=',
        ' NAME_TEMPLATE_OF_DATA_FRAMES=',
        ' DATA_RANGE=',
        ' ROTATION_AXIS=',
        ' OSCILLATION_RANGE=',
        ' X-RAY_WAVELENGTH=',
        ' FRACTION_OF_POLARIZATION=',
        ' DETECTOR=',
        ' NX=',
        ' ORGX=',
        ' DETECTOR_DISTANCE=',
        ' BEAM_DIVERGENCE_E.S.D.=',
        ' REFLECTING_RANGE_E.S.D.=',
        ' MAXIMUM_ERROR_OF_SPOT_POSITION=',
        ' MAXIMUM_ERROR_OF_SPINDLE_POSITION='
    ]
    _keywords_refined_params = [
        ' REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM',
        ' STANDARD DEVIATION OF SPOT    POSITION (PIXELS)',
        ' STANDARD DEVIATION OF SPINDLE POSITION (DEGREES)',
        ' SPACE GROUP NUMBER',
        ' UNIT CELL PARAMETERS',
        ' E.S.D. OF CELL PARAMETERS',
        ' CRYSTAL MOSAICITY (DEGREES)',
        ' DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM',
        ' CRYSTAL TO DETECTOR DISTANCE (mm)'
    ]

    def __init__(self, filename: str):
        super().__init__(filename, 'CORRECT')

    def _parse_lp(self):
        super()._parse_lp()

        inside_block_input_params = False
        inside_block_refined_params = False
        inside_block_isa = False
        input_params = []
        refined_params = []
        for line in self._lp_text.splitlines():
            # Termination conditions for each block
            if inside_block_input_params:
                if line == '':
                    inside_block_input_params = False
                    continue
                input_params += [line for keyword in self._keywords_input_params if line.startswith(keyword)]
            if inside_block_refined_params:
                if line == ' THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:':
                    inside_block_refined_params = False
                    continue
                refined_params += [line for keyword in self._keywords_refined_params if line.startswith(keyword)]
            if inside_block_isa:
                if line == '':
                    inside_block_isa = False
                    continue
                self._parse_isa(line)

            # Start conditions for each block
            if line.startswith(' INPUT PARAMETER VALUES'):
                inside_block_input_params = True
                continue
            if line.startswith('  REFINEMENT OF DIFFRACTION PARAMETERS USING ALL IMAGES'):
                inside_block_refined_params = True
                continue
            if line.startswith('     a        b          ISa'):
                inside_block_isa = True
                continue

        self._parse_input_params(input_params)
        self._parse_refined_params(refined_params)
        self._is_parsed = True
        self._is_processed = True

    def _parse_input_params(self, input_params: list[str]):
        block = 'correct/input_params'
        sep = '='

        for line in input_params:
            if line.startswith(' SPACE_GROUP_NUMBER='):
                self._input_space_group_number = self._parse_line_key_value(line=line, key='SPACE_GROUP_NUMBER',
                                                                            value_type='int', sep=sep, block=block)
            elif line.startswith(' UNIT_CELL_CONSTANTS='):
                self._input_unit_cell = self._parse_line_key_values(line=line, key='UNIT_CELL_CONSTANTS',
                                                                    values_type='float', sep=sep, block=block)
            elif line.startswith(' FRIEDEL\'S_LAW='):
                self._input_friedels_law = self._parse_line_key_value(line=line, key='FRIEDEL\'S_LAW',
                                                                      value_type='boolean', sep=sep, block=block)
            elif line.startswith(' NAME_TEMPLATE_OF_DATA_FRAMES='):
                self._input_image_template = self._parse_line_key_value(line=line, key='NAME_TEMPLATE_OF_DATA_FRAMES',
                                                                          value_type='string', sep=sep, block=block)
            elif line.startswith(' DATA_RANGE='):
                self._input_data_range = self._parse_line_key_values(line=line, key='DATA_RANGE',
                                                                    values_type='int', sep=sep, block=block)
            elif line.startswith(' ROTATION_AXIS='):
                self._input_rotation_axis = self._parse_line_key_values(line=line, key='ROTATION_AXIS',
                                                                    values_type='float', sep=sep, block=block)
            elif line.startswith(' OSCILLATION_RANGE='):
                self._input_oscillation_angle = self._parse_line_key_value(line=line, key='OSCILLATION_RANGE',
                                                                            value_type='float', sep=sep, block=block)
            elif line.startswith(' X-RAY_WAVELENGTH='):
                self._input_wavelength = self._parse_line_key_value(line=line, key='X-RAY_WAVELENGTH',
                                                                    value_type='float', sep=sep, block=block)
            elif line.startswith(' FRACTION_OF_POLARIZATION='):
                self._input_polarization_fraction = self._parse_line_key_value(line=line,
                                                                               key='FRACTION_OF_POLARIZATION',
                                                                               value_type='float', sep=sep, block=block)
            elif line.startswith(' DETECTOR='):
                self._input_detector = self._parse_line_key_value(line=line, key='DETECTOR',
                                                                  value_type='string', sep=sep, block=block)
            elif line.startswith(' NX='):
                self._input_no_pixels_x, self._input_no_pixels_y, self._input_pixel_size_x, self._input_pixel_size_y = \
                    self._parse_line_keys_values(line=line, keys=['NX', 'NY', 'QX', 'QY'],
                                                 value_types=['int', 'int', 'float', 'float'], sep=sep, block=block)
            elif line.startswith(' ORGX='):
                self._input_beam_center_x, self._input_beam_center_y = \
                    self._parse_line_keys_values(line=line, keys=['ORGX', 'ORGY'], value_types=['float', 'float'],
                                                 sep=sep, block=block)
            elif line.startswith(' DETECTOR_DISTANCE='):
                self._input_detector_distance = self._parse_line_key_value(line=line, key='DETECTOR_DISTANCE',
                                                                           value_type='float', sep=sep, block=block)
            elif line.startswith(' BEAM_DIVERGENCE_E.S.D.='):
                self._input_beam_divergence_esd = self._parse_line_key_value(line=line, key='BEAM_DIVERGENCE_E.S.D.',
                                                                             value_type='float', sep=sep, block=block)
            elif line.startswith(' REFLECTING_RANGE_E.S.D.='):
                self._input_reflecting_range_esd = self._parse_line_key_value(line=line, key='REFLECTING_RANGE_E.S.D.',
                                                                              value_type='float', sep=sep, block=block)
            elif line.startswith(' MAXIMUM_ERROR_OF_SPOT_POSITION='):
                self._input_maximum_error_of_spot_position = \
                    self._parse_line_key_value(line=line, key='MAXIMUM_ERROR_OF_SPOT_POSITION', value_type='float',
                                               sep=sep, block=block)
            elif line.startswith(' MAXIMUM_ERROR_OF_SPINDLE_POSITION='):
                self._input_maximum_error_of_spindle_position = \
                    self._parse_line_key_value(line=line, key='MAXIMUM_ERROR_OF_SPINDLE_POSITION', value_type='float',
                                               sep=sep, block=block)

    def _parse_refined_params(self, refined_params: list[str]):
        block = 'correct/refined_params'
        sep = ''

        for line in refined_params:
            if line.startswith(' REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM'):
                self._refined_indexed_spots = \
                self._parse_line_key_value(line=line, key='REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM',
                                           value_type='int', sep=sep, block=block)
            elif line.startswith(' STANDARD DEVIATION OF SPOT    POSITION (PIXELS)'):
                self._refined_deviation_spot_position = \
                self._parse_line_key_value(line=line, key='STANDARD DEVIATION OF SPOT    POSITION (PIXELS)',
                                           value_type='float', sep=sep, block=block)
            elif line.startswith(' STANDARD DEVIATION OF SPINDLE POSITION (DEGREES)'):
                self._refined_deviation_spindle_position = \
                self._parse_line_key_value(line=line, key='STANDARD DEVIATION OF SPINDLE POSITION (DEGREES)',
                                           value_type='float', sep=sep, block=block)
            elif line.startswith(' SPACE GROUP NUMBER'):
                self._refined_space_group_number = \
                self._parse_line_key_value(line=line, key='SPACE GROUP NUMBER',
                                           value_type='int', sep=sep, block=block)
            elif line.startswith(' UNIT CELL PARAMETERS'):
                self._refined_unit_cell = \
                self._parse_line_key_values(line=line, key='UNIT CELL PARAMETERS',
                                           values_type='float', sep=sep, block=block)
            elif line.startswith(' E.S.D. OF CELL PARAMETERS'):
                self._refined_unit_cell_esd = \
                self._parse_line_key_values(line=line, key='E.S.D. OF CELL PARAMETERS',
                                           values_type='sci-float', sep=sep, block=block)
            elif line.startswith(' CRYSTAL MOSAICITY (DEGREES)'):
                self._refined_mosaicity = \
                self._parse_line_key_value(line=line, key='CRYSTAL MOSAICITY (DEGREES)',
                                           value_type='float', sep=sep, block=block)
            elif line.startswith(' DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM'):
                self._refined_beam_center_x, self._refined_beam_center_y = \
                self._parse_line_key_values(line=line, key='DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM',
                                            values_type='float', sep=sep, block=block)
            elif line.startswith(' CRYSTAL TO DETECTOR DISTANCE (mm)'):
                self._refined_detector_distance = \
                self._parse_line_key_value(line=line, key='CRYSTAL TO DETECTOR DISTANCE (mm)',
                                           value_type='float', sep=sep, block=block)

    def _parse_isa(self, line: str):
        pattern = r'\s*((?:-?\d*\.\d*[e|E]?[-|+]?\d*\s*){3})'
        match = self._perform_regex(pattern, line, block='correct/isa')
        if match:
            results = [float(value.strip()) for value in match.group(1).split()]
        self._isa_params = results[0:2]
        self._isa = results[2]

    @property
    def input_space_group_number(self):
        return self._input_space_group_number

    @property
    def data(self):
        data = super().data
        data['input_params'] = {
            'space_group_number': self.input_space_group_number,
            'unit_cell': self._input_unit_cell,
            'friedels_law': self._input_friedels_law,
            'image_template': self._input_image_template,
            'data_range': self._input_data_range,
            'rotation_axis': self._input_rotation_axis,
            'oscillation_angle': self._input_oscillation_angle,
            'wavelength': self._input_wavelength,
            'polarization_fraction': self._input_polarization_fraction,
            'detector': self._input_detector,
            'no_pixels_x': self._input_no_pixels_x,
            'no_pixels_y': self._input_no_pixels_y,
            'pixel_size_x': self._input_pixel_size_x,
            'pixel_size_y': self._input_pixel_size_y,
            'beam_center_x': self._input_beam_center_x,
            'beam_center_y': self._input_beam_center_y,
            'detector_distance': self._input_detector_distance,
            'beam_divergence_esd': self._input_beam_divergence_esd,
            'reflecting_range_esd': self._input_reflecting_range_esd,
            'spot_position_error_max': self._input_maximum_error_of_spot_position,
            'spindle_position_error_max': self._input_maximum_error_of_spindle_position
        }
        data['refined_params'] = {
            'indexed_spots': self._refined_indexed_spots,
            'deviation_spot_position': self._refined_deviation_spot_position,
            'deviation_spindle_position': self._refined_deviation_spindle_position,
            'space_group_number': self._refined_space_group_number,
            'unit_cell': self._refined_unit_cell,
            'unit_cell_esd': self._refined_unit_cell_esd,
            'mosaicity': self._refined_mosaicity,
            'beam_center_x': self._refined_beam_center_x,
            'beam_center_y': self._refined_beam_center_y,
            'detector_distance': self._refined_detector_distance
        }
        data['isa'] = {
            'value': self._isa,
            'params': self._isa_params
        }
        return data