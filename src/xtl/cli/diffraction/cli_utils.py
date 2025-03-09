from enum import Enum
import re
from typing import Optional

from pyFAI.geometry import Geometry
from pyFAI.detectors import Detector

from xtl.diffraction.images.images import Image


class ZScale(Enum):
    LINEAR = 'linear'
    LOG = 'log'
    SQRT = 'sqrt'


class IntegrationErrorModel(Enum):
    NONE = 'None'
    POISSON = 'poisson'
    VARIANCE = 'variance'


class IntegrationRadialUnits(Enum):
    TWOTHETA_DEG = '2th_deg'
    Q_NM = 'q_nm'


def get_image_frames(images: list[str]) -> list[Image]:
    opened_images = []
    for i, img in enumerate(images):
        parts = img.split(':')
        if len(parts) == 1:
            file = parts[0]
            frame = 0
        elif len(parts) == 2:
            file = parts[0]
            if parts[1].isnumeric():
                frame = int(parts[1])
            else:
                raise ValueError(f'Invalid frame index for image [{i}]: {parts[1]!r}')
        else:
            raise ValueError(f'Invalid image format for image [{i}]: {img!r}')

        image = Image()
        try:
            image.open(file=file, frame=frame, is_eager=False)
        except Exception as e:
            raise ValueError(f'Failed to open image [{i}]: {img!r}') from e
        opened_images.append(image)

    return opened_images


def get_geometry_from_header(header: str) -> Geometry:
    """
    Return a pyFAI Geometry object from the header of an NPX file written by
    AzimuthalCrossCorrelatorQQ_1 or Integrator.
    """
    lines = []
    for line in header.splitlines():
        if line.startswith('pyFAI.Geometry'):
            contents = line.replace('pyFAI.Geometry.', '')
            key, value = contents.split(':')
            lines.append((key.strip(), value.strip()))

    kwargs = {}
    detector_config = {}
    for key, value in lines:
        if key in ['poni_version']:
            continue
        elif key == 'detector_config':
            if value.startswith('OrderedDict'):
                pixel1 = float(re.search(r"'pixel1', ([\d.e-]+)\)", value).group(1))
                pixel2 = float(re.search(r"'pixel2', ([\d.e-]+)\)", value).group(1))
                max_shape = list(map(int, re.search(r"'max_shape', \[(\d+), (\d+)\]", value).groups()))
                detector_config = {'pixel1': pixel1, 'pixel2': pixel2, 'max_shape': max_shape}
                continue
        try:
            value = float(value)
        except ValueError:
            pass
        kwargs[key] = value

    if detector_config:
        detector_config['detector'] = kwargs['detector']
        detector = Detector.from_dict(detector_config)
        kwargs['detector'] = detector
    return Geometry(**kwargs)


def get_radial_units_from_header(header: str) -> Optional[IntegrationRadialUnits]:
    for line in header.splitlines():
        if line.startswith('pyFAI.AzimuthalIntegrator.unit'):
            units = line.split(':')[-1].strip()
            if units in ['2th_deg', '2theta', '2th']:
                return IntegrationRadialUnits.TWOTHETA_DEG
            elif units in ['q_nm', 'q', 'q_nm^-1']:
                return IntegrationRadialUnits.Q_NM
    return None


def units_repr(units: IntegrationRadialUnits) -> Optional[tuple[str, str]]:
    if units == IntegrationRadialUnits.TWOTHETA_DEG:
        return '2\u03b8', '\u00b0'
    elif units == IntegrationRadialUnits.Q_NM:
        return 'q', 'nm\u207B\u00B9'
    return None
