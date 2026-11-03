from enum import Enum
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyFAI.geometry import Geometry

    from xtl.diffraction.images.images import Image
    from xtl.units.scattering.radial import RadialUnits


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


def get_image_frames(images: list[str]) -> list['Image']:
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

        from xtl.diffraction.images.images import Image

        image = Image()
        try:
            image.open(file=file, frame=frame, is_eager=False)
        except Exception as e:
            raise ValueError(f'Failed to open image [{i}]: {img!r}') from e
        opened_images.append(image)

    return opened_images


def get_geometry_from_header(header: str) -> 'Geometry':
    """
    Return a pyFAI Geometry object from the header of an NPX file written by
    AzimuthalCrossCorrelatorQQ_1 or Integrator.
    """
    import ast
    from pyFAI.geometry import Geometry

    config = {}
    for line in header.splitlines():
        key, _, value = line.partition(':')
        if not key.startswith('pyFAI.Geometry.'):
            continue
        key, value = key.removeprefix('pyFAI.Geometry.').strip(), value.strip()
        if key == 'detector_config':
            # repr() of a dict, or an OrderedDict in older pyFAI versions; enums are written as <Orientation.BottomRight: 3>
            value = re.sub(r'<\w+\.\w+: (-?\d+)>', r'\1', value)
            value = re.sub(r'^OrderedDict\((.*)\)$', r'\1', value)
            value = dict(ast.literal_eval(value)) if value else {}
        else:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # e.g. detector names
        config[key] = value

    geometry = Geometry()
    try:
        geometry.set_config(config)
    except TypeError:
        # Older pyFAI versions don't support detector orientation, but their implicit one is BottomRight (3)
        if config.get('detector_config', {}).pop('orientation', 3) != 3:
            raise
        geometry.set_config(config)
    return geometry


def get_radial_units_from_header(header: str) -> Optional['RadialUnits']:
    from xtl.units.scattering.radial import RadialUnits

    for line in header.splitlines():
        key, _, value = line.partition(':')
        # The prefix depends on the pyFAI version (e.g. pyFAI.integrator.azimuthal.AzimuthalIntegrator.unit) and is
        # pyFAI.AzimuthalIntegrator.unit in files written by older versions of xtl
        if key.startswith('pyFAI.') and key.endswith('AzimuthalIntegrator.unit'):
            units = value.strip().strip('()').split(',')[0].strip()  # 2D integrations: (radial, azimuthal)
            if units in ['2th_deg', '2theta', '2th']:
                return RadialUnits.TWOTHETA_DEG
            elif units in ['q_nm', 'q', 'q_nm^-1']:
                return RadialUnits.Q_NM
    return None
