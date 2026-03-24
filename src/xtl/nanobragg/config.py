from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import computed_field, PrivateAttr, model_validator

from xtl import Logger
from xtl.common.compatibility import PY310_OR_LESS
from xtl.common.options import Option, Options
from xtl.common.typed_vars import TypedList

if PY310_OR_LESS:
    class StrEnum(str, Enum):
        pass
else:
    from enum import StrEnum


logger = Logger(__name__)


class StructureOptions(Options):

    reflections: Path = \
        Option(
            path_exists=True,
            desc='Path to the reflections file (e.g., .mtz, .hkl, .cif)'
        )

    resolution: float = \
        Option(
            default=2.0,
            desc='Resolution cutoff for reflections (in Angstroms)'
        )


class CrystalShapeTransform(StrEnum):
    """
    Enumeration of shape-transform of the crystal. This is used to determine the shape of the spots in reciprocal space
    and the falloff of the intensity as a function of resolution.
    """
    SQUARE = 'square'
    """Crystal is modeled as a parallelepiped. Falloff is sinc(d) and anisotropic across h, k, l."""

    ROUND = 'round'
    """Crystal is modeled as an ellipsoid. Falloff is 3D spherical sinc-like function and isotropic across h, k, l."""

    GAUSSIAN = 'gaussian'
    """Intensity falloff is a Gaussian function of resolution."""

    GAUSSIAN_ARGCHK = 'gaussian_argchk'
    """Same as GAUSSIAN, but with some additional performance optimizations for GPUs"""

    TOPHAT = 'tophat'
    """Top-hat spot shape in reciprocal space."""


class CrystalOptions(Options):
    """
    Options about the crystal.
    """

    shape_transform: CrystalShapeTransform = \
        Option(
            default=CrystalShapeTransform.GAUSSIAN,
            desc='Shape transform of the crystal'
        )

    size: tuple[float, float, float] = \
        Option(
            default=(0., 0., 0.),
            desc='Size of the crystal in mm (x, y, z). Controls spot intensity.'
        )

    no_unit_cells: tuple[int, int, int] = \
        Option(
            default=(100, 100, 100),
            desc='Number of unit cells in the crystal along a, b, c. '
        )

    no_mosaic_domains: int = \
        Option(
            default=1,
            desc='Number of mosaic domains in the crystal'
        )

    mosaic_domains_spread: float = \
        Option(
            default=0.0,
            desc='Spread of mosaic domains in degrees'
        )

    orientation: tuple[float, float, float] = \
        Option(
            default=(0., 0., 0.),
            desc='Crystal orientation in degrees around the laboratory x, y, z reference axes.'
        )

    def randomize_orientation(self, seed: int = None) -> None:
        """
        Randomize the crystal orientation.

        :param seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.orientation = (
            np.random.random() * 360. - 180.,  # Between -180 and 180 degrees
            np.random.random() * 360. - 180.,
            np.random.random() * 360. - 180.
        )

    def model_post_init(self, context: Any, /) -> None:
        if self.size != CrystalOptions.model_fields['size'].default and \
                self.no_unit_cells != CrystalOptions.model_fields['no_unit_cells'].default:
            logger.warning('Both `size` and `no_unit_cells` are set. Only `size` will have an effect on the '
                           'simulation.')


class AmorphousMaterialOptions(Options):
    """
    Options about an amorphous material in the sample. The default `sample_size` has no thickness, so it is ignored
    during simulation.
    """

    name: str = \
        Option(
            desc='Name of the amorphous material'
        )

    density: float = \
        Option(
            gt=0.,
            desc='Density of the amorphous material in g/cm^3'
        )

    size: tuple[float, float, float] = \
        Option(
            default=(0., 0.001, 0.001),
            desc='Sample size in mm along z, y and x (thickness, height, width)'
        )

    molecular_weight: float = \
        Option(
            gt=0.,
            desc='Molecular weight of the amorphous material in g/mol'
        )


class AmorphousMaterials(StrEnum):
    WATER = 'water'
    AIR = 'air'
    HELIUM = 'helium'
    ICE = 'ice'
    NANOICE = 'nanoice'
    PARATONE_N = 'paratone-n'

    def get_options(self) -> AmorphousMaterialOptions:
        if self == AmorphousMaterials.WATER:
            # Density for water at 25 C
            #  https://en.wikipedia.org/wiki/Water
            return AmorphousMaterialOptions(
                name='water',
                density=0.997048,
                molecular_weight=18.015
            )
        elif self == AmorphousMaterials.AIR:
            # Average density/molecular weight for atmospheric air at 25 C
            #  https://en.wikipedia.org/wiki/Density_of_air
            #  https://en.wikipedia.org/wiki/Atmosphere_of_Earth#Composition
            return AmorphousMaterialOptions(
                name='air',
                density=0.001204,
                molecular_weight=28.95
            )
        elif self == AmorphousMaterials.HELIUM:
            # Density for helium at 20 C
            #  https://en.wikipedia.org/wiki/Helium
            return AmorphousMaterialOptions(
                name='helium',
                density=0.0001786,
                molecular_weight=4.0026
            )
        elif self in {AmorphousMaterials.ICE, AmorphousMaterials.NANOICE}:
            # Density for Ih ice at 93 K (-180 C)
            #  https://en.wikipedia.org/wiki/Ice#Physical_properties
            return AmorphousMaterialOptions(
                name='ice',
                density=0.9340,
                molecular_weight=18.015
            )
        elif self == AmorphousMaterials.PARATONE_N:
            # Density for Paratone-N at 15 C (best guess!)
            #  https://hamptonresearch.com/product-Parabar-10312-previously-known-as-Paratone-404.html
            return AmorphousMaterialOptions(
                name='paratone-n',
                density=0.88,
                molecular_weight=20_000.0
            )
        else:
            raise NotImplementedError(f'Material options for {self.value} not implemented')


class DetectorOptions(Options):

    type: str = \
        Option(
            choices={'simple'},
            desc='Detector type.'
        )

    distance: float = \
        Option(
            gt=0.,
            desc='Sample to detector distance in mm'
        )

    nx: int = \
        Option(
            gt=0,
            desc='Number of pixels along x-axis'
        )

    ny: int = \
        Option(
            gt=0,
            desc='Number of pixels along y-axis'
        )

    pixel_size: float = \
        Option(
            gt=0.,
            desc='Pixel size in mm (same across x and y)'
        )

    adc_offset: float = \
        Option(
            default=40.,
            desc='Analog-to-Digital converter (ADC) offset in analog-to-digital units (ADU)'
        )


class SimpleDetectorOptions(DetectorOptions):

    type: str = 'simple'


class BeamSpectrumOptions(Options):

    no_wavelengths: int = \
        Option(
            default=1,
            gt=0,
            desc='Number of wavelengths to simulate.'
        )

    wavelength_step: float = \
        Option(
            default=0.001,
            gt=0.,
            desc='Wavelength step in Angstroms'
        )

    _wavelength_mean: float | None = PrivateAttr(None)
    """Average wavelength in Angstroms"""

    _flux_total: float | None = PrivateAttr(None)
    """Total flux across all wavelengths in photons/s"""

    @computed_field(description='List of wavelengths to simulate, in Angstroms')
    def wavelength_list(self) -> list[float]:
        if self._wavelength_mean is None:
            return list()
        if self.no_wavelengths == 1:
            return [self._wavelength_mean]
        wavelengths = []
        if self.no_wavelengths % 2 == 1:
            # Odd number of wavelengths: symmetric around the mean
            for i in range(self.no_wavelengths // 2 + 1):
                if i == 0:
                    wavelengths.append(self._wavelength_mean)
                else:
                    wavelengths.append(self._wavelength_mean - i * self.wavelength_step)
                    wavelengths.append(self._wavelength_mean + i * self.wavelength_step)
        else:
            # Even number of wavelengths: symmetric around the mean but no wavelength at the mean
            for i in range(self.no_wavelengths // 2):
                wavelengths.append(self._wavelength_mean - (i + 0.5) * self.wavelength_step)
                wavelengths.append(self._wavelength_mean + (i + 0.5) * self.wavelength_step)
        return wavelengths

    @computed_field(description='List of fluxes for each wavelength in photons/s')
    def flux_list(self) -> list[float]:
        line_flux = self._flux_total / self.no_wavelengths if self._flux_total is not None else None
        return [line_flux] * self.no_wavelengths if line_flux is not None else []


class BeamOptions(Options):
    model_config = Options.model_config | {'extra': 'allow'}

    wavelength: float = \
        Option(
            default=1.,
            gt=0.,
            desc='X-ray wavelength in Angstroms'
        )

    flux: float = \
        Option(
            default=1e12,
            gt=0.,
            desc='X-ray flux in photons/s'
        )

    size: float = \
        Option(
            default=0.02,
            gt=0.,
            desc='Beam size in mm (same across x and y)'
        )

    _spectrum: BeamSpectrumOptions | None = PrivateAttr(None)

    @computed_field
    @property
    def spectrum(self) -> BeamSpectrumOptions:
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value: BeamSpectrumOptions | dict):
        if isinstance(value, dict):
            value = BeamSpectrumOptions(**value)
        elif not isinstance(value, BeamSpectrumOptions):
            raise ValueError('spectrum must be a BeamSpectrumOptions instance or a dict')
        self._spectrum = value

    @model_validator(mode='after')
    def _update_spectrum(self, context: Any, /) -> 'BeamOptions':
        spectrum = self.model_extra.pop('spectrum', {})
        if self._spectrum is None:
            self.spectrum = BeamSpectrumOptions(**spectrum)
        self._spectrum._wavelength_mean = self.wavelength
        self._spectrum._flux_total = self.flux
        return self


class CrystalOrientationMode(StrEnum):
    STATIONARY = 'stationary'
    RANDOM = 'random'


class ExperimentOptions(Options):

    exposure: float = \
        Option(
            default=0.01,
            gt=0.,
            desc='Exposure time in seconds'
        )

    oscillation_range: float = \
        Option(
            default=0.,
            ge=0.,
            desc='Oscillation range per frame in degrees. If 0, simulates a still image.'
        )

    crystal_orientation_mode: CrystalOrientationMode = \
        Option(
            default=CrystalOrientationMode.RANDOM,
            desc='Crystal orientation between frames.'
        )


class SimulationOptions(Options):

    no_images: int = \
        Option(
            default=1,
            gt=0,
            desc='Number of images to simulate.'
        )

    oversample: int = \
        Option(
            default=1,
            gt=0,
            desc='Number of subpixels to compute per pixel'
        )

    spot_scale: float = \
        Option(
            default=1.,
            gt=0.,
            desc='Scale factor for spot intensities'
        )

    seed: int = \
        Option(
            default_factory=lambda: np.random.randint(0, 10_000),
            desc='Random seed for reproducibility.'
        )

    mosaic_seed: int = \
        Option(
            default_factory=lambda: np.random.randint(0, 10_000),
            desc='Random seed for mosaic domain generation. Only used if `crystal.mosaic_domains_spread` > 0.'
        )


class NanoBraggOptions(Options):

    structure: StructureOptions

    crystal: CrystalOptions = \
        Option(
            default_factory=CrystalOptions,
            desc='Physical properties of the crystal'
        )

    amorphous_content: TypedList[AmorphousMaterialOptions] = \
        Option(
            default_factory=lambda: TypedList(AmorphousMaterialOptions),
            desc='List of amorphous materials in the sample, if any'
        )

    detector: DetectorOptions

    beam: BeamOptions

    experiment: ExperimentOptions = \
        Option(
            default_factory=ExperimentOptions,
            desc='Experimental parameters such as exposure time and oscillation range'
        )

    simulation: SimulationOptions = \
        Option(
            default_factory=SimulationOptions,
            desc='Simulation parameters such as number of images and oversampling'
        )
