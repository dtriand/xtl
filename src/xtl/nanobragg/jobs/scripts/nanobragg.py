"""
Driver script for simulating X-ray diffraction patterns using `simtbx.nanoBragg`.

This script provides a high-level interface for configuring and running nanoBragg simulations, including support for
background calculation from amorphous components, random crystal orientations, and GPU acceleration if available. The
simulated patterns can be saved in CBF format, as NumPy arrays, and/or plotted as PNG files.

Note that this script is intended to be run with a `cctbx` environment and not with the `xtl` environment.
"""
import argparse
from collections import ChainMap
from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Generic, Mapping, MutableMapping, TypeVar

try:
    import cctbx
    import dxtbx
    import iotbx
    from iotbx.reflection_file_reader import any_reflection_file
    import simtbx
    from scitbx.array_family import flex
    from simtbx.nanoBragg import shapetype
    from simtbx.nanoBragg.sim_data import SimData
    from simtbx.nanoBragg.utils import get_xray_beams
    import matplotlib.pyplot as plt
    import numpy as np
except (ImportError, ModuleNotFoundError) as e_:
    raise ImportError('Error importing required modules. '
                      'Ensure that simtbx.nanoBragg and its dependencies are installed.') from e_


logger = logging.getLogger(__name__)


K = TypeVar('K')
V = TypeVar('V')

class DeepChainMap(Generic[K, V], ChainMap[K, V]):
    """
    A recursive subclass of ChainMap

    Reproduced from: https://github.com/neutrinoceros/deep_chainmap
    """

    def __getitem__(self, key: K) -> V:
        submaps = [mapping for mapping in self.maps if key in mapping]
        if not submaps:
            return self.__missing__(key)
        if isinstance(submaps[0][key], Mapping):
            return DeepChainMap(*(submap[key] for submap in submaps))
        return super().__getitem__(key)

    def to_dict(self) -> dict[K, V]:
        d: dict[K, V] = {}
        for mapping in reversed(self.maps):
            self._depth_first_update(d, mapping)
        return d

    @classmethod
    def _depth_first_update(cls, target: dict[K, V], source: Mapping[K, V]) -> None:
        for key, val in source.items():
            if not isinstance(val, Mapping):
                target[key] = val
                continue

            if key not in target:
                target[key] = {}
            cls._depth_first_update(target[key], val)


class Simulator:

    _default_config = {
        'structure': {
            'resolution': 2.0,
        },
        'crystal': {
            'shape': 'gauss',
            'size': [0., 0., 0.],
            'no_unit_cells': [100, 100, 100],
            'no_mosaic_domains': 1,
            'mosaic_domain_spread': 0.,
        },
        'amorphous_content': [
            {
                'material': 'water',
                'density': 1.,
                'sample_size': [0., 0.001, 0.001],
                'molecular_weight': 18.
            }
        ],
        'detector': {
            'type': 'simple',
            'nx': 1024,
            'ny': 1024,
            'distance': 200.,
            'pixel_size': 0.075,
            'adc_offset': 40.,
        },
        'beam': {
            'wavelength': 1.,
            'flux': 1e12,
            'size': 0.02,
            'spectrum': {
                'no_wavelengths': 1,
                'wavelength_step': 0.001,
                'wavelength_list': [1.],
                'flux_list': [1e12]
            }
        },
        'experiment': {
            'exposure': 1e-2,
            'oscillation_range': 0.,
            'crystal_orientation_mode': 'stationary'
        },
        'simulation': {
            'no_images': 1,
            'oversample': 1,
            'spot_scale': 1,
            'include_noise': False,
            'include_background': False,
            'seed': -1,
            'mosaic_seed': -1
        }
    }

    def __init__(self, config: str | Path, **kwargs):
        """
        Wrapper around `simtbx.nanoBragg` to simulate X-ray diffraction patterns from nanocrystals in absolute scale.

        :param config: Path to JSON configuration file containing simulation parameters. See `_default_config` for
            expected structure.
        :param kwargs: Optional keyword arguments to override configuration parameters. These will be merged with the
            configuration file parameters, with `kwargs` taking precedence.
        """
        logger.info('Loading configuration... ')
        self._config = DeepChainMap(
            kwargs,
            self._load_config(config),
            deepcopy(self._default_config)
        )

        logger.info('Creating detector... ')
        self._detector = self._get_detector()
        logger.info('Creating beam... ')
        self._beam = self._get_beam()
        logger.info('Creating wavelength spectrum... ')
        self._wavelength_spectrum = self._get_wavelength_spectrum()
        logger.info('Loading structure factors... ')
        self._hkl = self._get_structure_factors()

        logger.info('Initializing nanoBragg simulator... ')
        self._simulator = self._get_nanobragg_simulator()

        if self.has_cuda:
            logger.info('CUDA available. Using GPU acceleration.')
        else:
            logger.info('CUDA not available. Using CPU simulation.')

        self._image = None

    @staticmethod
    def _load_config(config: str | Path) -> dict[str, Any]:
        """
        Load the configuration from a JSON file and return it as a dictionary.

        :param config: Path to JSON configuration file
        :return: Configuration parameters as a dictionary
        """
        config = Path(config)
        if not config.exists():
            raise FileNotFoundError(f'Configuration file not found: {config}')
        if config.suffix != '.json':
            raise ValueError(f'Configuration file must be a JSON file: {config}')

        try:
            options = json.loads(config.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f'Error parsing JSON configuration file: {config}') from e
        if not isinstance(options, dict):
            raise ValueError(f'Top-level JSON object must be a dictionary: {config}')

        return options

    @property
    def config(self) -> dict[str, Any]:
        """
        Get the configuration parameters as a dictionary.

        :return: Configuration parameters as a dictionary
        """
        return deepcopy(self._config.to_dict())

    @property
    def has_cuda(self) -> bool:
        """
        Check if the nanoBragg simulator has CUDA support for GPU acceleration.
        """
        if not hasattr(self, '_simulator'):
            raise RuntimeError('Simulator not initialized. Cannot determine CUDA availability.')
        return hasattr(self._simulator, 'add_nanoBragg_spots_cuda')

    def _get_detector(self) -> dxtbx.model.Detector:
        """
        Create a `dxtbx.model.Detector` object based on the configuration parameters.
        """
        config = self._config['detector']
        if config['type'] == 'simple':
            return SimData.simple_detector(
                detector_distance_mm=config['distance'],
                pixel_size_mm=config['pixel_size'],
                image_shape=(config['nx'], config['ny'])
            )
        else:
            raise ValueError(f'Unsupported detector type: {config["type"]}')

    def _get_beam(self) -> dxtbx.model.Beam:
        """
        Create a `dxtbx.model.Beam` object based on the configuration parameters.
        """
        config = self._config['beam']
        return dxtbx.model.BeamFactory.simple(wavelength=config['wavelength'])

    def _get_wavelength_spectrum(self) -> Any:
        """
        Create a flex_Beam array of wavelength components based on the configuration parameters.
        """
        if not hasattr(self, '_beam'):
            self._beam = self._get_beam()
        config = self._config['beam']['spectrum']
        spectrum = [(w, f) for w, f in zip(config['wavelength_list'], config['flux_list'])]
        return get_xray_beams(spectrum, self._beam)

    def _get_structure_factors(self) -> 'cctbx.miller.array':
        """
        Greate a `cctbx.miller.array` of structure factors based on the reflection file specified in the configuration
        parameters.
        """
        file = Path(self._config['structure']['reflections']).expanduser().resolve()
        if not file.exists():
            raise FileNotFoundError(f'Reflection file not found: {file}')

        try:
            reader = any_reflection_file(str(file))
            hkl = reader.as_miller_arrays()[0]
        except Exception as e:
            raise ValueError(f'Error reading reflection file: {file}') from e

        return hkl.amplitudes()

    def _get_crystal_shape(self) -> shapetype:
        """
        Get the crystal shape type based on the configuration parameters.
        """
        shapes = {
            'square': shapetype.Square,
            'round': shapetype.Round,
            'gauss': shapetype.Gauss,
            'gauss_star': shapetype.Gauss_star,
            'gauss_argchk': shapetype.Gauss_argchk,
            'tophat': shapetype.Tophat
        }
        return shapes.get(self._config['crystal']['shape'], shapes.get('gauss'))

    def _get_nanobragg_simulator(self) -> simtbx.nanoBragg.nanoBragg:
        """
        Get an initialized `simtbx.nanoBragg.nanoBragg` simulator object based on the configuration parameters.
        """
        simulator = simtbx.nanoBragg.nanoBragg(
            self._detector, self._beam
        )
        simulator.xray_beams = self._wavelength_spectrum
        simulator.Fhkl = self._hkl

        # Set crystal properties
        crystal = self._config['crystal']
        simulator.xtal_shape = self._get_crystal_shape()
        simulator.xtal_size_mm = crystal['size']
        simulator.Ncells_abs = crystal['no_unit_cells']
        # NB: Mosaic spread must be initialized prior to mosaic domains, otherwise it won't work
        simulator.mosaic_spread_deg = crystal['mosaic_domain_spread']
        simulator.mosaic_domains = crystal['no_mosaic_domains']

        # Set beam properties
        beam = self._config['beam']
        simulator.flux = beam['flux']
        simulator.beamsize_mm = beam['size']

        # Set detector properties
        simulator.adc_offset_adu = self._config['detector']['adc_offset']

        # Set experiment properties
        experiment = self._config['experiment']
        simulator.exposure_s = experiment['exposure']
        simulator.osc_deg = experiment['oscillation_range']

        # Set simulation properties
        simulation = self._config['simulation']
        simulator.oversample = simulation['oversample']
        simulator.spot_scale = simulation['spot_scale']
        simulator.seed = simulation['seed']
        simulator.mosaic_seed = simulation['mosaic_seed']

        return simulator

    @staticmethod
    def _load_stol_file(file: str | Path) -> np.ndarray:
        """
        Load a STOL file.

        :param file: Path to STOL file
        :return: STOL data as a NumPy array with shape (N, 2), where the first column is 1/2d (1/A) and the second
            column is the corresponding structure factor amplitude (e/A^3).
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f'STOL file not found: {file}')
        try:
            return np.loadtxt(str(file), comments='#')
        except Exception as e:
            raise ValueError(f'Error reading STOL file: {file}') from e

    def _get_radially_average_structure_factors(self, material: str | Path) -> flex.vec2_double:
        """
        Get the radially averaged structure factors for an amorphous material, either by name or from a custom STOL
        file.

        :param material: Name of the material (e.g. 'water') or path to a STOL file
        :return: Radially averaged structure factors as a flex.vec2_double array, where the first column is 1/2d (1/A)
            and the second column is the corresponding structure factor amplitude (e/A^3).
        """
        stol_dir = Path(__file__).parent / 'stol'
        materials = { file.name for file in stol_dir.glob('*.txt') }

        if material in materials:
            logger.info(f'Loading radial structure factors for {material}... ')
            stol = self._load_stol_file(stol_dir / f'{material}.txt')
        else:
            file = Path(material).expanduser().resolve()
            if file.exists():
                logger.info(f'Loading radial structure factors from {file}... ')
                stol = self._load_stol_file(file)
            else:
                raise ValueError(f'No radial structure factors found for material: {material}')

        return flex.vec2_double(stol.tolist())

    def calculate_background(self):
        """
        Calculate the background contribution from amorphous components based on the configuration parameters
        and add it to the simulator. If multiple components are specified, their contributions will be summed together.

        This is done on the CPU, as the nanoBragg GPU implementation does not currently support background calculation.
        """
        amorphous = self._config['amorphous_content']
        for amorph in amorphous:
            self._simulator.Fbg_vs_stol = self._get_radially_average_structure_factors(amorph['material'])
            self._simulator.amorphous_density_gcm3 = amorph['density']
            self._simulator.amorphous_sample_size_mm = amorph['sample_size']
            self._simulator.amorphous_molecular_weight_Da = amorph['molecular_weight']

            self._simulator.add_background()

    def simulate(self, n: int = None, output: str | Path = None, seed: int = None, mosaic_seed: int = None,
                 save_cbf: bool = True, save_npy: bool = False, plot: bool = False) -> None:
        """
        Simulate patterns and save them to disk.

        :param n: Number of images to simulate (default: value from config)
        :param output: Directory to save the simulated images (default: current working directory)
        :param seed: Random seed for reproducibility (default: value from config)
        :param mosaic_seed: Random seed for mosaic domain generation (default: value from config)
        :param save_cbf: Whether to save the simulated images in CBF format (default: True)
        :param save_npy: Whether to save the simulated images as NumPy arrays (default: False)
        :param plot: Whether to save plots of the simulated images as PNG files (default: False)
        """
        n = n or self._config['simulation']['no_images']
        if n <= 0:
            raise ValueError(f'Number of images to simulate must be a positive integer: {n}')

        seed = seed or self._config['simulation']['seed']
        mosaic_seed = mosaic_seed or self._config['simulation']['mosaic_seed']
        orientation = self._config['experiment']['crystal_orientation_mode'].lower()
        processor = 'GPU' if self.has_cuda else 'CPU'

        logger.debug('Setting random seeds: seed=%(seed)d, mosaic_seed=%(seed)d',
                     {'seed': seed, 'mosaic_seed': mosaic_seed})
        self._simulator.seed = seed
        self._simulator.mosaic_seed = mosaic_seed

        # Prepare output directory
        output = Path(output or Path.cwd())
        if not output.exists():
            logger.info('Creating output directory: %(output)s', {'output': output})
            output.mkdir(parents=True, exist_ok=True, mode=0o750)

        # Number of digits for file name padding
        digits = max(len(str(abs(n))), 4)

        # Log timings
        timings = {
            'frame': [], 'background': [], 'noise': [], 'save_cbf': [], 'save_npy': [],
            'plot': [], 'save_png': []
        }

        logger.info(f'Simulating %(n)d frames with %(orientation)s crystal orientation on the %(processor)s... ',
                    {'n': n, 'orientation': orientation, 'processor': processor})
        t0 = time.time()

        # Background calculation is always done on the CPU
        # We assume constant background across all images, so we only calculate it once
        logger.info('Calculating background... ')
        self.calculate_background()
        t_bkg = time.time()
        timings['background'].append(t_bkg - t0)

        # Spot calculation
        for i in range(n):
            logger.info('Simulating frame %(current)d/%(total)d... ', {'current': i + 1, 'total': n})
            t1 = time.time()

            if orientation == 'random' or i == 0:
                logger.debug('Randomizing crystal orientation...')
                self._simulator.randomize_orientation()

            if self.has_cuda:
                self._simulator.add_nanoBragg_spots_cuda()
            else:
                self._simulator.add_nanoBragg_spots()
            t_frame = time.time()
            timings['frame'].append(t_frame - t1)

            # Noise calculation is always done on the CPU
            if self._config['simulation']['include_noise']:
                logger.debug('Adding noise... ')
                self._simulator.add_noise()
                t_noise = time.time()
                timings['noise'].append(t_noise - t_frame)

            # Save the simulated image
            frame = output / f'image_{i + 1:0{digits}d}'
            if save_cbf:
                logger.debug('Saving CBF file... ')
                t2 = time.time()
                self._simulator.to_cbf(str(frame.with_suffix('.cbf')))
                t_cbf = time.time()
                timings['save_cbf'].append(t_cbf - t2)
            if save_npy:
                logger.debug('Saving NumPy file... ')
                t2 = time.time()
                self._image = self._simulator.raw_pixels.as_numpy_array()
                np.save(frame.with_suffix('.npy'), self._image)
                t_npy = time.time()
                timings['save_npy'].append(t_npy - t2)

            # Plot the simulated image
            if plot:
                if self._image is None:
                    logger.debug('Converting raw pixels to NumPy array for plotting... ')
                    self._image = self._simulator.raw_pixels.as_numpy_array()
                t2 = time.time()
                fig, _ = self.plot_image()
                t_plot = time.time()
                fig.savefig(frame.with_suffix('.png'), dpi=300)
                t_png = time.time()
                plt.close(fig)
                timings['plot'].append(t_plot - t2)
                timings['save_png'].append(t_png - t_plot)

        t_end = time.time()
        logger.info(f'Simulation completed in {(t_end - t0):,.2f} seconds.')

    def plot_image(self, vmin: float = None, vmax: float = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the last simulated image.

        :param vmin: Minimum intensity for color scaling (default: mean - std)
        :param vmax: Maximum intensity for color scaling (default: mean + 3*std)
        :return: Matplotlib figure and axes objects
        :raise RuntimeError: If no image data is available for plotting (i.e. if `simulate` has not been called yet)
        """
        if self._image is None:
            raise RuntimeError('No image data available for plotting. Simulate an image first.')

        if not vmin and not vmax:
            mean = np.nanmean(self._image)
            std = np.nanstd(self._image)
            vmin = mean - std
            vmax = mean + 3 * std

        beam_center = [c / self._config['detector']['pixel_size'] for c in self._simulator.beam_center_mm]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(self._image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        ax.scatter(*beam_center, s=70, marker='X', facecolor='red', edgecolors='white')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        ax.set_title('Simulated Diffraction Pattern')
        fig.colorbar(im, ax=ax, label='Intensity (ph)')
        fig.tight_layout()

        return fig, ax


def cli():
    """
    Command-line interface for running the nanoBragg simulator.
    """
    parser = argparse.ArgumentParser(
        description='Simulate X-ray diffraction patterns using simtbx.nanoBragg.',
        epilog=r'</> with <3 by \033[3;35m_dtriand\033[0m',
    )
    parser.add_argument(
        '-i', '--config',
        type=Path,
        required=True,
        help='Path to JSON configuration file containing simulation parameters.'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path.cwd(),
        help='Directory to save the simulated images (default: current working directory).'
    )
    parser.add_argument(
        '--fmt',
        type=str,
        choices=['cbf', 'npy', 'png'],
        nargs='+',
        default=['cbf'],
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging.'
    )

    args = parser.parse_args()
    if not args.config.exists():
        parser.error(f'Configuration file not found: {args.config}')

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Run the simulation
    simulator = Simulator(config=args.config)
    simulator.simulate(
        output=args.output,
        save_cbf='cbf' in args.fmt,
        save_npy='npy' in args.fmt,
        plot='png' in args.fmt
    )


if __name__ == '__main__':
    cli()
