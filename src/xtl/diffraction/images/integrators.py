from datetime import datetime
from functools import partial
from pathlib import Path

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm, to_rgba, LinearSegmentedColormap
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as _AzimuthalIntegrator
from pyFAI.containers import Integrate1dResult, Integrate2dResult

from .images import Image
from xtl.io.npx import NpxFile


class _Integrator:

    UNITS_2THETA_DEGREES = ['2theta', '2th', 'tth', 'ttheta', '2\u03B8',
                            '2theta_deg', '2th_deg', 'tth_deg', 'ttheta_deg']
    UNITS_2THETA_RADIANS = ['2theta_rad', '2th_rad', 'tth_rad', 'ttheta_rad']
    UNITS_Q_NM = ['q', 'q_nm', 'q_nm^-1']
    UNITS_Q_ANGSTROM = ['q_A', 'q_A^-1']
    SUPPORTED_UNITS_RADIAL = UNITS_2THETA_DEGREES + UNITS_Q_NM

    def __init__(self, image: Image):
        """
        Base integrator class.
        :param image: ``xtl.diffraction.images.image.Image`` instance to integrate from
        :raises ValueError: When a non ``Image`` instance is provided in ``image``
        :raises Exception: When the ``Image`` instance does not contain a pyFAI geometry
        """
        if not isinstance(image, Image):
            raise ValueError(f'Must be an Image instance, not {image.__class__.__name__}')
        self.image = image
        self.image.check_geometry()
        self.points_radial: int = None
        self.units_radial: str = '2theta_deg'
        self.units_radial_repr: str = '2\u03b8 (\u00b0)'
        self.masked_pixels_value: float = np.nan
        self.error_model: str = None
        self._integrator: _AzimuthalIntegrator = None
        self._is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """
        The initialization status of the integrator.
        :return: ``True`` if initialized, ``False`` if not.
        """
        return self._is_initialized

    def check_initialized(self) -> None:
        """
        Check if the integrator is ready for performing an integration.
        :raises Exception: When it hasn't been initialized
        :return:
        """
        if not self.is_initialized:
            raise Exception('No integration settings set. Run initialize() method first.')

    def _max_radial_pixels(self) -> int:
        """
        Number of pixels between 2\u03b8 = 0\u00b0 and 2\u03b8_max
        :return:
        """
        # Choose the longest distance (in pixels) from the beam center to detector edges
        x0, y0 = self.image.beam_center  # this is basically distance from the bottom left corner of the detector
        nx, ny = self.image.dimensions
        x1 = nx - x0  # and this is distance from the upper right corner of the detector
        y1 = ny - y0
        return int(max(x0, x1, y0, y1))


class AzimuthalIntegrator1D(_Integrator):

    def __init__(self, image: Image):
        super().__init__(image)
        self._results: Integrate1dResult = None

    def initialize(self, points_radial: int = None, units_radial: str = '2theta', masked_pixels_value: float = np.nan,
                   error_model: str = None) -> None:
        """
        Initialize settings for integrator. If ``points_radial`` is not specified, then the maximum number of pixels
        along the radial axis is chosen.

        :param int points_radial: Number of points along the radial axis
        :param str units_radial: Units for the radial axis (``'2theta'`` or ``'q'``)
        :param float masked_pixels_value: Value for masked or ignored pixels (default: ``numpy.nan``)
        :param str error_model: Error model for calculating intensities uncertainties (``None``, ``'poisson'``, or
                                ``'azimuthal'``)
        :raises ValueError: When an unsupported unit is provided to ``units_radial``
        :raises ValueError: When an unsupported model is provided to ``error_model``
        :return: None
        """
        if points_radial is None:
            # Automatic guess for number of points
            self.points_radial = self._max_radial_pixels()
        else:
            self.points_radial = int(points_radial)

        if units_radial in self.UNITS_2THETA_DEGREES:
            self.units_radial = '2th_deg'
            self.units_radial_repr = '2\u03b8 (\u00b0)'
        elif units_radial in self.UNITS_Q_NM:
            self.units_radial = 'q_nm^-1'
            self.units_radial_repr = 'Q (nm$^{-1}$)'
        else:
            raise ValueError(f'Unknown units: \'{units_radial}\'. Choose one from: '
                             f'{", ".join(self.SUPPORTED_UNITS_RADIAL)}')

        self.masked_pixels_value = float(masked_pixels_value)
        if error_model not in [None, 'poisson', 'azimuthal']:
            raise ValueError(f'Unknown error model. Must be one of: None, \'poisson\', \'azimuthal\'')
        self.error_model = error_model

        self._integrator = _AzimuthalIntegrator()
        self._integrator.set_config(self.image.geometry.get_config())  # pyFAI bug: to properly load detector info
        self._is_initialized = True

    def integrate(self, check: bool = True, keep: bool = True, **kwargs) -> Integrate1dResult:
        """
        Perform 1D azimuthal integration using ``pyFAI.AzimuthalIntegrator.integrate1d_ng`` on the ``Image``,
        excluding the regions defined in ``Image.mask``. Integration settings are defined from the initialize() method.
        This method is just a wrapper around the pyFAI method, without any additional initialization. The default
        pyFAI integration method is ``('bbox', 'csr', 'cython')`` (aka ``'csr'``).

        :param bool check: Check whether the integrator has already been initialized
        :param bool keep: Whether to store the integration results or return them
        :param kwargs: Any of the following pyFAI arguments: ``correctSolidAngle``, ``variance``, ``radial_range``,
                       ``azimuth_range``, ``delta_dummy``, ``polarization_factor``, ``dark``, ``flat``, ``method``,
                       ``safe``, ``normalization_factor``, ``metadata``. The default pyFAI values are chosen if not
                       provided.
        :raises Exception: When the integrator hasn't been initialized and ``check=True``
        :return: Integrate1dResult
        """
        if check:
            self.check_initialized()

        correctSolidAngle = kwargs.get('correctSolidAngle', True)
        variance = kwargs.get('variance', None)
        radial_range = kwargs.get('radial_range', None)
        azimuth_range = kwargs.get('azimuth_range', None)
        delta_dummy = kwargs.get('delta_dummy', None)
        polarization_factor = kwargs.get('polarization_factor', None)
        dark = kwargs.get('dark', None)
        flat = kwargs.get('flat', None)
        method = kwargs.get('method', 'csr')
        safe = kwargs.get('safe', True)
        normalization_factor = kwargs.get('normalization_factor', 1.0)
        metadata = kwargs.get('metadata', None)

        result = self._integrator.integrate1d_ng(data=self.image.data, npt=self.points_radial, filename=None,
                                                 correctSolidAngle=correctSolidAngle, variance=variance,
                                                 error_model=self.error_model, radial_range=radial_range,
                                                 azimuth_range=azimuth_range, mask=~self.image.mask.data,
                                                 dummy=self.masked_pixels_value, delta_dummy=delta_dummy,
                                                 polarization_factor=polarization_factor, dark=dark, flat=flat,
                                                 method=method, unit=self.units_radial, safe=safe,
                                                 normalization_factor=normalization_factor, metadata=metadata)
        # Note that the mask needs to be inverted! (for pyFAI True means mask that pixel)

        if keep:
            self._results = result
        return result

    @property
    def results(self):
        return self._results

    def _get_file_header(self) -> str:
        """
        Prepare a header for exporting integration results to file
        :return:
        """
        header = f'1D azimuthal integration using {self.__class__.__module__}.{self.__class__.__name__}\n'
        header += f'Integration performed on: {datetime.now()}\n'
        header += f'Filename: {self.image.file.resolve()}\n'
        header += f'Frame: {self.image.frame}\n\n'

        header += 'Geometry options:\n'
        header += '\n'.join(f'pyFAI.Geometry.{key}: {value}' for key, value in
                            self.image.geometry.get_config().items()) + '\n\n'

        header += 'Integration options:\n'
        for key in ['unit', 'error_model', 'method_called', 'method', 'compute_engine', 'has_mask_applied',
                    'has_flat_correction', 'has_dark_correction', 'has_solidangle_correction', 'polarization_factor',
                    'normalization_factor', 'metadata']:
            header += f'pyFAI.AzimuthalIntegrator.{key}: {getattr(self.results, key)}\n'
        header += f'pyFAI.AzimuthalIntegrator.npt_rad: {self.points_radial}\n'

        if self.results.sigma is not None:
            header += f'\nColumns: {self.units_radial_repr}, intensity (a.u.), sigma (a.u.)'
        else:
            header += f'\nColumns: {self.units_radial_repr}, intensity (a.u.)'
        return header

    def save(self, filename: str | Path, overwrite: bool = False, header: bool = True, delimiter: str = None,
             newline: str = None, comments: str = None, fmt: str = None, encoding: str = None) -> None:
        """
        Save integration results in a three-column, space-delimited .xye file (radial angle, intensity, intensity
        uncertainty). If no errors have been calculated during integration, the output file will contain only two
        columns.

        :param str | Path filename: Output filename. If the file extension is not .xye, it will be replaced.
        :param bool overwrite: Whether to overwrite the output file if it already exists.
        :param bool header: Whether to include metadata about the integration in the file header
        :param str delimiter: Delimiter between columns (default: ``'    '``)
        :param str newline: Newline character (default: ``'\n'``)
        :param str comments: Comments character (default: ``'# '``)
        :param str fmt: Formatting string (default: ``'%-15.15s'``)
        :param str encoding: File encoding (default: ``'utf-8'``)
        :return:
        """
        if self.results is None:
            raise Exception('No results to save. Run integrate() first.')

        file = Path(filename).with_suffix('.xye')
        if file.exists() and not overwrite:
            raise FileExistsError(f'File {file} already exists!')
        file.unlink(missing_ok=True)

        intensities, radial, sigma = self.results.intensity, self.results.radial, self.results.sigma
        data = np.array([radial, intensities]).T if sigma is None else np.array([radial, intensities, sigma]).T

        header = self._get_file_header() if header else ''
        if delimiter is None:
            delimiter = '    '
        if newline is None:
            newline = '\n'
        if comments is None:
            comments = '# '
        if fmt is None:
            fmt = '%-15.15s'
        if encoding is None:
            encoding = 'utf-8'
        np.savetxt(file, data, header=header, delimiter=delimiter, newline=newline, comments=comments, fmt=fmt,
                   encoding=encoding)

    def plot(self, ax: plt.Axes = plt.gca(), fig: plt.Figure = plt.gcf(), xlabel: str = None, ylabel: str = None,
             title: str = None, label: str = None, xscale: str = None, yscale: str = None,
             errors: bool = False) -> tuple[plt.Axes, plt.Figure]:
        """
        Prepare a plot of the 1D integration results. ``plt.show()`` must be called separately to display the plot.

        :param matplotlib.axes.Axes ax: Axes instance to draw into
        :param matplotlib.figure.Figure fig: Figure instance to draw into
        :param str xlabel: x-axis label (default: integration radial units)
        :param str ylabel: y-axis label (default: ``'Intensity (arbitrary units)'``)
        :param str title: Plot title (default: ``'1D azimuthal integration'``)
        :param str label: Line plot label for legend (default: ``None``)
        :param str xscale: x-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param str yscale: y-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param bool errors: Display intensity uncertainties as an error band (only if calculated during integration)
        :return:
        """
        if self.results is None:
            raise Exception('No results to plot. Run integrate() first.')
        if xlabel is None:
            xlabel = self.units_radial_repr
        if ylabel is None:
            ylabel = 'Intensity (arbitrary units)'
        if title is None:
            title = '1D azimuthal integration'

        axis_scales = ['linear', 'log', 'symlog', 'logit']
        if xscale is None:
            xscale = 'linear'
        if xscale not in axis_scales:
            raise ValueError(f'Invalid value for \'xscale\'. Must be one of: ' + ', '.join(axis_scales))
        if yscale is None:
            yscale = 'linear'
        if yscale not in axis_scales:
            raise ValueError(f'Invalid value for \'yscale\'. Must be one of: ' + ', '.join(axis_scales))

        intensities, radial = self.results.intensity, self.results.radial
        ax.plot(radial, intensities, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_title(title)

        if errors and self.results.sigma is not None:
            sigma = self.results.sigma
            ax.fill_between(radial, intensities - sigma, intensities + sigma, fc='gray', ec=None, alpha=0.5)
        return ax, fig


class AzimuthalIntegrator2D(_Integrator):

    def __init__(self, image: Image):
        super().__init__(image)
        self.points_azimuthal: int = None
        self.units_azimuthal: str = 'chi_deg'
        self.units_azimuthal_repr: str = '\u03c7 (\u00b0)'
        self._results: Integrate2dResult = None

    def initialize(self, points_radial: int = None, points_azimuthal: int = None, units_radial: str = '2theta',
                   masked_pixels_value: float = np.nan, error_model: str = None) -> None:
        """
        Initialize settings for integrator. If ``points_radial`` is not specified, then the maximum number of pixels
        along the radial axis is chosen. If ``points_azimuthal`` is not specified, then it is set to ~1 point/pixel at
        2\u03b8_max (determined at detector edge, not corners).

        :param int points_radial: Number of points along the radial axis
        :param int points_azimuthal: Number of points along the azimuthal axis
        :param str units_radial: Units for the radial axis (``'2theta'`` or ``'q'``)
        :param float masked_pixels_value: Value for masked or ignored pixels (default: ``numpy.nan``)
        :param str error_model: Error model for calculating intensities uncertainties (``None``, ``'poisson'``, or
                                ``'azimuthal'``)
        :raises ValueError: When an unsupported unit is provided to ``units_radial``
        :raises ValueError: When an unsupported model is provided to ``error_model``
        :return: None
        """
        if points_radial is None:
            # Automatic guess for radial number of points
            self.points_radial = self._max_radial_pixels()
        else:
            self.points_radial = int(points_radial)

        if points_azimuthal is None:
            # Automatic guess for azimuthal number of points
            radius = self._max_radial_pixels()  # 2theta_max radius in pixels
            perimeter = 2 * np.pi * radius  # 2theta_max perimeter in pixels
            self.points_azimuthal = int(perimeter)
        else:
            self.points_azimuthal = int(points_azimuthal)

        if units_radial in self.UNITS_2THETA_DEGREES:
            self.units_radial = '2th_deg'
            self.units_radial_repr = '2\u03b8 (\u00b0)'
        elif units_radial in self.UNITS_Q_NM:
            self.units_radial = 'q_nm^-1'
            self.units_radial_repr = 'Q (nm$^{-1}$)'
        else:
            raise ValueError(f'Unknown units: \'{units_radial}\'. Choose one from: '
                             f'{", ".join(self.SUPPORTED_UNITS_RADIAL)}')

        self.masked_pixels_value = float(masked_pixels_value)
        if error_model not in [None, 'poisson', 'azimuthal']:
            raise ValueError(f'Unknown error model. Must be one of: None, \'poisson\', \'azimuthal\'')
        self.error_model = error_model

        self._integrator = _AzimuthalIntegrator()
        self._integrator.set_config(self.image.geometry.get_config())  # pyFAI bug: to properly load detector info
        self._is_initialized = True

    def integrate(self, check=True, keep: bool = True, **kwargs) -> Integrate2dResult:
        """
        Perform 2D azimuthal integration using ``pyFAI.AzimuthalIntegrator.integrate2d_ng`` on the ``Image``,
        excluding the regions defined in ``Image.mask``. Integration settings are defined from the initialize() method.
        This method is just a wrapper around the pyFAI method, without any additional initialization. The default
        pyFAI integration method is ``('bbox', 'histogram', 'cython')`` (aka ``'bbox'``).

        :param bool check: Check whether the integrator has already been initialized
        :param bool keep: Whether to store the integration results or return them
        :param kwargs: Any of the following pyFAI arguments: ``correctSolidAngle``, ``variance``, ``radial_range``,
                       ``azimuth_range``, ``delta_dummy``, ``polarization_factor``, ``dark``, ``flat``, ``method``,
                       ``safe``, ``normalization_factor``, ``metadata``. The default pyFAI values are chosen if not
                       provided.
        :raises Exception: When the integrator hasn't been initialized and ``check=True``
        :return: Integrate2dResult
        """
        if check:
            self.check_initialized()

        correctSolidAngle = kwargs.get('correctSolidAngle', True)
        variance = kwargs.get('variance', None)
        radial_range = kwargs.get('radial_range', None)
        azimuth_range = kwargs.get('azimuth_range', None)
        delta_dummy = kwargs.get('delta_dummy', None)
        polarization_factor = kwargs.get('polarization_factor', None)
        dark = kwargs.get('dark', None)
        flat = kwargs.get('flat', None)
        method = kwargs.get('method', 'bbox')
        safe = kwargs.get('safe', True)
        normalization_factor = kwargs.get('normalization_factor', 1.0)
        metadata = kwargs.get('metadata', None)

        result = self._integrator.integrate2d_ng(data=self.image.data, npt_rad=self.points_radial,
                                                 npt_azim=self.points_azimuthal, filename=None,
                                                 correctSolidAngle=correctSolidAngle, variance=variance,
                                                 error_model=self.error_model, radial_range=radial_range,
                                                 azimuth_range=azimuth_range, mask=~self.image.mask.data,
                                                 dummy=self.masked_pixels_value,  delta_dummy=delta_dummy,
                                                 polarization_factor=polarization_factor, dark=dark, flat=flat,
                                                 method=method, unit=self.units_radial, safe=safe,
                                                 normalization_factor=normalization_factor, metadata=metadata)
        # Note that the mask needs to be inverted! (for pyFAI True means mask that pixel)

        if keep:
            self._results = result
        return result

    @property
    def results(self):
        return self._results

    def _get_file_header(self) -> str:
        """
        Prepare a header for exporting integration results to file
        :return:
        """
        header = f'2D azimuthal integration using {self.__class__.__module__}.{self.__class__.__name__}\n'
        header += f'Integration performed on: {datetime.now()}\n'
        header += f'Filename: {self.image.file.resolve()}\n'
        header += f'Frame: {self.image.frame}\n\n'

        header += 'Geometry options:\n'
        header += '\n'.join(f'pyFAI.Geometry.{key}: {value}' for key, value in
                            self.image.geometry.get_config().items()) + '\n\n'

        header += 'Integration options:\n'
        for key in ['unit', 'error_model', 'method_called', 'method', 'compute_engine', 'has_mask_applied',
                    'has_flat_correction', 'has_dark_correction', 'has_solidangle_correction', 'polarization_factor',
                    'normalization_factor', 'metadata']:
            header += f'pyFAI.AzimuthalIntegrator.{key}: {getattr(self.results, key)}\n'
        header += f'pyFAI.AzimuthalIntegrator.npt_rad: {self.points_radial}\n'
        header += f'pyFAI.AzimuthalIntegrator.npt_azim: {self.points_azimuthal}\n'

        if self.results.sigma is not None:
            header += f'\nDatasets: {self.units_radial}, {self.units_azimuthal}, intensity (a.u.), sigma (a.u.)'
        else:
            header += f'\nDatasets: {self.units_radial}, {self.units_azimuthal}, intensity (a.u.)'
        return header

    def save(self, filename: str | Path, overwrite: bool = False, header: bool = True):
        """
        Save integration results to a NPX file (numpy .npz + header). The following arrays are contained within the file
        radial angle, azimuthal angle, intensity, intensity uncertainty. The intensity and intensity uncertainty arrays
        are 2D arrays, while the angle arrays are 1D. If no errors have been calculated during integration, the output
        file will contain only three arrays.

        :param str | Path filename: Output filename. If the file extension is not .npx, it will be replaced.
        :param bool overwrite: Whether to overwrite the output file if it already exists.
        :param bool header: Whether to include metadata about the integration in the file header
        :return:
        """
        if self.results is None:
            raise Exception('No results to save. Run integrate() first.')

        file = Path(filename)
        if file.exists() and not overwrite:
            raise FileExistsError(f'File {file} already exists!')
        file.unlink(missing_ok=True)

        data = {
            'radial': self.results.radial,
            'azimuthal': self.results.azimuthal,
            'intensities': self.results.intensity
        }
        sigma = self.results.sigma
        if sigma is not None:
            data['sigma'] = sigma

        header = self._get_file_header() if header else None
        npx = NpxFile(header=header, **data)
        npx.save(file, compressed=True)

    def plot(self, ax: plt.Axes = plt.gca(), fig: plt.Figure = plt.gcf(), xlabel: str = None, ylabel: str = None,
             title: str = None, xscale: str = None, yscale: str = None, zscale: str = None, zmin: float = None,
             zmax: float = None, cmap: str = None, bad_value_color: str = None, overlay_mask: bool = False) \
            -> tuple[plt.Axes, plt.Figure, AxesImage]:
        """
        Prepare a plot of the 2D integration results. ``plt.show()`` must be called separately to display the plot.

        :param matplotlib.axes.Axes ax: Axes instance to draw into
        :param matplotlib.figure.Figure fig: Figure instance to draw into
        :param str xlabel: x-axis label (default: integration radial units)
        :param str ylabel: y-axis label (default: integration azimuthal units)
        :param str title: Plot title (default: ``'2D azimuthal integration'``)
        :param str xscale: x-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param str yscale: y-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param str zscale: z-axis scale, one from: ``'linear'``, ``'log'`` (default: ``'linear'``)
        :param float zmin: z-axis minimum value
        :param float zmax: z-axis maximum value
        :param str cmap: A Matplotlib colormap name to be used as the intensity scale
        :param str bad_value_color: The missing values color
        :param bool overlay_mask: Whether to overlay the image mask on top of the integration results
        :return:
        """
        if self.results is None:
            raise Exception('No results to plot. Run integrate() first.')
        if xlabel is None:
            xlabel = self.units_radial_repr
        if ylabel is None:
            ylabel = self.units_azimuthal_repr
        if title is None:
            title = '2D azimuthal integration'

        axis_scales = ['linear', 'log', 'symlog', 'logit']
        if xscale is None:
            xscale = 'linear'
        if xscale not in axis_scales:
            raise ValueError(f'Invalid value for \'xscale\'. Must be one of: ' + ', '.join(axis_scales))
        if yscale is None:
            yscale = 'linear'
        if yscale not in axis_scales:
            raise ValueError(f'Invalid value for \'yscale\'. Must be one of: ' + ', '.join(axis_scales))
        if zscale in [None, 'linear']:
            norm = partial(Normalize, clip=False)
        elif zscale in ['log', 'log10']:
            norm = partial(LogNorm, clip=False)
        else:
            raise ValueError(f'Invalid value for \'zscale\'. Must be one of: linear, log')

        if cmap is None:
            cmap = get_cmap(self.image.cmap)
        else:
            cmap = get_cmap(cmap)
        if bad_value_color is None:
            cmap.set_bad(color=self.image.cmap_bad_values, alpha=1.0)
        else:
            cmap.set_bad(color=bad_value_color, alpha=1.0)

        intensities, radial, azimuthal = self.results.intensity, self.results.radial, self.results.azimuthal
        img = ax.imshow(intensities, origin='lower', aspect='auto', interpolation='nearest', cmap=cmap,
                        norm=norm(vmin=zmin, vmax=zmax),
                        extent=(radial.min(), radial.max(), azimuthal.min(), azimuthal.max()))

        if overlay_mask:
            mask = ~np.isnan(intensities)
            mask_cmap = LinearSegmentedColormap.from_list(name='mask', N=2,
                                                          colors=[to_rgba(self.image.mask_color), (1, 1, 1, 0)])
            mask_alpha = self.image.mask_alpha
            ax.imshow(mask, origin='lower', aspect='auto', interpolation='nearest', cmap=mask_cmap, alpha=mask_alpha,
                      extent=(radial.min(), radial.max(), azimuthal.min(), azimuthal.max()))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_title(title)

        return ax, fig, img