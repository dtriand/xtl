from datetime import datetime
from functools import partial
from pathlib import Path

from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np

import xtl
from xtl.diffraction.images import Image
from xtl.io.npx import NpxFile


class _Correlator:

    UNITS_2THETA_DEGREES = ['2theta', '2th', 'tth', 'ttheta', '2\u03B8',
                            '2theta_deg', '2th_deg', 'tth_deg', 'ttheta_deg']
    UNITS_2THETA_RADIANS = ['2theta_rad', '2th_rad', 'tth_rad', 'ttheta_rad']
    UNITS_Q_NM = ['q', 'q_nm', 'q_nm^-1']
    UNITS_Q_ANGSTROM = ['q_A', 'q_A^-1']
    SUPPORTED_UNITS_RADIAL = UNITS_2THETA_DEGREES + UNITS_Q_NM

    def __init__(self, image: Image):
        self.image = image
        if not isinstance(image, Image):
            raise ValueError(f'Must be an Image instance, not {image.__class__.__name__}')
        self.image.check_geometry()

        # Units representation
        self.units_radial: str = '2theta_deg'
        self.units_radial_repr: str = '2\u03b8 (\u00b0)'
        self.units_azimuthal: str = 'delta_deg'
        self.units_azimuthal_repr: str = '\u0394 (\u00b0)'

        # Plotting options
        self.cmap = 'viridis'
        self.cmap_bad_values = 'white'
        self.symlog_linthresh = 0.05


class AzimuthalCrossCorrelatorQQ_1(_Correlator):

    def __init__(self, image: Image):
        """
        Calculates the intensity cross-correlation function along the azimuthal coordinate (``q_1`` = ``q_2`` = ``q``).
        This implementation relies on projecting the collected intensities from the cartesian coordinate space of the
        detector image to polar coordinates (*i.e.* azimuthal angle [``\u03c7``], radial distance [``2\u03b8`` or
        ``q``]) using ``pyFAI.integrate2d_ng()``.

        :param image:
        """
        super().__init__(image)

        self._ai2: 'xtl.diffraction.images.AzimuthalIntegrator2D' = None
        self._ccf: np.ndarray = None  # dim = (delta=azimuthal, radial)

    def _set_units(self):
        """
        Grabs the radial and azimuthal units from the 2D integrator.

        :return:
        """
        self.units_radial = self._ai2.units_radial
        self.units_radial_repr = self._ai2.units_radial_repr
        if self._ai2.units_azimuthal == 'chi_deg':
            self.units_azimuthal = 'delta_deg'
            self.units_azimuthal_repr = '\u0394 (\u00b0)'
        else:
            raise ValueError(f'Unknown azimuthal units: {self._ai2.units_azimuthal}')

    def _perform_azimuthal_integration(self, points_radial: int = 500, points_azimuthal: int = 500,
                                       units_radial: str = '2theta'):
        """
        Transform (and interpolate) intensities from cartesian to polar coordinates, using ``pyFAI.integrate2d_ng``.

        :param int points_radial:
        :param points_azimuthal:
        :param units_radial:
        :return:
        """
        if self.image.ai2 is None:
            self.image.initialize_azimuthal_integrator(dim=2)
        self._ai2 = self.image.ai2
        if not self._ai2.is_initialized:
            self._ai2.initialize(points_radial=points_radial, points_azimuthal=points_azimuthal,
                                 units_radial=units_radial)
        self._set_units()
        return self._ai2.integrate()

    @staticmethod
    def _calculate_delta_indices_array(points_azimuthal: int) -> np.ndarray:
        """
        Returns an 2D array of indices for all possible roll operations of a 1D array.

        :param int points_azimuthal: Size of the input 1D array
        :return:
        """
        di0 = np.arange(points_azimuthal)
        di = np.tile(di0, [points_azimuthal, 1])
        for i in range(points_azimuthal):
            di[i] = np.roll(di0, -i)
        return di

    def correlate(self, points_radial: int = 500, points_azimuthal: int = 360, units_radial: str = '2theta',
                  method: int = 0):
        """
        Calculate the intensity cross-correlation function (CCF) for the entire radial range. This uses interpolated
        intensities from ``pyFAI.integrate2d_ng`` as input for the calculations. The function being calculated is:

        .. math::
            CCF(q,\\Delta) = \\frac{\\langle I(q,\\chi) \\times I(q, \\chi + \\Delta) \\rangle_\\chi -
            \\langle I(q,\\chi) \\rangle_\\chi^2}{\\langle I(q,\\chi) \\rangle_\\chi^2}

        where ``q`` is the radial coordinate, ``\u03c7`` is the azimuthal coordinate and ``\u0394`` the offset in
        azimuthal coordinates.

        There are two methods available for performing this calculation:

        - Method ``0`` is faster and calculates the CCF for the entire image simultaneously. This, however, comes at the
          expense of higher memory consumption. If the number of interpolating points along the azimuthal and radial
          axes are set too high, this method can easily require several tens of GB of memory to run.
        - Method ``1`` is a bit slower, but calculates the CCF for every radial segment separately. If memory
          consumption is of concern, or method 0 runs out of memory while working with a very large array, then this
          method will work better.

        :param int points_radial: Number of points for intensity interpolation along the radial axis (default: ``500``)
        :param int points_azimuthal: Number of points for intensity interpolation along the azimuthal axis
                                     (default: ``360``, i.e. delta step of 1 degree).
        :param str units_radial: Units for the radial axis. Can be either ``'2theta'`` or ``'q'``
                                 (default: ``'2theta'``)
        :param int method: Calculation method. Can be either ``0`` or ``1``.
        :return:
        """
        supported_methods = [0, 1]
        if method not in supported_methods:
            raise ValueError('Unknown correlation method. Choose one from: ' +
                             ', '.join(str(i) for i in supported_methods))
        # Project intensities from cartesian to polar coordinates
        ai2 = self._perform_azimuthal_integration(points_radial=points_radial, points_azimuthal=points_azimuthal,
                                                  units_radial=units_radial)
        # Array of indices for calculating I(chi+delta) from I(chi)
        di = self._calculate_delta_indices_array(points_azimuthal=points_azimuthal)  # dim = (azimuthal, azimuthal)
        # Complete intensity array
        I = ai2.intensity  # dim = (azimuthal, radial)

        if method == 0:
            # Faster method: Calculate the CCF for all radial segments all at once, at the expense of using more memory.
            #  This method requires storing a 3D array of size azim * azim * radial.

            # Mean squared intensity per radial value
            I_mean_squared_radial = np.square(np.nanmean(I, axis=0))  # dim = (radial, )

            # Intensity array with all the possible azimuthal offsets (i.e. deltas)
            #  Note: This can be a very big array if the number of radial and azimuthal points is set too high!
            I_plus_delta = np.take(I, di, axis=0)  # dim = (delta=azim, azimuthal, radial)
            # Multiply the intensities for every delta with the original non-shifted intensities
            I_corr_prod = I_plus_delta * I  # dim = (delta=azim, azimuthal, radial)

            # Average correlation for all azimuthal angles
            I_corr_prod_radial = np.nanmean(I_corr_prod, axis=1)  # dim = (delta, radial)
            # Intensity-fluctuation cross-correlation function
            self._ccf = I_corr_prod_radial / I_mean_squared_radial - 1  # dim = (delta, radial)
        elif method == 1:
            # Slower method: Calculate the CCF for each radial segment separately. This method is a bit slower than the
            #  previous one, but requires significantly less memory, since the largest array that is stored is of size
            #  azim * azim

            # Initialize empty cross-correlation function array
            ccf = np.zeros((ai2.radial.size, ai2.azimuthal.size), dtype=I.dtype)
            # Iterate over radial segments
            for i, I_rad in enumerate(ai2.intensity.T):  # I_rad dim = (azimuthal, )
                # Mean squared intensity for segment
                I_mean_squared_radial = np.nanmean(I_rad) ** 2  # dim = scalar

                # Initialize intensity array with azimuthal offsets (i.e. delta)
                I_plus_delta = np.tile(I_rad, [points_azimuthal, 1])  # dim = (delta=azim, azimuthal)
                # Iterate over all delta offsets (which is equal to the azimuthal points)
                for j in range(points_azimuthal):
                    # Reindex intensities with the respective delta offsets
                    I_plus_delta[j] = I_plus_delta[j][di[j]]  # dim = (delta=azim, azimuthal)
                # Multiply the intensities for every delta with the original non-shifted intensities
                I_corr_prod = I_plus_delta * I_rad  # dim = (delta=azim, azimuthal)

                # Average correlation for all azimuthal angles
                I_corr_prod_radial = np.nanmean(I_corr_prod, axis=1)  # dim = (delta=azim, )
                # Intensity-fluctuation cross-correlation function for radial segment
                ccf[i] = I_corr_prod_radial / I_mean_squared_radial - 1  # dim for slice = (delta=azim, )

            # Transpose array to match the axes of the intensity array
            self._ccf = ccf.T  # dim = (delta, radial)
        return self.ccf

    @property
    def ccf(self):
        return self._ccf

    def _get_file_header(self) -> str:
        """
        Prepare a header for exporting integration results to file
        :return:
        """
        header = f'Azimuthal intensity cross-correlation function using ' + \
                 f'{self.__class__.__module__}.{self.__class__.__name__}\n'
        header += f'Calculation performed on: {datetime.now()}\n'
        header += f'Filename: {self.image.file.resolve()}\n'
        header += f'Frame: {self.image.frame}\n\n'

        header += 'Geometry options:\n'
        header += '\n'.join(f'pyFAI.Geometry.{key}: {value}' for key, value in
                            self.image.geometry.get_config().items()) + '\n\n'

        header += 'Integration options:\n'
        for key in ['unit', 'error_model', 'method_called', 'method', 'compute_engine', 'has_mask_applied',
                    'has_flat_correction', 'has_dark_correction', 'has_solidangle_correction', 'polarization_factor',
                    'normalization_factor', 'metadata']:
            header += f'pyFAI.AzimuthalIntegrator.{key}: {getattr(self._ai2.results, key)}\n'
        header += f'pyFAI.AzimuthalIntegrator.npt_rad: {self._ai2.points_radial}\n'
        header += f'pyFAI.AzimuthalIntegrator.npt_azim: {self._ai2.points_azimuthal}\n'

        header += f'\nDatasets: {self.units_radial}, {self.units_azimuthal}, cross-correlation function'
        return header

    def save(self, filename: str | Path, overwrite: bool = False, header: bool = True):
        """
        Save integration results to a NPX file (numpy .npz + header). The following arrays are contained within the file
        radial angle, azimuthal angle offset, CCF, intensity uncertainty. The CCF array is a 2D array, while the angle
        arrays are 1D.

        :param str | Path filename: Output filename. If the file extension is not .npx, it will be replaced.
        :param bool overwrite: Whether to overwrite the output file if it already exists.
        :param bool header: Whether to include metadata about the integration in the file header
        :return:
        """
        if self.ccf is None:
            raise Exception('No results to save. Run correlate() first.')

        file = Path(filename)
        if file.exists() and not overwrite:
            raise FileExistsError(f'File {file} already exists!')
        file.unlink(missing_ok=True)

        data = {
            'radial': self._ai2.results.radial,
            'delta': self._ai2.results.azimuthal,
            'ccf': self.ccf
        }

        header = self._get_file_header() if header else None
        npx = NpxFile(header=header, **data)
        npx.save(file, compressed=True)

    def plot(self, ax: plt.Axes = plt.gca(), fig: plt.Figure = plt.gcf(), xlabel: str = None, ylabel: str = None,
             title: str = None, xscale: str = None, yscale: str = None, zscale: str = None, zmin: float = None,
             zmax: float = None, cmap: str = None, bad_value_color: str = None) \
            -> tuple[plt.Axes, plt.Figure, AxesImage]:
        """
        Prepare a plot of the intensity CCF. ``plt.show()`` must be called separately to display the plot.

        :param matplotlib.axes.Axes ax: Axes instance to draw into
        :param matplotlib.figure.Figure fig: Figure instance to draw into
        :param str xlabel: x-axis label (default: integration radial units)
        :param str ylabel: y-axis label (default: integration azimuthal units)
        :param str title: Plot title (default: ``'Cross-correlation function'``)
        :param str xscale: x-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param str yscale: y-axis scale, one from: ``'linear'``, ``'log'``, ``'symlog'`` or ``'logit'`` (default:
                           ``'linear'``)
        :param str zscale: z-axis scale, one from: ``'linear'``, ``'log'`` or ``'symlog'`` (default: ``'linear'``)
        :param float zmin: z-axis minimum value
        :param float zmax: z-axis maximum value
        :param str cmap: A Matplotlib colormap name to be used as the CCF scale
        :param str bad_value_color: The missing values color
        :return:
        """
        if self.ccf is None:
            raise Exception('No results to plot. Run correlate() first.')
        if xlabel is None:
            xlabel = self.units_radial_repr
        if ylabel is None:
            ylabel = self.units_azimuthal_repr
        if title is None:
            title = 'Cross-correlation function'

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
        elif zscale in ['symlog']:
            norm = partial(SymLogNorm, linthresh=self.symlog_linthresh, clip=False)
        else:
            raise ValueError(f'Invalid value for \'zscale\'. Must be one of: linear, log, symlog')

        if cmap is None:
            cmap = get_cmap(self.cmap)
        else:
            cmap = get_cmap(cmap)
        if bad_value_color is None:
            cmap.set_bad(color=self.cmap_bad_values, alpha=1.0)
        else:
            cmap.set_bad(color=bad_value_color, alpha=1.0)

        ccf, radial, azimuthal = self.ccf, self._ai2.results.radial, self._ai2.results.azimuthal
        img = ax.imshow(ccf, origin='lower', aspect='auto', interpolation='nearest', cmap=cmap,
                        norm=norm(vmin=zmin, vmax=zmax),
                        extent=(radial.min(), radial.max(), azimuthal.min(), azimuthal.max()))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_title(title)

        return ax, fig, img


class AzimuthalCrossCorrelatorQQ_2(_Correlator):

    def __init__(self, image: Image):
        super().__init__(image=image)
        self.shells_img: np.ndarray = None
        self.pixels_per_shell = []

    def _set_units(self, units_radial: str, units_azimuthal: str):
        # if points_radial is None:
        #     # Automatic guess for number of points
        #     self.points_radial = self._max_radial_pixels()
        # else:
        #     self.points_radial = int(points_radial)

        if units_radial in self.UNITS_2THETA_DEGREES:
            self.units_radial = '2th_deg'
            self.units_radial_repr = '2\u03b8 (\u00b0)'
        elif units_radial in self.UNITS_Q_NM:
            self.units_radial = 'q_nm^-1'
            self.units_radial_repr = 'Q (nm$^{-1}$)'
        else:
            raise ValueError(f'Unknown units: \'{units_radial}\'. Choose one from: '
                             f'{", ".join(self.SUPPORTED_UNITS_RADIAL)}')

        if units_azimuthal == 'chi_deg':
            self.units_azimuthal = 'delta_deg'
            self.units_azimuthal_repr = '\u0394 (\u00b0)'
        else:
            raise ValueError(f'Unknown azimuthal units: {units_azimuthal}')

    def _get_pixel_geometry_data(self, apply_mask=False):
        pixel_geometry = self.image.geometry.corner_array(unit=self.units_radial, use_cython=True)
        # self.image.geometry.normalize_azimuth_range()
        # self.image.geometry.guess_npt_rad()
        
        # We need to grab the corner 0 pixel in order to perfectly align with the beam center
        radial = pixel_geometry[:, :, 0, 0]
        chi = np.rad2deg(pixel_geometry[:, :, 0, 1])
        if apply_mask:
            radial = np.where(self.image.mask.data, radial, np.nan)
            chi = np.where(self.image.mask.data, chi, np.nan)
        return radial, chi

    @staticmethod
    def _calculate_radial_shells(radial_array: np.ndarray, no_shells: int = 500):
        radial_min = np.nanmin(radial_array)
        radial_max = np.nanmax(radial_array)
        shells = np.linspace(radial_min, radial_max, no_shells)
        return shells

    def _q2r(self, q: np.ndarray, in_pixels: bool = True):
        # q in reverse nanometers
        # wavelength in meters (* 1e-9 in nanometers)
        # ttheta in degrees
        ttheta = np.rad2deg(2 * np.arcsin((q * self.image.geometry.wavelength / 1e-9) / (4 * np.pi)))
        # sample-to-detector distance in meters
        # radial distance in meters
        r = self.image.geometry.dist * np.tan(np.deg2rad(ttheta))
        if in_pixels:
            r /= np.mean([self.image.geometry.pixel1, self.image.geometry.pixel2])
        return r

    def _2th2r(self, tth: np.ndarray, in_pixels: bool = True):
        # ttheta in degrees
        # sample-to-detector distance in meters
        # radial distance in meters
        r = self.image.geometry.dist * np.tan(np.deg2rad(tth))
        if in_pixels:
            r /= np.mean([self.image.geometry.pixel1, self.image.geometry.pixel2])
        return r

    def _split_pixels_to_shells(self, no_shells: int = 500):
        radial, chi = self._get_pixel_geometry_data(apply_mask=True)

        self.shells_img = np.full_like(radial, np.nan)
        radial_shells = self._calculate_radial_shells(radial_array=radial, no_shells=no_shells+1)
        # radial_shells_in_pixels = self._q2r(radial_shells, in_pixels=True)
        # average_width = np.mean((radial_shells_in_pixels - np.roll(radial_shells_in_pixels, 1))[1:])
        # print(f'Average shell width: {average_width:.2f} px')
        for i, radial_low in enumerate(radial_shells[:-1]):
            radial_high = radial_shells[i + 1]

            ring = (radial >= radial_low) & (radial < radial_high)
            self.shells_img[ring] = np.random.random()

            intensities = np.extract(ring, self.image.data)
            azimuthals = np.extract(ring, chi)
            radials = np.extract(ring, radial)
            sorter = np.argsort(azimuthals)
            pixels = np.vstack((azimuthals[sorter], radials[sorter], intensities[sorter]))
            self.pixels_per_shell.append(pixels)

    @staticmethod
    def _calculate_delta_indices_array(points_azimuthal: int) -> np.ndarray:
        """
        Returns an 2D array of indices for all possible roll operations of a 1D array.

        :param int points_azimuthal: Size of the input 1D array
        :return:
        """
        di0 = np.arange(points_azimuthal)
        di = np.tile(di0, [points_azimuthal, 1])
        for i in range(points_azimuthal):
            di[i] = np.roll(di0, -i)
        return di

    def correlate(self, points_radial: int = 500, units_radial: str = '2theta', shell_range=(0, 100)):
        self._set_units(units_radial=units_radial, units_azimuthal='chi_deg')
        print('Splitting pixels to shells... ', end='')
        t1 = datetime.now()
        self._split_pixels_to_shells(no_shells=points_radial)
        t2 = datetime.now()
        print(f'Completed in {(t2 - t1).total_seconds():.3f} sec')

        print(f'Calculating CCF... ', end='')
        t3 = datetime.now()
        ccf = []
        if shell_range is None:
            shell_range = (0, -1)
        # Iterate over radial segments
        for i, (chi, radial, I) in enumerate(self.pixels_per_shell[shell_range[0]:shell_range[1]]):
            print(i, end='')
            points_azimuthal = len(chi)
            di = self._calculate_delta_indices_array(points_azimuthal=points_azimuthal)
            # Mean squared intensity for segment
            I_mean_squared_radial = np.nanmean(I) ** 2  # dim = scalar

            # Initialize intensity array with azimuthal offsets (i.e. delta)
            I_plus_delta = np.tile(I, [points_azimuthal, 1])  # dim = (delta=azim, azimuthal)
            # Iterate over all delta offsets (which is equal to the azimuthal points)
            for j in range(points_azimuthal):
                # Reindex intensities with the respective delta offsets
                I_plus_delta[j] = I_plus_delta[j][di[j]]  # dim = (delta=azim, azimuthal)
            # Multiply the intensities for every delta with the original non-shifted intensities
            I_corr_prod = I_plus_delta * I  # dim = (delta=azim, azimuthal)

            # Average correlation for all azimuthal angles
            I_corr_prod_radial = np.nanmean(I_corr_prod, axis=1)  # dim = (delta=azim, )
            # Intensity-fluctuation cross-correlation function for radial segment
            _ccf = I_corr_prod_radial / I_mean_squared_radial - 1  # dim for slice = (delta=azim, )

            ccf.append((chi, radial, _ccf))
            print('\b' * len(str(i)), end='')
        t4 = datetime.now()
        print(f'Completed in {(t4 - t3).total_seconds():.3f} sec')
        self.ccf = ccf

    def plot_shells(self):
        fig, ax = plt.subplots(1, 1)
        ax = plt.imshow(self.shells_img, cmap='prism')

    def plot_pixel_projection(self, shell_range: tuple[int, int] = (0, 10), zscale: str = 'linear',
                              ignore_radial_axis: bool = False):
        fig, ax = plt.subplots(1, 1)
        intensities = self.image.data_masked
        i_min = np.nanmin(intensities)
        i_max = np.nanmax(intensities)

        if zscale in ['linear', None]:
            norm = Normalize(vmin=i_min, vmax=i_max, clip=True)
        elif zscale in ['log', 'log10']:
            norm = LogNorm(vmin=i_min+1, vmax=i_max, clip=True)
        else:
            raise ValueError('Invalid zscale. Must be either \'linear\' or \'log\'')
        mapper = ScalarMappable(norm=norm, cmap=self.image.cmap)
        for chi, radial, intensities in self.pixels_per_shell[shell_range[0]:shell_range[1]]:
            ax.hlines(np.nanmin(radial), xmin=-180, xmax=180, linestyles='--', colors='grey')
            if ignore_radial_axis:
                radial = np.tile(np.nanmean(radial), len(radial))
            ax.scatter(chi, radial, c=mapper.to_rgba(intensities), marker='s')
            ax.text(185, np.nanmean(radial), s=f'{len(radial)}', ha='left', va='center')
        ax.hlines(np.nanmax(radial), xmin=-180, xmax=180, linestyles='--', colors='grey')
        ax.set_xlabel(self.units_azimuthal_repr)
        ax.set_ylabel(self.units_radial_repr)
        fig.colorbar(mapper, ax=ax)

    def plot_ccf(self):
        fig, ax = plt.subplots(1, 1)
        for chi, radial, ccf in self.ccf:
            ax.hlines(np.nanmin(radial), xmin=-180, xmax=180, linestyles='--', colors='grey')
            r_min, r_max = np.nanmin(radial), np.nanmax(radial)
            norm = Normalize(vmin=r_min, vmax=r_max, clip=True)
            ax.plot(chi, (r_max - r_min) * norm(ccf) + r_min, 'o-')
        ax.set_xlabel(self.units_azimuthal_repr)
        ax.set_ylabel(self.units_radial_repr)
