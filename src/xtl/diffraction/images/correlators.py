from functools import partial

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np

import xtl
from xtl.diffraction.images import Image


class _Correlator:
    ...


class AzimuthalCrossCorrelatorQQ(_Correlator):

    def __init__(self, image: Image):
        """
        Calculates the intensity cross-correlation function along the azimuthal coordinate (``q_1`` = ``q_2`` = ``q``).
        This implementation relies on projecting the collected intensities from the cartesian coordinate space of the
        detector image to polar coordinates (*i.e.* azimuthal angle [``\u03c7``], radial distance [``2\u03b8`` or
        ``q``]) using ``pyFAI.integrate2d_ng()``.

        :param image:
        """

        self.image = image
        self.image.check_geometry()
        self._ai2: 'xtl.diffraction.images.AzimuthalIntegrator2D' = None
        self._ccf: np.ndarray = None  # dim = (delta=azimuthal, radial)

        # Units representation
        self.units_radial: str = '2theta_deg'
        self.units_radial_repr: str = '2\u03b8 (\u00b0)'
        self.units_azimuthal: str = 'delta_deg'
        self.units_azimuthal_repr: str = '\u0394 (\u00b0)'

        # Plotting options
        self.cmap = 'viridis'
        self.cmap_bad_values = 'white'
        self.symlog_linthresh = 0.05

    def _set_units_repr(self):
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
        self._set_units_repr()
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
            raise Exception('No results to plot. Run integrate() first.')
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
