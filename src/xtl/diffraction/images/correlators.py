import numpy as np

from xtl.diffraction.images import Image


class _Correlator:
    ...


class AzimuthalCrossCorrelatorQQ(_Correlator):

    def __init__(self, parent: Image):
        """
        Calculates the intensity cross-correlation function along the azimuthal coordinate (``q_1`` = ``q_2`` = ``q``).
        This implementation relies on projecting the collected intensities from the cartesian coordinate space of the
        detector image to polar coordinates (*i.e.* azimuthal angle [``\u03c7``], radial distance [``2\u03b8`` or
        ``q``]) using ``pyFAI.integrate2d_ng()``.

        :param parent:
        """

        self.image = parent
        self.image.check_geometry()
        self._ccf: np.ndarray = None  # dim = (delta=azimuthal, radial)

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
        if not self.image.ai2.is_initialized:
            self.image.ai2.initialize(points_radial=points_radial, points_azimuthal=points_azimuthal,
                                      units_radial=units_radial)
        return self.image.ai2.integrate()

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
