import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.containers import Integrate1dResult, Integrate2dResult

from images import Image


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
        self._integrator: AzimuthalIntegrator = None
        self._is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """
        The initialization status of the integrator.
        :return: ``True`` if initialized, ``False`` if not.
        """
        if self._is_initialized:
            return True
        return False

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

        self._integrator = AzimuthalIntegrator(self.image.geometry.get_config())
        self._is_initialized = True

    def integrate(self, check=True, **kwargs) -> None:
        """
        Perform 1D azimuthal integration using ``pyFAI.AzimuthalIntegrator.integrate1d_ng`` on the ``Image``,
        excluding the regions defined in ``Image.mask``. Integration settings are defined from the initialize() method.
        This method is just a wrapper around the pyFAI method, without any additional initialization. The default
        pyFAI integration method is ``'csr'``.

        :param bool check: Check whether the integrator has already been initialized
        :param kwargs: Any of the following pyFAI arguments: ``correctSolidAngle``, ``variance``, ``radial_range``,
                       ``azimuth_range``, ``delta_dummy``, ``polarization_factor``, ``dark``, ``flat``, ``method``,
                       ``safe``, ``normalization_factor``, ``metadata``. The default pyFAI values are chosen if not
                       provided.
        :raises Exception: When the integrator hasn't been initialized and ``check=True``
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

        self._results = self._integrator.integrate1d_ng(data=self.image.data, npt=self.points_radial, filename=None,
                                                        correctSolidAngle=correctSolidAngle, variance=variance,
                                                        error_model=self.error_model, radial_range=radial_range,
                                                        azimuth_range=azimuth_range, mask=self.image.mask.data,
                                                        dummy=self.masked_pixels_value, delta_dummy=delta_dummy,
                                                        polarization_factor=polarization_factor, dark=dark, flat=flat,
                                                        method=method, unit=self.units_radial, safe=safe,
                                                        normalization_factor=normalization_factor, metadata=metadata)

    @property
    def results(self):
        return self._results


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

        self._integrator = AzimuthalIntegrator(self.image.geometry.get_config())
        self._is_initialized = True

    def integrate(self, check=True, **kwargs) -> None:
        """
        Perform 2D azimuthal integration using ``pyFAI.AzimuthalIntegrator.integrate2d_ng`` on the ``Image``,
        excluding the regions defined in ``Image.mask``. Integration settings are defined from the initialize() method.
        This method is just a wrapper around the pyFAI method, without any additional initialization. The default
        pyFAI integration method is ``'bbox'``.

        :param bool check: Check whether the integrator has already been initialized
        :param kwargs: Any of the following pyFAI arguments: ``correctSolidAngle``, ``variance``, ``radial_range``,
                       ``azimuth_range``, ``delta_dummy``, ``polarization_factor``, ``dark``, ``flat``, ``method``,
                       ``safe``, ``normalization_factor``, ``metadata``. The default pyFAI values are chosen if not
                       provided.
        :raises Exception: When the integrator hasn't been initialized and ``check=True``
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

        self._results = self._integrator.integrate2d_ng(data=self.image.data, npt_rad=self.points_radial,
                                                        npt_azim=self.points_azimuthal, filename=None,
                                                        correctSolidAngle=correctSolidAngle, variance=variance,
                                                        error_model=self.error_model, radial_range=radial_range,
                                                        azimuth_range=azimuth_range, mask=self.image.mask.data,
                                                        dummy=self.masked_pixels_value, delta_dummy=delta_dummy,
                                                        polarization_factor=polarization_factor, dark=dark, flat=flat,
                                                        method=method, unit=self.units_radial, safe=safe,
                                                        normalization_factor=normalization_factor, metadata=metadata)

    @property
    def results(self):
        return self._results
