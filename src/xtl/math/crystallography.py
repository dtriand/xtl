from typing_extensions import Literal

from .trig import sin, asin


def d_spacing_to_ttheta(d, wavelength, mode: Literal['d', 'r'] = 'd'):
    """
    Convert a d-spacing value to the corresponding 2theta.

    :param int or float d: d-spacing
    :param int or float wavelength: wavelength
    :param mode: 'd' for 2theta in degrees, 'r' for 2theta in radians
    :return: 2theta
    :rtype: float
    """
    return 2 * asin(wavelength / (2 * d), mode)


def ttheta_to_d_spacing(ttheta, wavelength, mode: Literal['d', 'r'] = 'd'):
    """
    Convert a 2theta value to the corresponding d-spacing.

    :param int or float ttheta: 2theta
    :param int or float wavelength: wavelength
    :param mode: 'd' for 2theta in degrees, 'r' for 2theta in radians
    :return: d-spacing
    :rtype: float
    """
    return wavelength / (2 * sin(ttheta / 2, mode))
