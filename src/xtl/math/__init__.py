import math
import decimal

from .crystallography import d_spacing_to_ttheta, ttheta_to_d_spacing


def round_value_and_esd(value, esd):
    """
    Round a value based on its esd. Can be used for pretty formatting of a (value, esd) pair.
    Example:

    .. code-block:: python

       >>> round_value_and_esd(12.3456, 0.01234)
       (12.35, 0.01, 2)

    :param int or float value:
    :param int or float esd:
    :return: rounded value, rounded esd, position of esd's first significant digit
    :rtype: tuple[int or float, int or float, int]
    """
    e = decimal.Decimal(value=esd)

    significant_digit = 1 - e.as_tuple().exponent - len(e.as_tuple().digits)
    # Positive for floats < 1, zero for 1 < float < 10, negative for floats > 10
    # e.g. 0.02 > sg = 2
    #      1.34 > sg = 0
    #     16.12 > sg = -1
    e_rounded = round(esd, significant_digit)
    v_rounded = round(value, significant_digit)
    return v_rounded, e_rounded, significant_digit


def si_units(value, suffix='', base=1000, digits=None):
    """
    Returns a value formatted with the appropriate SI prefix.

    :param int or float value:
    :param str suffix: unit
    :param int base: system base value
    :param int digits: Digits to round value at
    :return:
    """
    if value == 0:
        return f"{value} {suffix}"
    magnitude = int(math.floor(math.log(value, base)))
    scaled = value / math.pow(base, magnitude)
    if digits is not None:
        scaled = round(scaled, digits)
    if magnitude >= 0:
        if magnitude > 8:  # greater than yotta prefix
            raise NotImplemented
        prefix = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'][magnitude]
    else:
        if magnitude < -8:  # less than yocto prefix
            raise NotImplemented
        prefix = ['', 'm', '\u03bc', 'n', 'p', 'f', 'a', 'z', 'y'][abs(magnitude)]
    return f"{scaled} {prefix}{suffix}"

