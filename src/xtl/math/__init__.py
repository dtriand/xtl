import math


def d_spacing_to_ttheta(d, wavelength):
    return 2 * math.degrees(math.asin(wavelength/(2*d)))


def ttheta_to_d_spacing(ttheta, wavelength):
    return wavelength / (2 * math.sin(math.radians(ttheta/2)))
