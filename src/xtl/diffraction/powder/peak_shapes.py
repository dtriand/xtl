import numpy as np


_gaussian_fwhm_conversion_factor = np.sqrt(np.pi / np.log(2)) / 2


def gaussian(x, x0, ymax, fwhm):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    b = fwhm * _gaussian_fwhm_conversion_factor
    return ymax * np.exp(-np.pi * ((x - x0) / b) ** 2)
