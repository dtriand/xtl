from math import sqrt

from typing_extensions import Literal

from .trig import sin, asin, sin_d, cos_d


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


def d_hkl(hkl, cell):
    assert len(hkl) == 3
    a, b, c, alpha, beta, gamma = cell
    if hasattr(cell, 'crystal_system'):
        if cell.crystal_system == 'cubic':
            return d_hkl_cubic(hkl, a)
        elif cell.crystal_system == 'tetragonal':
            return d_hkl_tetragonal(hkl, a, c)
        elif cell.crystal_system == 'orthorhombic':
            return d_hkl_orthorhombic(hkl, a, b, c)
        elif cell.crystal_system == 'rhombohedral':
            return d_hkl_rhombohedral(hkl, a, alpha)
        elif cell.crystal_system == 'hexagonal':
            return d_hkl_hexagonal(hkl, a, c)
        elif cell.crystal_system == 'monoclinic':
            return d_hkl_monoclinic(hkl, a, b, c, beta)
        elif cell.crystal_system == 'triclinic':
            return d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma)
    else:
        if alpha == beta == gamma == 90.:
            a_, b_, c_ = sorted([a, b, c])
            if a_ == b_ == c_:  # cubic
                return d_hkl_cubic(hkl, a_)
            elif a_ == b_:  # tetragonal
                return d_hkl_tetragonal(hkl, a_, c_)
            else:  # orthorhombic
                return d_hkl_orthorhombic(hkl, a_, b_, c_)
        elif alpha == beta == gamma and a == b == c:  # rhombohedral
            return d_hkl_rhombohedral(hkl, a, alpha)
        elif alpha + beta + gamma == 300.:  # hexagonal (90+90+120)
            if a == b and alpha == beta == 90. and gamma == 120.:
                return d_hkl_hexagonal(hkl, a, c)
            elif a == c and alpha == gamma == 90. and beta == 120.:
                return d_hkl_hexagonal(hkl, a, b)
            elif b == c and beta == gamma == 90. and alpha == 120.:
                return d_hkl_hexagonal(hkl, b, a)
            else:
                return d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma)
        elif alpha == gamma == 90.:  # monoclinic
            return d_hkl_monoclinic(hkl, a, b, c, beta)
        elif alpha == beta == 90.:
            return d_hkl_monoclinic(hkl, a, c, b, gamma)
        elif beta == gamma == 90.:
            return d_hkl_monoclinic(hkl, b, a, c, alpha)
        else:
            return d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma)


def d_hkl_cubic(hkl, a):
    # 1 / d**2 = (h**2 + k**2 + l**2) / a**2
    return sqrt(a**2 / sum([*map(lambda x: x**2, hkl)]))


def d_hkl_tetragonal(hkl, a, c):
    h, k, l = hkl
    d_star_squared = (h**2 + k**2) / a**2 + l**2 / c**2
    return sqrt(d_star_squared)


def d_hkl_orthorhombic(hkl, a, b, c):
    h, k, l = hkl
    d_star_squared = h**2 / a**2 + k**2 / b**2 + l**2 / c**2
    return sqrt(d_star_squared)


def d_hkl_rhombohedral(hkl, a, alpha):
    h, k, l = hkl
    d_star_squared = ((h**2 + k**2 + l**2) * sin_d(alpha)**2
                      + 2 * (h * k + k * l + l * h) * (cos_d(alpha)**2 - cos_d(alpha))) / \
                     (a**2 * (1 - 3 * cos_d(alpha)**2 + 2 * cos_d(alpha)**3))
    return sqrt(d_star_squared)


def d_hkl_hexagonal(hkl, a, c):
    h, k, l = hkl
    d_star_squared = 4 * (h**2 + h * k + k**2) / (3 * a**2) + l**2 / c**2
    return sqrt(d_star_squared)


def d_hkl_monoclinic(hkl, a, b, c, beta):
    h, k, l = hkl
    d_star_squared = (1 / sin_d(beta)**2) * (h**2 / a**2 + (k**2 * sin_d(beta)**2) / b**2 + l**2 / c**2
                                             - 2 * h * l * cos_d(beta) / (a * c))
    return sqrt(d_star_squared)


def d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma):
    h, k, l = hkl
    s11 = b**2 * c**2 * sin_d(alpha)**2
    s22 = a**2 * c**2 * sin_d(beta)**2
    s33 = a**2 * b**2 * sin_d(gamma)**2
    s12 = a * b * c**2 * (cos_d(alpha) * cos_d(beta) - cos_d(gamma))
    s23 = a**2 * b * c * (cos_d(beta) * cos_d(gamma) - cos_d(alpha))
    s13 = a * b**2 * c * (cos_d(alpha) * cos_d(gamma) - cos_d(beta))
    V = a * b * c * sqrt(1 - cos_d(alpha)**2 - cos_d(beta)**2 - cos_d(gamma)**2
                         + 2 * cos_d(alpha) * cos_d(beta) * cos_d(gamma))

    d_star_squared = (1 / V**2) * (s11 * h**2 + s22 * k**2 + s33 * l**2
                                   + 2 * s12 * h * k + 2 * s23 * k * l + 2 * s13 * h * l)
    return sqrt(d_star_squared)
