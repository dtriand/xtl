import pytest
from pytest import approx

from xtl.math.crystallography import *


class TestConversions:

    def test_d_spacing_to_ttheta(self):
        assert d_spacing_to_ttheta(d=1, wavelength=1.3) == approx(81.0832, abs=1.e-04)
        assert d_spacing_to_ttheta(d=1, wavelength=1.3, mode='r') == approx(1.4151, abs=1.e-04)

    def test_ttheta_to_d_spacing(self):
        assert ttheta_to_d_spacing(ttheta=81.0832, wavelength=1.3) == approx(1, abs=1.e-04)
        assert ttheta_to_d_spacing(ttheta=1.4151, wavelength=1.3, mode='r') == approx(1, abs=1.e-04)


class TestDhkl:

    def test_dhkl(self):
        hkl = (1, 0, 0)
        assert d_hkl(hkl, (1, 1, 1, 90, 90, 90)) == approx(d_hkl_cubic(hkl, a=1))

        assert d_hkl(hkl, (1, 1, 2, 90, 90, 90)) == approx(d_hkl_tetragonal(hkl, a=1, c=2))
        assert d_hkl(hkl, (1, 2, 1, 90, 90, 90)) == approx(d_hkl_tetragonal(hkl, a=1, c=2))
        assert d_hkl(hkl, (2, 1, 1, 90, 90, 90)) == approx(d_hkl_tetragonal(hkl, a=1, c=2))

        assert d_hkl(hkl, (1, 2, 3, 90, 90, 90)) == approx(d_hkl_orthorhombic(hkl, a=1, b=2, c=3))
        assert d_hkl(hkl, (1, 3, 2, 90, 90, 90)) == approx(d_hkl_orthorhombic(hkl, a=1, b=2, c=3))
        assert d_hkl(hkl, (3, 2, 1, 90, 90, 90)) == approx(d_hkl_orthorhombic(hkl, a=1, b=2, c=3))

        assert d_hkl(hkl, (1, 1, 1, 60, 60, 60)) == approx(d_hkl_rhombohedral(hkl, a=1, alpha=60))

        assert d_hkl(hkl, (1, 1, 2, 90, 90, 120)) == approx(d_hkl_hexagonal(hkl, a=1, c=2))
        assert d_hkl(hkl, (1, 2, 1, 90, 120, 90)) == approx(d_hkl_hexagonal(hkl, a=1, c=2))
        assert d_hkl(hkl, (2, 1, 1, 120, 90, 90)) == approx(d_hkl_hexagonal(hkl, a=1, c=2))

        assert d_hkl(hkl, (1, 2, 3, 90, 110, 90)) == approx(d_hkl_monoclinic(hkl, a=1, b=2, c=3, beta=110))
        assert d_hkl(hkl, (1, 2, 3, 110, 90, 90)) == approx(d_hkl_monoclinic(hkl, a=2, b=1, c=3, beta=110))
        assert d_hkl(hkl, (1, 2, 3, 90, 90, 110)) == approx(d_hkl_monoclinic(hkl, a=1, b=3, c=2, beta=110))

        assert d_hkl(hkl, (1, 2, 3, 80, 100, 120)) == \
               approx(d_hkl_triclinic(hkl, a=1, b=2, c=3, alpha=80, beta=100, gamma=120))  # 80+100+120=300
        assert d_hkl(hkl, (1, 2, 3, 75, 85, 95)) == \
               approx(d_hkl_triclinic(hkl, a=1, b=2, c=3, alpha=75, beta=85, gamma=95))

    def test_cubic(self):
        assert d_hkl_cubic((1, 0, 0), a=1) == approx(1)
        assert d_hkl_cubic((2, 2, 1), a=3) == approx(1)

    def test_tetragonal(self):
        assert d_hkl_tetragonal((1, 0, 0), a=1, c=2) == approx(1)
        assert d_hkl_tetragonal((0, 0, 1), a=1, c=2) == approx(0.5)

        assert d_hkl_tetragonal((2, 2, 1), a=3, c=3) == approx(d_hkl_cubic((2, 2, 1), a=3))

    def test_orthorhombic(self):
        assert d_hkl_orthorhombic((1, 0, 0), a=1, b=2, c=3) == approx(1)
        assert d_hkl_orthorhombic((1, 2, 3), a=1, b=2, c=3) == approx(sqrt(3))

        assert d_hkl_orthorhombic((2, 2, 1), a=3, b=3, c=3) == approx(d_hkl_cubic((2, 2, 1), a=3))

    def test_rhombohedral(self):
        assert d_hkl_rhombohedral((1, 0, 0), a=1, alpha=60) == approx(sqrt(3 / 2))
        assert d_hkl_rhombohedral((1, 1, 0), a=1, alpha=60) == approx(sqrt(2))

    def test_hexagonal(self):
        assert d_hkl_hexagonal((1, 0, 0), a=1, c=2) == approx(2 / sqrt(3))
        assert d_hkl_hexagonal((1, 1, 0), a=1, c=2) == approx(2)

    def test_monoclinic(self):
        assert d_hkl_monoclinic((1, 0, 0), a=1, b=2, c=3, beta=120) == approx(2 / sqrt(3))
        assert d_hkl_monoclinic((0, 1, 0), a=1, b=2, c=3, beta=120) == approx(0.5)

        assert d_hkl_monoclinic((2, 3, 4), a=1, b=2, c=3, beta=90) == \
               approx(d_hkl_orthorhombic((2, 3, 4), a=1, b=2, c=3))

    def test_triclinic(self):
        assert d_hkl_triclinic((1, 0, 0), a=1, b=2, c=3, alpha=80, beta=100, gamma=120) == approx(1.16073095, abs=1.e-8)
        assert d_hkl_triclinic((2, 1, 1), a=1, b=2, c=3, alpha=75, beta=85, gamma=95) == approx(2.11156385, abs=1.e-8)

        assert d_hkl_triclinic((2, 2, 1), a=1, b=2, c=3, alpha=90, beta=110, gamma=90) == \
               approx(d_hkl_monoclinic((2, 2, 1), a=1, b=2, c=3, beta=110))
        assert d_hkl_triclinic((2, 2, 1), a=3, b=3, c=3, alpha=90, beta=90, gamma=90) == \
               approx(d_hkl_cubic((2, 2, 1), 3))
