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
