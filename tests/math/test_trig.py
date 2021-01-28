from math import pi, sqrt

import pytest
from pytest import approx

from xtl.math.trig import sin, sin_d, sin_r, asin, asin_d, asin_r
from xtl.math.trig import cos, cos_d, cos_r, acos, acos_d, acos_r
from xtl.math.trig import tan, tan_d, tan_r, atan, atan_d, atan_r


class TestSinAll:

    class TestSin:
        @pytest.mark.parametrize('angle, value, mode', [(90, 1, 'd'), (pi / 2, 1, 'r')])
        def test_sin(self, angle, value, mode):
            assert sin(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (90, 1)])
        def test_sin_d(self, angle, value):
            assert sin_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (pi / 2, 1)])
        def test_sin_r(self, angle, value):
            assert sin_r(angle) == approx(value)

    class TestASin:
        @pytest.mark.parametrize('angle, value, mode', [(1, 90, 'd'), (1, pi / 2, 'r')])
        def test_asin(self, angle, value, mode):
            assert asin(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (1, 90)])
        def test_asin_d(self, angle, value):
            assert asin_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (1, pi / 2)])
        def test_asin_r(self, angle, value):
            assert asin_r(angle) == approx(value)


class TestCosAll:

    class TestCos:
        @pytest.mark.parametrize('angle, value, mode', [(90, 0, 'd'), (pi / 2, 0, 'r')])
        def test_cos(self, angle, value, mode):
            assert cos(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 1), (90, 0)])
        def test_cos_d(self, angle, value):
            assert cos_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 1), (pi / 2, 0)])
        def test_cos_r(self, angle, value):
            assert cos_r(angle) == approx(value)

    class TestACos:
        @pytest.mark.parametrize('angle, value, mode', [(1, 0, 'd'), (0, pi / 2, 'r')])
        def test_acos(self, angle, value, mode):
            assert acos(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 90), (1, 0)])
        def test_acos_d(self, angle, value):
            assert acos_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, pi / 2), (1, 0)])
        def test_acos_r(self, angle, value):
            assert acos_r(angle) == approx(value)


class TestTanAll:

    class TestTan:
        @pytest.mark.parametrize('angle, value, mode', [(45, 1, 'd'), (pi / 4, 1, 'r')])
        def test_tan(self, angle, value, mode):
            assert tan(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (45, 1)])
        def test_tan_d(self, angle, value):
            assert tan_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (pi / 4, 1)])
        def test_tan_r(self, angle, value):
            assert tan_r(angle) == approx(value)

    class TestATan:
        @pytest.mark.parametrize('angle, value, mode', [(1, 45, 'd'), (sqrt(3), pi / 3, 'r')])
        def test_atan(self, angle, value, mode):
            assert atan(angle, mode) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (1, 45)])
        def test_atan_d(self, angle, value):
            assert atan_d(angle) == approx(value)

        @pytest.mark.parametrize('angle, value', [(0, 0), (1, pi / 4)])
        def test_atan_r(self, angle, value):
            assert atan_r(angle) == approx(value)

