from math import pi, sqrt

import pytest
from pytest import approx

from xtl.math.trig import sin, sin_d, sin_r, asin, asin_d, asin_r
from xtl.math.trig import cos, cos_d, cos_r, acos, acos_d, acos_r
from xtl.math.trig import tan, tan_d, tan_r, atan, atan_d, atan_r


class TestSinAll:

    class TestSin:
        def test_sin(self):
            assert sin(90, 'd') == 1
            assert sin(pi/2) == 1

        def test_sin_d(self):
            assert sin_d(0) == 0
            assert sin_d(90) == 1

        def test_sin_r(self):
            assert sin_r(0) == 0
            assert sin_r(pi/2) == 1

    class TestASin:
        def test_asin(self):
            assert asin(1, 'd') == 90
            assert asin(1) == pi/2

        def test_asin_d(self):
            assert asin_d(0) == 0
            assert asin_d(1) == 90

        def test_asin_r(self):
            assert asin_r(0) == 0
            assert asin_r(1) == pi/2


class TestCosAll:

    class TestCos:
        def test_cos(self):
            assert cos(90, 'd') == approx(0)
            assert cos(pi / 2) == approx(0)

        def test_cos_d(self):
            assert cos_d(0) == 1
            assert cos_d(90) == approx(0)

        def test_cos_r(self):
            assert cos_r(0) == 1
            assert cos_r(pi / 2) == approx(0)

    class TestACos:
        def test_acos(self):
            assert acos(1, 'd') == approx(0)
            assert acos(0) == pi / 2

        def test_acos_d(self):
            assert acos_d(0) == 90
            assert acos_d(1) == approx(0)

        def test_acos_r(self):
            assert acos_r(0) == pi / 2
            assert acos_r(1) == approx(0)


class TestTanAll:

    class TestTan:
        def test_tan(self):
            assert tan(45, 'd') == approx(1)
            assert tan(pi / 4) == approx(1)

        def test_tan_d(self):
            assert tan_d(0) == 0
            assert tan_d(45) == approx(1)

        def test_tan_r(self):
            assert tan_r(0) == 0
            assert tan_r(pi / 4) == approx(1)

    class TestATan:
        def test_acos(self):
            assert atan(1, 'd') == approx(45)
            assert atan(sqrt(3)) == pi / 3

        def test_atan_d(self):
            assert atan_d(0) == 0
            assert atan_d(1) == approx(45)

        def test_atan_r(self):
            assert atan_r(0) == 0
            assert atan_r(1) == approx(pi / 4)

