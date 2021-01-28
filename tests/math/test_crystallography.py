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

    @pytest.mark.parametrize('hkl', [(1, 0, 0), (2, -3, 4)])
    class TestDhklCrystalSystemDetermination:

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 1, 1, 90, 90, 90)])
        def test_dhkl_as_cubic(self, hkl, a, b, c, alpha, beta, gamma):
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_cubic(hkl, a))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 1, 2, 90, 90, 90),
                                                                 (1, 2, 1, 90, 90, 90),
                                                                 (2, 1, 1, 90, 90, 90)])
        def test_dhkl_as_tetragonal(self, hkl, a, b, c, alpha, beta, gamma):
            a_, b_, c_ = sorted((a, b, c))
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_tetragonal(hkl, a=a_, c=c_))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 2, 3, 90, 90, 90),
                                                                 (1, 3, 2, 90, 90, 90),
                                                                 (3, 2, 1, 90, 90, 90)])
        def test_dhkl_as_orthorhombic(self, hkl, a, b, c, alpha, beta, gamma):
            a_, b_, c_ = sorted((a, b, c))
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_orthorhombic(hkl, a=a_, b=b_, c=c_))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 1, 1, 60, 60, 60)])
        def test_dhkl_as_rhombohedral(self, hkl, a, b, c, alpha, beta, gamma):
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_rhombohedral(hkl, a, alpha))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 1, 2, 90, 90, 120),
                                                                 (1, 2, 1, 90, 120, 90),
                                                                 (2, 1, 1, 120, 90, 90)])
        def test_dhkl_as_hexagonal(self, hkl, a, b, c, alpha, beta, gamma):
            a_, b_, c_ = sorted((a, b, c))
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_hexagonal(hkl, a_, c_))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 2, 3, 90, 110, 90),
                                                                 (1, 2, 3, 110, 90, 90),
                                                                 (1, 2, 3, 90, 90, 110)])
        def test_dhkl_as_monoclinic(self, hkl, a, b, c, alpha, beta, gamma):
            angles = [alpha, beta, gamma]
            beta_index = [angles.index(angle) for angle in angles if angle != 90][0]

            a_, b_, c_ = a, b, c
            beta_ = angles[beta_index]
            if beta_index == 0:
                a_, b_, c_ = b_, a_, c_
            elif beta_index == 2:
                a_, b_, c_ = a_, c_, b_
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == approx(d_hkl_monoclinic(hkl, a_, b_, c_, beta_))

        @pytest.mark.parametrize('a, b, c, alpha, beta, gamma', [(1, 2, 3, 80, 100, 120),  # 80+100+120=300
                                                                 (1, 2, 3, 75, 85, 95)])
        def test_dhkl_as_triclinic(self, hkl, a, b, c, alpha, beta, gamma):
            assert d_hkl(hkl, (a, b, c, alpha, beta, gamma)) == \
                   approx(d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma))

    class TestDhklPerCrystalSystem:
        @pytest.mark.parametrize('hkl, a, d', [((1, 0, 0), 1, 1),
                                               ((2, 2, 1), 3, 1)])
        def test_cubic(self, hkl, a, d):
            assert d_hkl_cubic(hkl, a) == approx(d)

        @pytest.mark.parametrize('hkl, a, c, d', [((1, 0, 0), 1, 2, 1),
                                                  ((0, 0, 1), 1, 2, 0.5)])
        def test_tetragonal(self, hkl, a, c, d):
            assert d_hkl_tetragonal(hkl, a, c) == approx(d)

        @pytest.mark.parametrize('hkl, a, b, c, d', [((1, 0, 0), 1, 2, 3, 1),
                                                     ((1, 2, 3), 1, 2, 3, sqrt(3))])
        def test_orthorhombic(self, hkl, a, b, c, d):
            assert d_hkl_orthorhombic(hkl, a, b, c) == approx(d)

        @pytest.mark.parametrize('hkl, a, alpha, d', [((1, 0, 0), 1, 60, sqrt(3/2)),
                                                      ((1, 1, 0), 1, 60, sqrt(2))])
        def test_rhombohedral(self, hkl, a, alpha, d):
            assert d_hkl_rhombohedral(hkl, a, alpha) == approx(d)

        @pytest.mark.parametrize('hkl, a, c, d', [((1, 0, 0), 1, 2, 2/sqrt(3)),
                                                  ((1, 1, 0), 1, 2, 2)])
        def test_hexagonal(self, hkl, a, c, d):
            assert d_hkl_hexagonal(hkl, a, c) == approx(d)

        @pytest.mark.parametrize('hkl, a, b, c, beta, d', [((1, 0, 0), 1, 2, 3, 120, 2/sqrt(3)),
                                                           ((0, 1, 0), 1, 2, 3, 120, 0.5)])
        def test_monoclinic(self, hkl, a, b, c, beta, d):
            assert d_hkl_monoclinic(hkl, a, b, c, beta) == approx(d)

        @pytest.mark.parametrize('hkl, a, b, c, alpha, beta, gamma, d', [((1, 0, 0), 1, 2, 3, 80, 100, 120, 1.16073095),
                                                                         ((2, 1, 1), 1, 2, 3, 75, 85, 95, 2.11156385)])
        def test_triclinic(self, hkl, a, b, c, alpha, beta, gamma, d):
            assert d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma) == approx(d, abs=1.e-8)

    # @pytest.mark.parametrize('hkl', [(1, 0, 0), (2, -3, 4)])
    class TestDhklCellReduction:
        @pytest.mark.parametrize('hkl, a, c', [((2, 2, 1), 3, 3)])
        def test_tetragonal_reduction_to_cubic(self, hkl, a, c):
            assert d_hkl_tetragonal(hkl, a, c) == approx(d_hkl_cubic(hkl, a))

        @pytest.mark.parametrize('hkl, a, b, c', [((2, 2, 1), 3, 3, 3)])
        def test_orthorhombic_reduction_to_cubic(self, hkl, a, b, c):
            assert d_hkl_orthorhombic(hkl, a, b, c) == approx(d_hkl_cubic(hkl, a))

        @pytest.mark.parametrize('hkl, a, b, c, beta', [((2, 3, 4), 1, 2, 3, 90)])
        def test_monoclinic_reduction_to_orthorhombic(self, hkl, a, b, c, beta):
            assert d_hkl_monoclinic(hkl, a, b, c, beta) == d_hkl_orthorhombic(hkl, a, b, c)

        @pytest.mark.parametrize('hkl, a, b, c, alpha, beta, gamma', [((2, 2, 1), 1, 2, 3, 90, 110, 90)])
        def test_triclinic_reduction_to_monoclinic(self, hkl, a, b, c, alpha, beta, gamma):
            assert d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma) == approx(d_hkl_monoclinic(hkl, a, b, c, beta))

        @pytest.mark.parametrize('hkl, a, b, c, alpha, beta, gamma', [((2, 2, 1), 3, 3, 3, 90, 90, 90)])
        def test_triclinic_reduction_to_cubic(self, hkl, a, b, c, alpha, beta, gamma):
            assert d_hkl_triclinic(hkl, a, b, c, alpha, beta, gamma) == approx(d_hkl_cubic(hkl, a))
